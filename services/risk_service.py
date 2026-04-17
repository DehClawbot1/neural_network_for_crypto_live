"""
services/risk_service.py
─────────────────────────
Independent risk authority.

Rules
─────
1. RiskService can ONLY be called — never bypassed.
   Execution cannot proceed unless approve_entry() returns approved=True.
2. Risk decisions are based on portfolio state and hard limits only.
   They are NEVER based on model confidence scores.
3. When a trade is vetoed, RiskVetoError is raised.
   The caller MUST NOT catch RiskVetoError and continue silently.
4. All thresholds are configurable via env vars.
   Hard minimums are enforced in code regardless of env vars.

Usage
─────
    from services.risk_service import RiskService
    from services.types import RiskVetoError

    risk = RiskService(logs_dir="logs")
    try:
        decision = risk.approve_entry(signal, portfolio)
    except RiskVetoError as e:
        log_skip(signal, reason=e.reason, checks=e.checks_failed)
        continue  # explicit, not silent
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from services.types import (
    PortfolioState,
    RiskDecision,
    RiskOutcome,
    RiskVetoError,
)

logger = logging.getLogger(__name__)

# Hard limits — these cannot be overridden by env vars.
# They exist to prevent catastrophic single-session loss.
_HARD_MAX_SINGLE_POSITION_USDC = 500.0   # no single position > $500
_HARD_MAX_TOTAL_EXPOSURE_USDC  = 2000.0  # no total exposure > $2000
_HARD_MAX_DRAWDOWN_USDC        = 300.0   # halt if session loss > $300
_HARD_MIN_PRICE                = 0.02    # never buy below 2¢
_HARD_MAX_PRICE                = 0.97    # never buy above 97¢ (near-resolved)

# Per-family capital isolation. Each family has an independent notional pool
# so that a bad day in BTC can never drain weather's capital (or vice versa).
# Defaults split 60/40 of the hard total exposure; override with env vars.
_HARD_MAX_BTC_NOTIONAL_USDC     = 1200.0
_HARD_MAX_WEATHER_NOTIONAL_USDC = 800.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return float(default)


class RiskService:
    """
    Independent risk authority.

    This service reads portfolio state and signal metadata to decide whether
    a proposed trade is safe. It NEVER reads model scores.

    All soft limits are configurable via environment variables.
    Hard limits in this file cannot be overridden.
    """

    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = Path(logs_dir)
        # Optional dynamic per-family caps from the PortfolioAllocator.
        # Intersected with the static hard/soft caps — can only TIGHTEN,
        # never loosen. Set via set_family_pool_overrides() each cycle.
        self._family_pool_overrides: dict[str, float] = {}

    def set_family_pool_overrides(self, overrides: dict[str, float] | None) -> None:
        """
        Install per-family dynamic ceilings (in USDC) from the allocator.
        A family mapping to 0.0 is a hard stop for this cycle.
        Pass None or {} to clear.
        """
        cleaned: dict[str, float] = {}
        for k, v in (overrides or {}).items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f < 0:
                f = 0.0
            cleaned[str(k).lower()] = f
        self._family_pool_overrides = cleaned

    def _family_override_cap(self, market_family: str) -> float | None:
        if not self._family_pool_overrides:
            return None
        fam = "weather_temperature" if market_family.startswith("weather") else "btc"
        return self._family_pool_overrides.get(fam)

    # ── Primary API ───────────────────────────────────────────────────────────

    def approve_entry(
        self,
        signal: dict,
        portfolio: PortfolioState,
        *,
        raise_on_veto: bool = True,
    ) -> RiskDecision:
        """
        Approve or veto a proposed entry.

        Parameters
        ──────────
        signal       : signal row dict (must contain token_id, current_price, market_family)
        portfolio    : current PortfolioState snapshot
        raise_on_veto: if True (default), raises RiskVetoError on veto.
                       Set to False only in shadow/simulation mode.

        Returns
        ───────
        RiskDecision — always returned when approved.
        Raises RiskVetoError — when vetoed and raise_on_veto=True.
        """
        checks_run: list[str] = []
        checks_failed: list[str] = []

        token_id     = str(signal.get("token_id") or "")
        market_fam   = str(signal.get("market_family") or "btc").lower()
        price        = float(signal.get("current_price") or signal.get("fill_price") or 0.0)
        proposed_usdc = float(signal.get("proposed_size_usdc") or signal.get("size_usdc") or 0.0)

        # ── 1. Price bounds (hard) ─────────────────────────────────────────
        checks_run.append("price_bounds")
        if price < _HARD_MIN_PRICE:
            checks_failed.append("price_below_hard_min")
        if price > _HARD_MAX_PRICE:
            checks_failed.append("price_above_hard_max")

        # ── 2. Kill switch ─────────────────────────────────────────────────
        checks_run.append("kill_switch")
        if os.getenv("SESSION_KILL_SWITCH_ACTIVE", "false").strip().lower() in {"1", "true", "yes", "on"}:
            checks_failed.append("kill_switch_active")

        # ── 3. Session drawdown (hard) ─────────────────────────────────────
        checks_run.append("session_drawdown")
        if portfolio["max_drawdown_session"] >= _HARD_MAX_DRAWDOWN_USDC:
            checks_failed.append("hard_drawdown_limit_reached")
        soft_dd_limit = _env_float("RISK_MAX_SESSION_DRAWDOWN_USDC", 150.0)
        if portfolio["max_drawdown_session"] >= soft_dd_limit:
            checks_failed.append("soft_drawdown_limit_reached")

        # ── 4. Total exposure (hard) ───────────────────────────────────────
        checks_run.append("total_exposure")
        if portfolio["open_notional_usdc"] + proposed_usdc > _HARD_MAX_TOTAL_EXPOSURE_USDC:
            checks_failed.append("hard_total_exposure_exceeded")
        soft_exp_limit = _env_float("RISK_MAX_TOTAL_EXPOSURE_USDC", 500.0)
        if portfolio["open_notional_usdc"] + proposed_usdc > soft_exp_limit:
            checks_failed.append("soft_total_exposure_exceeded")

        # ── 5. Single position size (hard) ────────────────────────────────
        checks_run.append("position_size")
        if proposed_usdc > _HARD_MAX_SINGLE_POSITION_USDC:
            checks_failed.append("hard_single_position_size")

        # ── 6. Available balance ───────────────────────────────────────────
        checks_run.append("available_balance")
        if portfolio["available_usdc"] < proposed_usdc:
            checks_failed.append("insufficient_balance")
        min_reserve = _env_float("RISK_MIN_USDC_RESERVE", 5.0)
        if portfolio["available_usdc"] - proposed_usdc < min_reserve:
            checks_failed.append("below_min_reserve")

        # ── 7. Family-specific exposure (position count + capital pool) ────
        # Capital isolation: each family has its own notional pool, enforced
        # as a hard cap in code plus a soft cap via env. A loss spiral in one
        # family cannot consume the other family's capital.
        checks_run.append("family_exposure")
        max_btc_pos = int(_env_float("RISK_MAX_BTC_POSITIONS", 3.0))
        max_weather_pos = int(_env_float("RISK_MAX_WEATHER_POSITIONS", 2.0))
        # Dynamic allocator override (if any). A zero override is a hard stop.
        _override_cap = self._family_override_cap(market_fam)
        if _override_cap is not None:
            if _override_cap <= 0.0:
                checks_failed.append("family_allocator_stopped_out")
            else:
                _family_open = (
                    float(portfolio.get("weather_open_notional", 0.0) or 0.0)
                    if market_fam.startswith("weather")
                    else float(portfolio.get("btc_open_notional", 0.0) or 0.0)
                )
                if _family_open + proposed_usdc > _override_cap:
                    checks_failed.append("family_allocator_cap_exceeded")

        is_weather = market_fam.startswith("weather_temperature")
        if is_weather:
            if portfolio["weather_open_count"] >= max_weather_pos:
                checks_failed.append("weather_family_cap_reached")
            weather_notional = float(portfolio.get("weather_open_notional", 0.0) or 0.0)
            hard_weather = _env_float(
                "RISK_HARD_MAX_WEATHER_NOTIONAL_USDC", _HARD_MAX_WEATHER_NOTIONAL_USDC
            )
            if weather_notional + proposed_usdc > hard_weather:
                checks_failed.append("weather_family_notional_exceeded")
            soft_weather = _env_float("RISK_SOFT_MAX_WEATHER_NOTIONAL_USDC", 400.0)
            if weather_notional + proposed_usdc > soft_weather:
                checks_failed.append("weather_family_notional_soft_exceeded")
        else:
            if portfolio["btc_open_count"] >= max_btc_pos:
                checks_failed.append("btc_family_cap_reached")
            btc_notional = float(portfolio.get("btc_open_notional", 0.0) or 0.0)
            hard_btc = _env_float(
                "RISK_HARD_MAX_BTC_NOTIONAL_USDC", _HARD_MAX_BTC_NOTIONAL_USDC
            )
            if btc_notional + proposed_usdc > hard_btc:
                checks_failed.append("btc_family_notional_exceeded")
            soft_btc = _env_float("RISK_SOFT_MAX_BTC_NOTIONAL_USDC", 600.0)
            if btc_notional + proposed_usdc > soft_btc:
                checks_failed.append("btc_family_notional_soft_exceeded")

        # ── 8. Reconciliation rate guard ──────────────────────────────────
        # If >60% recent closes were operational/reconciliation, data quality
        # is too low to trust signal quality — reduce allowed size.
        checks_run.append("reconciliation_rate")
        rec_rate = portfolio.get("reconciliation_rate", 0.0)
        rec_rate_halt = _env_float("RISK_RECONCILIATION_HALT_RATE", 0.80)
        if rec_rate >= rec_rate_halt:
            checks_failed.append("reconciliation_rate_too_high")

        # ── Compute approved size ─────────────────────────────────────────
        approved_size = self._compute_approved_size(
            proposed_usdc, portfolio, rec_rate, market_family=market_fam
        )

        approved = len(checks_failed) == 0
        reason = checks_failed[0] if checks_failed else "all_checks_passed"

        decision: RiskDecision = {
            "outcome":       RiskOutcome.APPROVED if approved else RiskOutcome.VETOED,
            "approved":      approved,
            "reason":        reason,
            "max_size_usdc": approved_size if approved else 0.0,
            "checks_run":    checks_run,
            "checks_failed": checks_failed,
        }

        if not approved and raise_on_veto:
            raise RiskVetoError(
                reason=reason,
                checks_failed=checks_failed,
                context={
                    "token_id": token_id,
                    "proposed_usdc": proposed_usdc,
                    "price": price,
                    "open_notional": portfolio["open_notional_usdc"],
                    "available_usdc": portfolio["available_usdc"],
                },
            )

        return decision

    def approve_size(
        self,
        proposed_usdc: float,
        portfolio: PortfolioState,
        *,
        market_family: str = "btc",
    ) -> float:
        """
        Return the largest approved size ≤ proposed_usdc given current portfolio.
        Never raises — callers use this for sizing, not for entry approval.
        """
        rec_rate = portfolio.get("reconciliation_rate", 0.0)
        size = self._compute_approved_size(
            proposed_usdc, portfolio, rec_rate, market_family=market_family
        )
        return min(size, proposed_usdc)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_approved_size(
        self,
        proposed: float,
        portfolio: PortfolioState,
        rec_rate: float,
        *,
        market_family: str = "btc",
    ) -> float:
        """
        Compute the maximum safe position size given portfolio state.
        Applies hard caps, then soft reductions.
        """
        size = proposed

        # Hard single-position cap
        size = min(size, _HARD_MAX_SINGLE_POSITION_USDC)

        # Remaining capacity before hard total-exposure cap
        hard_remaining = max(0.0, _HARD_MAX_TOTAL_EXPOSURE_USDC - portfolio["open_notional_usdc"])
        size = min(size, hard_remaining)

        # Soft total-exposure cap
        soft_exp = _env_float("RISK_MAX_TOTAL_EXPOSURE_USDC", 500.0)
        soft_remaining = max(0.0, soft_exp - portfolio["open_notional_usdc"])
        size = min(size, soft_remaining)

        # Per-family pool cap (capital isolation)
        fam = str(market_family or "btc").lower()
        if fam.startswith("weather_temperature"):
            fam_open = float(portfolio.get("weather_open_notional", 0.0) or 0.0)
            hard_fam = _env_float(
                "RISK_HARD_MAX_WEATHER_NOTIONAL_USDC", _HARD_MAX_WEATHER_NOTIONAL_USDC
            )
            soft_fam = _env_float("RISK_SOFT_MAX_WEATHER_NOTIONAL_USDC", 400.0)
        else:
            fam_open = float(portfolio.get("btc_open_notional", 0.0) or 0.0)
            hard_fam = _env_float(
                "RISK_HARD_MAX_BTC_NOTIONAL_USDC", _HARD_MAX_BTC_NOTIONAL_USDC
            )
            soft_fam = _env_float("RISK_SOFT_MAX_BTC_NOTIONAL_USDC", 600.0)
        size = min(size, max(0.0, hard_fam - fam_open))
        size = min(size, max(0.0, soft_fam - fam_open))

        # Available balance cap (with reserve)
        reserve = _env_float("RISK_MIN_USDC_RESERVE", 5.0)
        size = min(size, max(0.0, portfolio["available_usdc"] - reserve))

        # Reconciliation penalty: if rec_rate is elevated, reduce size
        rec_rate_penalty_floor = _env_float("RISK_RECONCILIATION_PENALTY_FLOOR", 0.50)
        if rec_rate >= rec_rate_penalty_floor:
            # Linear reduction: at 50% rec_rate → 80% size; at 80% → 20% size
            fraction = 1.0 - (rec_rate - rec_rate_penalty_floor) / max(0.80 - rec_rate_penalty_floor, 0.01)
            size = size * max(0.10, min(1.0, fraction))

        return round(max(0.0, size), 4)

    def hard_limits(self) -> dict:
        """Return a summary of hard limits for logging/audit."""
        return {
            "hard_max_single_position_usdc": _HARD_MAX_SINGLE_POSITION_USDC,
            "hard_max_total_exposure_usdc":  _HARD_MAX_TOTAL_EXPOSURE_USDC,
            "hard_max_btc_notional_usdc":    _HARD_MAX_BTC_NOTIONAL_USDC,
            "hard_max_weather_notional_usdc": _HARD_MAX_WEATHER_NOTIONAL_USDC,
            "hard_max_drawdown_usdc":        _HARD_MAX_DRAWDOWN_USDC,
            "hard_min_price":                _HARD_MIN_PRICE,
            "hard_max_price":                _HARD_MAX_PRICE,
        }


if __name__ == "__main__":
    from services.types import RiskVetoError, PortfolioState

    risk = RiskService()

    good_portfolio: PortfolioState = {
        "open_count": 1,
        "open_notional_usdc": 20.0,
        "btc_open_count": 1,
        "btc_open_notional": 20.0,
        "weather_open_count": 0,
        "weather_open_notional": 0.0,
        "unrealized_pnl": 0.5,
        "realized_pnl_session": 1.2,
        "available_usdc": 100.0,
        "max_drawdown_session": 5.0,
        "reconciliation_rate": 0.10,
    }

    # Should approve
    signal = {"token_id": "tok1", "current_price": 0.55, "market_family": "btc",
              "proposed_size_usdc": 10.0}
    decision = risk.approve_entry(signal, good_portfolio)
    assert decision["approved"], f"Expected approval: {decision}"
    print(f"  Approved: max_size={decision['max_size_usdc']}")

    # Should veto: price too low
    signal2 = {**signal, "current_price": 0.01}
    try:
        risk.approve_entry(signal2, good_portfolio)
        assert False, "Should have raised"
    except RiskVetoError as e:
        print(f"  RiskVetoError correctly raised: {e.reason} ({e.checks_failed})")

    # Kill switch
    import os
    os.environ["SESSION_KILL_SWITCH_ACTIVE"] = "true"
    try:
        risk.approve_entry(signal, good_portfolio)
        assert False
    except RiskVetoError as e:
        assert "kill_switch_active" in e.checks_failed
        print(f"  Kill switch veto: {e.reason}")
    del os.environ["SESSION_KILL_SWITCH_ACTIVE"]

    print("risk_service self-test PASSED.")
