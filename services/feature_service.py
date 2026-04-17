"""
services/feature_service.py
─────────────────────────────
Pure, stateless feature scoring and signal sizing functions.

Rules
─────
1. Every function here is a pure transformation — no file I/O, no model calls,
   no side effects, no trading decisions.
2. Inputs are typed scalars or dicts. Outputs are typed scalars.
3. DataFaultError is raised if required fields are missing or uncoercible.
   Callers must not fall back to 0 — the candidate must be skipped.
4. SignalAction thresholds are configurable via env vars.
   The mapping from action → size multiplier is deterministic.
"""
from __future__ import annotations

import os
from typing import Optional

from services.types import DataFaultError, SignalAction


# ── Thresholds (soft, env-overridable) ───────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return float(default)


# Polymarket taker fee (round-trip cost on the way in)
_DEFAULT_TAKER_FEE_RATE = 0.01     # 1% one-way taker fee

# Action thresholds (can be overridden per session by env vars)
_CONFIDENCE_IGNORE_THRESHOLD     = lambda: _env_float("FEATURE_CONFIDENCE_IGNORE",     0.55)
_CONFIDENCE_LARGE_BUY_THRESHOLD  = lambda: _env_float("FEATURE_CONFIDENCE_LARGE_BUY",  0.68)
_CONFIDENCE_HIGH_CONV_THRESHOLD  = lambda: _env_float("FEATURE_CONFIDENCE_HIGH_CONV",  0.78)
_EDGE_IGNORE_THRESHOLD           = lambda: _env_float("FEATURE_EDGE_IGNORE",           0.02)
_EDGE_LARGE_BUY_THRESHOLD        = lambda: _env_float("FEATURE_EDGE_LARGE_BUY",        0.04)
_EV_IGNORE_THRESHOLD             = lambda: _env_float("FEATURE_EV_IGNORE",             0.005)

# Size multipliers by action
_SIZE_SMALL_BUY_FRAC    = lambda: _env_float("FEATURE_SIZE_SMALL_BUY_FRAC",    0.5)
_SIZE_LARGE_BUY_FRAC    = lambda: _env_float("FEATURE_SIZE_LARGE_BUY_FRAC",    1.0)
_SIZE_HIGH_CONV_FRAC    = lambda: _env_float("FEATURE_SIZE_HIGH_CONV_FRAC",    1.5)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_edge_score(p_tp_before_sl: float, expected_return: float) -> float:
    """
    Edge score = P(TP before SL) × expected return.

    Both inputs must be finite positive. Returns a scalar in [0, ∞).

    Raises
    ──────
    DataFaultError if either value is non-numeric or negative.
    """
    p = _require_non_negative_float(p_tp_before_sl, "p_tp_before_sl")
    r = _require_non_negative_float(expected_return, "expected_return")
    return round(p * r, 6)


def compute_ev_after_cost(
    expected_return: float,
    current_price: float,
    *,
    taker_fee_rate: float = _DEFAULT_TAKER_FEE_RATE,
) -> float:
    """
    Expected value after transaction costs.

    Formula: EV = expected_return - (taker_fee_rate / current_price)
    This expresses the fee as a fraction of the expected return at the
    current entry price — matching the Polymarket fee model.

    Returns a scalar (may be negative — callers should gate on EV > threshold).

    Raises
    ──────
    DataFaultError if current_price ≤ 0 or expected_return is non-numeric.
    """
    er = _require_non_negative_float(expected_return, "expected_return")
    if current_price <= 0:
        raise DataFaultError(
            f"current_price={current_price!r} is non-positive; cannot compute EV",
            field="current_price",
        )
    fee_drag = taker_fee_rate / current_price
    return round(er - fee_drag, 6)


def compute_action(
    confidence: float,
    edge_score: float,
    ev_after_cost: float,
) -> SignalAction:
    """
    Map model outputs to a discrete SignalAction.

    Decision tree (all conditions must hold for a BUY action):
    - If confidence < IGNORE → IGNORE
    - If edge_score < IGNORE_EDGE → IGNORE
    - If ev_after_cost < EV_IGNORE → IGNORE
    - If confidence ≥ HIGH_CONV and edge ≥ LARGE_BUY → HIGH_CONV_BUY
    - If confidence ≥ LARGE_BUY and edge ≥ LARGE_BUY → LARGE_BUY
    - Otherwise → SMALL_BUY

    Returns
    ───────
    SignalAction enum value.
    """
    conf_ign  = _CONFIDENCE_IGNORE_THRESHOLD()
    conf_lg   = _CONFIDENCE_LARGE_BUY_THRESHOLD()
    conf_hc   = _CONFIDENCE_HIGH_CONV_THRESHOLD()
    edge_ign  = _EDGE_IGNORE_THRESHOLD()
    edge_lg   = _EDGE_LARGE_BUY_THRESHOLD()
    ev_ign    = _EV_IGNORE_THRESHOLD()

    if confidence < conf_ign:
        return SignalAction.IGNORE
    if edge_score < edge_ign:
        return SignalAction.IGNORE
    if ev_after_cost < ev_ign:
        return SignalAction.IGNORE

    if confidence >= conf_hc and edge_score >= edge_lg:
        return SignalAction.HIGH_CONV_BUY
    if confidence >= conf_lg and edge_score >= edge_lg:
        return SignalAction.LARGE_BUY
    return SignalAction.SMALL_BUY


def compute_size_usdc(
    action: SignalAction,
    base_size_usdc: float,
    max_size_usdc: float,
    *,
    uncertainty_multiplier: float = 1.0,
) -> float:
    """
    Compute final position size from action and risk-approved max.

    The action multiplier scales base_size_usdc; an optional
    `uncertainty_multiplier` ∈ [0, 1] (typically from
    ConformalIntervalService.size_multiplier) further scales the result
    down for high-uncertainty predictions. Final size is then capped at
    max_size_usdc. IGNORE always returns 0.

    Parameters
    ──────────
    action                 : SignalAction from compute_action()
    base_size_usdc         : baseline size (e.g., RISK_BASE_SIZE_USDC)
    max_size_usdc          : ceiling from RiskService.approve_size()
    uncertainty_multiplier : [0, 1]; 1.0 disables (default). Smaller → smaller position.

    Returns
    ───────
    Float ≥ 0 rounded to 4 decimal places.
    """
    if action == SignalAction.IGNORE:
        return 0.0

    # Validate uncertainty multiplier rather than silently clamping — a
    # non-finite or out-of-range value means a bug upstream.
    try:
        um = float(uncertainty_multiplier)
    except (TypeError, ValueError) as exc:
        raise DataFaultError(
            f"uncertainty_multiplier={uncertainty_multiplier!r} is not a float",
            field="uncertainty_multiplier",
        ) from exc
    if um != um or um < 0.0 or um > 1.0:
        raise DataFaultError(
            f"uncertainty_multiplier={um!r} outside [0, 1]",
            field="uncertainty_multiplier",
        )

    multipliers = {
        SignalAction.SMALL_BUY:    _SIZE_SMALL_BUY_FRAC(),
        SignalAction.LARGE_BUY:    _SIZE_LARGE_BUY_FRAC(),
        SignalAction.HIGH_CONV_BUY: _SIZE_HIGH_CONV_FRAC(),
    }
    frac = multipliers.get(action, 0.5)
    size = base_size_usdc * frac * um
    size = min(size, max_size_usdc)
    return round(max(0.0, size), 4)


def score_signal_row(
    signal: dict,
    *,
    taker_fee_rate: float = _DEFAULT_TAKER_FEE_RATE,
    base_size_usdc: float = 10.0,
    max_size_usdc: float = 500.0,
    conformal_service: Optional["object"] = None,
) -> dict:
    """
    Convenience wrapper — scores a raw signal dict end-to-end.

    Reads: confidence, p_tp_before_sl, expected_return, current_price.
    Writes: edge_score, ev_after_cost, action (str), size_usdc.

    Returns the augmented signal dict (a shallow copy — does not mutate input).

    Raises
    ──────
    DataFaultError if any required field is missing or uncoercible.
    """
    out = dict(signal)

    confidence    = _require_float(signal, "confidence")
    p_tp          = _require_float(signal, "p_tp_before_sl")
    exp_return    = _require_float(signal, "expected_return")
    current_price = _require_float(signal, "current_price")

    edge  = compute_edge_score(p_tp, exp_return)
    ev    = compute_ev_after_cost(exp_return, current_price, taker_fee_rate=taker_fee_rate)
    action = compute_action(confidence, edge, ev)

    # Conformal sizing: if a fitted ConformalIntervalService is provided,
    # scale down for uncertain predictions. Absent service → full size (1.0).
    um = 1.0
    if conformal_service is not None and getattr(conformal_service, "is_fitted", False):
        um = float(conformal_service.size_multiplier(point_prediction=confidence))
        um = max(0.0, min(1.0, um))

    size   = compute_size_usdc(action, base_size_usdc, max_size_usdc, uncertainty_multiplier=um)

    out["edge_score"]              = edge
    out["ev_after_cost"]           = ev
    out["action"]                  = action
    out["uncertainty_multiplier"]  = round(um, 4)
    out["size_usdc"]               = size

    return out


def validate_signal_fields(signal: dict, required: list[str]) -> None:
    """
    Raise DataFaultError if any required field is missing from signal.

    Use this at the boundary of the signal pipeline before scoring.
    """
    missing = [f for f in required if f not in signal or signal[f] is None]
    if missing:
        raise DataFaultError(
            f"Signal row missing required fields: {missing}",
            field=missing[0],
            context={"token_id": signal.get("token_id"), "missing": missing},
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _require_float(signal: dict, field: str) -> float:
    val = signal.get(field)
    if val is None:
        raise DataFaultError(
            f"Signal field {field!r} is missing",
            field=field,
            context={"token_id": signal.get("token_id")},
        )
    try:
        return float(val)
    except (TypeError, ValueError) as exc:
        raise DataFaultError(
            f"Signal field {field!r}={val!r} cannot be coerced to float",
            field=field,
            context={"token_id": signal.get("token_id"), "raw_value": str(val)},
        ) from exc


def _require_non_negative_float(val, field: str) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError) as exc:
        raise DataFaultError(
            f"Field {field!r}={val!r} cannot be coerced to float",
            field=field,
        ) from exc
    if f < 0 or f != f:  # f != f catches NaN
        raise DataFaultError(
            f"Field {field!r}={val!r} must be a non-negative finite number",
            field=field,
        )
    return f


if __name__ == "__main__":
    # Self-test
    from services.types import SignalAction, DataFaultError

    # Edge score
    e = compute_edge_score(0.7, 0.10)
    assert abs(e - 0.07) < 1e-9, f"edge_score: {e}"

    # EV after cost
    ev = compute_ev_after_cost(0.10, current_price=0.50)
    # fee drag = 0.01 / 0.50 = 0.02; EV = 0.10 - 0.02 = 0.08
    assert abs(ev - 0.08) < 1e-9, f"ev_after_cost: {ev}"

    # Action dispatch
    assert compute_action(0.80, 0.07, 0.08) == SignalAction.HIGH_CONV_BUY
    assert compute_action(0.70, 0.07, 0.08) == SignalAction.LARGE_BUY
    assert compute_action(0.60, 0.03, 0.08) == SignalAction.IGNORE   # edge too low
    assert compute_action(0.40, 0.07, 0.08) == SignalAction.IGNORE   # confidence too low
    assert compute_action(0.60, 0.07, 0.001) == SignalAction.IGNORE  # EV too low

    # Size compute
    assert compute_size_usdc(SignalAction.IGNORE, 10.0, 500.0) == 0.0
    assert compute_size_usdc(SignalAction.SMALL_BUY, 10.0, 500.0) == 5.0   # 50% of 10
    assert compute_size_usdc(SignalAction.LARGE_BUY, 10.0, 500.0) == 10.0  # 100% of 10
    assert compute_size_usdc(SignalAction.HIGH_CONV_BUY, 10.0, 500.0) == 15.0  # 150% of 10
    assert compute_size_usdc(SignalAction.HIGH_CONV_BUY, 10.0, 12.0) == 12.0   # capped

    # Full row scorer
    sig = {
        "token_id": "tok1",
        "confidence": 0.80,
        "p_tp_before_sl": 0.70,
        "expected_return": 0.10,
        "current_price": 0.50,
    }
    result = score_signal_row(sig, base_size_usdc=10.0, max_size_usdc=500.0)
    assert result["action"] == SignalAction.HIGH_CONV_BUY, result["action"]
    assert result["size_usdc"] == 15.0, result["size_usdc"]

    # DataFaultError on missing field
    try:
        score_signal_row({"token_id": "t2", "confidence": 0.80})
        assert False, "should raise"
    except DataFaultError as e:
        assert "p_tp_before_sl" in str(e) or "p_tp_before_sl" == e.field

    print("feature_service self-test PASSED.")
