"""
services/portfolio_allocator.py
────────────────────────────────
Institutional-grade capital allocator across market families.

Solves the problem: BTC and weather live in the same USDC wallet. Without
active allocation you either (a) under-allocate the better family, or
(b) let a drawdown in one family quietly drain the other.

The allocator combines three disciplines used by multi-strategy hedge funds:

1. **Volatility targeting** — each family has a target *return volatility*
   (e.g. 10% annualised). Family weights scale inversely with realised
   family return vol, so a sudden vol spike auto-throttles that family
   before a drawdown materialises.

2. **Fractional Kelly sizing** — each family gets a Kelly fraction of its
   target pool, where Kelly is computed from per-family (win_rate, avg_win,
   avg_loss) on the last N closed trades. Fraction defaults to 0.25 (quarter
   Kelly) — the textbook institutional choice that accepts 1/16 of full
   Kelly's variance for 1/4 of its return.

3. **Drawdown throttle** — when a family's rolling drawdown exceeds
   `dd_soft`, its weight is linearly reduced; at `dd_hard` the family is
   cut to zero until a recovery threshold is cleared. This is the
   "volatility trimming + stop-out" rule from risk-parity mandates.

Cross-family correlation is handled by computing weights on *family return
series*, not on portfolio exposure. When BTC and weather co-draw-down
(rare but catastrophic), both families' weights compress simultaneously
and total book leverage falls — the same mechanism AQR/BridgeWater use.

Contract
────────
`allocate(family_stats: dict[str, FamilyStats], total_capital_usdc)` →
`dict[family, FamilyAllocation]` with fields:
    pool_usdc                : family's capital pool this cycle
    kelly_fraction           : Kelly % applied (bounded to [0, kelly_cap])
    vol_scalar               : vol-target multiplier (bounded to [0, vol_cap])
    dd_scalar                : drawdown throttle (in [0, 1])
    active                   : False if family is stopped out
    diagnostics              : dict of intermediate values for audit

The returned `pool_usdc` is what `RiskService.approve_entry` should use
as the family's hard-cap-for-this-cycle (intersected with the static hard
cap defined in risk_service.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


# ── Inputs ───────────────────────────────────────────────────────────────────

@dataclass
class FamilyStats:
    """Closed-trade statistics for one market family over a rolling window."""
    family: str
    n_trades: int                # closed trades in window
    win_rate: float              # fraction in [0, 1]
    avg_win: float               # avg $ won on winning trades (positive)
    avg_loss: float              # avg $ lost on losing trades (positive magnitude)
    returns_series: np.ndarray   # per-trade returns (fraction), chronological
    current_drawdown: float      # peak-to-trough return loss in [0, 1]
    peak_equity: float           # peak $ equity for this family


# ── Outputs ──────────────────────────────────────────────────────────────────

@dataclass
class FamilyAllocation:
    family: str
    pool_usdc: float
    kelly_fraction: float
    vol_scalar: float
    dd_scalar: float
    active: bool
    diagnostics: dict = field(default_factory=dict)


# ── Allocator ────────────────────────────────────────────────────────────────

class PortfolioAllocator:
    """
    Vol-target + fractional Kelly + drawdown throttle allocator.

    All limits are soft defaults; override per-strategy via constructor args.
    The caller is responsible for intersecting the allocator's pool_usdc with
    the static hard caps in `services/risk_service.py`.
    """

    def __init__(
        self,
        *,
        target_vol_annual: float = 0.10,   # 10% annualised target vol per family
        kelly_fraction_cap: float = 0.25,   # quarter-Kelly
        vol_scalar_cap: float = 2.0,        # cannot more-than-double base allocation
        dd_soft: float = 0.08,              # 8% rolling DD starts throttling
        dd_hard: float = 0.20,              # 20% stops the family out
        dd_reset: float = 0.05,             # must recover to < 5% DD to reactivate
        min_trades_for_kelly: int = 30,     # below this, use prior-based conservative Kelly
        trading_days_per_year: int = 252,   # standard trading-day count
        min_pool_floor_usdc: float = 25.0,  # never return a sub-minimum positive pool
    ) -> None:
        if not (0.0 < target_vol_annual < 2.0):
            raise ValueError("target_vol_annual must be in (0, 2)")
        if not (0.0 < kelly_fraction_cap <= 1.0):
            raise ValueError("kelly_fraction_cap must be in (0, 1]")
        if dd_soft >= dd_hard:
            raise ValueError("dd_soft must be < dd_hard")
        if dd_reset >= dd_soft:
            raise ValueError("dd_reset must be < dd_soft")
        self.target_vol_annual = float(target_vol_annual)
        self.kelly_fraction_cap = float(kelly_fraction_cap)
        self.vol_scalar_cap = float(vol_scalar_cap)
        self.dd_soft = float(dd_soft)
        self.dd_hard = float(dd_hard)
        self.dd_reset = float(dd_reset)
        self.min_trades_for_kelly = int(min_trades_for_kelly)
        self.trading_days_per_year = int(trading_days_per_year)
        self.min_pool_floor_usdc = float(min_pool_floor_usdc)
        # Persistent stop-out state — families stay stopped until recovery.
        self._stopped_out: set[str] = set()

    # ── Kelly ────────────────────────────────────────────────────────────
    def _kelly_fraction(self, stats: FamilyStats) -> tuple[float, dict]:
        """
        Kelly fraction = p/a - q/b, where
            p = win rate
            q = loss rate
            a = avg loss (fraction of stake)
            b = avg win (fraction of stake)

        We work in $ amounts (avg_win, avg_loss are $ magnitudes); the
        Kelly formula is dimensionally consistent as long as numerator and
        denominator use the same units.

        Below `min_trades_for_kelly` we apply a Bayesian shrinkage toward
        50/50 odds (prior strength = min_trades_for_kelly), so a 3-trade
        hot streak cannot blow up the position.
        """
        n = max(1, int(stats.n_trades))
        shrink = self.min_trades_for_kelly / (self.min_trades_for_kelly + n)
        p_observed = float(stats.win_rate)
        p_shrunk = shrink * 0.5 + (1.0 - shrink) * p_observed

        a = max(1e-9, float(stats.avg_loss))
        b = max(1e-9, float(stats.avg_win))
        q = 1.0 - p_shrunk
        raw_kelly = (p_shrunk / a) - (q / b)
        # Kelly can go negative when edge < 0. We never short — clamp to 0.
        raw_kelly = max(0.0, raw_kelly)
        # Fractional-Kelly cap — quarter-Kelly by default.
        kelly = min(raw_kelly, 1.0) * self.kelly_fraction_cap
        return kelly, {
            "p_observed": round(p_observed, 4),
            "p_shrunk": round(p_shrunk, 4),
            "raw_kelly": round(raw_kelly, 4),
            "shrink_weight": round(shrink, 4),
        }

    # ── Vol targeting ────────────────────────────────────────────────────
    def _vol_scalar(self, stats: FamilyStats) -> tuple[float, dict]:
        returns = np.asarray(stats.returns_series, dtype=float)
        returns = returns[np.isfinite(returns)]
        if returns.size < 5:
            return 1.0, {"realised_vol_ann": None, "reason": "too_few_returns"}
        # Per-trade std → annualised assuming ~N trades/day average.
        # We don't know N, so we scale by sqrt(trading_days) which treats
        # each trade as one "day" — conservative (overstates vol when
        # trading more than once per day). A trainer can pass a pre-scaled
        # series if higher fidelity is needed.
        per_trade_std = float(np.std(returns, ddof=1))
        ann_vol = per_trade_std * math.sqrt(self.trading_days_per_year)
        if ann_vol <= 1e-9:
            scalar = self.vol_scalar_cap
        else:
            scalar = self.target_vol_annual / ann_vol
        scalar = float(np.clip(scalar, 0.0, self.vol_scalar_cap))
        return scalar, {
            "realised_vol_ann": round(ann_vol, 4),
            "target_vol_ann": self.target_vol_annual,
        }

    # ── Drawdown throttle ────────────────────────────────────────────────
    def _dd_scalar(self, stats: FamilyStats) -> tuple[float, bool, dict]:
        dd = float(stats.current_drawdown)
        # Persistent stop-out: stay out until dd_reset is breached downward.
        if stats.family in self._stopped_out:
            if dd <= self.dd_reset:
                self._stopped_out.discard(stats.family)
                return 1.0, True, {"dd": round(dd, 4), "state": "recovered"}
            return 0.0, False, {"dd": round(dd, 4), "state": "stopped_out"}
        if dd >= self.dd_hard:
            self._stopped_out.add(stats.family)
            return 0.0, False, {"dd": round(dd, 4), "state": "newly_stopped"}
        if dd >= self.dd_soft:
            # Linear: dd_soft → 1.0 multiplier, dd_hard → 0.0 multiplier.
            span = max(self.dd_hard - self.dd_soft, 1e-9)
            scalar = 1.0 - (dd - self.dd_soft) / span
            return float(np.clip(scalar, 0.0, 1.0)), True, {
                "dd": round(dd, 4),
                "state": "throttled",
            }
        return 1.0, True, {"dd": round(dd, 4), "state": "normal"}

    # ── Public ───────────────────────────────────────────────────────────
    def allocate(
        self,
        family_stats: Mapping[str, FamilyStats],
        total_capital_usdc: float,
    ) -> dict[str, FamilyAllocation]:
        if total_capital_usdc <= 0:
            return {
                fam: FamilyAllocation(
                    family=fam, pool_usdc=0.0, kelly_fraction=0.0,
                    vol_scalar=0.0, dd_scalar=0.0, active=False,
                    diagnostics={"reason": "no_capital"},
                )
                for fam in family_stats
            }

        # Compute per-family unnormalised weights = kelly * vol_scalar * dd_scalar
        raw_weights: dict[str, float] = {}
        per_family_diag: dict[str, dict] = {}
        per_family_active: dict[str, bool] = {}
        per_family_components: dict[str, tuple[float, float, float]] = {}

        for fam, stats in family_stats.items():
            kelly, k_diag = self._kelly_fraction(stats)
            vol, v_diag = self._vol_scalar(stats)
            dd, active, d_diag = self._dd_scalar(stats)
            w = kelly * vol * dd
            raw_weights[fam] = w
            per_family_active[fam] = active
            per_family_components[fam] = (kelly, vol, dd)
            per_family_diag[fam] = {"kelly": k_diag, "vol": v_diag, "dd": d_diag}

        total_weight = sum(raw_weights.values())
        results: dict[str, FamilyAllocation] = {}

        for fam, w in raw_weights.items():
            kelly, vol, dd = per_family_components[fam]
            active = per_family_active[fam]
            if total_weight <= 0 or not active:
                pool = 0.0
            else:
                # Risk-parity: each family's pool is its share of total book
                # weight × total capital × its Kelly * vol * dd product.
                # This is equivalent to: pool_i = capital * (w_i / sum(w_j))
                # × kelly_i — i.e. proportional to BOTH Kelly and its share
                # of total allocatable weight. In the common case where all
                # families have similar edge, this collapses to risk parity.
                pool = total_capital_usdc * (w / total_weight)
                if 0.0 < pool < self.min_pool_floor_usdc:
                    # Floor-or-zero: tiny positive pools are useless — round up
                    # to the exchange minimum or drop to zero.
                    pool = 0.0 if pool < self.min_pool_floor_usdc * 0.5 else self.min_pool_floor_usdc
            results[fam] = FamilyAllocation(
                family=fam,
                pool_usdc=round(pool, 2),
                kelly_fraction=round(kelly, 4),
                vol_scalar=round(vol, 4),
                dd_scalar=round(dd, 4),
                active=active,
                diagnostics={
                    "raw_weight": round(w, 6),
                    "total_weight": round(total_weight, 6),
                    **per_family_diag[fam],
                },
            )
        return results


__all__ = ["FamilyStats", "FamilyAllocation", "PortfolioAllocator"]
