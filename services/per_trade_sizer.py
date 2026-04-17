"""
services/per_trade_sizer.py
───────────────────────────
Per-trade Kelly-based size multiplier with volatility scaling.

Complements the family-level PortfolioAllocator (which decides how much
capital each family gets) by computing a *per-signal* size multiplier
inside the family pool.

Formula
-------
For a binary outcome with probability p, net odds b (payoff per unit
staked on a win), full Kelly fraction:

    f* = (b*p - (1-p)) / b

We then apply:
- **Fractional Kelly** (default 0.25) to absorb parameter uncertainty.
- **Bayesian shrinkage** toward a 50/50 prior with virtual trades
  `prior_strength` to stabilise Kelly on small samples.
- **Volatility scaling**: multiply by target_vol / realised_vol, clamped.
- **Edge floor / ceiling**: skip trades with tiny edge, cap monster Kelly.

Returns a multiplier in [0, 1] that callers apply to a base size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SizeBreakdown:
    kelly_raw: float
    kelly_shrunk: float
    fractional_kelly: float
    vol_scalar: float
    edge: float
    multiplier: float     # final [0, 1] multiplier to apply to base size
    skip: bool
    reason: str


class PerTradeSizer:
    """
    Parameters
    ----------
    kelly_fraction : fraction of full Kelly to take (default 0.25).
    prior_strength : virtual trades for Bayesian shrinkage toward 0.5 win
                     rate (higher → more shrinkage). Default 30.
    min_edge : absolute edge (p - q) below which we skip the trade. Acts
               as a structural noise filter. Default 0.02.
    target_vol : target per-trade realised-return volatility (e.g. 0.04
                 = 4% std of per-trade returns).
    min_vol_scalar, max_vol_scalar : clamp for the vol multiplier.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        prior_strength: float = 30.0,
        min_edge: float = 0.02,
        target_vol: float = 0.04,
        min_vol_scalar: float = 0.25,
        max_vol_scalar: float = 1.25,
    ):
        if not (0.0 < kelly_fraction <= 1.0):
            raise ValueError("kelly_fraction must be in (0, 1]")
        if prior_strength < 0:
            raise ValueError("prior_strength must be >= 0")
        self.kelly_fraction = float(kelly_fraction)
        self.prior_strength = float(prior_strength)
        self.min_edge = float(min_edge)
        self.target_vol = float(target_vol)
        self.min_vol_scalar = float(min_vol_scalar)
        self.max_vol_scalar = float(max_vol_scalar)

    @staticmethod
    def _full_kelly(p: float, price: float) -> float:
        """
        For a Polymarket YES share bought at `price` with predicted true
        probability `p`:
            payoff-on-win    = (1 - price) / price
            loss-on-loss     = 1
            Kelly fraction   = p*b - q  all over b,  with  b = (1-price)/price
                             = (p - price) / (1 - price)       (algebra)

        Net edge = p - price. If p ≤ price, Kelly ≤ 0 → skip.
        """
        price = min(max(float(price), 1e-6), 1.0 - 1e-6)
        p = min(max(float(p), 0.0), 1.0)
        if p <= price:
            return 0.0
        return (p - price) / (1.0 - price)

    def _shrink(self, p: float, n_trades: int, win_rate: float | None) -> float:
        """
        Shrink `p` toward historical win rate using Bayesian update.
        Treats p as a sufficient statistic from the model and blends with
        a prior win rate weighted by prior_strength.
        """
        if win_rate is None or not np.isfinite(win_rate):
            win_rate = 0.5
        n = max(0, int(n_trades))
        weight_post = n / (n + self.prior_strength) if (n + self.prior_strength) > 0 else 0.0
        return weight_post * p + (1.0 - weight_post) * float(win_rate)

    def _vol_scalar(self, realised_vol: float | None) -> float:
        if realised_vol is None or not np.isfinite(realised_vol) or realised_vol <= 1e-6:
            return 1.0
        raw = self.target_vol / float(realised_vol)
        return float(max(self.min_vol_scalar, min(self.max_vol_scalar, raw)))

    def compute(
        self,
        *,
        predicted_probability: float,
        market_price: float,
        family_n_trades: int = 0,
        family_win_rate: float | None = None,
        family_realised_vol: float | None = None,
    ) -> SizeBreakdown:
        """
        Main entry. Returns a SizeBreakdown whose `multiplier` is intended
        to be multiplied onto the caller's base size_usdc.
        """
        p = float(predicted_probability)
        price = float(market_price)
        edge = p - price

        if not np.isfinite(p) or not np.isfinite(price):
            return SizeBreakdown(0, 0, 0, 0, 0, 0.0, True, "nan_input")

        if edge < self.min_edge:
            return SizeBreakdown(0, 0, 0, 0, edge, 0.0, True,
                                 f"edge_below_floor({edge:.3f}<{self.min_edge})")

        # Shrink model probability toward family empirical win rate.
        p_shrunk = self._shrink(p, family_n_trades, family_win_rate)
        kelly_raw = self._full_kelly(p, price)
        kelly_shrunk = self._full_kelly(p_shrunk, price)
        # Fractional Kelly on the SHRUNK probability — standard defensive move
        frac = self.kelly_fraction * kelly_shrunk
        frac = max(0.0, min(1.0, frac))

        vol_s = self._vol_scalar(family_realised_vol)
        multiplier = max(0.0, min(1.0, frac * vol_s))

        skip = multiplier <= 1e-4
        reason = "ok" if not skip else "multiplier_near_zero"
        return SizeBreakdown(
            kelly_raw=round(kelly_raw, 4),
            kelly_shrunk=round(kelly_shrunk, 4),
            fractional_kelly=round(frac, 4),
            vol_scalar=round(vol_s, 4),
            edge=round(edge, 4),
            multiplier=round(multiplier, 4),
            skip=skip,
            reason=reason,
        )


__all__ = ["PerTradeSizer", "SizeBreakdown"]
