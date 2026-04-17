"""
services/cost_impact_model.py
─────────────────────────────
Market-impact and slippage cost model.

What the old code had
---------------------
Flat fee subtraction. That underestimates real costs by 2-5× on any trade
that crosses more than one tick of the book, which on Polymarket is most
trades larger than a few hundred dollars.

What this adds
--------------
1. **Square-root impact law** (Almgren-Chriss / Kyle λ) — permanent price
   impact scales with sqrt(Q/ADV) where Q is order size and ADV is average
   daily volume. Industry-standard for equities and extensible to prediction
   markets via the liquidity-depth analog.

2. **Linear spread cost** — half-spread × size, the dominant cost for small
   orders that don't walk the book.

3. **Book-walking slippage** — when order size > top-of-book liquidity, walk
   the L2 ladder explicitly. Returns the actual volume-weighted fill price.

4. **Fee schedule** — Polymarket maker/taker fee plumbing with per-side
   granularity.

5. **Total-cost EV adjustment** — subtracts impact + slippage + fees from
   a raw edge estimate, returns an "EV-after-cost" signal for sizing.

Units
-----
All prices in [0, 1] (Polymarket YES-share convention). All costs returned
in the same units as price (subtract from edge). Sizes in USDC notional.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# ──────────────────────────── parameters ───────────────────────────────
@dataclass
class CostParams:
    taker_fee_bps: float = 0.0        # Polymarket is zero-fee for retail today
    maker_fee_bps: float = 0.0
    # Square-root impact coefficient. Calibrated from regression
    # realised_slippage ~ k * sqrt(Q / ADV)
    impact_coef_k: float = 0.10
    # Minimum half-spread cost floor (captures adverse selection on tiny orders)
    min_half_spread: float = 0.005
    # Maximum plausible impact (caps pathological cases where ADV is stale)
    max_impact: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)


# ───────────────────────────── impact ──────────────────────────────────
def square_root_impact(
    size_usdc: float,
    adv_usdc: float,
    *,
    k: float = 0.10,
    cap: float = 0.05,
) -> float:
    """
    Permanent price impact (in price units) = k * sqrt(Q / ADV).
    Clamped to `cap`. Returns 0 when ADV is not provided.
    """
    size = max(0.0, float(size_usdc))
    adv = max(0.0, float(adv_usdc))
    if adv <= 0.0 or size <= 0.0:
        return 0.0
    raw = float(k) * float(np.sqrt(size / adv))
    return float(min(cap, max(0.0, raw)))


# ───────────────────────────── book walk ───────────────────────────────
@dataclass
class FillResult:
    avg_fill_price: float
    filled_shares: float
    unfilled_shares: float
    levels_consumed: int
    slippage_vs_top: float   # avg_fill - best_price (buy side); negated for sell
    notional_usdc: float

    def to_dict(self) -> dict:
        return asdict(self)


def walk_the_book(
    levels: list[tuple[float, float]],
    size_shares: float,
    side: str = "buy",
) -> FillResult:
    """
    Consume L2 liquidity `levels = [(price, size_shares), ...]` sorted from
    best to worst. Returns the volume-weighted fill and residual.

    - BUY side: expects ascending asks; slippage = fill - best_ask.
    - SELL side: expects descending bids; slippage = best_bid - fill.

    When the book is exhausted, returns what was filled + unfilled residual.
    """
    side = str(side).lower()
    remaining = max(0.0, float(size_shares))
    filled = 0.0
    notional = 0.0
    best_price = None
    consumed = 0
    for price, available in levels:
        if remaining <= 0:
            break
        price = float(price); available = max(0.0, float(available))
        if best_price is None:
            best_price = price
        take = min(remaining, available)
        if take <= 0:
            continue
        filled += take
        notional += take * price
        remaining -= take
        consumed += 1
    if filled <= 0:
        return FillResult(0.0, 0.0, float(size_shares), 0, 0.0, 0.0)
    avg_fill = notional / filled
    if best_price is None:
        best_price = avg_fill
    slip = (avg_fill - best_price) if side == "buy" else (best_price - avg_fill)
    return FillResult(
        avg_fill_price=round(avg_fill, 6),
        filled_shares=round(filled, 4),
        unfilled_shares=round(max(0.0, remaining), 4),
        levels_consumed=consumed,
        slippage_vs_top=round(slip, 6),
        notional_usdc=round(notional, 4),
    )


# ─────────────────────────── aggregate API ─────────────────────────────
@dataclass
class TradeCostEstimate:
    impact: float
    half_spread: float
    fees: float
    total_cost: float
    edge_after_cost: float
    breakeven_edge: float
    viable: bool

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_trade_cost(
    *,
    size_usdc: float,
    price: float,
    spread: float,
    adv_usdc: float,
    edge: float,
    side: str = "buy",
    params: CostParams | None = None,
) -> TradeCostEstimate:
    """
    End-to-end cost estimate for a prospective trade.

    - `spread` is the top-of-book bid-ask spread in price units.
    - `adv_usdc` is market average daily volume (USDC).
    - `edge` is the raw alpha (p_model - price) BEFORE cost.

    Returns whether the trade is viable (edge > total cost) and the
    residual EV after all costs.
    """
    p = params or CostParams()
    half_spread = max(0.5 * float(max(0.0, spread)), p.min_half_spread)
    impact = square_root_impact(size_usdc, adv_usdc, k=p.impact_coef_k, cap=p.max_impact)
    fee_bps = p.taker_fee_bps if side.lower() == "buy" else p.maker_fee_bps
    fees = (fee_bps / 10_000.0) * float(price)
    total = float(half_spread) + float(impact) + float(fees)
    edge_net = float(edge) - total
    return TradeCostEstimate(
        impact=round(impact, 6),
        half_spread=round(half_spread, 6),
        fees=round(fees, 6),
        total_cost=round(total, 6),
        edge_after_cost=round(edge_net, 6),
        breakeven_edge=round(total, 6),
        viable=bool(edge_net > 0),
    )


# ──────────────────────── calibration helper ───────────────────────────
def calibrate_impact_coefficient(
    realised_slippage: np.ndarray,
    size_usdc: np.ndarray,
    adv_usdc: np.ndarray,
) -> dict:
    """
    Fit the square-root impact coefficient from a pile of closed trades:
        slippage_i ≈ k * sqrt(Q_i / ADV_i)
    OLS through origin on transformed variables. Returns k and R².
    """
    s = np.asarray(realised_slippage, dtype=float)
    q = np.asarray(size_usdc, dtype=float)
    a = np.asarray(adv_usdc, dtype=float)
    m = np.isfinite(s) & np.isfinite(q) & np.isfinite(a) & (a > 0) & (q > 0)
    if m.sum() < 30:
        return {"k": None, "r2": None, "n": int(m.sum()), "status": "insufficient"}
    x = np.sqrt(q[m] / a[m])
    y = s[m]
    # OLS through origin: k = (x·y) / (x·x)
    xx = float((x * x).sum())
    if xx <= 0:
        return {"k": None, "r2": None, "n": int(m.sum()), "status": "degenerate"}
    k = float((x * y).sum() / xx)
    y_hat = k * x
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"k": round(k, 5), "r2": round(float(r2), 4),
            "n": int(m.sum()), "status": "ok"}


__all__ = [
    "CostParams",
    "square_root_impact",
    "walk_the_book",
    "FillResult",
    "estimate_trade_cost",
    "TradeCostEstimate",
    "calibrate_impact_coefficient",
]
