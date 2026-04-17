"""Smart order router: adaptive aggressive/passive execution + TWAP slicing.

Replaces naive single-shot market orders with a cost-aware slicer that:

1. **Estimates impact** per clip using the square-root model from
   ``services.cost_impact_model`` (calibrated coefficient k from
   ``logs/impact_k.json`` if available).
2. **Picks a strategy**:
   * ``PASSIVE``   -- single clip, post-only / limit at mid+tick, when book is
     thick and the full size impacts < ``passive_bps_budget``.
   * ``TWAP``      -- split into N equal slices across a time window when the
     single-shot impact exceeds the passive budget but stays below the hard
     kill-switch budget.
   * ``ABORT``     -- even the sliced path breaches the kill-switch budget;
     signal caller to skip or wait for liquidity to replenish.
3. **Emits a plan** as a list of child orders (``ChildOrder``): size, target
   price, delay-from-start seconds, mode. The caller (trade_manager /
   execution_service) consumes the plan and fires clips at the scheduled
   offsets.

This is a planner, not an executor -- no network calls, no side effects, pure
function of the inputs. Integration: call ``plan_execution(...)`` right before
submitting the order and fan out the returned plan through the existing
``execution_service``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import math


class RoutingDecision(str, Enum):
    PASSIVE = "PASSIVE"
    TWAP = "TWAP"
    ABORT = "ABORT"


@dataclass
class ChildOrder:
    size: float
    target_price: float
    delay_seconds: float
    mode: str  # "LIMIT_POST_ONLY" | "LIMIT_TAKER" | "MARKET"


@dataclass
class ExecutionPlan:
    decision: RoutingDecision
    children: List[ChildOrder] = field(default_factory=list)
    est_total_cost_bps: float = 0.0
    est_impact_bps: float = 0.0
    est_spread_bps: float = 0.0
    notes: str = ""

    @property
    def n_clips(self) -> int:
        return len(self.children)


def _square_root_impact_bps(notional: float, adv: float, k: float = 0.10) -> float:
    """Impact in basis points for a trade of size ``notional`` vs ADV ``adv``."""
    if adv <= 0 or notional <= 0:
        return 0.0
    return float(k * math.sqrt(max(notional, 0.0) / max(adv, 1e-9)) * 1e4)


def plan_execution(
    *,
    side: str,
    notional_usd: float,
    mid_price: float,
    best_bid: float,
    best_ask: float,
    top_book_usd: float,
    adv_usd: float,
    impact_k: float = 0.10,
    passive_bps_budget: float = 15.0,
    abort_bps_budget: float = 80.0,
    twap_max_clips: int = 6,
    twap_window_seconds: float = 180.0,
    tick_size: float = 0.001,
) -> ExecutionPlan:
    """Produce an execution plan.

    Parameters
    ----------
    side : "BUY" or "SELL"
    notional_usd : target order notional in USD
    top_book_usd : USD sitting on the best bid (for SELL) or best ask (for BUY)
    adv_usd : average daily volume of the market in USD
    impact_k : square-root impact coefficient (from cost_impact_model calibration)
    passive_bps_budget : if single-shot impact <= this, do it passive
    abort_bps_budget : if even sliced impact > this, abort
    """
    side_u = str(side).strip().upper()
    if mid_price <= 0 or notional_usd <= 0:
        return ExecutionPlan(decision=RoutingDecision.ABORT, notes="invalid_inputs")

    spread = max(best_ask - best_bid, 0.0)
    spread_bps = (spread / mid_price) * 1e4 if mid_price > 0 else 0.0

    # --- Path 1: passive single clip if the book can absorb us comfortably ---
    single_shot_impact_bps = _square_root_impact_bps(notional_usd, adv_usd, impact_k)
    # Book depth check: if our notional is below the resting top-of-book size,
    # we can almost certainly cross at the touch with negligible extra impact.
    book_covers = top_book_usd >= notional_usd * 0.8
    if single_shot_impact_bps <= passive_bps_budget and book_covers:
        target = best_bid + tick_size if side_u == "BUY" else best_ask - tick_size
        target = max(target, tick_size)
        return ExecutionPlan(
            decision=RoutingDecision.PASSIVE,
            children=[ChildOrder(
                size=notional_usd / mid_price,
                target_price=float(target),
                delay_seconds=0.0,
                mode="LIMIT_POST_ONLY",
            )],
            est_total_cost_bps=spread_bps * 0.5 + single_shot_impact_bps,
            est_impact_bps=single_shot_impact_bps,
            est_spread_bps=spread_bps,
            notes="single_clip_passive",
        )

    # --- Path 3: even sliced, cost exceeds kill switch -> abort ---
    # With TWAP we split notional into N equal clips, so per-clip impact scales
    # as sqrt(1/N) and total (summed linearly) as N * k*sqrt(notional/N/adv) =
    # k*sqrt(N*notional/adv) = sqrt(N) * single_shot. That's WORSE summed, but
    # the point of TWAP is spreading in *time* so the market refills between
    # clips -- we model that as effective ADV scaling by the window fraction.
    # TWAP benefit: spreading over a window lets the book refill between clips.
    # We model this by comparing each clip's notional to FULL ADV (not ADV
    # scaled down by window fraction) -- the slicing itself is the liquidity
    # concession, not the window length.
    effective_adv = adv_usd
    # Per-clip impact with N clips:
    best_plan: Optional[ExecutionPlan] = None
    for n in range(2, max(twap_max_clips, 2) + 1):
        clip_notional = notional_usd / n
        clip_impact_bps = _square_root_impact_bps(clip_notional, effective_adv / n, impact_k)
        total_impact_bps = clip_impact_bps  # average cost per dollar, not sum
        total_cost_bps = spread_bps * 0.5 + total_impact_bps
        if total_cost_bps <= passive_bps_budget * 1.5:
            # Good enough -- build the plan with this N
            children: List[ChildOrder] = []
            step = twap_window_seconds / n
            for i in range(n):
                target = best_ask if side_u == "BUY" else best_bid
                children.append(ChildOrder(
                    size=(clip_notional / mid_price),
                    target_price=float(target),
                    delay_seconds=float(i * step),
                    mode="LIMIT_TAKER",
                ))
            best_plan = ExecutionPlan(
                decision=RoutingDecision.TWAP,
                children=children,
                est_total_cost_bps=total_cost_bps,
                est_impact_bps=total_impact_bps,
                est_spread_bps=spread_bps,
                notes=f"twap_{n}_clips_{int(twap_window_seconds)}s",
            )
            break
    if best_plan is not None:
        return best_plan

    # Try the max-clip plan even if it exceeds soft budget, as long as under kill switch
    n = twap_max_clips
    clip_notional = notional_usd / n
    clip_impact_bps = _square_root_impact_bps(clip_notional, effective_adv / n, impact_k)
    total_cost_bps = spread_bps * 0.5 + clip_impact_bps
    if total_cost_bps <= abort_bps_budget:
        step = twap_window_seconds / n
        children = []
        for i in range(n):
            target = best_ask if side_u == "BUY" else best_bid
            children.append(ChildOrder(
                size=(clip_notional / mid_price),
                target_price=float(target),
                delay_seconds=float(i * step),
                mode="LIMIT_TAKER",
            ))
        return ExecutionPlan(
            decision=RoutingDecision.TWAP,
            children=children,
            est_total_cost_bps=total_cost_bps,
            est_impact_bps=clip_impact_bps,
            est_spread_bps=spread_bps,
            notes=f"twap_maxclips_{n}_overbudget",
        )

    return ExecutionPlan(
        decision=RoutingDecision.ABORT,
        est_total_cost_bps=total_cost_bps,
        est_impact_bps=clip_impact_bps,
        est_spread_bps=spread_bps,
        notes="cost_exceeds_abort_budget",
    )
