"""
services/order_lifecycle.py
────────────────────────────
Explicit state machine for the order lifecycle.

Every order is an OrderLifecycle object. State transitions are validated
against ORDER_TRANSITIONS — any invalid transition raises TransitionError
immediately. There is no silent state mutation.

Usage
─────
    lc = OrderLifecycle(order_id="abc123", token_id="...", side=OrderSide.BUY)
    lc.transition(OrderState.SUBMITTED)       # OK: PENDING → SUBMITTED
    lc.transition(OrderState.FILLED, fill=…)  # OK: SUBMITTED → FILLED
    lc.transition(OrderState.PENDING)         # raises TransitionError

    # Inspect
    lc.state          # OrderState.FILLED
    lc.fill           # FillRecord | None
    lc.is_terminal    # True
    lc.history        # [(ts, from_state, to_state), ...]
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from services.types import (
    FillRecord,
    OrderSide,
    OrderState,
    ORDER_TRANSITIONS,
    TransitionError,
)


class OrderLifecycle:
    """
    Single order state machine.

    Invariants
    ──────────
    1. state is always a valid OrderState.
    2. Every state change appends to history.
    3. No transition is permitted unless listed in ORDER_TRANSITIONS.
    4. Terminal states cannot transition to any state, including themselves.
    5. fill is only set when transitioning to FILLED or PARTIAL; accessing
       fill on a non-filled order returns None (never raises).
    """

    __slots__ = (
        "order_id", "token_id", "side", "requested_price", "requested_shares",
        "requested_usdc", "_state", "fill", "history", "created_at",
        "submitted_at", "filled_at", "cancelled_at", "error_message",
    )

    def __init__(
        self,
        order_id: str,
        token_id: str,
        side: OrderSide,
        requested_price: float = 0.0,
        requested_shares: float = 0.0,
        requested_usdc: float = 0.0,
    ) -> None:
        self.order_id = order_id
        self.token_id = token_id
        self.side = side
        self.requested_price = requested_price
        self.requested_shares = requested_shares
        self.requested_usdc = requested_usdc
        self._state: OrderState = OrderState.PENDING
        self.fill: Optional[FillRecord] = None
        self.history: list[tuple[str, OrderState, OrderState]] = []
        self.created_at: str = _now_iso()
        self.submitted_at: Optional[str] = None
        self.filled_at: Optional[str] = None
        self.cancelled_at: Optional[str] = None
        self.error_message: Optional[str] = None

    # ── State property ────────────────────────────────────────────────────────

    @property
    def state(self) -> OrderState:
        return self._state

    @property
    def is_terminal(self) -> bool:
        return len(ORDER_TRANSITIONS[self._state]) == 0

    @property
    def is_filled(self) -> bool:
        return self._state in (OrderState.FILLED, OrderState.PARTIAL)

    @property
    def is_live(self) -> bool:
        """True if the order is still active on the exchange."""
        return self._state in (OrderState.SUBMITTED, OrderState.PARTIAL)

    # ── Transition ────────────────────────────────────────────────────────────

    def transition(
        self,
        new_state: OrderState,
        fill: Optional[FillRecord] = None,
        error_message: Optional[str] = None,
    ) -> "OrderLifecycle":
        """
        Advance the state machine.

        Raises TransitionError immediately if the transition is not valid.
        Never silently ignores invalid transitions.
        """
        allowed = ORDER_TRANSITIONS[self._state]
        if new_state not in allowed:
            raise TransitionError(self._state, new_state)

        ts = _now_iso()
        self.history.append((ts, self._state, new_state))

        # Set timestamps for key transitions.
        if new_state == OrderState.SUBMITTED:
            self.submitted_at = ts
        elif new_state in (OrderState.FILLED, OrderState.PARTIAL) and fill is not None:
            self.fill = fill
            if new_state == OrderState.FILLED:
                self.filled_at = ts
        elif new_state == OrderState.CANCELLED:
            self.cancelled_at = ts
        elif new_state == OrderState.ERROR:
            self.error_message = error_message or "unspecified error"

        self._state = new_state
        return self

    def mark_submitted(self) -> "OrderLifecycle":
        return self.transition(OrderState.SUBMITTED)

    def mark_filled(self, fill: FillRecord) -> "OrderLifecycle":
        return self.transition(OrderState.FILLED, fill=fill)

    def mark_partial(self, fill: FillRecord) -> "OrderLifecycle":
        return self.transition(OrderState.PARTIAL, fill=fill)

    def mark_cancelled(self) -> "OrderLifecycle":
        return self.transition(OrderState.CANCELLED)

    def mark_rejected(self) -> "OrderLifecycle":
        return self.transition(OrderState.REJECTED)

    def mark_expired(self) -> "OrderLifecycle":
        return self.transition(OrderState.EXPIRED)

    def mark_error(self, message: str) -> "OrderLifecycle":
        return self.transition(OrderState.ERROR, error_message=message)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "order_id":         self.order_id,
            "token_id":         self.token_id,
            "side":             self.side.value,
            "requested_price":  self.requested_price,
            "requested_shares": self.requested_shares,
            "requested_usdc":   self.requested_usdc,
            "state":            self._state.value,
            "fill":             dict(self.fill) if self.fill else None,
            "created_at":       self.created_at,
            "submitted_at":     self.submitted_at,
            "filled_at":        self.filled_at,
            "cancelled_at":     self.cancelled_at,
            "error_message":    self.error_message,
            "history": [
                {"ts": ts, "from": f.value, "to": t.value}
                for ts, f, t in self.history
            ],
        }

    def __repr__(self) -> str:
        return (
            f"OrderLifecycle(order_id={self.order_id!r}, "
            f"token={self.token_id!r}, side={self.side.value}, "
            f"state={self._state.value})"
        )


# ── Registry for active orders ────────────────────────────────────────────────

class OrderRegistry:
    """
    In-memory registry of active order lifecycles.

    Rules
    ─────
    - Every submitted order is registered here.
    - Terminal orders are moved to _closed (kept for diagnostics this session).
    - lookup() raises KeyError if the order_id is unknown.
    """

    def __init__(self) -> None:
        self._active: dict[str, OrderLifecycle] = {}
        self._closed: dict[str, OrderLifecycle] = {}

    def register(self, lc: OrderLifecycle) -> None:
        if lc.order_id in self._active or lc.order_id in self._closed:
            raise ValueError(f"Order {lc.order_id!r} already registered")
        self._active[lc.order_id] = lc

    def lookup(self, order_id: str) -> OrderLifecycle:
        if order_id in self._active:
            return self._active[order_id]
        if order_id in self._closed:
            return self._closed[order_id]
        raise KeyError(f"Unknown order_id: {order_id!r}")

    def advance(self, order_id: str, new_state: OrderState, **kwargs) -> OrderLifecycle:
        """Transition an order and move to closed if terminal."""
        lc = self.lookup(order_id)
        lc.transition(new_state, **kwargs)
        if lc.is_terminal:
            self._closed[order_id] = self._active.pop(order_id)
        return lc

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def active_orders(self) -> list[OrderLifecycle]:
        return list(self._active.values())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    # Self-test
    from services.types import FillRecord, OrderSide, OrderState, TransitionError

    lc = OrderLifecycle("ord-001", "tok-123", OrderSide.BUY, requested_price=0.55, requested_usdc=10.0)
    assert lc.state == OrderState.PENDING
    assert not lc.is_terminal

    lc.mark_submitted()
    assert lc.state == OrderState.SUBMITTED

    fill: FillRecord = {
        "order_id": "ord-001",
        "fill_price": 0.553,
        "fill_shares": 18.08,
        "fill_notional": 10.0,
        "fill_ts": _now_iso(),
    }
    lc.mark_filled(fill)
    assert lc.state == OrderState.FILLED
    assert lc.is_terminal
    assert lc.fill["fill_price"] == 0.553

    # Test invalid transition is blocked
    try:
        lc.mark_cancelled()
        assert False, "should have raised"
    except TransitionError as e:
        assert e.from_state == OrderState.FILLED
        print(f"  TransitionError correctly raised: {e}")

    # Registry
    reg = OrderRegistry()
    lc2 = OrderLifecycle("ord-002", "tok-456", OrderSide.BUY)
    reg.register(lc2)
    reg.advance("ord-002", OrderState.SUBMITTED)
    reg.advance("ord-002", OrderState.REJECTED)
    assert reg.active_count == 0  # moved to closed

    print("order_lifecycle self-test PASSED.")
