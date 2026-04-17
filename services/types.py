"""
services/types.py
─────────────────
Canonical types, enums, and exceptions for the service layer.

Rule: no trading logic here — only contracts.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Optional
from typing_extensions import TypedDict


# ── Enumerations ─────────────────────────────────────────────────────────────

class OrderState(Enum):
    """Explicit order lifecycle states. Every transition must be validated."""
    PENDING    = "pending"      # constructed, not submitted to exchange
    SUBMITTED  = "submitted"    # sent to exchange, awaiting acknowledgement
    PARTIAL    = "partial"      # exchange reports partial fill
    FILLED     = "filled"       # fully filled
    CANCELLED  = "cancelled"    # cancel confirmed by exchange
    REJECTED   = "rejected"     # exchange rejected the order
    EXPIRED    = "expired"      # timed out waiting for fill/cancel
    ERROR      = "error"        # unexpected fault — requires human review


class OrderSide(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class SignalAction(Enum):
    IGNORE        = 0
    SMALL_BUY     = 1
    LARGE_BUY     = 2
    HIGH_CONV_BUY = 3


class RiskOutcome(Enum):
    APPROVED = "approved"
    VETOED   = "vetoed"


class FillStatus(Enum):
    PENDING    = "pending"
    PARTIAL    = "partial"
    COMPLETE   = "complete"
    FAILED     = "failed"


# ── Valid state transitions (state machine) ───────────────────────────────────
#
# Only the transitions listed here are permitted.
# Attempting any other transition raises TransitionError.

ORDER_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.PENDING:   frozenset({OrderState.SUBMITTED, OrderState.ERROR}),
    OrderState.SUBMITTED: frozenset({
        OrderState.PARTIAL,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.ERROR,
    }),
    OrderState.PARTIAL:   frozenset({
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
        OrderState.ERROR,
    }),
    OrderState.FILLED:    frozenset(),   # terminal
    OrderState.CANCELLED: frozenset(),   # terminal
    OrderState.REJECTED:  frozenset(),   # terminal
    OrderState.EXPIRED:   frozenset(),   # terminal
    OrderState.ERROR:     frozenset(),   # terminal — requires human review
}


# ── TypedDicts ────────────────────────────────────────────────────────────────

class RiskDecision(TypedDict):
    outcome:        RiskOutcome
    approved:       bool
    reason:         str
    max_size_usdc:  float
    checks_run:     list[str]
    checks_failed:  list[str]


class FillRecord(TypedDict):
    order_id:       str
    fill_price:     float
    fill_shares:    float
    fill_notional:  float
    fill_ts:        str          # ISO UTC


class ExecutionResult(TypedDict):
    success:        bool
    order_id:       Optional[str]
    final_state:    OrderState
    fill:           Optional[FillRecord]
    error_code:     Optional[str]
    error_message:  Optional[str]
    attempts:       int
    latency_ms:     float


class PortfolioState(TypedDict):
    open_count:             int
    open_notional_usdc:     float
    btc_open_count:         int
    btc_open_notional:      float
    weather_open_count:     int
    weather_open_notional:  float
    unrealized_pnl:         float
    realized_pnl_session:   float
    available_usdc:         float
    max_drawdown_session:   float
    reconciliation_rate:    float   # fraction of closes flagged operational


class ScoredSignal(TypedDict):
    token_id:       str
    market_key:     str
    market_family:  str
    confidence:     float
    edge_score:     float
    expected_return: float
    ev_after_cost:  float
    action:         SignalAction
    size_usdc:      float
    model_version:  str


class NormalizedMarket(TypedDict):
    token_id:       str
    condition_id:   str
    market_slug:    str
    market_family:  str
    current_price:  float
    liquidity:      float
    volume_24h:     float
    time_left_sec:  float
    snapshot_ts:    str          # ISO UTC
    is_open:        bool


# ── Exceptions ────────────────────────────────────────────────────────────────

class ServiceError(Exception):
    """Base class for all service-layer errors."""
    def __init__(self, message: str, code: str = "SERVICE_ERROR", context: dict | None = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}


class TransitionError(ServiceError):
    """Raised when an invalid order state transition is attempted."""
    def __init__(self, from_state: OrderState, to_state: OrderState):
        super().__init__(
            f"Invalid order state transition: {from_state.value} → {to_state.value}",
            code="INVALID_TRANSITION",
            context={"from": from_state.value, "to": to_state.value},
        )
        self.from_state = from_state
        self.to_state = to_state


class RiskVetoError(ServiceError):
    """
    Raised by RiskService when a trade is vetoed.
    CRITICAL: this must NOT be caught and silently continued in the live path.
    Callers must explicitly handle it.
    """
    def __init__(self, reason: str, checks_failed: list[str], context: dict | None = None):
        super().__init__(
            f"Risk veto: {reason} [failed: {', '.join(checks_failed)}]",
            code="RISK_VETO",
            context=context or {},
        )
        self.reason = reason
        self.checks_failed = checks_failed


class ExecutionFaultError(ServiceError):
    """
    Raised when the execution path encounters an unrecoverable fault.
    CRITICAL: this must NOT be caught and silently continued in the live path.
    The order MUST be moved to OrderState.ERROR and the cycle MUST halt.
    """
    def __init__(self, message: str, order_id: str | None = None, context: dict | None = None):
        super().__init__(message, code="EXECUTION_FAULT", context=context or {})
        self.order_id = order_id


class ModelFaultError(ServiceError):
    """
    Raised when model inference fails in a way that makes the score unreliable.
    CRITICAL: do NOT fall back to score=0 and continue. Log and skip the candidate.
    """
    def __init__(self, message: str, model_stage: str = "", context: dict | None = None):
        super().__init__(message, code="MODEL_FAULT", context=context or {})
        self.model_stage = model_stage


class DataFaultError(ServiceError):
    """
    Raised when upstream data is stale, missing, or corrupt in a critical field.
    Callers must not use the data; skip the affected candidate.
    """
    def __init__(self, message: str, field: str = "", context: dict | None = None):
        super().__init__(message, code="DATA_FAULT", context=context or {})
        self.field = field
