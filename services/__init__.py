"""
services/
─────────
High-assurance service layer for the Polymarket trading bot.

Each service has a single responsibility and a clean typed interface.
Services are imported by supervisor.py; they never import from supervisor.py.

Architecture rule: supervisor.py is the orchestrator. Services are the authorities.

Services
────────
- types              : shared enums, TypedDicts, exceptions
- order_lifecycle    : explicit order state machine + registry
- risk_service       : independent risk authority (approve_entry / approve_size)
- portfolio_service  : read-only portfolio state from logs CSVs
- execution_service  : order placement with lifecycle tracking
- feature_service    : pure feature scoring and signal sizing functions
- model_inference_service : immutable model loading and safe inference
- market_data_service     : fetch and normalize Polymarket market data
"""
from services.types import (
    OrderState,
    OrderSide,
    SignalAction,
    RiskOutcome,
    RiskDecision,
    FillStatus,
    FillRecord,
    ExecutionResult,
    PortfolioState,
    ScoredSignal,
    NormalizedMarket,
    ServiceError,
    TransitionError,
    RiskVetoError,
    ExecutionFaultError,
    ModelFaultError,
    DataFaultError,
    ORDER_TRANSITIONS,
)

__all__ = [
    # Enums
    "OrderState",
    "OrderSide",
    "SignalAction",
    "RiskOutcome",
    "FillStatus",
    # TypedDicts
    "RiskDecision",
    "FillRecord",
    "ExecutionResult",
    "PortfolioState",
    "ScoredSignal",
    "NormalizedMarket",
    # State machine
    "ORDER_TRANSITIONS",
    # Exceptions
    "ServiceError",
    "TransitionError",
    "RiskVetoError",
    "ExecutionFaultError",
    "ModelFaultError",
    "DataFaultError",
]
