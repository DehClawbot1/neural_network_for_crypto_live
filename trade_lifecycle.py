from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

from pnl_engine import PNLEngine


class TradeState(str, Enum):
    NEW_SIGNAL = "NEW_SIGNAL"
    ENTERED = "ENTERED"
    OPEN = "OPEN"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    CLOSED = "CLOSED"
    RESOLVED = "RESOLVED"


@dataclass
class TradeLifecycle:
    market: str
    token_id: str | None
    condition_id: str | None
    outcome_side: str
    logs_dir: str = "logs"
    state: TradeState = TradeState.NEW_SIGNAL
    size_usdc: float = 0.0
    shares: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    opened_at: str | None = None
    closed_at: str | None = None
    close_reason: str | None = None
    confidence_at_entry: float = 0.0
    signal_label: str = "UNKNOWN"
    ledger: list = field(default_factory=list)

    def _write_execution_event(self, payload: dict):
        logs_path = Path(self.logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        # Keep lifecycle events isolated from execution_log.csv so fill/trade logs
        # keep a stable schema for dashboards, dataset builders, and reconciliations.
        events_file = logs_path / "trade_events.csv"
        pd.DataFrame([payload]).to_csv(events_file, mode="a", header=not events_file.exists(), index=False)

    def on_signal(self, signal_row: dict):
        payload = {
            "event": "signal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market": self.market,
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "outcome_side": self.outcome_side,
        }
        self.ledger.append({**payload, "signal": signal_row})
        self._write_execution_event(payload)
        self.state = TradeState.NEW_SIGNAL
        self.confidence_at_entry = float(signal_row.get("confidence", 0.0) or 0.0)
        self.signal_label = str(signal_row.get("signal_label", "UNKNOWN") or "UNKNOWN")

    def enter(self, size_usdc: float, entry_price: float):
        self.size_usdc = float(size_usdc)
        self.entry_price = float(entry_price)
        self.current_price = float(entry_price)
        self.shares = PNLEngine.shares_from_capital(size_usdc, entry_price)
        self.opened_at = datetime.now(timezone.utc).isoformat()
        self.state = TradeState.ENTERED
        self.peak_price = self.entry_price
        payload = {
            "event": "enter",
            "timestamp": self.opened_at,
            "market": self.market,
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "outcome_side": self.outcome_side,
            "size_usdc": self.size_usdc,
            "entry_price": self.entry_price,
            "shares": self.shares,
        }
        self.ledger.append(payload)
        self._write_execution_event(payload)

    def update_market(self, live_price: float):
        self.current_price = float(live_price)
        self.peak_price = max(float(getattr(self, "peak_price", self.entry_price or self.current_price)), self.current_price)
        self.unrealized_pnl = self.shares * (self.current_price - self.entry_price)
        if self.state in [TradeState.ENTERED, TradeState.PARTIAL_EXIT]:
            self.state = TradeState.OPEN
        self.ledger.append({
            "event": "mark",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "live_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
        })
        return self.unrealized_pnl

    def partial_exit(self, fraction: float, exit_price: float):
        fraction = max(0.0, min(1.0, float(fraction)))
        if fraction <= 0 or self.shares <= 0:
            return 0.0
        exited_shares = self.shares * fraction
        pnl = exited_shares * (float(exit_price) - float(self.entry_price))
        self.realized_pnl += pnl
        self.shares = max(0.0, self.shares - exited_shares)
        self.size_usdc = max(0.0, self.shares * float(self.entry_price))
        self.current_price = float(exit_price)
        self.unrealized_pnl = self.shares * (self.current_price - self.entry_price)
        self.state = TradeState.PARTIAL_EXIT if self.shares > 0 else TradeState.CLOSED
        self.ledger.append({
            "event": "partial_exit",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fraction": fraction,
            "exit_price": exit_price,
            "remaining_shares": self.shares,
            "realized_pnl": pnl,
        })
        if self.shares <= 0:
            self.closed_at = datetime.now(timezone.utc).isoformat()
        return pnl

    def close(self, exit_price: float, reason: str = "policy_exit"):
        pnl = self.shares * (float(exit_price) - float(self.entry_price))
        self.realized_pnl += pnl
        self.current_price = float(exit_price)
        self.unrealized_pnl = 0.0
        self.closed_at = datetime.now(timezone.utc).isoformat()
        # self.shares = 0.0 # BUG 1 FIX: Keep shares intact so supervisor knows how much to sell
        # self.size_usdc = 0.0
        self.state = TradeState.CLOSED
        self.close_reason = reason
        self.ledger.append({
            "event": "close",
            "timestamp": self.closed_at,
            "exit_price": exit_price,
            "realized_pnl": pnl,
            "close_reason": reason,
        })
        self._write_execution_event({
            "event": "close",
            "timestamp": self.closed_at,
            "market": self.market,
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "outcome_side": self.outcome_side,
            "exit_price": exit_price,
            "realized_pnl": self.realized_pnl,
            "close_reason": reason,
        })
        return pnl

    def resolve(self, token_won: bool):
        pnl = PNLEngine.resolution_pnl(self.size_usdc, self.entry_price, token_won)
        self.realized_pnl = pnl
        self.unrealized_pnl = 0.0
        self.state = TradeState.RESOLVED
        self.closed_at = datetime.now(timezone.utc).isoformat()
        self.ledger.append({
            "event": "resolve",
            "timestamp": self.closed_at,
            "token_won": token_won,
            "realized_pnl": self.realized_pnl,
        })
        return self.realized_pnl
