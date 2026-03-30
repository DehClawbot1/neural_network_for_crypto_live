from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging


@dataclass
class RiskDecision:
    allowed: bool
    reason: str


class LiveRiskManager:
    """
    Live-test pre-trade and operational risk controls with optional DB logging.
    """

    def __init__(self, db=None, max_position_size=100.0, max_open_orders=10, max_daily_loss=200.0, max_spread=0.05, cooldown_after_loss_minutes=15, max_failed_orders=10):
        self.db = db
        self.max_position_size = max_position_size
        self.max_open_orders = max_open_orders
        self.max_daily_loss = max_daily_loss
        self.max_spread = max_spread
        self.cooldown_after_loss_minutes = cooldown_after_loss_minutes
        self.max_failed_orders = max_failed_orders
        self.last_loss_time = None
        self.failed_orders = 0
        self.kill_switch = False

    def _evaluate(self, price, size, spread=None, open_orders=0, daily_pnl=0.0):
        if self.kill_switch:
            return RiskDecision(False, "kill_switch_enabled")
        if float(size or 0.0) > self.max_position_size: # BUG FIX 10
            return RiskDecision(False, "max_position_size_exceeded")
        if open_orders >= self.max_open_orders:
            return RiskDecision(False, "max_open_orders_exceeded")
        if daily_pnl <= -abs(self.max_daily_loss):
            return RiskDecision(False, "max_daily_loss_hit")
        if spread is not None and float(spread) > self.max_spread:
            return RiskDecision(False, "spread_too_wide")
        if self.last_loss_time and datetime.now(timezone.utc) - self.last_loss_time < timedelta(minutes=self.cooldown_after_loss_minutes):
            return RiskDecision(False, "cooldown_after_loss")
        if self.failed_orders >= self.max_failed_orders:
            return RiskDecision(False, "circuit_breaker_failed_orders")
        return RiskDecision(True, "ok")

    def pre_trade_check(self, token_id=None, price=0.0, size=0.0, spread=None, open_orders=0, daily_pnl=0.0):
        decision = self._evaluate(price=price, size=size, spread=spread, open_orders=open_orders, daily_pnl=daily_pnl)
        if not decision.allowed and self.db is not None:
            try:
                detail = f"price={price}, size={size}, spread={spread}, daily_pnl={daily_pnl}"
                self.db.execute(
                    "INSERT INTO risk_events (token_id, event_type, detail) VALUES (?, ?, ?)",
                    (str(token_id) if token_id is not None else None, decision.reason, detail),
                )
                if hasattr(self.db, "commit"): self.db.commit() # BUG FIX 6: Ensure audit logs save
            except Exception as exc:
                logging.error("Failed to log risk event to DB: %s", exc)
        return decision

    def record_failed_order(self):
        self.failed_orders += 1

    def record_successful_order(self):
        self.failed_orders = 0 # BUG FIX 7: Decay circuit breaker on success

    def record_loss(self):
        self.last_loss_time = datetime.now(timezone.utc)

    def reset_failed_orders(self):
        self.failed_orders = 0

    def activate_kill_switch(self):
        self.kill_switch = True

    def deactivate_kill_switch(self):
        self.kill_switch = False
        self.failed_orders = 0


