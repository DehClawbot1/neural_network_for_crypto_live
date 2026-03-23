from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RiskDecision:
    allowed: bool
    reason: str


class LiveRiskManager:
    """
    Live-test pre-trade and operational risk controls.
    """

    def __init__(self, max_position_size=100.0, max_open_orders=10, max_daily_loss=200.0, max_spread=0.05, cooldown_after_loss_minutes=15, max_failed_orders=3):
        self.max_position_size = max_position_size
        self.max_open_orders = max_open_orders
        self.max_daily_loss = max_daily_loss
        self.max_spread = max_spread
        self.cooldown_after_loss_minutes = cooldown_after_loss_minutes
        self.max_failed_orders = max_failed_orders
        self.last_loss_time = None
        self.failed_orders = 0
        self.kill_switch = False

    def pre_trade_check(self, price, size, spread=None, open_orders=0, daily_pnl=0.0):
        if self.kill_switch:
            return RiskDecision(False, "kill_switch_enabled")
        if float(size) > self.max_position_size:
            return RiskDecision(False, "max_position_size_exceeded")
        if open_orders >= self.max_open_orders:
            return RiskDecision(False, "max_open_orders_exceeded")
        if daily_pnl <= -abs(self.max_daily_loss):
            return RiskDecision(False, "max_daily_loss_hit")
        if spread is not None and float(spread) > self.max_spread:
            return RiskDecision(False, "spread_too_wide")
        if self.last_loss_time and datetime.utcnow() - self.last_loss_time < timedelta(minutes=self.cooldown_after_loss_minutes):
            return RiskDecision(False, "cooldown_after_loss")
        if self.failed_orders >= self.max_failed_orders:
            return RiskDecision(False, "circuit_breaker_failed_orders")
        return RiskDecision(True, "ok")

    def record_failed_order(self):
        self.failed_orders += 1

    def record_loss(self):
        self.last_loss_time = datetime.utcnow()

    def activate_kill_switch(self):
        self.kill_switch = True

    def deactivate_kill_switch(self):
        self.kill_switch = False
        self.failed_orders = 0

