from live_risk_manager import LiveRiskManager


def test_cooldown_after_loss_blocks_new_trade():
    risk = LiveRiskManager(cooldown_after_loss_minutes=15)
    risk.record_loss()
    decision = risk.pre_trade_check(price=0.5, size=10, spread=0.01, open_orders=0, daily_pnl=0)
    assert decision.allowed is False
    assert decision.reason == "cooldown_after_loss"


def test_failed_order_circuit_breaker_blocks_trading():
    risk = LiveRiskManager(max_failed_orders=2)
    risk.record_failed_order()
    risk.record_failed_order()
    decision = risk.pre_trade_check(price=0.5, size=10, spread=0.01, open_orders=0, daily_pnl=0)
    assert decision.allowed is False
    assert decision.reason == "circuit_breaker_failed_orders"


def test_kill_switch_can_be_reset():
    risk = LiveRiskManager()
    risk.activate_kill_switch()
    blocked = risk.pre_trade_check(price=0.5, size=10, spread=0.01, open_orders=0, daily_pnl=0)
    assert blocked.allowed is False
    assert blocked.reason == "kill_switch_enabled"

    risk.failed_orders = 2
    risk.deactivate_kill_switch()
    allowed = risk.pre_trade_check(price=0.5, size=10, spread=0.01, open_orders=0, daily_pnl=0)
    assert allowed.allowed is True
    assert allowed.reason == "ok"
    assert risk.failed_orders == 0
