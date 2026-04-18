import tempfile
import os
from pathlib import Path

import pytest

from live_risk_manager import LiveRiskManager


@pytest.fixture
def isolated_risk():
    resources = []

    def _make(**kwargs):
        tmpdir = tempfile.TemporaryDirectory()
        old_cwd = os.getcwd()
        old_risk_tier = os.environ.get("RISK_TIER")
        os.chdir(tmpdir.name)
        os.environ["RISK_TIER"] = "SCALED"
        lockfile_path = Path(tmpdir.name) / "operational_lockout.lock"
        risk = LiveRiskManager(lockfile_path=lockfile_path, **kwargs)
        resources.append((tmpdir, old_cwd, old_risk_tier))
        return risk

    yield _make

    while resources:
        tmpdir, old_cwd, old_risk_tier = resources.pop()
        os.chdir(old_cwd)
        if old_risk_tier is None:
            os.environ.pop("RISK_TIER", None)
        else:
            os.environ["RISK_TIER"] = old_risk_tier
        tmpdir.cleanup()


def _risk(**kwargs):
    tmpdir = tempfile.TemporaryDirectory()
    old_cwd = os.getcwd()
    os.chdir(tmpdir.name)
    lockfile_path = Path(tmpdir.name) / "operational_lockout.lock"
    risk = LiveRiskManager(lockfile_path=lockfile_path, **kwargs)
    risk._tmpdir = tmpdir
    risk._old_cwd = old_cwd
    return risk


def test_cooldown_after_loss_blocks_new_trade(isolated_risk):
    risk = isolated_risk(cooldown_after_loss_minutes=15)
    risk.record_loss()
    decision = risk.pre_trade_check(price=0.5, size=10, spread=0.01, open_orders=0, daily_pnl=0)
    assert decision.allowed is False
    assert decision.reason == "cooldown_after_loss"


def test_failed_order_circuit_breaker_blocks_trading(isolated_risk):
    risk = isolated_risk(max_failed_orders=2)
    risk.record_failed_order()
    risk.record_failed_order()
    decision = risk.pre_trade_check(price=0.5, size=10, spread=0.01, open_orders=0, daily_pnl=0)
    assert decision.allowed is False
    assert decision.reason == "kill_switch_enabled"


def test_kill_switch_can_be_reset(isolated_risk):
    risk = isolated_risk()
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
