import unittest
import tempfile
import os
from pathlib import Path

from pnl_engine import PNLEngine
from live_risk_manager import LiveRiskManager


class TestCoreTradingLogic(unittest.TestCase):
    def _risk(self, **kwargs):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        old_cwd = os.getcwd()
        old_risk_tier = os.environ.get("RISK_TIER")
        os.chdir(tmpdir.name)
        self.addCleanup(lambda: os.chdir(old_cwd))
        os.environ["RISK_TIER"] = "SCALED"
        self.addCleanup(
            lambda: (
                os.environ.__setitem__("RISK_TIER", old_risk_tier)
                if old_risk_tier is not None
                else os.environ.pop("RISK_TIER", None)
            )
        )
        lockfile_path = Path(tmpdir.name) / "operational_lockout.lock"
        return LiveRiskManager(lockfile_path=lockfile_path, **kwargs)

    def test_outcome_token_pnl(self):
        pnl = PNLEngine.mark_to_market_pnl(4.0, 0.40, 0.70)
        self.assertAlmostEqual(pnl, 3.0, places=6)

    def test_resolution_pnl_win(self):
        pnl = PNLEngine.resolution_pnl(4.0, 0.40, token_won=True)
        self.assertAlmostEqual(pnl, 6.0, places=6)

    def test_risk_manager_blocks_large_position(self):
        risk = self._risk(max_position_size=50)
        decision = risk.pre_trade_check(price=0.5, size=100, spread=0.01, open_orders=0, daily_pnl=0)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "max_position_size_exceeded")

    def test_risk_manager_no_longer_owns_spread_veto(self):
        risk = self._risk(max_spread=0.02)
        decision = risk.pre_trade_check(price=0.5, size=10, spread=0.05, open_orders=0, daily_pnl=0)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "ok")


if __name__ == "__main__":
    unittest.main()
