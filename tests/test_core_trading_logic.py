import unittest

from pnl_engine import PNLEngine
from live_risk_manager import LiveRiskManager


class TestCoreTradingLogic(unittest.TestCase):
    def test_outcome_token_pnl(self):
        pnl = PNLEngine.mark_to_market_pnl(4.0, 0.40, 0.70)
        self.assertAlmostEqual(pnl, 3.0, places=6)

    def test_resolution_pnl_win(self):
        pnl = PNLEngine.resolution_pnl(4.0, 0.40, token_won=True)
        self.assertAlmostEqual(pnl, 6.0, places=6)

    def test_risk_manager_blocks_large_position(self):
        risk = LiveRiskManager(max_position_size=50)
        decision = risk.pre_trade_check(price=0.5, size=100, spread=0.01, open_orders=0, daily_pnl=0)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "max_position_size_exceeded")

    def test_risk_manager_blocks_wide_spread(self):
        risk = LiveRiskManager(max_spread=0.02)
        decision = risk.pre_trade_check(price=0.5, size=10, spread=0.05, open_orders=0, daily_pnl=0)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "spread_too_wide")


if __name__ == "__main__":
    unittest.main()
