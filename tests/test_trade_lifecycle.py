import unittest

from trade_lifecycle import TradeLifecycle, TradeState


class TestTradeLifecycle(unittest.TestCase):
    def test_enter_hold_close_flow(self):
        tl = TradeLifecycle(market="BTC Test", token_id="1", condition_id="abc", outcome_side="YES")
        tl.on_signal({"market": "BTC Test"})
        tl.enter(size_usdc=4.0, entry_price=0.40)
        self.assertEqual(tl.state, TradeState.ENTERED)
        tl.update_market(0.70)
        self.assertEqual(tl.state, TradeState.OPEN)
        self.assertGreater(tl.unrealized_pnl, 0)
        tl.close(0.70)
        self.assertEqual(tl.state, TradeState.CLOSED)
        self.assertGreater(tl.realized_pnl, 0)

    def test_partial_exit(self):
        tl = TradeLifecycle(market="BTC Test", token_id="1", condition_id="abc", outcome_side="YES")
        tl.enter(size_usdc=4.0, entry_price=0.40)
        pnl = tl.partial_exit(0.5, 0.70)
        self.assertEqual(tl.state, TradeState.PARTIAL_EXIT)
        self.assertGreater(pnl, 0)
        self.assertGreaterEqual(tl.shares, 0)

    def test_tracks_open_trade_pain_metrics(self):
        tl = TradeLifecycle(market="BTC Test", token_id="1", condition_id="abc", outcome_side="YES")
        tl.enter(size_usdc=4.0, entry_price=0.40)
        tl.update_market(0.36)
        tl.update_market(0.34)
        self.assertLessEqual(tl.max_adverse_excursion_pct, -0.10)
        self.assertGreaterEqual(tl.fast_adverse_move_count, 1)
        self.assertGreaterEqual(tl.max_drawdown_from_peak_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
