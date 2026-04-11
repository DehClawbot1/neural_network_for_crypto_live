import unittest
import json

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

    def test_on_signal_persists_full_entry_snapshot_for_learning(self):
        tl = TradeLifecycle(market="BTC Test", token_id="1", condition_id="abc", outcome_side="YES")
        tl.on_signal(
            {
                "market": "BTC Test",
                "confidence": 0.42,
                "btc_live_index_price": 68123.4,
                "reddit_sentiment": -0.21,
                "open_positions_count": 3,
            }
        )

        snapshot = json.loads(tl.entry_signal_snapshot_json)
        self.assertEqual(tl.entry_signal_snapshot_feature_count, len(snapshot))
        self.assertEqual(snapshot["btc_live_index_price"], 68123.4)
        self.assertEqual(snapshot["reddit_sentiment"], -0.21)
        self.assertEqual(snapshot["open_positions_count"], 3)

    def test_on_signal_handles_nan_integer_like_fields(self):
        tl = TradeLifecycle(market="BTC Test", token_id="1", condition_id="abc", outcome_side="YES")
        tl.on_signal(
            {
                "market": "BTC Test",
                "btc_predicted_direction": float("nan"),
                "performance_governor_level": float("nan"),
            }
        )

        self.assertEqual(tl.entry_btc_predicted_direction, 0)
        self.assertEqual(tl.performance_governor_level, 0)


    def test_trade_lifecycle_dict_contains_expected_fields(self):
        tl = TradeLifecycle(market="BTC Test", token_id="1", condition_id="abc", outcome_side="YES")
        d = tl.__dict__
        for key in ["market", "token_id", "condition_id", "outcome_side", "state", "size_usdc", "ledger"]:
            self.assertIn(key, d)

    def test_ledger_default_is_empty_list(self):
        tl = TradeLifecycle(market="X", token_id=None, condition_id=None, outcome_side="YES")
        self.assertIsInstance(tl.ledger, list)
        self.assertEqual(len(tl.ledger), 0)


def test_serialize_signal_snapshot_with_simple_dict():
    from trade_lifecycle import serialize_signal_snapshot

    json_str, count = serialize_signal_snapshot({"a": 1, "b": "hello"})
    self_parsed = __import__("json").loads(json_str)
    assert self_parsed == {"a": 1, "b": "hello"}
    assert count == 2


def test_serialize_signal_snapshot_with_empty_dict():
    from trade_lifecycle import serialize_signal_snapshot

    json_str, count = serialize_signal_snapshot({})
    assert count == 0


if __name__ == "__main__":
    unittest.main()
