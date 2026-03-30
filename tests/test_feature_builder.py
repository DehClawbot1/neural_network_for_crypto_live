import unittest

from feature_builder import FeatureBuilder


class TestFeatureLogic(unittest.TestCase):
    def test_wallet_history_stat_updates(self):
        builder = FeatureBuilder()
        wallet = "0xwhale1"

        builder.update_wallet_history(
            {
                "trader_wallet": wallet,
                "size": 1000,
                "future_return": 0.10,
                "tp_before_sl_60m": 1,
                "market_title": "BTC Test",
            }
        )

        builder.update_wallet_history(
            {
                "trader_wallet": wallet,
                "size": 2000,
                "future_return": -0.05,
                "tp_before_sl_60m": 0,
                "market_title": "BTC Test",
            }
        )

        stats = builder.wallet_stats[wallet]
        self.assertEqual(stats["trade_count"], 2)
        self.assertEqual(stats["avg_size"], 1500.0)
        self.assertEqual(stats["win_rate"], 0.5)
        self.assertEqual(stats["tp_precision"], 0.5)
        self.assertEqual(stats["same_market_history"], 2)

    def test_normalized_size_clipping(self):
        builder = FeatureBuilder()
        wallet = "0xwhale2"
        builder.wallet_stats[wallet] = {"avg_size": 100.0}

        norm_size = builder._normalized_trade_size(wallet, 1000.0)
        self.assertLessEqual(norm_size, 1.0)
        self.assertGreaterEqual(norm_size, 0.0)


if __name__ == "__main__":
    unittest.main()
