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

    def test_build_feature_row_preserves_open_source_btc_monitoring(self):
        builder = FeatureBuilder()
        signal = {
            "trader_wallet": "0xwhale3",
            "size": 500,
            "price": 0.61,
            "timestamp": "2026-04-02T03:00:00Z",
            "outcome_side": "UP",
            "btc_fee_pressure_score": 0.82,
            "btc_mempool_congestion_score": 0.67,
            "btc_network_activity_score": 0.74,
            "btc_network_stress_score": 0.71,
            "btc_fee_fastest_satvb": 12,
            "btc_fee_hour_satvb": 7,
            "btc_difficulty_change_pct": 4.4,
            "btc_mempool_tx_count": 245000,
            "btc_mempool_vsize": 14000000,
            "onchain_network_health": "BUSY",
        }
        market = {
            "condition_id": "cond_1",
            "yes_token_id": "1",
            "no_token_id": "2",
            "best_bid": 0.60,
            "best_ask": 0.62,
            "liquidity": 25000,
            "volume": 90000,
            "last_trade_price": 0.605,
        }

        row = builder.build_feature_row(signal, market)

        self.assertEqual(row["btc_fee_pressure_score"], 0.82)
        self.assertEqual(row["btc_mempool_congestion_score"], 0.67)
        self.assertEqual(row["btc_network_activity_score"], 0.74)
        self.assertEqual(row["btc_network_stress_score"], 0.71)
        self.assertEqual(row["btc_fee_fastest_satvb"], 12.0)
        self.assertEqual(row["onchain_network_health"], "BUSY")


if __name__ == "__main__":
    unittest.main()
