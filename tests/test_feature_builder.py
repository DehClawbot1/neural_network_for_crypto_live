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
            "alligator_alignment": "BULLISH",
            "alligator_bullish": True,
            "alligator_bearish": False,
            "adx_value": 27.4,
            "adx_threshold": 18.0,
            "adx_trending": True,
            "anchored_vwap": 66123.0,
            "price_vs_anchored_vwap": 0.0084,
            "price_above_anchored_vwap": True,
            "price_below_anchored_vwap": False,
            "btc_trend_bias": "LONG",
            "btc_trend_confluence": 1.0,
            "latest_bullish_fractal": 66550.0,
            "latest_bearish_fractal": 65910.0,
            "long_fractal_breakout": True,
            "short_fractal_breakout": False,
            "fractal_trigger_direction": "LONG",
            "fractal_entry_ready": True,
            "wallet_quality_score": 0.77,
            "wallet_watchlist_approved": True,
            "wallet_agreement_score": 0.81,
            "wallet_conflict_with_stronger": False,
            "wallet_state_gate_pass": True,
            "source_wallet_position_event": "NEW_ENTRY",
            "source_wallet_net_position_increased": True,
            "source_wallet_current_net_exposure": 500.0,
            "source_wallet_average_entry": 0.61,
            "source_wallet_current_direction": "YES",
            "source_wallet_direction_confidence": 0.83,
            "source_wallet_size_delta_ratio": 0.64,
            "source_wallet_freshness_score": 0.92,
            "source_wallet_fresh": True,
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
        self.assertEqual(row["alligator_alignment"], "BULLISH")
        self.assertEqual(row["adx_value"], 27.4)
        self.assertTrue(row["price_above_anchored_vwap"])
        self.assertEqual(row["btc_trend_bias"], "LONG")
        self.assertEqual(row["latest_bullish_fractal"], 66550.0)
        self.assertTrue(row["long_fractal_breakout"])
        self.assertTrue(row["fractal_entry_ready"])
        self.assertEqual(row["wallet_quality_score"], 0.77)
        self.assertEqual(row["wallet_agreement_score"], 0.81)
        self.assertEqual(row["wallet_support_strength"], 0.0)
        self.assertEqual(row["source_wallet_position_event"], "NEW_ENTRY")
        self.assertTrue(row["source_wallet_net_position_increased"])
        self.assertEqual(row["wallet_state_confidence"], 0.83)
        self.assertEqual(row["source_wallet_direction_confidence"], 0.83)
        self.assertEqual(row["wallet_size_change_score"], 0.64)
        self.assertTrue(row["source_wallet_fresh"])

    def test_build_feature_row_preserves_wallet_gate_reason_and_support_metrics(self):
        builder = FeatureBuilder()
        signal = {
            "trader_wallet": "0xwhale4",
            "size": 120,
            "price": 0.44,
            "timestamp": "2026-04-08T18:00:00Z",
            "outcome_side": "YES",
            "wallet_watchlist_approved": True,
            "wallet_quality_score": 0.64,
            "wallet_agreement_score": 0.38,
            "wallet_conflict_with_stronger": True,
            "wallet_stronger_conflict_score": 0.79,
            "wallet_support_strength": 0.49,
            "wallet_state_gate_pass": False,
            "wallet_state_gate_reason": "conflict_with_stronger_wallet",
            "source_wallet_position_event": "SCALE_IN",
            "source_wallet_direction_confidence": 0.76,
            "source_wallet_size_delta_ratio": 0.13,
            "source_wallet_freshness_score": 0.91,
            "source_wallet_fresh": True,
        }

        row = builder.build_feature_row(signal, {"condition_id": "cond_2", "yes_token_id": "3", "no_token_id": "4"})

        self.assertEqual(row["wallet_state_gate_reason"], "conflict_with_stronger_wallet")
        self.assertEqual(row["wallet_support_strength"], 0.49)
        self.assertEqual(row["wallet_stronger_conflict_score"], 0.79)
        self.assertEqual(row["source_wallet_direction_confidence"], 0.76)

    def test_build_feature_row_preserves_open_position_context_for_retraining(self):
        builder = FeatureBuilder()
        signal = {
            "trader_wallet": "0xwhale5",
            "size": 180,
            "price": 0.57,
            "timestamp": "2026-04-09T05:00:00Z",
            "outcome_side": "YES",
            "open_positions_count": 2,
            "open_positions_negotiated_value_total": 5.70,
            "open_positions_max_payout_total": 14.20,
            "open_positions_current_value_total": 5.42,
            "open_positions_unrealized_pnl_total": -0.28,
            "open_positions_unrealized_pnl_pct_total": -0.0491228,
            "open_positions_avg_to_now_price_change_pct_mean": -0.036,
            "open_positions_avg_to_now_price_change_pct_min": -0.081,
            "open_positions_avg_to_now_price_change_pct_max": 0.012,
            "open_positions_winner_count": 1,
            "open_positions_loser_count": 1,
        }

        row = builder.build_feature_row(signal, {"condition_id": "cond_3", "yes_token_id": "5", "no_token_id": "6"})

        self.assertEqual(row["open_positions_count"], 2)
        self.assertEqual(row["open_positions_negotiated_value_total"], 5.70)
        self.assertEqual(row["open_positions_current_value_total"], 5.42)
        self.assertAlmostEqual(row["open_positions_unrealized_pnl_pct_total"], -0.0491228)
        self.assertEqual(row["open_positions_avg_to_now_price_change_pct_min"], -0.081)
        self.assertEqual(row["open_positions_winner_count"], 1)
        self.assertEqual(row["open_positions_loser_count"], 1)


if __name__ == "__main__":
    unittest.main()
