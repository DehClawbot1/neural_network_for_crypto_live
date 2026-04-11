import tempfile
import unittest
from pathlib import Path

import pandas as pd
from trade_lifecycle import serialize_signal_snapshot

from contract_target_builder import ContractTargetBuilder


class TestContractTargetBuilder(unittest.TestCase):
    def test_build_from_token_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            pd.DataFrame([
                {
                    "market": "BTC Test",
                    "timestamp": "2026-03-22T00:00:00Z",
                    "token_id": "1",
                    "side": "YES",
                    "confidence": 0.8,
                    "btc_live_index_price": 68111.0,
                    "reddit_sentiment": -0.14,
                    "open_positions_count": 2,
                    "open_positions_unrealized_pnl_pct_total": -0.04,
                }
            ]).to_csv(logs / "signals.csv", index=False)
            pd.DataFrame([
                {"question": "BTC Test", "yes_token_id": "1", "no_token_id": "2"}
            ]).to_csv(logs / "markets.csv", index=False)
            pd.DataFrame([
                {"token_id": "1", "timestamp": "2026-03-22T00:00:00Z", "price": 0.40},
                {"token_id": "1", "timestamp": "2026-03-22T00:10:00Z", "price": 0.50},
                {"token_id": "1", "timestamp": "2026-03-22T00:20:00Z", "price": 0.60},
            ]).to_csv(logs / "clob_price_history.csv", index=False)

            df = ContractTargetBuilder(logs_dir=logs).build()
            self.assertFalse(df.empty)
            self.assertIn("forward_return_15m", df.columns)
            self.assertIn("tp_before_sl_60m", df.columns)
            self.assertEqual(float(df.iloc[0]["btc_live_index_price"]), 68111.0)
            self.assertEqual(float(df.iloc[0]["reddit_sentiment"]), -0.14)
            self.assertEqual(int(df.iloc[0]["open_positions_count"]), 2)
            self.assertAlmostEqual(float(df.iloc[0]["open_positions_unrealized_pnl_pct_total"]), -0.04)

    def test_build_from_token_history_merges_kalman_and_regime_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            pd.DataFrame([
                {
                    "market": "BTC Test",
                    "timestamp": "2026-03-22T00:00:00Z",
                    "token_id": "1",
                    "side": "YES",
                    "confidence": 0.8,
                }
            ]).to_csv(logs / "signals.csv", index=False)
            pd.DataFrame([
                {"question": "BTC Test", "yes_token_id": "1", "no_token_id": "2"}
            ]).to_csv(logs / "markets.csv", index=False)
            pd.DataFrame([
                {"token_id": "1", "timestamp": "2026-03-22T00:00:00Z", "price": 0.40},
                {"token_id": "1", "timestamp": "2026-03-22T00:10:00Z", "price": 0.50},
                {"token_id": "1", "timestamp": "2026-03-22T00:20:00Z", "price": 0.60},
            ]).to_csv(logs / "clob_price_history.csv", index=False)
            pd.DataFrame([
                {
                    "btc_live_timestamp": "2026-03-21T23:59:30Z",
                    "btc_live_mark_price_kalman": 68123.4,
                    "btc_live_return_5m_kalman": 0.0041,
                }
            ]).to_csv(logs / "btc_live_snapshot.csv", index=False)
            pd.DataFrame([
                {
                    "technical_timestamp": "2026-03-21T23:59:40Z",
                    "btc_market_regime_label": "volatile",
                    "btc_market_regime_score": 0.69,
                    "btc_market_regime_weight_stage2": 0.45,
                }
            ]).to_csv(logs / "technical_regime_snapshot.csv", index=False)

            df = ContractTargetBuilder(logs_dir=logs).build()
            self.assertFalse(df.empty)
            self.assertEqual(round(float(df.iloc[0]["btc_live_mark_price_kalman"]), 1), 68123.4)
            self.assertEqual(df.iloc[0]["btc_market_regime_label"], "volatile")
            self.assertAlmostEqual(float(df.iloc[0]["btc_market_regime_weight_stage2"]), 0.45)

    def test_build_from_token_history_backfills_missing_signal_features_from_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            pd.DataFrame([
                {
                    "market": "BTC Test",
                    "timestamp": "2026-03-22T00:00:00Z",
                    "token_id": "1",
                    "side": "YES",
                }
            ]).to_csv(logs / "signals.csv", index=False)
            snapshot_json, feature_count = serialize_signal_snapshot(
                {
                    "market": "BTC Test",
                    "timestamp": "2026-03-22T00:00:00Z",
                    "token_id": "1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "trader_wallet": "0xabc",
                    "wallet_quality_score": 0.78,
                }
            )
            pd.DataFrame([
                {
                    "position_id": "1|cond-1|YES",
                    "market": "BTC Test",
                    "market_title": "BTC Test",
                    "token_id": "1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "opened_at": "2026-03-22T00:00:00Z",
                    "status": "OPEN",
                    "entry_signal_snapshot_json": snapshot_json,
                    "entry_signal_snapshot_feature_count": feature_count,
                    "entry_signal_snapshot_version": 1,
                }
            ]).to_csv(logs / "positions.csv", index=False)
            pd.DataFrame([
                {"question": "BTC Test", "yes_token_id": "1", "no_token_id": "2", "condition_id": "cond-1"}
            ]).to_csv(logs / "markets.csv", index=False)
            pd.DataFrame([
                {"token_id": "1", "timestamp": "2026-03-22T00:00:00Z", "price": 0.40},
                {"token_id": "1", "timestamp": "2026-03-22T00:10:00Z", "price": 0.50},
                {"token_id": "1", "timestamp": "2026-03-22T00:20:00Z", "price": 0.60},
            ]).to_csv(logs / "clob_price_history.csv", index=False)

            df = ContractTargetBuilder(logs_dir=logs).build()
            self.assertFalse(df.empty)
            self.assertEqual(round(float(df.iloc[0]["wallet_quality_score"]), 2), 0.78)
            self.assertEqual(df.iloc[0]["trader_wallet"], "0xabc")

    def test_build_prefers_historical_dataset_backbone_for_feature_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            pd.DataFrame([
                {
                    "market": "BTC Test",
                    "timestamp": "2026-03-22T00:00:00Z",
                    "token_id": "1",
                    "side": "YES",
                    "confidence": 0.8,
                    "wallet_trade_count_30d": None,
                    "btc_live_index_price": None,
                }
            ]).to_csv(logs / "signals.csv", index=False)
            pd.DataFrame([
                {
                    "market_title": "BTC Test",
                    "timestamp": "2026-03-22T00:00:00Z",
                    "token_id": "1",
                    "outcome_side": "YES",
                    "wallet_trade_count_30d": 7.0,
                    "btc_live_index_price": 68111.0,
                    "spread": 0.02,
                }
            ]).to_csv(logs / "historical_dataset.csv", index=False)
            pd.DataFrame([
                {"question": "BTC Test", "yes_token_id": "1", "no_token_id": "2"}
            ]).to_csv(logs / "markets.csv", index=False)
            pd.DataFrame([
                {"token_id": "1", "timestamp": "2026-03-22T00:00:00Z", "price": 0.40},
                {"token_id": "1", "timestamp": "2026-03-22T00:10:00Z", "price": 0.50},
                {"token_id": "1", "timestamp": "2026-03-22T00:20:00Z", "price": 0.60},
            ]).to_csv(logs / "clob_price_history.csv", index=False)

            df = ContractTargetBuilder(logs_dir=logs).build()

            self.assertFalse(df.empty)
            self.assertEqual(float(df.iloc[0]["wallet_trade_count_30d"]), 7.0)
            self.assertEqual(float(df.iloc[0]["btc_live_index_price"]), 68111.0)
            self.assertEqual(float(df.iloc[0]["spread"]), 0.02)


if __name__ == "__main__":
    unittest.main()
