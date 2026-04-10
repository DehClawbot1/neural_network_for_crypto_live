import tempfile
import unittest
from pathlib import Path

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
