import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from shadow_purgatory import ShadowPurgatory


class TestShadowPurgatory(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmp_dir.name) / "test_shadow.csv"
        self.mock_bundle = {
            "model": MagicMock(),
            "features": ["trader_win_rate", "normalized_trade_size"],
        }
        self.mock_bundle["model"].predict_proba.return_value = np.array([[0.2, 0.8]])

        bundle_patcher = patch("shadow_purgatory.joblib.load", return_value=self.mock_bundle)
        self.addCleanup(bundle_patcher.stop)
        bundle_patcher.start()

        self.purgatory = ShadowPurgatory(
            model_bundle_path="fake_path.pkl",
            log_path=self.log_path,
        )

    def test_log_intent_veto_doa(self):
        signal = {
            "token_id": "1",
            "timestamp": "2026-03-23T12:00:00Z",
            "price": 0.5,
            "market_title": "BTC Test",
        }
        features = pd.DataFrame([
            {"trader_win_rate": 0.8, "normalized_trade_size": 0.5}
        ])

        with patch.object(self.purgatory, "_get_bucket_slippage", return_value=500.0), \
             patch.object(self.purgatory, "_get_reachable_price", return_value=(0.51, 2)):
            prob = self.purgatory.log_intent(signal, features)

        self.assertEqual(prob, 0.0)
        df = pd.read_csv(self.log_path)
        self.assertEqual(df.iloc[-1]["outcome"], "DOA")
        self.assertGreater(float(df.iloc[-1]["expected_slip_bps"]), 0)

    def test_resolve_purgatory_path_logic(self):
        row = {
            "token_id": "1",
            "shadow_entry_price": 0.5,
            "timestamp": "2025-03-23T12:00:00+00:00",
        }
        mock_trades = [
            {"timestamp": 1742731500, "price": 0.45},
            {"timestamp": 1742731800, "price": 0.55},
        ]

        with patch.object(self.purgatory.clob, "get_trades_with_retry", return_value=mock_trades):
            outcome, pnl, count = self.purgatory._check_path(row)

        self.assertEqual(outcome, "SL")
        self.assertEqual(pnl, -0.03)
        self.assertEqual(count, 1)

    def tearDown(self):
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
