"""
Tests for BTC forecast walk-forward evaluator (btc_forecast_eval.py).
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_forecast_eval import BTCForecastEvaluator


class TestBTCForecastEvaluator(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Use very short horizon for testing (2 seconds)
        self.evaluator = BTCForecastEvaluator(
            logs_dir=self.tmpdir,
            horizon_seconds=2,
            confidence_threshold=0.52,
        )

    def _make_prediction(self, direction=1, confidence=0.65, predicted_return=0.002):
        return {
            "btc_predicted_direction": direction,
            "btc_predicted_return_15": predicted_return,
            "btc_forecast_confidence": confidence,
            "btc_forecast_ready": True,
            "btc_mtf_agreement": 0.8,
            "btc_mtf_n_agree": 3,
            "btc_mtf_n_total": 3,
            "btc_mtf_source": "multi_timeframe",
        }

    def test_record_prediction_adds_to_pending(self):
        pred = self._make_prediction()
        self.evaluator.record_prediction(pred, current_price=66000.0)
        self.assertEqual(self.evaluator.pending_count, 1)

    def test_record_skips_not_ready(self):
        pred = {"btc_forecast_ready": False}
        self.evaluator.record_prediction(pred, current_price=66000.0)
        self.assertEqual(self.evaluator.pending_count, 0)

    def test_evaluate_matured_correct_bullish(self):
        """Predict UP, price goes UP → correct."""
        pred = self._make_prediction(direction=1, confidence=0.7)
        self.evaluator.record_prediction(pred, current_price=66000.0)

        # Wait for horizon to pass
        time.sleep(2.5)

        results = self.evaluator.evaluate_matured(current_price=66500.0)  # price went UP
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["correct"])
        self.assertEqual(results[0]["actual_direction"], 1)
        self.assertGreater(results[0]["actual_return"], 0)
        self.assertGreater(results[0]["pnl_pct"], 0)

    def test_evaluate_matured_incorrect_bullish(self):
        """Predict UP, price goes DOWN → incorrect."""
        pred = self._make_prediction(direction=1, confidence=0.7)
        self.evaluator.record_prediction(pred, current_price=66000.0)

        time.sleep(2.5)

        results = self.evaluator.evaluate_matured(current_price=65500.0)  # price went DOWN
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["correct"])
        self.assertEqual(results[0]["actual_direction"], -1)
        self.assertLess(results[0]["pnl_pct"], 0)

    def test_evaluate_matured_bearish(self):
        """Predict DOWN, price goes DOWN → correct."""
        pred = self._make_prediction(direction=-1, confidence=0.6)
        self.evaluator.record_prediction(pred, current_price=66000.0)

        time.sleep(2.5)

        results = self.evaluator.evaluate_matured(current_price=65800.0)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["correct"])
        self.assertGreater(results[0]["pnl_pct"], 0)  # short is profitable when price drops

    def test_pending_cleared_after_evaluation(self):
        pred = self._make_prediction()
        self.evaluator.record_prediction(pred, current_price=66000.0)
        self.assertEqual(self.evaluator.pending_count, 1)

        time.sleep(2.5)
        self.evaluator.evaluate_matured(current_price=66100.0)
        self.assertEqual(self.evaluator.pending_count, 0)

    def test_immature_predictions_stay_pending(self):
        pred = self._make_prediction()
        self.evaluator.record_prediction(pred, current_price=66000.0)

        # Don't wait — evaluate immediately
        results = self.evaluator.evaluate_matured(current_price=66100.0)
        self.assertEqual(len(results), 0)
        self.assertEqual(self.evaluator.pending_count, 1)

    def test_rolling_stats_empty(self):
        stats = self.evaluator.rolling_stats()
        self.assertEqual(stats["n_evaluated"], 0)
        self.assertEqual(stats["accuracy_pct"], 0.0)

    def test_rolling_stats_after_evaluations(self):
        # Record and evaluate 3 predictions: 2 correct, 1 wrong
        for i, (direction, exit_price) in enumerate([
            (1, 66200.0),   # correct (UP, price went up)
            (1, 65800.0),   # wrong (UP, price went down)
            (-1, 65500.0),  # correct (DOWN, price went down)
        ]):
            pred = self._make_prediction(direction=direction, confidence=0.65)
            self.evaluator.record_prediction(pred, current_price=66000.0)
            time.sleep(2.5)
            self.evaluator.evaluate_matured(current_price=exit_price)

        stats = self.evaluator.rolling_stats()
        self.assertEqual(stats["n_evaluated"], 3)
        self.assertAlmostEqual(stats["accuracy_pct"], 66.67, delta=1.0)

    def test_csv_persistence(self):
        pred = self._make_prediction()
        self.evaluator.record_prediction(pred, current_price=66000.0)
        time.sleep(2.5)
        self.evaluator.evaluate_matured(current_price=66100.0)

        # Check CSV was created
        eval_path = Path(self.tmpdir) / "btc_forecast_eval.csv"
        self.assertTrue(eval_path.exists())
        self.assertGreater(eval_path.stat().st_size, 0)

        # Load and verify
        import pandas as pd
        df = pd.read_csv(eval_path)
        self.assertEqual(len(df), 1)
        self.assertIn("predicted_direction", df.columns)
        self.assertIn("actual_direction", df.columns)
        self.assertIn("correct", df.columns)
        self.assertIn("pnl_pct", df.columns)

    def test_confident_vs_unconfident(self):
        # Confident prediction (0.65 > 0.52)
        pred1 = self._make_prediction(direction=1, confidence=0.65)
        self.evaluator.record_prediction(pred1, current_price=66000.0)
        time.sleep(2.5)
        results = self.evaluator.evaluate_matured(current_price=66200.0)
        self.assertTrue(results[0]["is_confident"])
        self.assertTrue(results[0]["confident_correct"])

        # Unconfident prediction (0.50 < 0.52)
        pred2 = self._make_prediction(direction=1, confidence=0.50)
        self.evaluator.record_prediction(pred2, current_price=66000.0)
        time.sleep(2.5)
        results = self.evaluator.evaluate_matured(current_price=66200.0)
        self.assertFalse(results[0]["is_confident"])
        self.assertIsNone(results[0]["confident_correct"])

    def test_neutral_prediction_not_counted_as_correct(self):
        pred = self._make_prediction(direction=0, confidence=0.3)
        self.evaluator.record_prediction(pred, current_price=66000.0)
        time.sleep(2.5)
        results = self.evaluator.evaluate_matured(current_price=66200.0)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["correct"])
        self.assertEqual(results[0]["pnl_pct"], 0.0)

    def test_summary(self):
        summary = self.evaluator.summary()
        self.assertIn("pending_predictions", summary)
        self.assertIn("n_evaluated", summary)
        self.assertIn("eval_log_path", summary)


if __name__ == "__main__":
    unittest.main()
