"""
Tests for BTC price forecast pipeline:
  - btc_price_dataset.py (feature engineering + labelling)
  - btc_forecast_model.py (training + prediction)
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# sklearn has a known scipy version incompatibility on some environments
_sklearn_available = False
try:
    from sklearn.metrics import mean_absolute_error  # noqa: F401
    _sklearn_available = True
except (ImportError, TypeError):
    pass

_skip_sklearn = unittest.skipUnless(_sklearn_available, "sklearn/scipy version incompatibility")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_price_dataset import BTCPriceDatasetBuilder
from btc_forecast_model import BTCForecastModel


def _make_synthetic_candles(n: int = 500, start_price: float = 50000.0) -> pd.DataFrame:
    """Generate synthetic BTC-like OHLCV candles for testing."""
    np.random.seed(42)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    returns = np.random.normal(0.0001, 0.005, n)
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
    open_ = close * (1 + np.random.normal(0, 0.001, n))
    volume = np.random.uniform(100, 10000, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestBTCPriceDatasetBuilder(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.builder = BTCPriceDatasetBuilder(logs_dir=self.tmpdir)
        self.candles = _make_synthetic_candles(500)

    def test_build_from_candles_produces_features_and_labels(self):
        df = self.builder.build_from_candles(self.candles)
        self.assertGreater(len(df), 0, "Should produce non-empty dataset")

        # Check feature columns exist
        for col in ["rsi_14", "macd", "atr_pct", "bb_position", "adx", "return_1"]:
            self.assertIn(col, df.columns, f"Missing feature: {col}")

        # Check label columns exist
        for col in ["fwd_return_15", "fwd_up_15", "fwd_direction_15"]:
            self.assertIn(col, df.columns, f"Missing label: {col}")

    def test_no_nan_in_output(self):
        df = self.builder.build_from_candles(self.candles)
        nan_counts = df.isna().sum()
        bad = nan_counts[nan_counts > 0]
        self.assertEqual(len(bad), 0, f"NaN found in columns: {bad.to_dict()}")

    def test_labels_are_bounded(self):
        df = self.builder.build_from_candles(self.candles)
        # Binary labels should be 0 or 1
        for h in ["5", "15", "60"]:
            col = f"fwd_up_{h}"
            if col in df.columns:
                vals = df[col].unique()
                self.assertTrue(set(vals).issubset({0, 1}), f"{col} should be binary: {vals}")

    def test_too_few_candles_returns_empty(self):
        small = self.candles.head(50)
        df = self.builder.build_from_candles(small)
        self.assertEqual(len(df), 0)

    def test_append_to_disk_and_load(self):
        df = self.builder.build_from_candles(self.candles)
        rows_written = self.builder.append_to_disk(df)
        self.assertGreater(rows_written, 0)

        loaded = self.builder.load_dataset()
        self.assertEqual(len(loaded), len(df))

    def test_build_from_csv(self):
        csv_path = Path(self.tmpdir) / "test_candles.csv"
        self.candles.to_csv(csv_path, index=False)
        df = self.builder.build_from_csv(csv_path)
        self.assertGreater(len(df), 0)

    def test_column_name_variants(self):
        """Test that alternative column names (Open, Close, etc.) are handled."""
        alt = self.candles.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        df = self.builder.build_from_candles(alt)
        self.assertGreater(len(df), 0)


class TestBTCForecastModel(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.weights_dir = os.path.join(self.tmpdir, "weights")
        self.logs_dir = os.path.join(self.tmpdir, "logs")
        self.candles = _make_synthetic_candles(500)
        self.builder = BTCPriceDatasetBuilder(logs_dir=self.logs_dir)
        self.dataset = self.builder.build_from_candles(self.candles)

    @_skip_sklearn
    def test_train_produces_metrics(self):
        model = BTCForecastModel(weights_dir=self.weights_dir, logs_dir=self.logs_dir)
        metrics = model.train(self.dataset)
        self.assertNotIn("error", metrics, f"Training failed: {metrics}")
        self.assertIn("mae", metrics)
        self.assertIn("direction_accuracy", metrics)
        self.assertGreater(metrics["train_rows"], 0)

    @_skip_sklearn
    def test_predict_after_train(self):
        model = BTCForecastModel(weights_dir=self.weights_dir, logs_dir=self.logs_dir)
        model.train(self.dataset)

        # Predict from last feature row
        pred = model.predict(self.dataset.iloc[-1])
        self.assertTrue(pred["btc_forecast_ready"])
        self.assertIn(pred["btc_predicted_direction"], [-1, 0, 1])
        self.assertIsInstance(pred["btc_predicted_return_15"], float)
        self.assertGreater(pred["btc_forecast_confidence"], 0.0)

    def test_predict_without_model_returns_defaults(self):
        model = BTCForecastModel(weights_dir=self.weights_dir, logs_dir=self.logs_dir)
        pred = model.predict({"rsi_14": 50.0})
        self.assertFalse(pred["btc_forecast_ready"])

    @_skip_sklearn
    def test_model_persistence(self):
        """Train, save, reload from disk, predict."""
        model = BTCForecastModel(weights_dir=self.weights_dir, logs_dir=self.logs_dir)
        model.train(self.dataset)
        pred1 = model.predict(self.dataset.iloc[-1])

        # Load fresh instance from same weights dir
        model2 = BTCForecastModel(weights_dir=self.weights_dir, logs_dir=self.logs_dir)
        self.assertTrue(model2.is_ready, "Model should load from disk")
        pred2 = model2.predict(self.dataset.iloc[-1])
        self.assertEqual(pred1["btc_predicted_direction"], pred2["btc_predicted_direction"])

    @_skip_sklearn
    def test_train_rejects_tiny_dataset(self):
        tiny = self.dataset.head(10)
        model = BTCForecastModel(weights_dir=self.weights_dir, logs_dir=self.logs_dir)
        metrics = model.train(tiny)
        self.assertIn("error", metrics)


if __name__ == "__main__":
    unittest.main()
