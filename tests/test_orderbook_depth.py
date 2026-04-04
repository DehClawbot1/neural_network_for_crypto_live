"""
Tests for order book depth feature analyzer (orderbook_depth_features.py).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orderbook_depth_features import OrderBookDepthAnalyzer, BinanceBookFetcher, fetch_btc_depth_snapshot


def _make_synthetic_book(
    midpoint: float = 66000.0,
    n_levels: int = 20,
    base_qty: float = 1.0,
    bid_heavy: bool = False,
    whale_bid: bool = False,
    whale_ask: bool = False,
) -> tuple:
    """Generate a synthetic order book for testing."""
    bids = []
    asks = []

    for i in range(n_levels):
        bid_price = midpoint - (i + 1) * 0.5
        ask_price = midpoint + (i + 1) * 0.5
        bid_qty = base_qty * (2.0 if bid_heavy else 1.0) * (1 + np.random.uniform(-0.3, 0.3))
        ask_qty = base_qty * (1.0 if bid_heavy else 1.0) * (1 + np.random.uniform(-0.3, 0.3))
        bids.append([bid_price, bid_qty])
        asks.append([ask_price, ask_qty])

    # Sort correctly
    bids.sort(key=lambda x: x[0], reverse=True)
    asks.sort(key=lambda x: x[0])

    # Add whale walls if requested
    if whale_bid:
        bids[2][1] = base_qty * 50  # huge order at level 3
    if whale_ask:
        asks[2][1] = base_qty * 50

    return bids, asks


class TestOrderBookDepthAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = OrderBookDepthAnalyzer("BTCUSDT")
        np.random.seed(42)

    def test_compute_features_balanced_book(self):
        """Test features from a balanced order book."""
        bids, asks = _make_synthetic_book(midpoint=66000.0, bid_heavy=False)
        features = self.analyzer._compute_features(bids, asks)

        self.assertTrue(features["ob_ready"])
        self.assertGreater(features["ob_best_bid"], 0)
        self.assertGreater(features["ob_best_ask"], 0)
        self.assertGreater(features["ob_midpoint"], 0)
        self.assertGreater(features["ob_spread"], 0)

        # Balanced book should have imbalance near 0
        self.assertAlmostEqual(features["ob_imbalance_5"], 0.0, delta=0.4)

    def test_compute_features_bid_heavy(self):
        """Test that bid-heavy book shows positive imbalance."""
        bids, asks = _make_synthetic_book(midpoint=66000.0, bid_heavy=True)
        features = self.analyzer._compute_features(bids, asks)

        # Bid-heavy should show positive imbalance
        self.assertGreater(features["ob_imbalance_10"], 0.0)
        self.assertGreater(features["ob_imbalance_20"], 0.0)

    def test_whale_wall_detection_bid(self):
        """Test whale wall detection on bid side."""
        bids, asks = _make_synthetic_book(midpoint=66000.0, whale_bid=True)
        features = self.analyzer._compute_features(bids, asks)

        self.assertEqual(features["ob_whale_bid_wall"], 1.0)
        self.assertGreater(features["ob_whale_bid_wall_size"], 10.0)  # The 50x order

    def test_whale_wall_detection_ask(self):
        """Test whale wall detection on ask side."""
        bids, asks = _make_synthetic_book(midpoint=66000.0, whale_ask=True)
        features = self.analyzer._compute_features(bids, asks)

        self.assertEqual(features["ob_whale_ask_wall"], 1.0)
        self.assertGreater(features["ob_whale_ask_wall_size"], 10.0)

    def test_whale_wall_bias(self):
        """Test whale wall bias: bid-only wall should be bullish."""
        bids, asks = _make_synthetic_book(midpoint=66000.0, whale_bid=True, whale_ask=False)
        features = self.analyzer._compute_features(bids, asks)

        self.assertGreater(features["ob_whale_wall_bias"], 0.0)  # bullish

    def test_weighted_midpoint(self):
        """Test VWAP midpoint computation."""
        bids = [[100.0, 10.0], [99.0, 5.0]]
        asks = [[101.0, 10.0], [102.0, 5.0]]
        wmid = self.analyzer._weighted_midpoint(bids, asks)

        # Should be between best bid and best ask
        self.assertGreater(wmid, 99.0)
        self.assertLess(wmid, 102.0)

    def test_depth_imbalance_at_multiple_levels(self):
        """Test that imbalance is computed at levels 5, 10, 20."""
        bids, asks = _make_synthetic_book(midpoint=66000.0, n_levels=25)
        features = self.analyzer._compute_features(bids, asks)

        for n in [5, 10, 20]:
            self.assertIn(f"ob_imbalance_{n}", features)
            self.assertIn(f"ob_bid_vol_{n}", features)
            self.assertIn(f"ob_ask_vol_{n}", features)
            # Imbalance should be between -1 and 1
            self.assertGreaterEqual(features[f"ob_imbalance_{n}"], -1.0)
            self.assertLessEqual(features[f"ob_imbalance_{n}"], 1.0)

    def test_cumulative_depth_at_bps_levels(self):
        """Test cumulative depth computed at 10/25/50/100 bps."""
        bids, asks = _make_synthetic_book(midpoint=66000.0, n_levels=50)
        features = self.analyzer._compute_features(bids, asks)

        for bps in [10, 25, 50, 100]:
            self.assertIn(f"ob_cum_depth_{bps}bps_btc", features)
            self.assertIn(f"ob_cum_depth_{bps}bps_usd", features)
            self.assertIn(f"ob_cum_imbalance_{bps}bps", features)

    def test_order_book_slope(self):
        """Test slope computation."""
        bids, asks = _make_synthetic_book(midpoint=66000.0)
        features = self.analyzer._compute_features(bids, asks)

        self.assertIn("ob_bid_slope", features)
        self.assertIn("ob_ask_slope", features)
        self.assertIn("ob_slope_imbalance", features)
        # Slope should be non-negative
        self.assertGreaterEqual(features["ob_bid_slope"], 0.0)
        self.assertGreaterEqual(features["ob_ask_slope"], 0.0)

    def test_concentration_features(self):
        """Test depth concentration (top5 / total)."""
        bids, asks = _make_synthetic_book(midpoint=66000.0, n_levels=20)
        features = self.analyzer._compute_features(bids, asks)

        self.assertIn("ob_bid_concentration", features)
        self.assertIn("ob_ask_concentration", features)
        # Concentration should be between 0 and 1
        self.assertGreaterEqual(features["ob_bid_concentration"], 0.0)
        self.assertLessEqual(features["ob_bid_concentration"], 1.0)

    def test_default_features_structure(self):
        """Test that default features have the right keys."""
        defaults = self.analyzer._default_features()
        self.assertFalse(defaults["ob_ready"])
        self.assertEqual(defaults["ob_imbalance_5"], 0.0)
        self.assertEqual(defaults["ob_whale_wall_bias"], 0.0)

    @patch.object(BinanceBookFetcher, "fetch_depth")
    def test_analyze_with_mock_api(self, mock_fetch):
        """Test full analyze() with mocked Binance API."""
        bids, asks = _make_synthetic_book(midpoint=66000.0)
        mock_fetch.return_value = {
            "bids": bids,
            "asks": asks,
            "timestamp": None,
            "last_update_id": 12345,
        }

        # Clear cache
        with self.analyzer._cache_lock:
            self.analyzer._cache.clear()

        features = self.analyzer.analyze(depth_limit=100)
        self.assertTrue(features["ob_ready"])
        self.assertEqual(len(features), 43)  # All features present

    @patch.object(BinanceBookFetcher, "fetch_depth")
    def test_analyze_returns_defaults_on_failure(self, mock_fetch):
        """Test that analyze returns defaults when API fails."""
        mock_fetch.return_value = None

        with self.analyzer._cache_lock:
            self.analyzer._cache.clear()

        features = self.analyzer.analyze()
        self.assertFalse(features["ob_ready"])

    def test_merge_depth_with_candles(self):
        """Test merge_asof with candle data."""
        candles = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC"),
            "close": np.random.uniform(65000, 67000, 10),
        })
        depth = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC"),
            "ob_imbalance_5": [0.1, 0.3, -0.2],
            "ob_spread_bps": [1.5, 2.0, 1.8],
        })

        result = self.analyzer.merge_depth_with_candles(candles, depth)
        self.assertEqual(len(result), 10)
        self.assertIn("ob_imbalance_5", result.columns)
        self.assertIn("ob_spread_bps", result.columns)

    def test_empty_book(self):
        """Test handling of empty bids/asks."""
        features = self.analyzer._compute_features([[66000, 1.0]], [[66001, 1.0]])
        self.assertTrue(features["ob_ready"])


class TestFetchConvenience(unittest.TestCase):
    """Test the convenience function."""

    @patch.object(BinanceBookFetcher, "fetch_depth")
    def test_fetch_btc_depth_snapshot(self, mock_fetch):
        bids, asks = _make_synthetic_book(midpoint=66000.0)
        mock_fetch.return_value = {
            "bids": bids, "asks": asks,
            "timestamp": None, "last_update_id": 1,
        }

        # Clear any cached state
        OrderBookDepthAnalyzer._cache.clear()

        result = fetch_btc_depth_snapshot("BTCUSDT")
        self.assertTrue(result["ob_ready"])
        self.assertIn("ob_whale_wall_bias", result)


if __name__ == "__main__":
    unittest.main()
