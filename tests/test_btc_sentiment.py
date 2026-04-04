"""
Tests for BTC sentiment feature pipeline (btc_sentiment_features.py).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_sentiment_features import BTCSentimentFeatures


def _make_synthetic_candles(n: int = 200, start_price: float = 50000.0) -> pd.DataFrame:
    """Generate synthetic BTC-like OHLCV candles for testing."""
    np.random.seed(42)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    close = start_price * np.cumprod(1 + np.random.normal(0.0001, 0.005, n))
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close * (1 + np.random.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.002, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.002, n))),
        "close": close,
        "volume": np.random.uniform(100, 10000, n),
    })


class TestBTCSentimentFeatures(unittest.TestCase):

    def setUp(self):
        self.fetcher = BTCSentimentFeatures()
        self.candles = _make_synthetic_candles(200)

    def test_empty_fgi_returns_correct_columns(self):
        df = self.fetcher._empty_fgi()
        self.assertIn("timestamp", df.columns)
        self.assertIn("fgi_value", df.columns)
        self.assertEqual(len(df), 0)

    def test_empty_gtrends_returns_correct_columns(self):
        df = self.fetcher._empty_gtrends()
        self.assertIn("timestamp", df.columns)
        self.assertIn("gtrends_bitcoin", df.columns)
        self.assertEqual(len(df), 0)

    def test_empty_twitter_returns_correct_columns(self):
        df = self.fetcher._empty_twitter()
        self.assertIn("timestamp", df.columns)
        self.assertIn("twitter_sentiment", df.columns)
        self.assertIn("twitter_post_count", df.columns)
        self.assertEqual(len(df), 0)

    def test_empty_reddit_returns_correct_columns(self):
        df = self.fetcher._empty_reddit()
        self.assertIn("timestamp", df.columns)
        self.assertIn("reddit_sentiment", df.columns)
        self.assertIn("reddit_post_count", df.columns)
        self.assertEqual(len(df), 0)

    def test_compute_derived_features_fgi(self):
        """Test derived features from Fear & Greed Index values."""
        df = pd.DataFrame({
            "fgi_value": [10, 15, 20, 30, 50, 70, 80, 85, 90, 50] * 5,
        })
        result = self.fetcher._compute_derived_features(df)

        self.assertIn("fgi_extreme_fear", result.columns)
        self.assertIn("fgi_extreme_greed", result.columns)
        self.assertIn("fgi_contrarian", result.columns)
        self.assertIn("fgi_momentum", result.columns)
        self.assertIn("fgi_normalized", result.columns)

        # Check extreme fear flag (FGI < 20)
        self.assertEqual(result["fgi_extreme_fear"].iloc[0], 1.0)  # FGI=10
        self.assertEqual(result["fgi_extreme_fear"].iloc[4], 0.0)  # FGI=50

        # Check extreme greed flag (FGI > 80)
        self.assertEqual(result["fgi_extreme_greed"].iloc[7], 1.0)  # FGI=85
        self.assertEqual(result["fgi_extreme_greed"].iloc[3], 0.0)  # FGI=30

        # Check normalized range
        self.assertTrue((result["fgi_normalized"] >= 0).all())
        self.assertTrue((result["fgi_normalized"] <= 1).all())

    def test_compute_derived_features_gtrends(self):
        """Test derived features from Google Trends values."""
        df = pd.DataFrame({
            "gtrends_bitcoin": list(range(30, 80)) + list(range(80, 30, -1)),
        })
        result = self.fetcher._compute_derived_features(df)

        self.assertIn("gtrends_zscore", result.columns)
        self.assertIn("gtrends_spike", result.columns)
        self.assertIn("gtrends_momentum", result.columns)
        self.assertIn("gtrends_normalized", result.columns)

    def test_compute_derived_features_twitter(self):
        """Test derived features from Twitter/X sentiment values."""
        df = pd.DataFrame({
            "twitter_sentiment": np.random.uniform(-0.6, 0.6, 50),
        })
        result = self.fetcher._compute_derived_features(df)

        self.assertIn("twitter_sentiment_zscore", result.columns)
        self.assertIn("twitter_bullish", result.columns)
        self.assertIn("twitter_bearish", result.columns)
        self.assertIn("twitter_sentiment_momentum", result.columns)

    @patch("btc_sentiment_features._get_vader")
    @patch.object(BTCSentimentFeatures, "_fetch_nitter_posts")
    def test_fetch_twitter_sentiment_aggregates_posts(self, mock_fetch_nitter_posts, mock_get_vader):
        mock_vader = MagicMock()
        mock_vader.polarity_scores.side_effect = [
            {"compound": 0.6, "pos": 0.5, "neg": 0.1},
            {"compound": 0.4, "pos": 0.4, "neg": 0.1},
        ]
        mock_get_vader.return_value = mock_vader
        mock_fetch_nitter_posts.side_effect = [
            [
                {
                    "timestamp": pd.Timestamp("2025-01-10T08:00:00Z"),
                    "text": "Bitcoin looks strong today, very bullish setup",
                    "engagement_proxy": 4.0,
                },
                {
                    "timestamp": pd.Timestamp("2025-01-10T12:00:00Z"),
                    "text": "BTC breakout confirmed, buyers in control",
                    "engagement_proxy": 3.0,
                },
            ],
            [],
        ]

        result = self.fetcher.fetch_twitter_sentiment(search_terms=["bitcoin", "btc"], per_term_limit=10)

        self.assertEqual(len(result), 1)
        self.assertIn("twitter_sentiment", result.columns)
        self.assertIn("twitter_post_count", result.columns)
        self.assertEqual(int(result["twitter_post_count"].iloc[0]), 2)

    def test_compute_derived_features_reddit(self):
        """Test derived features from Reddit sentiment values."""
        df = pd.DataFrame({
            "reddit_sentiment": np.random.uniform(-0.5, 0.5, 50),
        })
        result = self.fetcher._compute_derived_features(df)

        self.assertIn("reddit_sentiment_zscore", result.columns)
        self.assertIn("reddit_bullish", result.columns)
        self.assertIn("reddit_bearish", result.columns)
        self.assertIn("reddit_sentiment_momentum", result.columns)

    @patch.object(BTCSentimentFeatures, "fetch_fear_greed_history")
    @patch.object(BTCSentimentFeatures, "fetch_google_trends")
    @patch.object(BTCSentimentFeatures, "fetch_twitter_sentiment")
    def test_fetch_all_and_merge_with_mock_data(self, mock_twitter, mock_gtrends, mock_fgi):
        """Test merge pipeline with mocked API data."""
        # Mock FGI data (daily)
        fgi_dates = pd.date_range("2024-12-01", periods=60, freq="D", tz="UTC")
        mock_fgi.return_value = pd.DataFrame({
            "timestamp": fgi_dates,
            "fgi_value": np.random.randint(20, 80, 60),
            "fgi_class": ["Neutral"] * 60,
        })

        # Mock Google Trends data (weekly)
        gt_dates = pd.date_range("2024-12-01", periods=10, freq="W", tz="UTC")
        mock_gtrends.return_value = pd.DataFrame({
            "timestamp": gt_dates,
            "gtrends_bitcoin": np.random.randint(30, 70, 10),
        })

        tw_dates = pd.date_range("2024-12-20", periods=14, freq="D", tz="UTC")
        mock_twitter.return_value = pd.DataFrame({
            "timestamp": tw_dates,
            "twitter_sentiment": np.random.uniform(-0.3, 0.4, len(tw_dates)),
            "twitter_post_count": np.random.randint(10, 120, len(tw_dates)),
            "twitter_sentiment_pos": np.random.uniform(0.1, 0.4, len(tw_dates)),
            "twitter_sentiment_neg": np.random.uniform(0.05, 0.3, len(tw_dates)),
            "twitter_engagement_proxy": np.random.uniform(1.0, 5.0, len(tw_dates)),
        })

        result = self.fetcher.fetch_all_and_merge(
            self.candles,
            fetch_trends=True,
            fetch_twitter=True,
            fetch_reddit=False,
        )

        # Should have all original columns plus sentiment columns
        self.assertGreater(len(result.columns), len(self.candles.columns))
        self.assertIn("fgi_value", result.columns)
        self.assertIn("fgi_extreme_fear", result.columns)
        self.assertIn("fgi_contrarian", result.columns)
        self.assertIn("gtrends_bitcoin", result.columns)
        self.assertIn("twitter_sentiment", result.columns)
        self.assertIn("twitter_bullish", result.columns)
        self.assertEqual(len(result), len(self.candles))

    def test_fetch_all_and_merge_no_timestamp(self):
        """Should return unchanged df if no timestamp column."""
        df = pd.DataFrame({"close": [100, 200, 300]})
        result = self.fetcher.fetch_all_and_merge(df, fetch_trends=False, fetch_reddit=False)
        self.assertEqual(len(result), 3)

    @patch.object(BTCSentimentFeatures, "fetch_fear_greed_history")
    def test_fetch_all_handles_empty_apis(self, mock_fgi):
        """Should gracefully handle when APIs return empty."""
        mock_fgi.return_value = pd.DataFrame(columns=["timestamp", "fgi_value", "fgi_class"])

        result = self.fetcher.fetch_all_and_merge(
            self.candles,
            fetch_trends=False,
            fetch_twitter=False,
            fetch_reddit=False,
        )

        # Should still have fgi_value column (filled with NaN)
        self.assertIn("fgi_value", result.columns)
        self.assertIn("twitter_sentiment", result.columns)
        self.assertEqual(len(result), len(self.candles))

    def test_fetch_current_snapshot_structure(self):
        """Test that snapshot returns a dict with expected keys when FGI works."""
        with patch.object(self.fetcher, "_session") as mock_session:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "data": [
                    {"value": "25", "value_classification": "Extreme Fear", "timestamp": "1712188800"},
                    {"value": "30", "value_classification": "Fear", "timestamp": "1712102400"},
                    {"value": "28", "value_classification": "Fear", "timestamp": "1712016000"},
                ]
            }
            mock_session.get.return_value = mock_resp

            twitter_df = pd.DataFrame({
                "timestamp": pd.to_datetime(["2025-01-10"], utc=True),
                "twitter_sentiment": [0.31],
                "twitter_post_count": [42],
                "twitter_sentiment_pos": [0.34],
                "twitter_sentiment_neg": [0.11],
                "twitter_engagement_proxy": [3.5],
            })

            # Patch social fetches to avoid real API calls
            with patch.object(self.fetcher, "fetch_twitter_sentiment", return_value=twitter_df):
                with patch.object(self.fetcher, "fetch_reddit_sentiment", return_value=self.fetcher._empty_reddit()):
                    result = self.fetcher.fetch_current_snapshot()

            self.assertIn("twitter_sentiment", result)
            self.assertEqual(result["twitter_post_count"], 42.0)
            self.assertEqual(result["twitter_bullish"], 1.0)
            self.assertEqual(result["twitter_bearish"], 0.0)
            self.assertIn("fgi_value", result)
            self.assertEqual(result["fgi_value"], 25)
            self.assertEqual(result["fgi_extreme_fear"], 0.0)  # 25 >= 20, not extreme fear
            self.assertEqual(result["fgi_extreme_greed"], 0.0)
            self.assertIn("fgi_momentum", result)
            self.assertIn("fgi_momentum_3d", result)


class TestVADERSentiment(unittest.TestCase):
    """Test VADER sentiment scoring on crypto-related text."""

    def test_vader_loads(self):
        from btc_sentiment_features import _get_vader
        vader = _get_vader()
        # VADER should be available (we installed nltk in setup)
        if vader is None:
            self.skipTest("VADER not available")

    def test_vader_scores_bullish_text(self):
        from btc_sentiment_features import _get_vader
        vader = _get_vader()
        if vader is None:
            self.skipTest("VADER not available")

        scores = vader.polarity_scores("Bitcoin is mooning! Great news for crypto, huge gains ahead!")
        self.assertGreater(scores["compound"], 0.0)

    def test_vader_scores_bearish_text(self):
        from btc_sentiment_features import _get_vader
        vader = _get_vader()
        if vader is None:
            self.skipTest("VADER not available")

        scores = vader.polarity_scores("Bitcoin crash incoming, terrible market, massive losses expected")
        self.assertLess(scores["compound"], 0.0)


if __name__ == "__main__":
    unittest.main()
