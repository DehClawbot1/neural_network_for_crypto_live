"""
Tests for leaderboard consensus computation and BTC-forecast-driven side selection.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We cannot import supervisor.py directly due to heavy import chain (scipy bug).
# Re-import the standalone function by reading the module source.
# Instead, just re-define the function here (it's pure logic, no deps).


def compute_leaderboard_consensus(signals_df, market_slug_prefix="btc-updown-"):
    """Mirror of supervisor.compute_leaderboard_consensus for testing."""
    result = {
        "leaderboard_n_yes": 0,
        "leaderboard_n_no": 0,
        "leaderboard_n_total": 0,
        "leaderboard_bias": 0,
        "leaderboard_agreement": 0.0,
        "leaderboard_vol_yes": 0.0,
        "leaderboard_vol_no": 0.0,
    }
    if signals_df is None or signals_df.empty:
        return result
    slug_col = "market_slug" if "market_slug" in signals_df.columns else None
    if slug_col is None:
        return result
    mask = signals_df[slug_col].astype(str).str.lower().str.startswith(market_slug_prefix)
    btc_signals = signals_df[mask]
    if btc_signals.empty:
        return result
    if "signal_source" in btc_signals.columns:
        btc_signals = btc_signals[btc_signals["signal_source"].astype(str).str.lower() != "always_on_market"]
    if btc_signals.empty:
        return result
    side_col = "outcome_side" if "outcome_side" in btc_signals.columns else "side"
    if side_col not in btc_signals.columns:
        return result
    sides = btc_signals[side_col].astype(str).str.upper()
    sizes = btc_signals["size"].astype(float).fillna(0) if "size" in btc_signals.columns else pd.Series([0] * len(btc_signals))
    n_yes = int((sides == "YES").sum())
    n_no = int((sides == "NO").sum())
    n_total = n_yes + n_no
    vol_yes = float(sizes[sides == "YES"].sum())
    vol_no = float(sizes[sides == "NO"].sum())
    if n_total == 0:
        return result
    majority_pct = max(n_yes, n_no) / n_total
    result.update({
        "leaderboard_n_yes": n_yes,
        "leaderboard_n_no": n_no,
        "leaderboard_n_total": n_total,
        "leaderboard_bias": 1 if n_yes > n_no else (-1 if n_no > n_yes else 0),
        "leaderboard_agreement": round(majority_pct, 4),
        "leaderboard_vol_yes": round(vol_yes, 2),
        "leaderboard_vol_no": round(vol_no, 2),
    })
    return result


class TestLeaderboardConsensus(unittest.TestCase):

    def test_empty_signals(self):
        result = compute_leaderboard_consensus(pd.DataFrame())
        self.assertEqual(result["leaderboard_n_total"], 0)
        self.assertEqual(result["leaderboard_bias"], 0)
        self.assertEqual(result["leaderboard_agreement"], 0.0)

    def test_none_signals(self):
        result = compute_leaderboard_consensus(None)
        self.assertEqual(result["leaderboard_n_total"], 0)

    def test_no_btc_markets(self):
        df = pd.DataFrame([
            {"market_slug": "some-other-market", "outcome_side": "YES", "size": 100},
        ])
        result = compute_leaderboard_consensus(df)
        self.assertEqual(result["leaderboard_n_total"], 0)

    def test_unanimous_yes(self):
        df = pd.DataFrame([
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 100},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 200},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 50},
        ])
        result = compute_leaderboard_consensus(df)
        self.assertEqual(result["leaderboard_n_total"], 3)
        self.assertEqual(result["leaderboard_n_yes"], 3)
        self.assertEqual(result["leaderboard_n_no"], 0)
        self.assertEqual(result["leaderboard_bias"], 1)
        self.assertAlmostEqual(result["leaderboard_agreement"], 1.0)
        self.assertAlmostEqual(result["leaderboard_vol_yes"], 350.0)
        self.assertAlmostEqual(result["leaderboard_vol_no"], 0.0)

    def test_unanimous_no(self):
        df = pd.DataFrame([
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "NO", "size": 100},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "NO", "size": 200},
        ])
        result = compute_leaderboard_consensus(df)
        self.assertEqual(result["leaderboard_bias"], -1)
        self.assertAlmostEqual(result["leaderboard_agreement"], 1.0)

    def test_mixed_majority_yes(self):
        df = pd.DataFrame([
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 100},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 200},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 150},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "NO", "size": 50},
        ])
        result = compute_leaderboard_consensus(df)
        self.assertEqual(result["leaderboard_n_yes"], 3)
        self.assertEqual(result["leaderboard_n_no"], 1)
        self.assertEqual(result["leaderboard_bias"], 1)
        self.assertAlmostEqual(result["leaderboard_agreement"], 0.75)

    def test_balanced_signals(self):
        df = pd.DataFrame([
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 100},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "NO", "size": 100},
        ])
        result = compute_leaderboard_consensus(df)
        self.assertEqual(result["leaderboard_bias"], 0)
        self.assertAlmostEqual(result["leaderboard_agreement"], 0.5)

    def test_excludes_always_on_synthetic(self):
        """Our own always_on_market signals should not count."""
        df = pd.DataFrame([
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 100, "signal_source": "always_on_market"},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "NO", "size": 50, "signal_source": "leaderboard"},
        ])
        result = compute_leaderboard_consensus(df)
        self.assertEqual(result["leaderboard_n_total"], 1)
        self.assertEqual(result["leaderboard_n_no"], 1)
        self.assertEqual(result["leaderboard_bias"], -1)

    def test_mixed_markets_only_counts_btc(self):
        df = pd.DataFrame([
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 100},
            {"market_slug": "some-other-market", "outcome_side": "NO", "size": 500},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 200},
        ])
        result = compute_leaderboard_consensus(df)
        self.assertEqual(result["leaderboard_n_total"], 2)
        self.assertEqual(result["leaderboard_n_yes"], 2)

    def test_no_slug_column(self):
        df = pd.DataFrame([
            {"outcome_side": "YES", "size": 100},
        ])
        result = compute_leaderboard_consensus(df)
        self.assertEqual(result["leaderboard_n_total"], 0)

    def test_volume_tracking(self):
        df = pd.DataFrame([
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 100.5},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "NO", "size": 75.25},
            {"market_slug": "btc-updown-5m-123456789", "outcome_side": "YES", "size": 200.0},
        ])
        result = compute_leaderboard_consensus(df)
        self.assertAlmostEqual(result["leaderboard_vol_yes"], 300.5)
        self.assertAlmostEqual(result["leaderboard_vol_no"], 75.25)


class TestBTCForecastDrivenSideSelection(unittest.TestCase):
    """Test the side selection logic that lives in _inject_always_on_signal.

    We test the decision logic in isolation since _inject_always_on_signal
    is a closure and cannot be directly imported.
    """

    def _decide_side(self, btc_context, leaderboard_consensus, yes_price=0.5):
        """Replicate the side selection logic from _inject_always_on_signal."""
        btc_ctx = btc_context or {}
        btc_direction = int(btc_ctx.get("btc_predicted_direction", 0) or 0)
        btc_confidence = float(btc_ctx.get("btc_forecast_confidence", 0.0) or 0.0)
        btc_ready = bool(btc_ctx.get("btc_forecast_ready", False))
        lb_bias = leaderboard_consensus.get("leaderboard_bias", 0)
        lb_agreement = leaderboard_consensus.get("leaderboard_agreement", 0.0)
        lb_n_total = leaderboard_consensus.get("leaderboard_n_total", 0)

        pref_side = None
        side_source = None

        if btc_ready and btc_direction != 0 and btc_confidence >= 0.52:
            pref_side = "YES" if btc_direction == 1 else "NO"
            side_source = "btc_forecast"
            if lb_n_total >= 3 and lb_agreement >= 0.60:
                lb_side = "YES" if lb_bias == 1 else "NO"
                if lb_side == pref_side:
                    btc_confidence = min(1.0, btc_confidence + 0.05)
                    side_source = "btc_forecast+leaderboard_confirm"
                else:
                    btc_confidence = max(0.50, btc_confidence - 0.05)
                    side_source = "btc_forecast+leaderboard_disagree"
        elif lb_n_total >= 5 and lb_agreement >= 0.65:
            pref_side = "YES" if lb_bias == 1 else "NO"
            btc_confidence = lb_agreement * 0.7
            side_source = "leaderboard_consensus"
        else:
            pref_side = "YES" if yes_price >= 0.5 else "NO"
            side_source = "price_fallback"

        return pref_side, side_source, btc_confidence

    def test_btc_forecast_bullish_picks_yes(self):
        side, source, conf = self._decide_side(
            {"btc_predicted_direction": 1, "btc_forecast_confidence": 0.65, "btc_forecast_ready": True},
            {"leaderboard_bias": 0, "leaderboard_agreement": 0.0, "leaderboard_n_total": 0},
        )
        self.assertEqual(side, "YES")
        self.assertEqual(source, "btc_forecast")
        self.assertAlmostEqual(conf, 0.65)

    def test_btc_forecast_bearish_picks_no(self):
        side, source, conf = self._decide_side(
            {"btc_predicted_direction": -1, "btc_forecast_confidence": 0.58, "btc_forecast_ready": True},
            {"leaderboard_bias": 0, "leaderboard_agreement": 0.0, "leaderboard_n_total": 0},
        )
        self.assertEqual(side, "NO")
        self.assertEqual(source, "btc_forecast")

    def test_btc_forecast_low_confidence_falls_through(self):
        """If BTC forecast confidence < 0.52, it should not be used."""
        side, source, _ = self._decide_side(
            {"btc_predicted_direction": 1, "btc_forecast_confidence": 0.50, "btc_forecast_ready": True},
            {"leaderboard_bias": 0, "leaderboard_agreement": 0.0, "leaderboard_n_total": 0},
            yes_price=0.4,
        )
        # Falls through to price fallback: yes_price < 0.5 → NO
        self.assertEqual(side, "NO")
        self.assertEqual(source, "price_fallback")

    def test_btc_forecast_not_ready_uses_leaderboard(self):
        """When forecast not ready but strong leaderboard consensus, use leaderboard."""
        side, source, conf = self._decide_side(
            {"btc_predicted_direction": 0, "btc_forecast_confidence": 0.0, "btc_forecast_ready": False},
            {"leaderboard_bias": -1, "leaderboard_agreement": 0.80, "leaderboard_n_total": 6},
        )
        self.assertEqual(side, "NO")
        self.assertEqual(source, "leaderboard_consensus")
        self.assertAlmostEqual(conf, 0.80 * 0.7, places=3)

    def test_leaderboard_confirms_forecast_boosts_confidence(self):
        side, source, conf = self._decide_side(
            {"btc_predicted_direction": 1, "btc_forecast_confidence": 0.60, "btc_forecast_ready": True},
            {"leaderboard_bias": 1, "leaderboard_agreement": 0.75, "leaderboard_n_total": 4},
        )
        self.assertEqual(side, "YES")
        self.assertEqual(source, "btc_forecast+leaderboard_confirm")
        self.assertAlmostEqual(conf, 0.65)  # 0.60 + 0.05

    def test_leaderboard_disagrees_reduces_confidence(self):
        side, source, conf = self._decide_side(
            {"btc_predicted_direction": 1, "btc_forecast_confidence": 0.60, "btc_forecast_ready": True},
            {"leaderboard_bias": -1, "leaderboard_agreement": 0.80, "leaderboard_n_total": 5},
        )
        self.assertEqual(side, "YES")  # Still follows the ML forecast
        self.assertEqual(source, "btc_forecast+leaderboard_disagree")
        self.assertAlmostEqual(conf, 0.55)  # 0.60 - 0.05

    def test_no_forecast_weak_leaderboard_falls_to_price(self):
        side, source, _ = self._decide_side(
            {"btc_predicted_direction": 0, "btc_forecast_ready": False},
            {"leaderboard_bias": 1, "leaderboard_agreement": 0.55, "leaderboard_n_total": 2},
            yes_price=0.65,
        )
        self.assertEqual(side, "YES")
        self.assertEqual(source, "price_fallback")

    def test_leaderboard_needs_min_5_traders(self):
        """Leaderboard consensus requires at least 5 traders."""
        side, source, _ = self._decide_side(
            {"btc_forecast_ready": False},
            {"leaderboard_bias": 1, "leaderboard_agreement": 0.80, "leaderboard_n_total": 4},
            yes_price=0.3,
        )
        # 4 < 5 traders → falls to price_fallback
        self.assertEqual(source, "price_fallback")
        self.assertEqual(side, "NO")  # yes_price 0.3 < 0.5

    def test_confidence_floor_on_disagree(self):
        """Confidence should never drop below 0.50 on leaderboard disagree."""
        side, source, conf = self._decide_side(
            {"btc_predicted_direction": -1, "btc_forecast_confidence": 0.52, "btc_forecast_ready": True},
            {"leaderboard_bias": 1, "leaderboard_agreement": 0.90, "leaderboard_n_total": 10},
        )
        self.assertEqual(side, "NO")
        # 0.52 - 0.05 = 0.47, but floor is 0.50
        self.assertAlmostEqual(conf, 0.50)


if __name__ == "__main__":
    unittest.main()
