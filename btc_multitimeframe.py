"""
BTC Multi-Timeframe Forecast System

Combines predictions from 3 timeframes (15m, 1h, 4h) into a single
high-confidence directional signal. Higher timeframes are more accurate
(less noise) and provide trend context for lower timeframe decisions.

Architecture:
  15m model → short-term momentum & micro-entries
  1h model  → medium-term trend direction
  4h model  → macro trend & regime (most accurate)

  Combined signal = weighted vote with confidence gating.
  Only signals when multiple timeframes agree.

Usage:
  from btc_multitimeframe import BTCMultiTimeframeForecaster
  forecaster = BTCMultiTimeframeForecaster()
  forecaster.train_all()           # train 15m/1h/4h models
  result = forecaster.predict()    # combined signal
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from btc_forecast_model import BTCForecastModel
from btc_price_dataset import BTCPriceDatasetBuilder

logger = logging.getLogger(__name__)

# Timeframe weights for ensemble — higher TF gets more weight (less noise)
_TF_WEIGHTS = {
    "15m": 0.25,
    "1h": 0.35,
    "4h": 0.40,
}

# Minimum confidence to include a timeframe's vote
_MIN_CONFIDENCE = 0.52

# Minimum agreement score to emit a signal (0-1 scale)
_MIN_AGREEMENT = 0.55


class BTCMultiTimeframeForecaster:
    """
    Manages 3 BTC forecast models (15m, 1h, 4h) and combines their
    predictions into a single directional signal.
    """

    def __init__(self, weights_dir: str = "weights", logs_dir: str = "logs"):
        self.weights_dir = Path(weights_dir)
        self.logs_dir = Path(logs_dir)

        # Each timeframe gets its own model subdirectory
        self.models: dict[str, BTCForecastModel] = {}
        self.builders: dict[str, BTCPriceDatasetBuilder] = {}

        for tf in ("15m", "1h", "4h"):
            tf_weights = self.weights_dir / f"btc_forecast_{tf}"
            tf_weights.mkdir(parents=True, exist_ok=True)
            self.models[tf] = BTCForecastModel(
                weights_dir=str(tf_weights),
                logs_dir=str(self.logs_dir),
            )
            self.builders[tf] = BTCPriceDatasetBuilder(logs_dir=str(self.logs_dir))

        # Also keep the main 15m model for backward compat
        self._main_model = BTCForecastModel(
            weights_dir=str(self.weights_dir),
            logs_dir=str(self.logs_dir),
        )

        self._last_prediction: dict = {}
        self._last_predict_ts: float = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_all(
        self,
        candle_paths: dict[str, str | Path] | None = None,
        enrich_derivatives: bool = True,
        enrich_sentiment: bool = False,
    ) -> dict[str, dict]:
        """
        Train models for each timeframe.

        Args:
            candle_paths: dict mapping timeframe -> CSV path
                e.g. {"15m": "data/BTCUSDT_15m_730d.csv", ...}
            enrich_derivatives: whether to fetch derivatives data
            enrich_sentiment: whether to fetch sentiment data (FGI, Google Trends, Twitter/X, Reddit)

        Returns:
            dict of timeframe -> training metrics
        """
        if candle_paths is None:
            candle_paths = self._auto_discover_candle_files()

        results = {}
        for tf, csv_path in candle_paths.items():
            if tf not in self.models:
                logger.warning("Unknown timeframe %s, skipping", tf)
                continue

            csv_path = Path(csv_path)
            if not csv_path.exists():
                logger.warning("Candle file not found for %s: %s", tf, csv_path)
                continue

            logger.info("=== Training %s model from %s ===", tf, csv_path)

            # Build dataset
            candle_df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

            # Enrich with derivatives (only for 15m — API returns 15m granularity)
            if enrich_derivatives and tf == "15m":
                try:
                    from btc_onchain_features import BTCDerivativesFeatures
                    fetcher = BTCDerivativesFeatures()
                    candle_df = fetcher.fetch_all_and_merge(candle_df, period="15m")
                except Exception as exc:
                    logger.warning("Derivatives enrichment failed for %s: %s", tf, exc)

            # Enrich with sentiment (FGI is daily → works for all timeframes)
            if enrich_sentiment:
                try:
                    from btc_sentiment_features import BTCSentimentFeatures
                    sentiment = BTCSentimentFeatures()
                    candle_df = sentiment.fetch_all_and_merge(
                        candle_df,
                        fetch_trends=True,
                        fetch_twitter=True,
                        fetch_reddit=False,  # Reddit is real-time only, not useful for historical training
                    )
                except Exception as exc:
                    logger.warning("Sentiment enrichment failed for %s: %s", tf, exc)

            # Choose appropriate target horizon per timeframe
            target_return, target_dir = self._get_targets_for_tf(tf)

            dataset = self.builders[tf].build_from_candles(candle_df)
            if dataset.empty:
                logger.error("Failed to build dataset for %s", tf)
                results[tf] = {"error": "empty dataset"}
                continue

            logger.info("%s dataset: %d rows x %d cols", tf, len(dataset), len(dataset.columns))

            # Train the model
            metrics = self.models[tf].train(
                dataset,
                target_return_col=target_return,
                target_dir_col=target_dir,
            )
            results[tf] = metrics

            logger.info(
                "%s model: DirAcc=%.2f%% ConfDirAcc=%.2f%% ClsAcc=%.2f%%",
                tf,
                metrics.get("direction_accuracy", 0) * 100,
                metrics.get("confident_direction_accuracy", 0) * 100,
                metrics.get("classifier_accuracy", 0) * 100,
            )

        return results

    def _get_targets_for_tf(self, tf: str) -> tuple[str, str]:
        """Return appropriate target columns for each timeframe."""
        # 15m model predicts 15-candle ahead (3.75 hours)
        # 1h model predicts 15-candle ahead (15 hours)
        # 4h model predicts 15-candle ahead (60 hours / 2.5 days)
        # All use the same horizon-15 labels but at different granularities
        return "fwd_return_15", "fwd_up_15"

    def _auto_discover_candle_files(self) -> dict[str, Path]:
        """Look for candle CSVs in data/ directory."""
        data_dir = Path("data")
        paths = {}
        for tf in ("15m", "1h", "4h"):
            candidates = list(data_dir.glob(f"BTCUSDT_{tf}_*d.csv"))
            if candidates:
                # Pick the largest file (most data)
                paths[tf] = max(candidates, key=lambda p: p.stat().st_size)
                logger.info("Auto-discovered %s candles: %s", tf, paths[tf])
        return paths

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, candle_dfs: dict[str, pd.DataFrame] | None = None) -> dict:
        """
        Combine predictions from all timeframes into a single signal.

        Args:
            candle_dfs: optional dict of timeframe -> candle DataFrame for live prediction.
                        If None, uses cached models' predict with empty features.

        Returns:
            dict with combined prediction including per-timeframe breakdown.
        """
        predictions = {}

        for tf, model in self.models.items():
            if not model.is_ready:
                continue

            if candle_dfs and tf in candle_dfs:
                pred = model.predict_from_candles(candle_dfs[tf])
            else:
                pred = model.predict({})  # returns defaults if no data

            if pred.get("btc_forecast_ready"):
                predictions[tf] = pred

        if not predictions:
            # Fall back to main 15m model
            if self._main_model.is_ready:
                main_pred = self._main_model.predict({})
                if main_pred.get("btc_forecast_ready"):
                    return {**main_pred, "btc_mtf_source": "main_15m_only"}
            return self._default_result()

        # --- Weighted combination ---
        combined = self._combine_predictions(predictions)
        self._last_prediction = combined
        self._last_predict_ts = time.time()
        return combined

    def predict_from_candles(self, candle_dfs: dict[str, pd.DataFrame]) -> dict:
        """Convenience: predict from candle DataFrames per timeframe."""
        return self.predict(candle_dfs=candle_dfs)

    def _combine_predictions(self, predictions: dict[str, dict]) -> dict:
        """
        Weighted vote across timeframes with confidence gating.

        Logic:
          1. Each timeframe votes direction (-1 or +1) weighted by its weight * confidence
          2. Only include timeframes where confidence > threshold
          3. Agreement score = how much the weighted votes agree (0-1)
          4. Final direction = sign of weighted sum
          5. Combined confidence = agreement * avg confidence
        """
        weighted_direction = 0.0
        weighted_return = 0.0
        total_weight = 0.0
        confidences = []
        tf_details = {}

        for tf, pred in predictions.items():
            direction = pred.get("btc_predicted_direction", 0)
            confidence = pred.get("btc_forecast_confidence", 0.5)
            ret = pred.get("btc_predicted_return_15", 0.0)
            weight = _TF_WEIGHTS.get(tf, 0.2)

            tf_details[f"btc_mtf_{tf}_direction"] = direction
            tf_details[f"btc_mtf_{tf}_confidence"] = round(confidence, 4)
            tf_details[f"btc_mtf_{tf}_return"] = round(ret, 6)

            # Only include confident predictions
            if confidence >= _MIN_CONFIDENCE:
                weighted_direction += weight * direction * confidence
                weighted_return += weight * ret
                total_weight += weight
                confidences.append(confidence)

        if total_weight == 0:
            return {**self._default_result(), **tf_details, "btc_mtf_source": "no_confident_tf"}

        # Normalise
        avg_direction = weighted_direction / total_weight
        avg_return = weighted_return / total_weight
        avg_confidence = float(np.mean(confidences)) if confidences else 0.5

        # Agreement: how strongly aligned are the votes? (0 = split, 1 = unanimous)
        agreement = abs(avg_direction)

        # Final direction
        final_direction = 1 if avg_direction > 0 else -1 if avg_direction < 0 else 0

        # Combined confidence = agreement * average confidence
        combined_confidence = min(1.0, agreement * avg_confidence)

        # How many timeframes agree?
        directions = [p.get("btc_predicted_direction", 0) for p in predictions.values()]
        n_agree = sum(1 for d in directions if d == final_direction)
        n_total = len(directions)

        # Override to neutral if agreement is too low
        if agreement < _MIN_AGREEMENT:
            final_direction = 0
            combined_confidence = combined_confidence * 0.5

        result = {
            "btc_predicted_return_15": round(avg_return, 6),
            "btc_predicted_direction": final_direction,
            "btc_forecast_confidence": round(combined_confidence, 4),
            "btc_forecast_ready": True,
            "btc_mtf_agreement": round(agreement, 4),
            "btc_mtf_n_agree": n_agree,
            "btc_mtf_n_total": n_total,
            "btc_mtf_source": "multi_timeframe",
            **tf_details,
        }
        return result

    @staticmethod
    def _default_result() -> dict:
        return {
            "btc_predicted_return_15": 0.0,
            "btc_predicted_direction": 0,
            "btc_forecast_confidence": 0.0,
            "btc_forecast_ready": False,
            "btc_mtf_agreement": 0.0,
            "btc_mtf_n_agree": 0,
            "btc_mtf_n_total": 0,
            "btc_mtf_source": "unavailable",
        }

    @property
    def is_ready(self) -> bool:
        """True if at least 2 timeframes are ready."""
        ready_count = sum(1 for m in self.models.values() if m.is_ready)
        return ready_count >= 2 or self._main_model.is_ready

    @property
    def model_meta(self) -> dict:
        return {tf: m.model_meta for tf, m in self.models.items() if m.is_ready}
