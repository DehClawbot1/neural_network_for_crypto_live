"""
BTC Price Forecast Model

LightGBM-based model that predicts:
  - btc_predicted_return_15  (regression: expected 15-candle forward return)
  - btc_predicted_direction  (classification: -1 / 0 / +1)
  - btc_forecast_confidence  (probability calibration from the classifier)

Designed to be:
  1. Trained offline from btc_price_dataset.py output
  2. Called live each supervisor cycle via predict()
  3. Periodically retrained by the Retrainer
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns used by the model (must match btc_price_dataset.py output)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "return_1", "return_5", "return_15",
    "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_9", "ema_21",
    "close_to_sma_20", "close_to_sma_50", "close_to_sma_200",
    "ema_9_21_cross",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "atr_14", "atr_pct",
    "bb_position", "bb_width_pct",
    "adx",
    "stoch_rsi_k", "stoch_rsi_d",
    "volume_ratio",
    "realized_vol_20", "realized_vol_60",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
]

# Relative features (no absolute price levels — prevents data leakage)
RELATIVE_FEATURE_COLS = [c for c in FEATURE_COLS if c not in ("sma_10", "sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "atr_14")]


class BTCForecastModel:
    """
    Gradient-boosted BTC price direction + return predictor.

    Uses LightGBM when available, falls back to sklearn GradientBoosting.
    """

    def __init__(self, weights_dir: str = "weights", logs_dir: str = "logs"):
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._regressor = None
        self._classifier = None
        self._feature_cols = list(RELATIVE_FEATURE_COLS)
        self._is_lgbm = False
        self._model_meta: dict = {}
        self._last_prediction: dict = {}
        self._last_predict_ts: float = 0.0

        # Try to load existing model on init
        self._try_load()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        target_return_col: str = "fwd_return_15",
        target_dir_col: str = "fwd_direction_15",
        test_fraction: float = 0.15,
    ) -> dict:
        """
        Train both regressor and classifier on labelled dataset.
        Uses walk-forward split (last N% as test) — NOT random split.

        Returns a metrics dict.
        """
        if df is None or df.empty:
            logger.error("BTCForecastModel.train: empty dataset")
            return {"error": "empty dataset"}

        # Ensure feature columns exist
        available = [c for c in self._feature_cols if c in df.columns]
        if len(available) < 5:
            logger.error("BTCForecastModel.train: too few feature columns (%d)", len(available))
            return {"error": "too few features"}
        self._feature_cols = available

        # Drop rows with NaN in features or target
        cols_needed = available + [target_return_col, target_dir_col]
        clean = df[cols_needed].dropna()
        if len(clean) < 100:
            logger.error("BTCForecastModel.train: only %d clean rows (need >=100)", len(clean))
            return {"error": f"only {len(clean)} rows"}

        X = clean[available].values
        y_reg = clean[target_return_col].values.astype(float)
        y_cls = clean[target_dir_col].values.astype(int)

        # Walk-forward split
        split_idx = int(len(X) * (1.0 - test_fraction))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
        y_cls_train, y_cls_test = y_cls[:split_idx], y_cls[split_idx:]

        logger.info(
            "BTCForecastModel: training on %d rows, testing on %d rows, %d features",
            len(X_train), len(X_test), len(available),
        )

        regressor, classifier, is_lgbm = self._create_models()
        regressor.fit(X_train, y_reg_train)
        classifier.fit(X_train, y_cls_train)

        # Evaluate
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            accuracy_score,
            classification_report,
        )

        reg_pred = regressor.predict(X_test)
        cls_pred = classifier.predict(X_test)

        mae = float(mean_absolute_error(y_reg_test, reg_pred))
        rmse = float(np.sqrt(mean_squared_error(y_reg_test, reg_pred)))
        accuracy = float(accuracy_score(y_cls_test, cls_pred))

        # Directional accuracy (did we get up/down right?)
        dir_correct = np.sign(reg_pred) == np.sign(y_reg_test)
        dir_accuracy = float(dir_correct.mean())

        metrics = {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": int(len(available)),
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "direction_accuracy": round(dir_accuracy, 4),
            "classifier_accuracy": round(accuracy, 4),
            "backend": "lightgbm" if is_lgbm else "sklearn",
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }

        logger.info(
            "BTCForecastModel trained: MAE=%.6f RMSE=%.6f DirAcc=%.2f%% ClsAcc=%.2f%%",
            mae, rmse, dir_accuracy * 100, accuracy * 100,
        )

        # Store
        self._regressor = regressor
        self._classifier = classifier
        self._is_lgbm = is_lgbm
        self._model_meta = metrics

        # Persist
        self._save()
        self._save_metrics(metrics)

        return metrics

    # ------------------------------------------------------------------
    # Prediction (live inference)
    # ------------------------------------------------------------------

    def predict(self, features: dict | pd.Series | pd.DataFrame) -> dict:
        """
        Predict BTC price direction and return from a single feature row.

        Returns dict with:
            btc_predicted_return_15, btc_predicted_direction,
            btc_forecast_confidence, btc_forecast_ready
        """
        default = {
            "btc_predicted_return_15": 0.0,
            "btc_predicted_direction": 0,
            "btc_forecast_confidence": 0.0,
            "btc_forecast_ready": False,
        }

        if self._regressor is None or self._classifier is None:
            return default

        try:
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            elif isinstance(features, pd.Series):
                features = features.to_frame().T

            # Align columns
            available = [c for c in self._feature_cols if c in features.columns]
            if len(available) < 5:
                return default

            X = features[available].values.astype(float)
            if np.isnan(X).any():
                return default

            reg_pred = float(self._regressor.predict(X)[0])
            cls_pred = int(self._classifier.predict(X)[0])

            # Confidence from classifier probability
            confidence = 0.5
            if hasattr(self._classifier, "predict_proba"):
                proba = self._classifier.predict_proba(X)[0]
                confidence = float(np.max(proba))

            result = {
                "btc_predicted_return_15": round(reg_pred, 6),
                "btc_predicted_direction": cls_pred,
                "btc_forecast_confidence": round(confidence, 4),
                "btc_forecast_ready": True,
            }
            self._last_prediction = result
            self._last_predict_ts = time.time()
            return result

        except Exception as exc:
            logger.warning("BTCForecastModel.predict failed: %s", exc)
            return default

    def predict_from_candles(self, candle_df: pd.DataFrame) -> dict:
        """
        Convenience: build features from raw candles and predict.
        Uses the last row of features for prediction.
        """
        from btc_price_dataset import BTCPriceDatasetBuilder
        builder = BTCPriceDatasetBuilder(logs_dir=str(self.logs_dir))
        features_df = builder.build_from_candles(candle_df)
        if features_df.empty:
            return self.predict({})
        return self.predict(features_df.iloc[-1])

    @property
    def is_ready(self) -> bool:
        return self._regressor is not None and self._classifier is not None

    @property
    def model_meta(self) -> dict:
        return dict(self._model_meta)

    # ------------------------------------------------------------------
    # Model creation (LightGBM with sklearn fallback)
    # ------------------------------------------------------------------

    def _create_models(self):
        """Create regressor + classifier. Prefer LightGBM, fall back to sklearn."""
        try:
            import lightgbm as lgb
            regressor = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbosity=-1,
            )
            classifier = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbosity=-1,
            )
            logger.info("BTCForecastModel: using LightGBM backend")
            return regressor, classifier, True
        except (ImportError, ModuleNotFoundError):
            pass

        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        regressor = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
        )
        classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
        )
        logger.info("BTCForecastModel: using sklearn GradientBoosting backend (install lightgbm for better performance)")
        return regressor, classifier, False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        """Save model to disk using joblib."""
        try:
            import joblib
            reg_path = self.weights_dir / "btc_forecast_regressor.joblib"
            cls_path = self.weights_dir / "btc_forecast_classifier.joblib"
            meta_path = self.weights_dir / "btc_forecast_meta.json"

            joblib.dump(self._regressor, reg_path)
            joblib.dump(self._classifier, cls_path)

            meta = {
                **self._model_meta,
                "feature_cols": self._feature_cols,
                "is_lgbm": self._is_lgbm,
            }
            meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            logger.info("BTCForecastModel: saved to %s", self.weights_dir)
        except Exception as exc:
            logger.warning("BTCForecastModel: save failed: %s", exc)

    def _try_load(self):
        """Load model from disk if available."""
        try:
            import joblib
            reg_path = self.weights_dir / "btc_forecast_regressor.joblib"
            cls_path = self.weights_dir / "btc_forecast_classifier.joblib"
            meta_path = self.weights_dir / "btc_forecast_meta.json"

            if not reg_path.exists() or not cls_path.exists():
                return

            self._regressor = joblib.load(reg_path)
            self._classifier = joblib.load(cls_path)

            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._model_meta = meta
                self._feature_cols = meta.get("feature_cols", self._feature_cols)
                self._is_lgbm = meta.get("is_lgbm", False)

            logger.info(
                "BTCForecastModel: loaded from %s (backend=%s, features=%d)",
                self.weights_dir,
                "lightgbm" if self._is_lgbm else "sklearn",
                len(self._feature_cols),
            )
        except Exception as exc:
            logger.debug("BTCForecastModel: no saved model loaded: %s", exc)

    def _save_metrics(self, metrics: dict):
        """Append training metrics to log."""
        metrics_file = self.logs_dir / "btc_forecast_train_log.csv"
        try:
            row = pd.DataFrame([metrics])
            safe_csv_append(metrics_file, row)
        except Exception as exc:
            logger.debug("BTCForecastModel: metrics log failed: %s", exc)
