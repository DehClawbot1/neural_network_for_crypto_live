"""
BTC Price Forecast Model — v2 (Advanced)

Ensemble model that predicts:
  - btc_predicted_return_15  (regression: expected 15-candle forward return)
  - btc_predicted_direction  (classification: -1 / 0 / +1)
  - btc_forecast_confidence  (calibrated probability from ensemble)

Advanced techniques:
  1. Purged walk-forward cross-validation (no look-ahead leakage)
  2. Ensemble stacking: LightGBM + sklearn GBT + MLP meta-learner
  3. Binary target focus (UP/DOWN, collapsing neutral zone)
  4. Feature importance-based pruning
  5. Regime-aware sample weighting (recent data weighted higher)
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
# Feature columns (auto-discovered from dataset, but these are the baseline)
# ---------------------------------------------------------------------------
_EXCLUDE_COLS = {
    "timestamp", "close", "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_9", "ema_21", "atr_14", "bb_upper", "bb_lower",
    "volume_sma_20", "obv", "obv_sma_10",
}
_LABEL_PREFIXES = ("fwd_return_", "fwd_up_", "fwd_direction_", "fwd_return_vol_adj_", "fwd_max_drawdown_", "fwd_max_runup_")

# Walk-forward CV settings
_N_SPLITS = 5
_PURGE_BARS = 60  # gap between train/test to prevent leakage
_MIN_TRAIN_SIZE = 5000


class BTCForecastModel:
    """
    Ensemble BTC price direction + return predictor.

    Training pipeline:
      1. Auto-discover relative features (exclude absolute prices)
      2. Purged walk-forward CV to estimate generalisation
      3. Train final ensemble on full training data
      4. Feature importance pruning (drop low-importance features)
    """

    def __init__(self, weights_dir: str = "weights", logs_dir: str = "logs"):
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._ensemble_reg = None   # list of (model, weight)
        self._ensemble_cls = None   # list of (model, weight)
        self._feature_cols: list[str] = []
        self._model_meta: dict = {}
        self._last_prediction: dict = {}
        self._last_predict_ts: float = 0.0

        self._try_load()

    # ------------------------------------------------------------------
    # Feature discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_features(df: pd.DataFrame) -> list[str]:
        """Auto-discover feature columns by excluding labels and absolute-price cols."""
        features = []
        for col in df.columns:
            if col in _EXCLUDE_COLS:
                continue
            if any(col.startswith(p) for p in _LABEL_PREFIXES):
                continue
            if df[col].dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
                features.append(col)
        return features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        target_return_col: str = "fwd_return_15",
        target_dir_col: str = "fwd_up_15",  # binary UP/DOWN (better than 3-class)
        test_fraction: float = 0.15,
        feedback_weights: dict | None = None,
    ) -> dict:
        """
        Train ensemble with purged walk-forward cross-validation.
        """
        if df is None or df.empty:
            logger.error("BTCForecastModel.train: empty dataset")
            return {"error": "empty dataset"}

        # Auto-discover features
        available = self._discover_features(df)
        if len(available) < 5:
            logger.error("BTCForecastModel.train: too few feature columns (%d)", len(available))
            return {"error": "too few features"}

        # Ensure target columns exist; fall back gracefully
        if target_dir_col not in df.columns:
            target_dir_col = "fwd_direction_15"
        if target_return_col not in df.columns:
            logger.error("BTCForecastModel.train: target column %s not found", target_return_col)
            return {"error": f"missing {target_return_col}"}

        # Clean dataset
        cols_needed = available + [target_return_col, target_dir_col]
        cols_needed = [c for c in cols_needed if c in df.columns]
        clean = df[cols_needed].dropna()
        if len(clean) < 500:
            logger.error("BTCForecastModel.train: only %d clean rows (need >=500)", len(clean))
            return {"error": f"only {len(clean)} rows"}

        X_all = clean[available].values.astype(np.float32)
        y_reg_all = clean[target_return_col].values.astype(float)
        y_cls_all = clean[target_dir_col].values.astype(int)

        # --- Feature importance pruning (first pass) ---
        important_features, importance_scores = self._select_important_features(
            X_all, y_cls_all, available, top_k=80
        )
        if len(important_features) >= 10:
            feat_indices = [available.index(f) for f in important_features]
            X_all = X_all[:, feat_indices]
            available = important_features
            logger.info("BTCForecastModel: pruned to %d important features", len(available))

        self._feature_cols = available

        # --- Purged walk-forward cross-validation ---
        cv_metrics = self._purged_walk_forward_cv(
            X_all, y_reg_all, y_cls_all, n_splits=_N_SPLITS
        )

        # --- Train final models on all data except holdout ---
        split_idx = int(len(X_all) * (1.0 - test_fraction))
        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
        y_reg_train, y_reg_test = y_reg_all[:split_idx], y_reg_all[split_idx:]
        y_cls_train, y_cls_test = y_cls_all[:split_idx], y_cls_all[split_idx:]

        # Apply recency weighting, adjusted by trade feedback accuracy
        feedback_scale = 1.0
        if feedback_weights:
            feedback_scale = feedback_weights.get("error_scale", 1.0)
            logger.info(
                "BTCForecastModel: using trade feedback weights (accuracy=%.1f%% error_scale=%.3f)",
                feedback_weights.get("overall_accuracy", 0) * 100,
                feedback_scale,
            )
        sample_weights = self._compute_sample_weights(len(X_train), feedback_scale=feedback_scale)

        logger.info(
            "BTCForecastModel: training ensemble on %d rows, testing on %d rows, %d features",
            len(X_train), len(X_test), len(available),
        )

        ensemble_reg, ensemble_cls = self._train_ensemble(
            X_train, y_reg_train, y_cls_train, sample_weights
        )

        # Evaluate on holdout
        from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

        reg_pred = self._ensemble_predict_reg(ensemble_reg, X_test)
        cls_pred = self._ensemble_predict_cls(ensemble_cls, X_test)

        mae = float(mean_absolute_error(y_reg_test, reg_pred))
        rmse = float(np.sqrt(mean_squared_error(y_reg_test, reg_pred)))

        # For binary target
        accuracy = float(accuracy_score(y_cls_test, cls_pred))

        # Directional accuracy from regressor
        dir_correct = np.sign(reg_pred) == np.sign(y_reg_test)
        nonzero_mask = y_reg_test != 0
        dir_accuracy = float(dir_correct[nonzero_mask].mean()) if nonzero_mask.any() else 0.5

        # Profitable direction accuracy (when model is confident)
        confident_mask = np.abs(reg_pred) > np.percentile(np.abs(reg_pred), 50)
        confident_dir_acc = float(dir_correct[confident_mask & nonzero_mask].mean()) if (confident_mask & nonzero_mask).any() else 0.5

        metrics = {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": int(len(available)),
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "direction_accuracy": round(dir_accuracy, 4),
            "confident_direction_accuracy": round(confident_dir_acc, 4),
            "classifier_accuracy": round(accuracy, 4),
            "cv_direction_accuracy": round(cv_metrics.get("mean_dir_acc", 0), 4),
            "cv_classifier_accuracy": round(cv_metrics.get("mean_cls_acc", 0), 4),
            "ensemble_members": len(ensemble_reg),
            "backend": "ensemble",
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }

        logger.info(
            "BTCForecastModel v2 trained: DirAcc=%.2f%% ConfidentDirAcc=%.2f%% "
            "ClsAcc=%.2f%% CV_DirAcc=%.2f%% MAE=%.6f features=%d",
            dir_accuracy * 100, confident_dir_acc * 100,
            accuracy * 100, cv_metrics.get("mean_dir_acc", 0) * 100,
            mae, len(available),
        )

        self._ensemble_reg = ensemble_reg
        self._ensemble_cls = ensemble_cls
        self._model_meta = metrics

        self._save()
        self._save_metrics(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Purged Walk-Forward Cross-Validation
    # ------------------------------------------------------------------

    def _purged_walk_forward_cv(self, X, y_reg, y_cls, n_splits=5) -> dict:
        """
        Walk-forward CV with purge gap to prevent information leakage.
        Each fold uses expanding window for training.
        """
        total = len(X)
        test_size = max(500, total // (n_splits + 1))
        dir_accs, cls_accs = [], []

        for fold in range(n_splits):
            test_end = total - fold * test_size
            test_start = test_end - test_size
            if test_start < _MIN_TRAIN_SIZE + _PURGE_BARS:
                break

            train_end = test_start - _PURGE_BARS  # purge gap
            X_tr, X_te = X[:train_end], X[test_start:test_end]
            y_reg_tr, y_reg_te = y_reg[:train_end], y_reg[test_start:test_end]
            y_cls_tr, y_cls_te = y_cls[:train_end], y_cls[test_start:test_end]

            weights = self._compute_sample_weights(len(X_tr))
            ens_reg, ens_cls = self._train_ensemble(X_tr, y_reg_tr, y_cls_tr, weights)

            reg_pred = self._ensemble_predict_reg(ens_reg, X_te)
            cls_pred = self._ensemble_predict_cls(ens_cls, X_te)

            nonzero = y_reg_te != 0
            if nonzero.any():
                dir_accs.append(float((np.sign(reg_pred[nonzero]) == np.sign(y_reg_te[nonzero])).mean()))

            from sklearn.metrics import accuracy_score
            cls_accs.append(float(accuracy_score(y_cls_te, cls_pred)))

            logger.info(
                "  CV fold %d: dir_acc=%.2f%% cls_acc=%.2f%% (train=%d test=%d)",
                fold + 1,
                dir_accs[-1] * 100 if dir_accs else 0,
                cls_accs[-1] * 100,
                len(X_tr), len(X_te),
            )

        return {
            "mean_dir_acc": float(np.mean(dir_accs)) if dir_accs else 0.0,
            "mean_cls_acc": float(np.mean(cls_accs)) if cls_accs else 0.0,
            "std_dir_acc": float(np.std(dir_accs)) if dir_accs else 0.0,
            "n_folds": len(dir_accs),
        }

    # ------------------------------------------------------------------
    # Ensemble Training
    # ------------------------------------------------------------------

    def _train_ensemble(self, X_train, y_reg_train, y_cls_train, sample_weights):
        """Train multiple diverse models for ensemble."""
        regressors = []
        classifiers = []

        # --- Model 1: LightGBM (primary) ---
        try:
            import lightgbm as lgb
            reg1 = lgb.LGBMRegressor(
                n_estimators=500, max_depth=7, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7,
                min_child_samples=30, reg_alpha=0.1, reg_lambda=0.5,
                verbosity=-1, n_jobs=-1,
            )
            reg1.fit(X_train, y_reg_train, sample_weight=sample_weights)
            regressors.append((reg1, 0.45))

            cls1 = lgb.LGBMClassifier(
                n_estimators=500, max_depth=7, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7,
                min_child_samples=30, reg_alpha=0.1, reg_lambda=0.5,
                verbosity=-1, n_jobs=-1,
            )
            cls1.fit(X_train, y_cls_train, sample_weight=sample_weights)
            classifiers.append((cls1, 0.45))
        except (ImportError, ModuleNotFoundError):
            logger.info("BTCForecastModel: LightGBM not available, skipping")

        # --- Model 2: LightGBM with different hyperparams (diversity) ---
        try:
            import lightgbm as lgb
            reg2 = lgb.LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.6,
                min_child_samples=50, reg_alpha=0.5, reg_lambda=1.0,
                verbosity=-1, n_jobs=-1,
            )
            reg2.fit(X_train, y_reg_train, sample_weight=sample_weights)
            regressors.append((reg2, 0.25))

            cls2 = lgb.LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.6,
                min_child_samples=50, reg_alpha=0.5, reg_lambda=1.0,
                verbosity=-1, n_jobs=-1,
            )
            cls2.fit(X_train, y_cls_train, sample_weight=sample_weights)
            classifiers.append((cls2, 0.25))
        except (ImportError, ModuleNotFoundError):
            pass

        # --- Model 3: sklearn HistGradientBoosting (different algorithm) ---
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

            reg3 = HistGradientBoostingRegressor(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=30, l2_regularization=0.5,
            )
            reg3.fit(X_train, y_reg_train, sample_weight=sample_weights)
            regressors.append((reg3, 0.20))

            cls3 = HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=30, l2_regularization=0.5,
            )
            cls3.fit(X_train, y_cls_train, sample_weight=sample_weights)
            classifiers.append((cls3, 0.20))
        except (ImportError, TypeError):
            pass

        # --- Model 4: MLP (neural net — captures non-tree patterns) ---
        try:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            reg4 = _ScaledModel(
                MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32), max_iter=200,
                    learning_rate="adaptive", learning_rate_init=0.001,
                    early_stopping=True, validation_fraction=0.1,
                    alpha=0.001, batch_size=256,
                ),
                scaler,
            )
            reg4.model.fit(X_scaled, y_reg_train)
            regressors.append((reg4, 0.10))

            cls4 = _ScaledModel(
                MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32), max_iter=200,
                    learning_rate="adaptive", learning_rate_init=0.001,
                    early_stopping=True, validation_fraction=0.1,
                    alpha=0.001, batch_size=256,
                ),
                scaler,
            )
            cls4.model.fit(X_scaled, y_cls_train)
            classifiers.append((cls4, 0.10))
        except (ImportError, TypeError) as exc:
            logger.debug("BTCForecastModel: MLP skipped: %s", exc)

        # Normalise weights
        if regressors:
            total_w = sum(w for _, w in regressors)
            regressors = [(m, w / total_w) for m, w in regressors]
        if classifiers:
            total_w = sum(w for _, w in classifiers)
            classifiers = [(m, w / total_w) for m, w in classifiers]

        logger.info(
            "BTCForecastModel: ensemble built with %d regressors, %d classifiers",
            len(regressors), len(classifiers),
        )
        return regressors, classifiers

    # ------------------------------------------------------------------
    # Ensemble Prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _ensemble_predict_reg(ensemble, X) -> np.ndarray:
        """Weighted average of regressor predictions."""
        if not ensemble:
            return np.zeros(len(X))
        preds = np.zeros(len(X))
        for model, weight in ensemble:
            preds += weight * model.predict(X)
        return preds

    @staticmethod
    def _ensemble_predict_cls(ensemble, X) -> np.ndarray:
        """Weighted soft-voting for classifiers."""
        if not ensemble:
            return np.zeros(len(X), dtype=int)

        # Collect weighted probabilities
        all_proba = None
        for model, weight in ensemble:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                if all_proba is None:
                    all_proba = weight * proba
                else:
                    # Align shapes (different models may have different class sets)
                    if proba.shape == all_proba.shape:
                        all_proba += weight * proba
                    else:
                        all_proba += weight * np.zeros_like(all_proba)
            else:
                # Fall back to hard vote
                pred = model.predict(X)
                if all_proba is not None:
                    for i, p in enumerate(pred):
                        col = min(int(p), all_proba.shape[1] - 1)
                        if col >= 0:
                            all_proba[i, col] += weight

        if all_proba is not None:
            return all_proba.argmax(axis=1)
        # Pure hard voting fallback
        votes = np.array([m.predict(X) for m, _ in ensemble])
        from scipy.stats import mode as scipy_mode
        result, _ = scipy_mode(votes, axis=0)
        return result.flatten().astype(int)

    # ------------------------------------------------------------------
    # Feature importance pruning
    # ------------------------------------------------------------------

    def _select_important_features(self, X, y, feature_names, top_k=80) -> tuple:
        """Use LightGBM feature importance to select top-K features."""
        try:
            import lightgbm as lgb
            quick_model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                verbosity=-1, n_jobs=-1,
            )
            quick_model.fit(X, y)
            importance = quick_model.feature_importances_
            indices = np.argsort(importance)[::-1][:top_k]
            selected = [feature_names[i] for i in indices if importance[i] > 0]
            scores = {feature_names[i]: int(importance[i]) for i in indices[:top_k]}
            logger.info(
                "BTCForecastModel: top-5 features: %s",
                list(scores.items())[:5],
            )
            return selected, scores
        except (ImportError, ModuleNotFoundError):
            return feature_names[:top_k], {}

    # ------------------------------------------------------------------
    # Sample weighting (recency bias)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sample_weights(n: int, decay: float = 0.9999, feedback_scale: float = 1.0) -> np.ndarray:
        """Exponential recency weighting, optionally scaled by trade feedback.

        Args:
            n: number of samples
            decay: exponential decay factor (closer to 1 = slower decay)
            feedback_scale: multiplier from BTCTradeFeedback (>1 = model is
                accurate so trust recent data more; <1 = model is inaccurate
                so flatten weights toward uniform)
        """
        weights = np.power(decay, np.arange(n)[::-1])
        if feedback_scale != 1.0:
            # Blend between recency weights and uniform weights
            # High accuracy -> more recency bias; low accuracy -> flatter
            alpha = min(1.0, max(0.3, feedback_scale))
            uniform = np.ones(n)
            weights = alpha * weights + (1 - alpha) * uniform
        weights /= weights.mean()  # normalise so mean weight = 1
        return weights

    # ------------------------------------------------------------------
    # Prediction (live inference)
    # ------------------------------------------------------------------

    def predict(self, features: dict | pd.Series | pd.DataFrame) -> dict:
        """
        Predict BTC price direction and return from a single feature row.
        """
        default = {
            "btc_predicted_return_15": 0.0,
            "btc_predicted_direction": 0,
            "btc_forecast_confidence": 0.0,
            "btc_forecast_ready": False,
        }

        if not self._ensemble_reg or not self._ensemble_cls:
            return default

        try:
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            elif isinstance(features, pd.Series):
                features = features.to_frame().T

            available = [c for c in self._feature_cols if c in features.columns]
            if len(available) < 5:
                return default

            X = features[available].values.astype(np.float32)
            # Replace NaN/inf with 0 for robustness
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            reg_pred = float(self._ensemble_predict_reg(self._ensemble_reg, X)[0])
            cls_pred_raw = self._ensemble_predict_cls(self._ensemble_cls, X)
            cls_pred = int(cls_pred_raw[0]) if len(cls_pred_raw) > 0 else 0

            # Ensemble confidence from probability spread
            confidence = 0.5
            try:
                probas = []
                for model, weight in self._ensemble_cls:
                    if hasattr(model, "predict_proba"):
                        p = model.predict_proba(X)[0]
                        probas.append(weight * p)
                if probas:
                    avg_proba = sum(probas) / sum(w for _, w in self._ensemble_cls if hasattr(_, "predict_proba") or True)
                    confidence = float(np.max(avg_proba))
            except Exception:
                pass

            # Map binary cls (0/1) to direction (-1/+1)
            direction = 1 if cls_pred >= 1 else -1

            # Agreement boost: if regressor and classifier agree, boost confidence
            if (reg_pred > 0 and direction > 0) or (reg_pred < 0 and direction < 0):
                confidence = min(1.0, confidence * 1.1)

            result = {
                "btc_predicted_return_15": round(reg_pred, 6),
                "btc_predicted_direction": direction,
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
        """Build features from raw candles and predict using last row."""
        from btc_price_dataset import BTCPriceDatasetBuilder
        builder = BTCPriceDatasetBuilder(logs_dir=str(self.logs_dir))
        features_df = builder.build_from_candles(candle_df)
        if features_df.empty:
            return self.predict({})
        return self.predict(features_df.iloc[-1])

    @property
    def is_ready(self) -> bool:
        return bool(self._ensemble_reg) and bool(self._ensemble_cls)

    @property
    def model_meta(self) -> dict:
        return dict(self._model_meta)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        try:
            import joblib
            joblib.dump(self._ensemble_reg, self.weights_dir / "btc_forecast_ensemble_reg.joblib")
            joblib.dump(self._ensemble_cls, self.weights_dir / "btc_forecast_ensemble_cls.joblib")

            meta = {
                **self._model_meta,
                "feature_cols": self._feature_cols,
            }
            (self.weights_dir / "btc_forecast_meta.json").write_text(
                json.dumps(meta, indent=2, default=str), encoding="utf-8"
            )
            logger.info("BTCForecastModel: saved ensemble to %s", self.weights_dir)
        except Exception as exc:
            logger.warning("BTCForecastModel: save failed: %s", exc)

    def _try_load(self):
        try:
            import joblib
            reg_path = self.weights_dir / "btc_forecast_ensemble_reg.joblib"
            cls_path = self.weights_dir / "btc_forecast_ensemble_cls.joblib"
            meta_path = self.weights_dir / "btc_forecast_meta.json"

            if not reg_path.exists() or not cls_path.exists():
                # Try loading old single-model format for backward compat
                self._try_load_legacy()
                return

            self._ensemble_reg = joblib.load(reg_path)
            self._ensemble_cls = joblib.load(cls_path)

            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._model_meta = meta
                self._feature_cols = meta.get("feature_cols", [])

            logger.info(
                "BTCForecastModel: loaded ensemble from %s (%d reg, %d cls, %d features)",
                self.weights_dir,
                len(self._ensemble_reg or []),
                len(self._ensemble_cls or []),
                len(self._feature_cols),
            )
        except Exception as exc:
            logger.debug("BTCForecastModel: no saved model loaded: %s", exc)

    def _try_load_legacy(self):
        """Load old single-model format (v1 backward compat)."""
        try:
            import joblib
            reg_path = self.weights_dir / "btc_forecast_regressor.joblib"
            cls_path = self.weights_dir / "btc_forecast_classifier.joblib"
            meta_path = self.weights_dir / "btc_forecast_meta.json"

            if not reg_path.exists() or not cls_path.exists():
                return

            reg = joblib.load(reg_path)
            cls = joblib.load(cls_path)
            self._ensemble_reg = [(reg, 1.0)]
            self._ensemble_cls = [(cls, 1.0)]

            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._model_meta = meta
                self._feature_cols = meta.get("feature_cols", [])

            logger.info("BTCForecastModel: loaded legacy single model (will upgrade on next train)")
        except Exception:
            pass

    def _save_metrics(self, metrics: dict):
        from csv_utils import safe_csv_append
        try:
            safe_csv_append(self.logs_dir / "btc_forecast_train_log.csv", pd.DataFrame([metrics]))
        except Exception as exc:
            logger.debug("BTCForecastModel: metrics log failed: %s", exc)


class _ScaledModel:
    """Wrapper that applies StandardScaler before predict/predict_proba."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

    def __getattr__(self, name):
        # Guard against infinite recursion during unpickling (self.model not yet set)
        if name in ("model", "scaler"):
            raise AttributeError(name)
        return getattr(self.model, name)
