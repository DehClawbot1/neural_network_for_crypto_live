from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from brain_paths import filter_frame_for_brain, resolve_brain_context

from model_feature_catalog import (
    WEATHER_FORECAST_EDGE_FEATURES,
    WEATHER_MARKET_STRUCTURE_FEATURES,
    WEATHER_WALLET_COPY_FEATURES,
)
from model_feature_safety import drop_all_nan_features


class WeatherTemperatureCentroidModel:
    """
    Lightweight classifier that stays independent of sklearn/scipy.
    It scores each class by inverse distance to that class centroid.
    """

    def __init__(self, *, features: list[str], feature_medians: dict[str, float], class_centroids: dict[int, dict[str, float]], class_priors: dict[int, float]):
        self.features = list(features)
        self.feature_medians = {str(k): float(v) for k, v in (feature_medians or {}).items()}
        self.class_centroids = {
            int(label): {str(k): float(v) for k, v in centroid.items()}
            for label, centroid in (class_centroids or {}).items()
        }
        self.class_priors = {int(label): float(prior) for label, prior in (class_priors or {}).items()}

    def _prepare_frame(self, X) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        for feature in self.features:
            if feature not in frame.columns:
                frame[feature] = self.feature_medians.get(feature, 0.0)
        frame = frame[self.features].apply(pd.to_numeric, errors="coerce")
        for feature in self.features:
            frame[feature] = frame[feature].fillna(self.feature_medians.get(feature, 0.0))
        return frame

    def predict_proba(self, X):
        frame = self._prepare_frame(X)
        labels = sorted(self.class_centroids.keys())
        if not labels:
            return np.zeros((len(frame.index), 2), dtype=float)
        if labels == [0]:
            return np.tile(np.array([[1.0, 0.0]], dtype=float), (len(frame.index), 1))
        if labels == [1]:
            return np.tile(np.array([[0.0, 1.0]], dtype=float), (len(frame.index), 1))

        rows = []
        for _, row in frame.iterrows():
            scores = {}
            for label in labels:
                centroid = self.class_centroids.get(label, {})
                distance = 0.0
                for feature in self.features:
                    distance += abs(float(row.get(feature, 0.0)) - float(centroid.get(feature, self.feature_medians.get(feature, 0.0))))
                prior = max(self.class_priors.get(label, 0.5), 1e-6)
                scores[label] = prior / max(distance, 1e-6)
            total = sum(scores.values()) or 1.0
            rows.append([scores.get(0, 0.0) / total, scores.get(1, 0.0) / total])
        return np.asarray(rows, dtype=float)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= probabilities[:, 0]).astype(int)


class WeatherTemperatureTrainer:
    """
    Train a weather-temperature model from contract-level labeled data.
    """

    FEATURE_COLUMNS = (
        WEATHER_WALLET_COPY_FEATURES
        + WEATHER_MARKET_STRUCTURE_FEATURES
        + WEATHER_FORECAST_EDGE_FEATURES
    )

    def __init__(self, logs_dir="logs", weights_dir="weights", *, brain_context=None, brain_id=None, market_family=None, shared_logs_dir="logs", shared_weights_dir="weights"):
        if brain_context is None and (brain_id or market_family):
            brain_context = resolve_brain_context(
                market_family,
                brain_id=brain_id,
                shared_logs_dir=shared_logs_dir,
                shared_weights_dir=shared_weights_dir,
            )
        self.brain_context = brain_context
        self.logs_dir = Path(brain_context.logs_dir if brain_context is not None else logs_dir)
        self.weights_dir = Path(brain_context.weights_dir if brain_context is not None else weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.model_file = self.weights_dir / "weather_temperature_model.joblib"

    def _safe_read(self):
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def train(self):
        df = self._safe_read()
        if df.empty or "target_up" not in df.columns:
            return None, None
        if self.brain_context is not None:
            df = filter_frame_for_brain(df, self.brain_context)
            if df.empty or "target_up" not in df.columns:
                return None, None

        family_series = df.get("market_family", pd.Series("", index=df.index)).astype(str).str.lower()
        df = df[family_series.str.startswith("weather_temperature")].copy()
        if df.empty:
            return None, None

        candidates = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        usable_features, _ = drop_all_nan_features(df, candidates, context="weather_temperature_trainer")
        if not usable_features:
            return None, None

        X = df[usable_features].apply(pd.to_numeric, errors="coerce")
        usable_features = [col for col in usable_features if X[col].notna().any()]
        if not usable_features:
            return None, None
        X = X[usable_features]
        y = pd.to_numeric(df["target_up"], errors="coerce").fillna(0).astype(int)

        feature_medians = {
            feature: float(value)
            for feature, value in X.median(numeric_only=True).fillna(0.0).to_dict().items()
        }
        X = X.fillna(feature_medians)
        class_centroids = {
            int(label): {
                feature: float(value)
                for feature, value in X.loc[y == label, usable_features].median(numeric_only=True).fillna(0.0).to_dict().items()
            }
            for label in sorted(y.unique())
        }
        class_priors = {
            int(label): float(prior)
            for label, prior in y.value_counts(normalize=True).to_dict().items()
        }
        model = WeatherTemperatureCentroidModel(
            features=usable_features,
            feature_medians=feature_medians,
            class_centroids=class_centroids,
            class_priors=class_priors,
        )
        joblib.dump(
            {
                "model": model,
                "features": usable_features,
                "market_family": "weather_temperature",
                "model_kind": "weather_temperature_centroid",
                "feature_set": "weather_temperature_hybrid",
                "scaling": "median_fill_only",
                "regularization": "centroid_distance",
                "market_family": "weather_temperature",
            },
            self.model_file,
        )
        return model, usable_features
