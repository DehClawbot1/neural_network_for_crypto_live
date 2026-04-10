import os
from pathlib import Path

import joblib
import pandas as pd
from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
from model_feature_safety import drop_all_nan_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SupervisedTrainer:
    """
    Train a supervised BTC-direction model from aligned historical data.
    """

    FEATURE_COLUMNS = DEFAULT_TABULAR_FEATURE_COLUMNS

    def __init__(self, logs_dir="logs", weights_dir="weights"):
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.logs_dir / "aligned_dataset.csv"
        self.model_file = self.weights_dir / "btc_direction_model.joblib"

    def _safe_read(self):
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _train_sparse_classifier(self, X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
        regularization_c = float(os.getenv("SUPERVISED_L1_C", "0.35") or 0.35)
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=42,
                        max_iter=2000,
                        C=max(1e-4, regularization_c),
                    ),
                ),
            ]
        )
        model.fit(X, y)
        coef = getattr(model.named_steps["clf"], "coef_", None)
        nonzero = int((abs(coef) > 1e-12).sum()) if coef is not None else 0
        metadata = {
            "model_kind": "logistic_l1",
            "regularization": "l1",
            "nonzero_feature_count": nonzero,
            "fallback_used": False,
        }
        return model, metadata

    def _train_fallback_classifier(self, X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]
        )
        model.fit(X, y)
        metadata = {
            "model_kind": "random_forest_fallback",
            "regularization": "none",
            "nonzero_feature_count": None,
            "fallback_used": True,
        }
        return model, metadata

    def train(self):
        df = self._safe_read()
        if df.empty:
            return None, None

        candidates = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        usable_features, _ = drop_all_nan_features(df, candidates, context="supervised_trainer")
        if not usable_features or "target_up" not in df.columns:
            return None, None

        X = df[usable_features].apply(pd.to_numeric, errors="coerce")
        usable_features = [col for col in usable_features if X[col].notna().any()]
        if not usable_features:
            return None, None
        X = X[usable_features]
        y = df["target_up"].astype(int)
        if y.nunique(dropna=True) < 2:
            return None, None

        metadata = {}
        try:
            model, metadata = self._train_sparse_classifier(X, y)
        except Exception:
            model, metadata = self._train_fallback_classifier(X, y)
        joblib.dump({"model": model, "features": usable_features, **metadata}, self.model_file)
        return model, usable_features

