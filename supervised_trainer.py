from pathlib import Path

import joblib
import pandas as pd
from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
from model_feature_safety import drop_all_nan_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]
        )
        model.fit(X, y)
        joblib.dump({"model": model, "features": usable_features}, self.model_file)
        return model, usable_features

