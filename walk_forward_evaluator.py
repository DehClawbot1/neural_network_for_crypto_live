from pathlib import Path

import numpy as np
import pandas as pd
from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
from model_feature_safety import drop_all_nan_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


class WalkForwardEvaluator:
    """
    Simple walk-forward evaluation to reduce time leakage.
    Research/paper-trading only.
    """

    FEATURE_COLUMNS = DEFAULT_TABULAR_FEATURE_COLUMNS

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.output_file = self.logs_dir / "walk_forward_eval.csv"

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _numeric_feature_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        normalized = {}
        truthy = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "on": 1.0, "off": 0.0}
        for column in columns:
            series = df[column]
            if pd.api.types.is_bool_dtype(series):
                normalized[column] = series.astype(float)
                continue
            if pd.api.types.is_numeric_dtype(series):
                normalized[column] = pd.to_numeric(series, errors="coerce")
                continue
            text = series.astype(str).str.strip().str.lower()
            mapped = text.map(truthy)
            numeric = pd.to_numeric(series, errors="coerce")
            normalized[column] = numeric.where(numeric.notna(), mapped)
        return pd.DataFrame(normalized, index=df.index)

    def evaluate(self, train_ratio=0.7):
        df = self._safe_read(self.dataset_file)
        if df.empty or "target_up" not in df.columns:
            return pd.DataFrame()

        candidates = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        if not candidates:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        if train_df.empty or test_df.empty:
            return pd.DataFrame()
        usable, _ = drop_all_nan_features(train_df, candidates, context="walk_forward_evaluator")
        if not usable:
            return pd.DataFrame()

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]
        )
        train_x = self._numeric_feature_frame(train_df, usable)
        test_x = self._numeric_feature_frame(test_df, usable)
        model.fit(train_x, train_df["target_up"].astype(int))
        preds = model.predict(test_x)
        acc = accuracy_score(test_df["target_up"].astype(int), preds)

        result = pd.DataFrame([
            {
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "accuracy": acc,
                "positive_rate": float(np.mean(preds)),
            }
        ])
        result.to_csv(self.output_file, index=False)
        return result

