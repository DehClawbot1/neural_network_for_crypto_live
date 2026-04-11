from pathlib import Path

import pandas as pd
from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
from model_feature_safety import drop_all_nan_features
from schema import ALIASES
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline


class TimeSplitTrainer:
    """
    Train classifier/regressor with time-based splits instead of random splits.
    Research/paper-trading only.
    """

    FEATURE_COLUMNS = DEFAULT_TABULAR_FEATURE_COLUMNS

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.output_file = self.logs_dir / "time_split_eval.csv"

    def _safe_read(self):
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
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

    def run(self):
        df = self._safe_read()
        if df.empty or "target_up" not in df.columns:
            return pd.DataFrame()

        candidates = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        if not candidates:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        n = len(df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        if train_df.empty or val_df.empty or test_df.empty:
            return pd.DataFrame()
        usable, _ = drop_all_nan_features(train_df, candidates, context="time_split_trainer")
        if not usable:
            return pd.DataFrame()

        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ])
        train_x = self._numeric_feature_frame(train_df, usable)
        val_x = self._numeric_feature_frame(val_df, usable)
        test_x = self._numeric_feature_frame(test_df, usable)

        clf.fit(train_x, train_df["target_up"].astype(int))
        val_pred = clf.predict(val_x)
        test_pred = clf.predict(test_x)

        result = {
            "val_accuracy": accuracy_score(val_df["target_up"].astype(int), val_pred),
            "test_accuracy": accuracy_score(test_df["target_up"].astype(int), test_pred),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        }

        target_return_col = next((c for c in ALIASES["forward_return_15m"] if c in df.columns), None)
        if target_return_col is not None:
            reg = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
            ])
            reg.fit(train_x, train_df[target_return_col])
            val_reg = reg.predict(val_x)
            test_reg = reg.predict(test_x)
            result["val_return_rmse"] = mean_squared_error(val_df[target_return_col], val_reg) ** 0.5
            result["test_return_rmse"] = mean_squared_error(test_df[target_return_col], test_reg) ** 0.5

        output = pd.DataFrame([result])
        output.to_csv(self.output_file, index=False)
        return output

