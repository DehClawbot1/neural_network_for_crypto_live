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
    Walk-forward evaluation with PURGING and EMBARGO to eliminate time leakage
    from overlapping feature windows and autocorrelated residuals.

    Concepts (López de Prado, Advances in Financial ML, ch. 7):
    - PURGE: drop training samples whose *label window* overlaps any test
      sample's *feature window*. Without this, the model trains on data that
      encodes post-test-sample information.
    - EMBARGO: further drop training samples within `embargo_bars` of each
      test fold boundary, to suppress short-term autocorrelation leakage at
      the seam.

    Research/paper-trading only.
    """

    FEATURE_COLUMNS = DEFAULT_TABULAR_FEATURE_COLUMNS

    def __init__(
        self,
        logs_dir="logs",
        *,
        embargo_bars: int = 24,
        label_window_bars: int = 1,
        n_splits: int = 1,
    ):
        self.logs_dir = Path(logs_dir)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.output_file = self.logs_dir / "walk_forward_eval.csv"
        if embargo_bars < 0:
            raise ValueError("embargo_bars must be >= 0")
        if label_window_bars < 1:
            raise ValueError("label_window_bars must be >= 1")
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.embargo_bars = int(embargo_bars)
        self.label_window_bars = int(label_window_bars)
        self.n_splits = int(n_splits)

    @staticmethod
    def purge_train_indices(
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        *,
        embargo_bars: int,
        label_window_bars: int,
    ) -> np.ndarray:
        """
        Remove training rows whose label window [i, i+label_window_bars) overlaps
        any test row, AND rows within `embargo_bars` of the test-fold boundary.

        Assumes indices are positional (0..N-1) after sorting by time.
        """
        if test_idx.size == 0:
            return train_idx
        test_min = int(test_idx.min())
        test_max = int(test_idx.max())
        # Purge: any train row i whose [i, i+label_window_bars-1] intersects [test_min, test_max]
        purge_lo = test_min - label_window_bars + 1
        purge_hi = test_max
        # Embargo: forbid train rows in a band around the test fold
        embargo_lo = test_min - embargo_bars
        embargo_hi = test_max + embargo_bars
        lo = min(purge_lo, embargo_lo)
        hi = max(purge_hi, embargo_hi)
        mask = (train_idx < lo) | (train_idx > hi)
        return train_idx[mask]

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

        df = df.reset_index(drop=True)
        split_idx = int(len(df) * train_ratio)
        all_idx = np.arange(len(df))
        test_idx = all_idx[split_idx:]
        train_idx_raw = all_idx[:split_idx]
        train_idx = self.purge_train_indices(
            train_idx_raw,
            test_idx,
            embargo_bars=self.embargo_bars,
            label_window_bars=self.label_window_bars,
        )
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
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
                "train_rows_raw": int(train_idx_raw.size),
                "purged_rows": int(train_idx_raw.size - train_idx.size),
                "embargo_bars": self.embargo_bars,
                "label_window_bars": self.label_window_bars,
                "test_rows": len(test_df),
                "accuracy": acc,
                "positive_rate": float(np.mean(preds)),
            }
        ])
        result.to_csv(self.output_file, index=False)
        return result

