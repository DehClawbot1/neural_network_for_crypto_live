from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Stage2TemporalModels:
    """
    Stage 2 temporal baseline using lagged sequence features.
    This is a practical intermediate step before heavier deep sequence models.
    """

    def __init__(self, logs_dir="logs", weights_dir="weights"):
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.logs_dir / "sequence_dataset.csv"
        self.classifier_file = self.weights_dir / "stage2_temporal_classifier.joblib"
        self.regressor_file = self.weights_dir / "stage2_temporal_regressor.joblib"
        self.metrics_file = self.logs_dir / "stage2_temporal_eval.csv"

    def _safe_read(self):
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _balance_binary_frame(self, df, target_col):
        if target_col not in df.columns or df.empty:
            return df
        counts = df[target_col].fillna(0).astype(int).value_counts()
        if len(counts) < 2:
            return df
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()
        majority_df = df[df[target_col].fillna(0).astype(int) == majority_class]
        minority_df = df[df[target_col].fillna(0).astype(int) == minority_class]
        if minority_df.empty or majority_df.empty:
            return df
        minority_upsampled = resample(minority_df, replace=True, n_samples=len(majority_df), random_state=42)
        balanced = pd.concat([majority_df, minority_upsampled], ignore_index=True)
        return balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    def train(self):
        df = self._safe_read()
        if df.empty:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        target_cls = "tp_before_sl_60m" if "tp_before_sl_60m" in df.columns else None
        target_reg = "forward_return_15m" if "forward_return_15m" in df.columns else None
        feature_cols = [c for c in df.columns if "_lag_" in c or c in ["recent_wallet_activity_5", "recent_yes_ratio_5"]]
        if not feature_cols:
            return pd.DataFrame()

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        if train_df.empty or test_df.empty:
            return pd.DataFrame()

        metrics = {}

        if target_cls:
            balanced_train_df = self._balance_binary_frame(train_df, target_cls)
            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=300)),
            ])
            clf.fit(balanced_train_df[feature_cols], balanced_train_df[target_cls].fillna(0).astype(int))
            preds = clf.predict(test_df[feature_cols])
            metrics["temporal_test_accuracy"] = accuracy_score(test_df[target_cls].fillna(0).astype(int), preds)
            joblib.dump({"model": clf, "features": feature_cols}, self.classifier_file)

        if target_reg:
            reg = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42, max_iter=300)),
            ])
            reg.fit(train_df[feature_cols], train_df[target_reg].fillna(0.0))
            preds = reg.predict(test_df[feature_cols])
            metrics["temporal_test_rmse"] = mean_squared_error(test_df[target_reg].fillna(0.0), preds) ** 0.5
            joblib.dump({"model": reg, "features": feature_cols}, self.regressor_file)

        result = pd.DataFrame([metrics]) if metrics else pd.DataFrame()
        if not result.empty:
            result.to_csv(self.metrics_file, index=False)
        return result
