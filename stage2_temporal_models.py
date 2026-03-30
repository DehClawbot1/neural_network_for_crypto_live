from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
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

    def _stationarize_features(self, df, feature_cols):
        out = df.copy()
        price_cols = [c for c in feature_cols if "price" in c.lower()]
        for col in price_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = np.log1p(out[col].clip(lower=0))
        for prefix in ["volume", "liquidity"]:
            cols = [c for c in feature_cols if prefix in c.lower()]
            for col in cols:
                series = pd.to_numeric(out[col], errors="coerce")
                out[col] = (series - series.mean()) / (series.std(ddof=0) + 1e-9)
        return out

    def train(self):
        df = self._safe_read()
        if df.empty:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        target_cls = "tp_before_sl_60m" if "tp_before_sl_60m" in df.columns else None
        target_reg = "forward_return_15m" if "forward_return_15m" in df.columns else None
        base_features = [
            c for c in [
                "entry_price",
                "spread",
                "normalized_trade_size",
                "wallet_trade_count_30d",
                "wallet_alpha_30d_y",
                "wallet_signal_precision_tp_y",
                "wallet_avg_forward_return_15m_y",
            ] if c in df.columns
        ]
        lag_features = [c for c in df.columns if "_lag_" in c]
        context_features = [c for c in ["recent_token_activity_5", "recent_yes_ratio_5"] if c in df.columns]
        feature_cols = base_features + lag_features + context_features
        if not feature_cols:
            return pd.DataFrame()

        df = self._stationarize_features(df, feature_cols)
        if len(df) < 6:
            return pd.DataFrame()
        metrics = {}
        n_splits = min(5, len(df) - 1)
        if n_splits < 2:
            return pd.DataFrame()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        if target_cls:
            cls_scores = []
            precision_scores = []
            recall_scores = []
            top_precision_scores = []
            last_clf = None
            for train_idx, test_idx in tscv.split(df):
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                if train_df.empty or test_df.empty:
                    continue
                balanced_train_df = self._balance_binary_frame(train_df, target_cls)
                clf = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler()),
                    ("model", MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=300, learning_rate_init=1e-3, alpha=1e-4, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15)),
                ])
                clf.fit(balanced_train_df[feature_cols], balanced_train_df[target_cls].fillna(0).astype(int))
                y_test = test_df[target_cls].fillna(0).astype(int)
                preds = clf.predict(test_df[feature_cols])
                cls_scores.append(accuracy_score(y_test, preds))
                precision_scores.append(precision_score(y_test, preds, zero_division=0))
                recall_scores.append(recall_score(y_test, preds, zero_division=0))
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(test_df[feature_cols])[:, 1]
                    top_k = max(1, int(len(proba) * 0.1))
                    top_idx = np.argsort(proba)[-top_k:]
                    top_precision_scores.append(precision_score(y_test.iloc[top_idx], preds[top_idx], zero_division=0))
                last_clf = clf
            if cls_scores:
                metrics["temporal_walk_forward_accuracy"] = float(sum(cls_scores) / len(cls_scores))
                metrics["temporal_walk_forward_precision"] = float(sum(precision_scores) / len(precision_scores))
                metrics["temporal_walk_forward_recall"] = float(sum(recall_scores) / len(recall_scores))
                if top_precision_scores:
                    metrics["temporal_precision_at_top_10pct"] = float(sum(top_precision_scores) / len(top_precision_scores))
            if last_clf is not None:
                joblib.dump({"model": last_clf, "features": feature_cols}, self.classifier_file)

        if target_reg:
            reg_scores = []
            last_reg = None
            for train_idx, test_idx in tscv.split(df):
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                if train_df.empty or test_df.empty:
                    continue
                reg = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler()),
                    ("model", MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42, max_iter=300, learning_rate_init=1e-3, alpha=1e-4, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15)),
                ])
                reg.fit(train_df[feature_cols], train_df[target_reg].fillna(0.0))
                preds = reg.predict(test_df[feature_cols])
                reg_scores.append(mean_squared_error(test_df[target_reg].fillna(0.0), preds) ** 0.5)
                last_reg = reg
            if reg_scores:
                metrics["temporal_walk_forward_rmse"] = float(sum(reg_scores) / len(reg_scores))
            if last_reg is not None:
                joblib.dump({"model": last_reg, "features": feature_cols}, self.regressor_file)

        result = pd.DataFrame([metrics]) if metrics else pd.DataFrame()
        if not result.empty:
            result.to_csv(self.metrics_file, index=False)
        return result

