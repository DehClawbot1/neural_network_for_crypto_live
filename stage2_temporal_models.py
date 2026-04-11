from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import warnings
from brain_paths import filter_frame_for_brain, normalize_market_family, resolve_brain_context
from model_feature_safety import drop_all_nan_features
from feature_treatment_policy import features_by_kind, features_for_scope, log_audit
from return_calibration import fit_return_calibration, transform_return_targets


def _load_sklearn_temporal_components():
    from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "ConvergenceWarning": ConvergenceWarning,
        "MLPClassifier": MLPClassifier,
        "MLPRegressor": MLPRegressor,
        "Pipeline": Pipeline,
        "SimpleImputer": SimpleImputer,
        "StandardScaler": StandardScaler,
        "TimeSeriesSplit": TimeSeriesSplit,
        "UndefinedMetricWarning": UndefinedMetricWarning,
        "accuracy_score": accuracy_score,
        "mean_squared_error": mean_squared_error,
        "precision_score": precision_score,
        "recall_score": recall_score,
    }


class Stage2TemporalModels:
    """
    Stage 2 temporal baseline using lagged sequence features.
    This is a practical intermediate step before heavier deep sequence models.
    """

    def __init__(self, logs_dir="logs", weights_dir="weights", *, brain_context=None, brain_id=None, market_family=None, shared_logs_dir="logs", shared_weights_dir="weights"):
        if brain_context is None and (brain_id or market_family):
            brain_context = resolve_brain_context(
                market_family,
                brain_id=brain_id,
                shared_logs_dir=shared_logs_dir,
                shared_weights_dir=shared_weights_dir,
            )
        self.brain_context = brain_context
        self.market_family = (
            brain_context.market_family
            if brain_context is not None
            else (normalize_market_family(market_family) or "btc")
        )
        self.logs_dir = Path(brain_context.logs_dir if brain_context is not None else logs_dir)
        self.weights_dir = Path(brain_context.weights_dir if brain_context is not None else weights_dir)
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
        minority_upsampled = minority_df.sample(n=len(majority_df), replace=True, random_state=42)
        balanced = pd.concat([majority_df, minority_upsampled], ignore_index=True)
        return balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    def _stationarize_features(self, df, feature_cols):
        """Apply policy-driven transforms before the sklearn pipeline.

        ``log_scale`` features get ``log1p``; ``robust_scale`` features get
        median/IQR normalisation.  Everything else is left for the pipeline's
        ``StandardScaler`` to handle.
        """
        out = df.copy()
        log_cols = features_by_kind("log_scale", feature_cols)
        for col in log_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = np.log1p(out[col].clip(lower=0))
        robust_cols = features_by_kind("robust_scale", feature_cols)
        for col in robust_cols:
            series = pd.to_numeric(out[col], errors="coerce")
            median = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            out[col] = (series - median) / (iqr + 1e-9)
        return out

    def _pick_first_existing(self, df, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

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

    @staticmethod
    def _can_use_mlp_early_stopping(n_rows: int, validation_fraction: float = 0.15) -> bool:
        try:
            n_rows = int(n_rows)
        except Exception:
            return False
        if n_rows <= 0:
            return False
        # MLP early-stopping uses an internal validation split and, for regressors,
        # computes R2 on that split. Require at least 2 validation samples.
        n_val = int(np.ceil(n_rows * float(validation_fraction)))
        return n_val >= 2

    def train(self):
        sklearn = _load_sklearn_temporal_components()
        SimpleImputer = sklearn["SimpleImputer"]
        TimeSeriesSplit = sklearn["TimeSeriesSplit"]
        Pipeline = sklearn["Pipeline"]
        StandardScaler = sklearn["StandardScaler"]
        MLPClassifier = sklearn["MLPClassifier"]
        MLPRegressor = sklearn["MLPRegressor"]
        ConvergenceWarning = sklearn["ConvergenceWarning"]
        UndefinedMetricWarning = sklearn["UndefinedMetricWarning"]
        accuracy_score = sklearn["accuracy_score"]
        mean_squared_error = sklearn["mean_squared_error"]
        precision_score = sklearn["precision_score"]
        recall_score = sklearn["recall_score"]

        df = self._safe_read()
        if df.empty:
            return pd.DataFrame()
        if self.brain_context is not None:
            df = filter_frame_for_brain(df, self.brain_context)
            if df.empty:
                return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        target_cls = "tp_before_sl_60m" if "tp_before_sl_60m" in df.columns else None
        target_reg = "forward_return_15m" if "forward_return_15m" in df.columns else None
        base_features = []
        base_features.extend([c for c in ["entry_price", "spread", "normalized_trade_size", "wallet_trade_count_30d"] if c in df.columns])
        alias_groups = [
            ["wallet_alpha_30d_y", "wallet_alpha_30d"],
            ["wallet_signal_precision_tp_y", "wallet_signal_precision_tp"],
            ["wallet_avg_forward_return_15m_y", "wallet_avg_forward_return_15m"],
        ]
        for group in alias_groups:
            picked = self._pick_first_existing(df, group)
            if picked:
                base_features.append(picked)
        lag_features = [c for c in df.columns if "_lag_" in c]
        context_features = [
            c
            for c in [
                "recent_token_activity_5",
                "recent_yes_ratio_5",
                "btc_fee_pressure_score",
                "btc_mempool_congestion_score",
                "btc_network_activity_score",
                "btc_network_stress_score",
            ]
            if c in df.columns
        ]
        feature_cols = base_features + lag_features + context_features
        if not feature_cols:
            return pd.DataFrame()
        feature_cols = features_for_scope("nn", feature_cols)
        feature_cols, _ = drop_all_nan_features(df, feature_cols, context="stage2_temporal_models")
        if not feature_cols:
            return pd.DataFrame()

        log_audit(feature_cols)
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
                fold_feature_cols, _ = drop_all_nan_features(
                    balanced_train_df,
                    feature_cols,
                    context="stage2_temporal_models_fold",
                )
                if not fold_feature_cols:
                    continue
                X_train = self._numeric_feature_frame(balanced_train_df, fold_feature_cols)
                X_test = self._numeric_feature_frame(test_df, fold_feature_cols)
                y_train = balanced_train_df[target_cls].fillna(0).astype(int)
                class_counts = y_train.value_counts()
                use_early_stopping_cls = self._can_use_mlp_early_stopping(len(balanced_train_df), validation_fraction=0.15)
                if len(class_counts) < 2 or class_counts.min() < 2:
                    use_early_stopping_cls = False
                clf = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("model", MLPClassifier(
                        hidden_layer_sizes=(48, 24),
                        random_state=42,
                        max_iter=600,
                        learning_rate_init=1e-3,
                        alpha=1e-4,
                        early_stopping=use_early_stopping_cls,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        tol=1e-3,
                    )),
                ])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    clf.fit(X_train, y_train)
                y_test = test_df[target_cls].fillna(0).astype(int)
                preds = clf.predict(X_test)
                cls_scores.append(accuracy_score(y_test, preds))
                precision_scores.append(precision_score(y_test, preds, zero_division=0))
                recall_scores.append(recall_score(y_test, preds, zero_division=0))
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_test)[:, 1]
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
                joblib.dump(
                    {
                        "model": last_clf,
                        "features": feature_cols,
                        "model_kind": "stage2_temporal_classifier",
                        "feature_set": "temporal_sequence",
                        "scaling": "standard",
                        "regularization": "mlp_alpha",
                        "market_family": self.market_family,
                    },
                    self.classifier_file,
                )

        if target_reg:
            reg_scores = []
            last_reg = None
            return_calibration = fit_return_calibration(pd.to_numeric(df[target_reg], errors="coerce").fillna(0.0))
            for train_idx, test_idx in tscv.split(df):
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                if train_df.empty or test_df.empty:
                    continue
                fold_feature_cols, _ = drop_all_nan_features(
                    train_df,
                    feature_cols,
                    context="stage2_temporal_models_fold",
                )
                if not fold_feature_cols:
                    continue
                X_train = self._numeric_feature_frame(train_df, fold_feature_cols)
                X_test = self._numeric_feature_frame(test_df, fold_feature_cols)
                use_early_stopping_reg = self._can_use_mlp_early_stopping(len(train_df), validation_fraction=0.15)
                reg = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("model", MLPRegressor(
                        hidden_layer_sizes=(48, 24),
                        random_state=42,
                        max_iter=600,
                        learning_rate_init=1e-3,
                        alpha=1e-4,
                        early_stopping=use_early_stopping_reg,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        tol=1e-3,
                    )),
                ])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    train_target = pd.to_numeric(train_df[target_reg], errors="coerce").fillna(0.0)
                    reg.fit(X_train, transform_return_targets(train_target, return_calibration))
                preds = reg.predict(X_test)
                from return_calibration import calibrate_return_predictions
                calibrated_preds = calibrate_return_predictions(preds, return_calibration, index=test_df.index)
                reg_scores.append(mean_squared_error(pd.to_numeric(test_df[target_reg], errors="coerce").fillna(0.0), calibrated_preds) ** 0.5)
                last_reg = reg
            if reg_scores:
                metrics["temporal_walk_forward_rmse"] = float(sum(reg_scores) / len(reg_scores))
            if last_reg is not None:
                joblib.dump(
                    {
                        "model": last_reg,
                        "features": feature_cols,
                        "return_calibration": return_calibration,
                        "model_kind": "stage2_temporal_regressor",
                        "feature_set": "temporal_sequence",
                        "scaling": "standard",
                        "regularization": "mlp_alpha",
                        "market_family": self.market_family,
                    },
                    self.regressor_file,
                )

        result = pd.DataFrame([metrics]) if metrics else pd.DataFrame()
        if not result.empty:
            result.to_csv(self.metrics_file, index=False)
        return result
