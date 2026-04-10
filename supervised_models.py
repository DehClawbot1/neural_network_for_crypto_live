import logging
import os
from pathlib import Path

import pandas as pd
from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
from model_feature_safety import drop_all_nan_features
from return_calibration import fit_return_calibration, transform_return_targets

logger = logging.getLogger(__name__)

_MIN_NONZERO_FEATURES = int(os.getenv("MIN_NONZERO_FEATURES", "3") or 3)


def _load_sklearn_supervised():
    """Lazy-import sklearn to avoid import-time failures."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "joblib": joblib,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "SGDClassifier": SGDClassifier,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


try:
    from hardware_config import get_sklearn_jobs
    _N_JOBS = get_sklearn_jobs()
except ImportError:
    _N_JOBS = -1


class SupervisedModels:
    """
    Supervised TP/SL classifier + return regressor with sparse regularization.

    Guardrails:
    - Rejects degenerate all-zero L1/Lasso models.
    - Elastic Net beside pure L1 to avoid over-pruning correlated features.
    """

    FEATURE_COLUMNS = DEFAULT_TABULAR_FEATURE_COLUMNS

    def __init__(self, logs_dir="logs", weights_dir="weights"):
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.classifier_file = self.weights_dir / "tp_classifier.joblib"
        self.regressor_file = self.weights_dir / "return_regressor.joblib"

    def _safe_read(self):
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    # ── classifiers ─────────────────────────────────────────────────

    def _train_sparse_classifier(self, X, y):
        sklearn = _load_sklearn_supervised()
        regularization_c = float(os.getenv("TP_CLASSIFIER_L1_C", os.getenv("SUPERVISED_L1_C", "0.35")) or 0.35)
        model = sklearn["Pipeline"]([
            ("imputer", sklearn["SimpleImputer"](strategy="median")),
            ("scaler", sklearn["StandardScaler"]()),
            ("model", sklearn["LogisticRegression"](
                penalty="l1", solver="liblinear", class_weight="balanced",
                random_state=42, max_iter=2000, C=max(1e-4, regularization_c),
            )),
        ])
        model.fit(X, y)
        coef = getattr(model.named_steps["model"], "coef_", None)
        nonzero = int((abs(coef) > 1e-12).sum()) if coef is not None else 0

        if nonzero < _MIN_NONZERO_FEATURES:
            logger.warning("L1 classifier degenerate (%d nonzero). Trying Elastic Net.", nonzero)
            return self._train_elastic_net_classifier(X, y)

        return model, {
            "model_kind": "logistic_l1",
            "regularization": "l1",
            "nonzero_feature_count": nonzero,
            "fallback_used": False,
        }

    def _train_elastic_net_classifier(self, X, y):
        sklearn = _load_sklearn_supervised()
        l1_ratio = float(os.getenv("TP_CLASSIFIER_ELASTIC_L1_RATIO", "0.7") or 0.7)
        alpha = float(os.getenv("TP_CLASSIFIER_ELASTIC_ALPHA", "0.001") or 0.001)
        model = sklearn["Pipeline"]([
            ("imputer", sklearn["SimpleImputer"](strategy="median")),
            ("scaler", sklearn["StandardScaler"]()),
            ("model", sklearn["SGDClassifier"](
                loss="log_loss", penalty="elasticnet",
                l1_ratio=max(0.0, min(1.0, l1_ratio)),
                alpha=max(1e-6, alpha),
                class_weight="balanced", random_state=42, max_iter=2000,
            )),
        ])
        model.fit(X, y)
        coef = getattr(model.named_steps["model"], "coef_", None)
        nonzero = int((abs(coef) > 1e-12).sum()) if coef is not None else 0

        if nonzero < _MIN_NONZERO_FEATURES:
            logger.warning("Elastic Net classifier also degenerate (%d nonzero). Falling back to RF.", nonzero)
            return self._train_classifier_fallback(X, y)

        return model, {
            "model_kind": "logistic_elastic_net",
            "regularization": f"elasticnet(l1_ratio={l1_ratio})",
            "nonzero_feature_count": nonzero,
            "fallback_used": False,
        }

    def _train_classifier_fallback(self, X, y):
        sklearn = _load_sklearn_supervised()
        model = sklearn["Pipeline"]([
            ("imputer", sklearn["SimpleImputer"](strategy="median")),
            ("model", sklearn["RandomForestClassifier"](
                n_estimators=250, random_state=42, class_weight="balanced", n_jobs=_N_JOBS,
            )),
        ])
        model.fit(X, y)
        return model, {
            "model_kind": "random_forest_fallback",
            "regularization": "none",
            "nonzero_feature_count": None,
            "fallback_used": True,
        }

    # ── regressors ──────────────────────────────────────────────────

    def _train_lasso_regressor(self, X, y):
        sklearn = _load_sklearn_supervised()
        alpha = float(os.getenv("RETURN_REGRESSOR_LASSO_ALPHA", "0.0015") or 0.0015)
        model = sklearn["Pipeline"]([
            ("imputer", sklearn["SimpleImputer"](strategy="median")),
            ("scaler", sklearn["StandardScaler"]()),
            ("model", sklearn["Lasso"](alpha=max(1e-6, alpha), max_iter=5000, random_state=42)),
        ])
        model.fit(X, y)
        coef = getattr(model.named_steps["model"], "coef_", None)
        nonzero = int((abs(coef) > 1e-12).sum()) if coef is not None else 0

        if nonzero < _MIN_NONZERO_FEATURES:
            logger.warning("Lasso degenerate (%d nonzero). Trying Elastic Net regressor.", nonzero)
            return self._train_elastic_net_regressor(X, y)

        return model, {
            "model_kind": "lasso",
            "regularization": "l1",
            "nonzero_feature_count": nonzero,
            "fallback_used": False,
        }

    def _train_elastic_net_regressor(self, X, y):
        sklearn = _load_sklearn_supervised()
        alpha = float(os.getenv("RETURN_REGRESSOR_ELASTIC_ALPHA", "0.001") or 0.001)
        l1_ratio = float(os.getenv("RETURN_REGRESSOR_ELASTIC_L1_RATIO", "0.7") or 0.7)
        model = sklearn["Pipeline"]([
            ("imputer", sklearn["SimpleImputer"](strategy="median")),
            ("scaler", sklearn["StandardScaler"]()),
            ("model", sklearn["ElasticNet"](
                alpha=max(1e-6, alpha),
                l1_ratio=max(0.0, min(1.0, l1_ratio)),
                max_iter=5000, random_state=42,
            )),
        ])
        model.fit(X, y)
        coef = getattr(model.named_steps["model"], "coef_", None)
        nonzero = int((abs(coef) > 1e-12).sum()) if coef is not None else 0

        if nonzero < _MIN_NONZERO_FEATURES:
            logger.warning("Elastic Net regressor degenerate (%d nonzero). Falling back to RF.", nonzero)
            return self._train_regressor_fallback(X, y)

        return model, {
            "model_kind": "elastic_net",
            "regularization": f"elasticnet(l1_ratio={l1_ratio})",
            "nonzero_feature_count": nonzero,
            "fallback_used": False,
        }

    def _train_regressor_fallback(self, X, y):
        sklearn = _load_sklearn_supervised()
        model = sklearn["Pipeline"]([
            ("imputer", sklearn["SimpleImputer"](strategy="median")),
            ("model", sklearn["RandomForestRegressor"](n_estimators=250, random_state=42, n_jobs=_N_JOBS)),
        ])
        model.fit(X, y)
        return model, {
            "model_kind": "random_forest_fallback",
            "regularization": "none",
            "nonzero_feature_count": None,
            "fallback_used": True,
        }

    # ── main train ──────────────────────────────────────────────────

    def train(self):
        df = self._safe_read()
        if df.empty:
            return None

        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp", kind="stable")

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx] if split_idx > 0 else df
        if train_df.empty:
            return None

        candidates = [c for c in self.FEATURE_COLUMNS if c in train_df.columns]
        usable, _ = drop_all_nan_features(train_df, candidates, context="supervised_models")
        if not usable:
            return None

        X = train_df[usable].apply(pd.to_numeric, errors="coerce")
        usable = [col for col in usable if X[col].notna().any()]
        if not usable:
            return None
        X = X[usable]

        sklearn = _load_sklearn_supervised()
        joblib = sklearn["joblib"]

        if "tp_before_sl_60m" in train_df.columns:
            y_clf = train_df["tp_before_sl_60m"].fillna(0).astype(int)
            if y_clf.nunique(dropna=True) >= 2:
                try:
                    clf, clf_meta = self._train_sparse_classifier(X, y_clf)
                except Exception:
                    clf, clf_meta = self._train_classifier_fallback(X, y_clf)
                joblib.dump({"model": clf, "features": usable, **clf_meta}, self.classifier_file)

        if "forward_return_15m" in train_df.columns:
            target_returns = pd.to_numeric(train_df["forward_return_15m"], errors="coerce").fillna(0.0)
            return_calibration = fit_return_calibration(target_returns)
            transformed_returns = transform_return_targets(target_returns, return_calibration)
            try:
                reg, reg_meta = self._train_lasso_regressor(X, transformed_returns)
            except Exception:
                reg, reg_meta = self._train_regressor_fallback(X, transformed_returns)
            joblib.dump(
                {"model": reg, "features": usable, "return_calibration": return_calibration, **reg_meta},
                self.regressor_file,
            )

        return usable
