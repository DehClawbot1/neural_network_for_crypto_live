import logging
import os
from pathlib import Path

import pandas as pd
from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
from model_feature_safety import drop_all_nan_features

logger = logging.getLogger(__name__)

_MIN_NONZERO_FEATURES = int(os.getenv("MIN_NONZERO_FEATURES", "3") or 3)


def _dedupe_feature_names(feature_names):
    seen = set()
    out = []
    for name in feature_names or []:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _load_sklearn_supervised():
    """Lazy-import sklearn to avoid import-time failures."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "joblib": joblib,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "SGDClassifier": SGDClassifier,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


class SupervisedTrainer:
    """
    Train a supervised BTC-direction model from aligned historical data.

    Guardrails:
    - Rejects degenerate all-zero L1 models (falls back to Elastic Net).
    - Elastic Net (l1_ratio=0.7) runs beside pure L1 when L1 is too sparse.
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

    def _train_sparse_classifier(self, X, y):
        sklearn = _load_sklearn_supervised()
        Pipeline = sklearn["Pipeline"]
        SimpleImputer = sklearn["SimpleImputer"]
        StandardScaler = sklearn["StandardScaler"]
        LogisticRegression = sklearn["LogisticRegression"]

        regularization_c = float(os.getenv("SUPERVISED_L1_C", "0.35") or 0.35)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1", solver="liblinear", class_weight="balanced",
                random_state=42, max_iter=2000, C=max(1e-4, regularization_c),
            )),
        ])
        model.fit(X, y)
        coef = getattr(model.named_steps["clf"], "coef_", None)
        nonzero = int((abs(coef) > 1e-12).sum()) if coef is not None else 0

        # guardrail: reject degenerate all-zero model
        if nonzero < _MIN_NONZERO_FEATURES:
            logger.warning(
                "L1 model degenerate (%d nonzero features < %d minimum). "
                "Falling back to Elastic Net.",
                nonzero, _MIN_NONZERO_FEATURES,
            )
            return self._train_elastic_net_classifier(X, y)

        return model, {
            "model_kind": "logistic_l1",
            "regularization": "l1",
            "nonzero_feature_count": nonzero,
            "fallback_used": False,
        }

    def _train_elastic_net_classifier(self, X, y):
        sklearn = _load_sklearn_supervised()
        Pipeline = sklearn["Pipeline"]
        SimpleImputer = sklearn["SimpleImputer"]
        StandardScaler = sklearn["StandardScaler"]
        SGDClassifier = sklearn["SGDClassifier"]

        l1_ratio = float(os.getenv("SUPERVISED_ELASTIC_L1_RATIO", "0.7") or 0.7)
        alpha = float(os.getenv("SUPERVISED_ELASTIC_ALPHA", "0.001") or 0.001)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(
                loss="log_loss", penalty="elasticnet",
                l1_ratio=max(0.0, min(1.0, l1_ratio)),
                alpha=max(1e-6, alpha),
                class_weight="balanced", random_state=42, max_iter=2000,
            )),
        ])
        model.fit(X, y)
        coef = getattr(model.named_steps["clf"], "coef_", None)
        nonzero = int((abs(coef) > 1e-12).sum()) if coef is not None else 0

        if nonzero < _MIN_NONZERO_FEATURES:
            logger.warning(
                "Elastic Net also degenerate (%d nonzero). Falling back to RF.",
                nonzero,
            )
            return self._train_fallback_classifier(X, y)

        return model, {
            "model_kind": "logistic_elastic_net",
            "regularization": f"elasticnet(l1_ratio={l1_ratio})",
            "nonzero_feature_count": nonzero,
            "fallback_used": False,
        }

    def _train_fallback_classifier(self, X, y):
        sklearn = _load_sklearn_supervised()
        Pipeline = sklearn["Pipeline"]
        SimpleImputer = sklearn["SimpleImputer"]
        RandomForestClassifier = sklearn["RandomForestClassifier"]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ])
        model.fit(X, y)
        return model, {
            "model_kind": "random_forest_fallback",
            "regularization": "none",
            "nonzero_feature_count": None,
            "fallback_used": True,
        }

    def train(self):
        df = self._safe_read()
        if df.empty:
            return None, None

        candidates = _dedupe_feature_names([col for col in self.FEATURE_COLUMNS if col in df.columns])
        usable_features, _ = drop_all_nan_features(df, candidates, context="supervised_trainer")
        if not usable_features or "target_up" not in df.columns:
            return None, None

        usable_features = _dedupe_feature_names(usable_features)
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

        sklearn = _load_sklearn_supervised()
        sklearn["joblib"].dump({"model": model, "features": usable_features, **metadata}, self.model_file)
        return model, usable_features
