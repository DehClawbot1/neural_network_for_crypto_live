from pathlib import Path
import warnings

import pandas as pd
from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
from model_feature_safety import drop_all_nan_features
from feature_treatment_policy import features_for_scope, log_audit
from return_calibration import fit_return_calibration, transform_return_targets


def _load_sklearn_stage1():
    """Lazy-import sklearn and optional tree libraries."""
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    try:
        from hardware_config import get_sklearn_jobs, get_lightgbm_params
        n_jobs = get_sklearn_jobs()
        lgb_extra = get_lightgbm_params()
    except ImportError:
        n_jobs = -1
        lgb_extra = {}

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception:
        LGBMClassifier = None
        LGBMRegressor = None

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except Exception:
        CatBoostClassifier = None
        CatBoostRegressor = None

    return {
        "joblib": joblib,
        "CalibratedClassifierCV": CalibratedClassifierCV,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "SimpleImputer": SimpleImputer,
        "Pipeline": Pipeline,
        "n_jobs": n_jobs,
        "lgb_extra": lgb_extra,
        "LGBMClassifier": LGBMClassifier,
        "LGBMRegressor": LGBMRegressor,
        "CatBoostClassifier": CatBoostClassifier,
        "CatBoostRegressor": CatBoostRegressor,
    }


class Stage1Models:
    """
    Stronger tabular ensemble stage with calibrated classification outputs.
    Falls back to sklearn if optional libraries are unavailable.

    BUG FIX I: Now uses all available CPU cores via n_jobs parameter.
    """

    FEATURE_COLUMNS = DEFAULT_TABULAR_FEATURE_COLUMNS

    def __init__(self, logs_dir="logs", weights_dir="weights"):
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.classifier_file = self.weights_dir / "stage1_tp_classifier.joblib"
        self.regressor_file = self.weights_dir / "stage1_return_regressor.joblib"
        self.importance_file = self.logs_dir / "feature_importance.csv"

    def _safe_read(self):
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _usable_features(self, df):
        candidates = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        candidates = features_for_scope("tree", candidates)
        usable, _ = drop_all_nan_features(df, candidates, context="stage1_models")
        return usable

    def _build_classifier(self, cv=3):
        sk = _load_sklearn_stage1()
        Pipeline = sk["Pipeline"]
        SimpleImputer = sk["SimpleImputer"]
        if sk["LGBMClassifier"] is not None:
            lgb_params = {
                "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                "subsample": 0.9, "colsample_bytree": 0.9, "random_state": 42,
                "n_jobs": sk["n_jobs"], "verbose": -1,
            }
            for k, v in sk["lgb_extra"].items():
                if k != "n_jobs":
                    lgb_params[k] = v
            base = sk["LGBMClassifier"](**lgb_params)
            model = sk["CalibratedClassifierCV"](base, method="sigmoid", cv=cv) if cv else base
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        if sk["CatBoostClassifier"] is not None:
            base = sk["CatBoostClassifier"](iterations=300, learning_rate=0.05, depth=6, verbose=False, thread_count=sk["n_jobs"])
            model = sk["CalibratedClassifierCV"](base, method="sigmoid", cv=cv) if cv else base
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", sk["HistGradientBoostingClassifier"](max_depth=6, learning_rate=0.05, random_state=42)),
        ])

    def _build_regressor(self):
        sk = _load_sklearn_stage1()
        Pipeline = sk["Pipeline"]
        SimpleImputer = sk["SimpleImputer"]
        if sk["LGBMRegressor"] is not None:
            lgb_params = {
                "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                "random_state": 42, "n_jobs": sk["n_jobs"], "verbose": -1,
            }
            for k, v in sk["lgb_extra"].items():
                if k != "n_jobs":
                    lgb_params[k] = v
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", sk["LGBMRegressor"](**lgb_params))])
        if sk["CatBoostRegressor"] is not None:
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", sk["CatBoostRegressor"](iterations=300, learning_rate=0.05, depth=6, verbose=False, thread_count=sk["n_jobs"])),
            ])
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", sk["HistGradientBoostingRegressor"](max_depth=6, learning_rate=0.05, random_state=42)),
        ])

    def _write_feature_importance(self, feature_names, fitted_model):
        model = fitted_model.named_steps.get("model")
        if hasattr(model, "feature_importances_"):
            values = list(model.feature_importances_)
        elif hasattr(model, "base_estimator") and hasattr(model.base_estimator, "feature_importances_"):
            values = list(model.base_estimator.feature_importances_)
        else:
            values = [0.0] * len(feature_names)
        pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False).to_csv(self.importance_file, index=False)

    def _safe_cv_folds(self, y) -> int | None:
        """Return the max CV folds we can safely use (2 or 3), or None to skip calibration."""
        counts = y.value_counts()
        if counts.empty or int(counts.min()) < 2:
            return None
        return min(3, int(counts.min()))

    def train(self):
        log_audit()

        df = self._safe_read()
        if df.empty:
            return None

        usable = self._usable_features(df)
        if not usable:
            return None

        X = df[usable]
        sk = _load_sklearn_stage1()
        joblib = sk["joblib"]

        if "tp_before_sl_60m" in df.columns:
            y_cls = df["tp_before_sl_60m"].fillna(0).astype(int)
            cv_folds = self._safe_cv_folds(y_cls)
            clf = self._build_classifier(cv=cv_folds)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                    category=UserWarning,
                )
                clf.fit(X, y_cls)
            joblib.dump({"model": clf, "features": usable}, self.classifier_file)
            self._write_feature_importance(usable, clf)

        if "forward_return_15m" in df.columns:
            target_returns = pd.to_numeric(df["forward_return_15m"], errors="coerce").fillna(0.0)
            return_calibration = fit_return_calibration(target_returns)
            reg = self._build_regressor()
            reg.fit(X, transform_return_targets(target_returns, return_calibration))
            joblib.dump({"model": reg, "features": usable, "return_calibration": return_calibration}, self.regressor_file)

        return usable
