from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None


class Stage1Models:
    """
    Stronger tabular ensemble stage with calibrated classification outputs.
    Falls back to sklearn if optional libraries are unavailable.
    """

    FEATURE_COLUMNS = [
        "current_price",
        "spread",
        "liquidity_score",
        "volume_score",
        "probability_momentum",
        "volatility_score",
        "wallet_trade_count_30d",
        "wallet_alpha_30d",
        "wallet_avg_forward_return_15m",
        "wallet_signal_precision_tp",
        "wallet_recent_streak",
        "whale_pressure",
        "market_structure_score",
        "btc_spot_return_5m",
        "btc_spot_return_15m",
        "btc_realized_vol_15m",
        "btc_volume_proxy",
    ]

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
        return [c for c in self.FEATURE_COLUMNS if c in df.columns]

    def _build_classifier(self):
        if LGBMClassifier is not None:
            base = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CalibratedClassifierCV(base, method="sigmoid", cv=3)),
            ])
        if CatBoostClassifier is not None:
            base = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=False)
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CalibratedClassifierCV(base, method="sigmoid", cv=3)),
            ])
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, random_state=42)),
        ])

    def _build_regressor(self):
        if LGBMRegressor is not None:
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)),
            ])
        if CatBoostRegressor is not None:
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=False)),
            ])
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)),
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

    def train(self):
        df = self._safe_read()
        if df.empty:
            return None

        usable = self._usable_features(df)
        if not usable:
            return None

        X = df[usable]

        if "tp_before_sl_60m" in df.columns:
            clf = self._build_classifier()
            clf.fit(X, df["tp_before_sl_60m"].fillna(0).astype(int))
            joblib.dump({"model": clf, "features": usable}, self.classifier_file)
            self._write_feature_importance(usable, clf)

        if "forward_return_15m" in df.columns:
            reg = self._build_regressor()
            reg.fit(X, df["forward_return_15m"].fillna(0.0))
            joblib.dump({"model": reg, "features": usable}, self.regressor_file)

        return usable
