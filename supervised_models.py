from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class SupervisedModels:
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
        "whale_pressure",
        "market_structure_score",
    ]

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

    def train(self):
        df = self._safe_read()
        if df.empty:
            return None

        usable = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        if not usable:
            return None

        if "tp_before_sl_60m" in df.columns:
            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
            ])
            clf.fit(df[usable], df["tp_before_sl_60m"].fillna(0).astype(int))
            joblib.dump({"model": clf, "features": usable}, self.classifier_file)

        if "forward_return_15m" in df.columns:
            reg = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=250, random_state=42)),
            ])
            reg.fit(df[usable], df["forward_return_15m"].fillna(0.0))
            joblib.dump({"model": reg, "features": usable}, self.regressor_file)

        return usable

