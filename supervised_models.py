from pathlib import Path

import joblib
import pandas as pd
from model_feature_safety import drop_all_nan_features
from return_calibration import fit_return_calibration, transform_return_targets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── BUG FIX I: Use hardware config for parallelism ──
try:
    from hardware_config import get_sklearn_jobs
    _N_JOBS = get_sklearn_jobs()
except ImportError:
    _N_JOBS = -1


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

        if "tp_before_sl_60m" in train_df.columns:
            # ── BUG FIX I: n_jobs uses all available cores ──
            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=_N_JOBS)),
            ])
            clf.fit(train_df[usable], train_df["tp_before_sl_60m"].fillna(0).astype(int))
            joblib.dump({"model": clf, "features": usable}, self.classifier_file)

        if "forward_return_15m" in train_df.columns:
            target_returns = pd.to_numeric(train_df["forward_return_15m"], errors="coerce").fillna(0.0)
            return_calibration = fit_return_calibration(target_returns)
            reg = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=_N_JOBS)),
            ])
            reg.fit(train_df[usable], transform_return_targets(target_returns, return_calibration))
            joblib.dump({"model": reg, "features": usable, "return_calibration": return_calibration}, self.regressor_file)

        return usable
