from pathlib import Path

import pandas as pd
from schema import ALIASES
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline


class TimeSplitTrainer:
    """
    Train classifier/regressor with time-based splits instead of random splits.
    Research/paper-trading only.
    """

    FEATURE_COLUMNS = [
        "trader_win_rate",
        "normalized_trade_size",
        "current_price",
        "time_left",
        "liquidity_score",
        "volume_score",
        "probability_momentum",
        "volatility_score",
        "whale_pressure",
        "market_structure_score",
        "spread",
        "wallet_trade_count_30d",
        "wallet_alpha_30d",
        "wallet_avg_forward_return_15m",
    ]

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.output_file = self.logs_dir / "time_split_eval.csv"

    def _safe_read(self):
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def run(self):
        df = self._safe_read()
        if df.empty or "target_up" not in df.columns:
            return pd.DataFrame()

        usable = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        if not usable:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        n = len(df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        if train_df.empty or val_df.empty or test_df.empty:
            return pd.DataFrame()

        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ])
        clf.fit(train_df[usable], train_df["target_up"].astype(int))
        val_pred = clf.predict(val_df[usable])
        test_pred = clf.predict(test_df[usable])

        result = {
            "val_accuracy": accuracy_score(val_df["target_up"].astype(int), val_pred),
            "test_accuracy": accuracy_score(test_df["target_up"].astype(int), test_pred),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        }

        target_return_col = next((c for c in ALIASES["forward_return_15m"] if c in df.columns), None)
        if target_return_col is not None:
            reg = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
            ])
            reg.fit(train_df[usable], train_df[target_return_col])
            val_reg = reg.predict(val_df[usable])
            test_reg = reg.predict(test_df[usable])
            result["val_return_rmse"] = mean_squared_error(val_df[target_return_col], val_reg) ** 0.5
            result["test_return_rmse"] = mean_squared_error(test_df[target_return_col], test_reg) ** 0.5

        output = pd.DataFrame([result])
        output.to_csv(self.output_file, index=False)
        return output

