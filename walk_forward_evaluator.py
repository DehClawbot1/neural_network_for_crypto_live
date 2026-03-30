from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


class WalkForwardEvaluator:
    """
    Simple walk-forward evaluation to reduce time leakage.
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
    ]

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.output_file = self.logs_dir / "walk_forward_eval.csv"

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def evaluate(self, train_ratio=0.7):
        df = self._safe_read(self.dataset_file)
        if df.empty or "target_up" not in df.columns:
            return pd.DataFrame()

        usable = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        if not usable:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        if train_df.empty or test_df.empty:
            return pd.DataFrame()

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]
        )
        model.fit(train_df[usable], train_df["target_up"].astype(int))
        preds = model.predict(test_df[usable])
        acc = accuracy_score(test_df["target_up"].astype(int), preds)

        result = pd.DataFrame([
            {
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "accuracy": acc,
                "positive_rate": float(np.mean(preds)),
            }
        ])
        result.to_csv(self.output_file, index=False)
        return result

