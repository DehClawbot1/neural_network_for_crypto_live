from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class SupervisedTrainer:
    """
    Train a supervised BTC-direction model from aligned historical data.
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

    def train(self):
        df = self._safe_read()
        if df.empty:
            return None, None

        usable_features = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        if not usable_features or "target_up" not in df.columns:
            return None, None

        X = df[usable_features]
        y = df["target_up"].astype(int)

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]
        )
        model.fit(X, y)
        joblib.dump({"model": model, "features": usable_features}, self.model_file)
        return model, usable_features

