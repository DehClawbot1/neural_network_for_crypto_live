import logging
import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MetaModelTrainer:
    @staticmethod
    def _get_clean_features(df):
        allowed_patterns = [
            r'^entry_price(_lag_\d+)?$',
            r'^spread(_lag_\d+)?$',
            r'^wallet_.*_30d$',
            r'^wallet_signal_precision.*$',
            r'^btc_spot_return_.*$',
            r'^recent_.*_5$',
            r'^normalized_trade_size$',
        ]
        features = [c for c in df.columns if any(re.match(p, c) for p in allowed_patterns)]
        forbidden = ['horizon', 'mfe', 'mae', 'pnl', 'exit', 'hit', 'target', 'within']
        features = [f for f in features if not any(word in f.lower() for word in forbidden)]
        return features

    @staticmethod
    def _sanitize_sequence_data(df):
        targets_to_kill = [c for c in df.columns if 'tp_hit' in c or 'forward_return' in c or 'sl_hit' in c]
        if 'tp_before_sl_60m' in targets_to_kill:
            targets_to_kill.remove('tp_before_sl_60m')
        df = df.drop(columns=[c for c in targets_to_kill if c in df.columns], errors='ignore')
        cols_to_drop = [c for c in df.columns if c.endswith('_y')]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        df.columns = [c[:-2] if c.endswith('_x') else c for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def __init__(self, data_path="logs/sequence_dataset.csv", model_dir="weights"):
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.target = "tp_before_sl_60m"
        self.ignored_cols = [
            "timestamp", "token_id", "condition_id", "market_slug", "trade_id", "market_url", "anchor_timestamp",
            "trader_wallet", "trader_wallet_x", "trader_wallet_y",
            "market_title", "market_title_x", "market_title_y",
            "outcome_side", "outcome_side_x", "outcome_side_y",
            "order_side", "trade_side", "entry_intent", "side",
        ]

    def _prepare_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset missing at {self.data_path}")

        df = pd.read_csv(self.data_path, engine="python", on_bad_lines="skip")
        df = self._sanitize_sequence_data(df)
        if df.empty or self.target not in df.columns:
            raise ValueError("Sequence dataset is empty or missing target column.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        y = df[self.target].fillna(0).astype(int)
        X = df.drop(columns=[self.target] + [c for c in self.ignored_cols if c in df.columns], errors="ignore")
        X = X.select_dtypes(include=["number", "bool"]).copy()
        clean_features = self._get_clean_features(X)
        if not clean_features:
            raise ValueError("No clean past-only features available for meta-model training.")
        X = X[clean_features].copy()

        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        if X_train.empty or X_test.empty:
            raise ValueError("Not enough sequence rows for chronological train/test split.")

        return X_train, X_test, y_train, y_test, list(X.columns)

    def train_meta_model(self):
        logging.info("Starting Gold Standard meta-model training...")
        X_train, X_test, y_train, y_test, feature_names = self._prepare_data()

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        test_results = pd.DataFrame({"prob": probs, "actual": y_test.values})
        top_k = max(1, int(len(test_results) * 0.1))
        top_10_percent = test_results.sort_values("prob", ascending=False).head(top_k)
        precision_at_k = float(top_10_percent["actual"].mean())
        roc_auc = float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else float("nan")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        bundle = {
            "model": model,
            "features": feature_names,
            "target": self.target,
            "precision_top_10": precision_at_k,
            "roc_auc": roc_auc,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "created_at": timestamp,
        }
        joblib.dump(bundle, self.model_dir / f"meta_model_bundle_{timestamp}.pkl")
        self._update_registry(timestamp, precision_at_k, roc_auc, feature_names, len(X_train), len(X_test))

        logging.info("Model Training Complete.")
        logging.info("ROC-AUC: %.4f", roc_auc)
        logging.info("Precision @ Top 10%% (Execution Zone): %.4f", precision_at_k)
        return model, precision_at_k

    def _update_registry(self, ts, precision, roc_auc, features, train_rows, test_rows):
        registry_path = self.model_dir / "model_registry.csv"
        new_row = pd.DataFrame([{
            "timestamp": ts,
            "precision_top_10": precision,
            "roc_auc": roc_auc,
            "feature_count": len(features),
            "model_type": "RandomForestClassifier",
            "target": self.target,
            "train_rows": train_rows,
            "test_rows": test_rows,
        }])
        new_row.to_csv(registry_path, mode="a", header=not registry_path.exists(), index=False)


if __name__ == "__main__":
    trainer = MetaModelTrainer()
    trainer.train_meta_model()
