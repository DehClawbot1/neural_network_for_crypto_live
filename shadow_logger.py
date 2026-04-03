import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from csv_utils import safe_csv_append


class ShadowLogger:
    def __init__(self, model_bundle_path=None, log_path="logs/shadow_results.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.bundle_path = self._resolve_bundle_path(model_bundle_path)
        self.bundle = joblib.load(self.bundle_path)
        self.model = self.bundle["model"]
        self.features = self.bundle["features"]

    @staticmethod
    def _resolve_bundle_path(model_bundle_path=None):
        if model_bundle_path:
            return Path(model_bundle_path)
        bundles = sorted(Path("weights").glob("meta_model_bundle_*.pkl"))
        if not bundles:
            raise FileNotFoundError("No meta_model_bundle_*.pkl found in weights/")
        return bundles[-1]

    def log_entry(self, signal_dict, features_df):
        if features_df is None or features_df.empty:
            return None
        working = features_df.copy()
        missing = [f for f in self.features if f not in working.columns]
        if missing:
            logging.warning("ShadowLogger: imputing missing model features for live scoring: %s", missing[:10])
            for feature_name in missing:
                if feature_name == "recent_yes_ratio_5":
                    working[feature_name] = 0.5
                else:
                    working[feature_name] = 0.0
        X = working[self.features].copy()
        meta_prob = float(self.model.predict_proba(X)[:, 1][0])
        shadow_row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market": signal_dict.get("market_title", signal_dict.get("market")),
            "market_slug": signal_dict.get("market_slug"),
            "condition_id": signal_dict.get("condition_id"),
            "token_id": signal_dict.get("token_id"),
            "wallet": signal_dict.get("trader_wallet", signal_dict.get("wallet_copied")),
            "side": signal_dict.get("outcome_side", signal_dict.get("side")),
            "entry_price": signal_dict.get("price", signal_dict.get("entry_price")),
            "signal_confidence": signal_dict.get("confidence", 0),
            "meta_prob": round(meta_prob, 6),
            "is_top_decile": int(meta_prob >= 0.85),
            "outcome": "PENDING",
            "realized_return": None,
        }
        safe_csv_append(self.log_path, pd.DataFrame([shadow_row]))
        logging.info("👻 Shadow Entry: %s | Meta-Prob: %.2f%%", shadow_row["market"], meta_prob * 100)
        return meta_prob

    def resolve_outcomes(self, clob_client=None):
        if not self.log_path.exists():
            return
        df = pd.read_csv(self.log_path, engine="python", on_bad_lines="skip")
        pending = df[df["outcome"] == "PENDING"].copy() if "outcome" in df.columns else pd.DataFrame()
        for idx, row in pending.iterrows():
            try:
                entry_time = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
            except Exception:
                continue
            if datetime.now(timezone.utc) - entry_time <= timedelta(minutes=60):
                continue
            outcome, ret = self._check_realized_outcome(row, clob_client)
            df.at[idx, "outcome"] = outcome
            df.at[idx, "realized_return"] = ret
        df.to_csv(self.log_path, index=False)

    def _check_realized_outcome(self, row, clob_client=None):
        return "EXPIRED", 0.0
