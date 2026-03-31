import logging
from pathlib import Path

import pandas as pd


class SequenceFeatureBuilder:
    """
    Build lag/sequence-style features from token price history and recent signal flow.
    Stage 2 starts with explicit lag features before deeper sequence models.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.contract_targets_file = self.logs_dir / "contract_targets.csv"
        self.output_file = self.logs_dir / "sequence_dataset.csv"

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def build(self, lags=(1, 2, 3, 5, 10)):
        df = self._safe_read(self.contract_targets_file)
        if df.empty or "token_id" not in df.columns:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values(["token_id", "timestamp"]).reset_index(drop=True)

        potential_cols = [
            "entry_price",
            "wallet_trade_count_30d",
            "wallet_alpha_30d",
            "wallet_signal_precision_tp",
            "btc_spot_return_5m",
            "btc_spot_return_15m",
            "spread",
            "current_price",
            "normalized_trade_size",
        ]

        base_cols = []
        for c in potential_cols:
            if c in df.columns:
                if df[c].notnull().any():
                    base_cols.append(c)
                else:
                    logging.warning("SequenceFeatureBuilder: Skipping '%s' - source column is all-null.", c)

        if not base_cols:
            logging.error("SequenceFeatureBuilder: No valid base columns with data found. Dataset aborted.")
            return pd.DataFrame()

        parts = []
        lag_cols = []
        for token_id, group in df.groupby("token_id"):
            group = group.copy().reset_index(drop=True)
            for col in base_cols:
                for lag in lags:
                    l_name = f"{col}_lag_{lag}"
                    group[l_name] = group[col].shift(lag)
                    if l_name not in lag_cols:
                        lag_cols.append(l_name)
            if "trader_wallet" in group.columns:
                group["recent_token_activity_5"] = group["trader_wallet"].rolling(5).count()
            if "side" in group.columns:
                group["recent_yes_ratio_5"] = (group["side"].astype(str).str.upper() == "YES").rolling(5).mean()
            parts.append(group)

        combined = pd.concat(parts, ignore_index=True)
        result = combined.dropna(subset=lag_cols).reset_index(drop=True)
        logging.info("SequenceFeatureBuilder: Generated %s sequence rows using %s base features.", len(result), len(base_cols))
        return result

    def write(self):
        df = self.build()
        # Always rewrite the output file so stale sequence datasets do not
        # survive cycles where no valid sequence rows were produced.
        if df is None:
            df = pd.DataFrame()
        df.to_csv(self.output_file, index=False)
        return df
