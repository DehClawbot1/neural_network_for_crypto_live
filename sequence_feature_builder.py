import logging
from pathlib import Path

import pandas as pd
from brain_paths import filter_frame_for_brain, resolve_brain_context
from model_feature_catalog import SEQUENCE_BASE_COLUMNS


class SequenceFeatureBuilder:
    """
    Build lag/sequence-style features from token price history and recent signal flow.
    Stage 2 starts with explicit lag features before deeper sequence models.
    """

    def __init__(self, logs_dir="logs", *, brain_context=None, brain_id=None, market_family=None, shared_logs_dir="logs", shared_weights_dir="weights"):
        if brain_context is None and (brain_id or market_family):
            brain_context = resolve_brain_context(
                market_family,
                brain_id=brain_id,
                shared_logs_dir=shared_logs_dir,
                shared_weights_dir=shared_weights_dir,
            )
        self.brain_context = brain_context
        self.logs_dir = Path(brain_context.logs_dir if brain_context is not None else logs_dir)
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
        if self.brain_context is not None:
            df = filter_frame_for_brain(df, self.brain_context)
            if df.empty:
                return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values(["token_id", "timestamp"]).reset_index(drop=True)

        potential_cols = SEQUENCE_BASE_COLUMNS

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
        for col in base_cols:
            for lag in lags:
                lag_cols.append(f"{col}_lag_{lag}")
        for token_id, group in df.groupby("token_id"):
            group = group.copy().reset_index(drop=True)
            new_cols = {}
            for col in base_cols:
                for lag in lags:
                    new_cols[f"{col}_lag_{lag}"] = group[col].shift(lag)
            if "trader_wallet" in group.columns:
                new_cols["recent_token_activity_5"] = group["trader_wallet"].rolling(5).count()
            if "side" in group.columns:
                new_cols["recent_yes_ratio_5"] = (group["side"].astype(str).str.upper() == "YES").rolling(5).mean()
            group = pd.concat([group, pd.DataFrame(new_cols, index=group.index)], axis=1)
            parts.append(group)

        combined = pd.concat(parts, ignore_index=True)
        result = combined.dropna(subset=lag_cols).reset_index(drop=True)
        if self.brain_context is not None and not result.empty:
            result = filter_frame_for_brain(result, self.brain_context)
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
