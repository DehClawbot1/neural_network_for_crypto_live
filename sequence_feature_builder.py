import logging
import os
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
        self.historical_dataset_file = self.logs_dir / "historical_dataset.csv"
        self.output_file = self.logs_dir / "sequence_dataset.csv"

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def build(self, lags=(1, 2, 3, 5, 10)):
        target_df = self._safe_read(self.contract_targets_file)
        history_df = self._safe_read(self.historical_dataset_file)

        if self.brain_context is not None:
            if not history_df.empty:
                history_df = filter_frame_for_brain(history_df, self.brain_context)
            if not target_df.empty:
                target_df = filter_frame_for_brain(target_df, self.brain_context)

        df = history_df if not history_df.empty else target_df
        if df.empty or "token_id" not in df.columns:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values(["token_id", "timestamp"]).reset_index(drop=True)

        potential_cols = SEQUENCE_BASE_COLUMNS

        base_cols = []
        skipped_all_null = []
        for c in potential_cols:
            if c in df.columns:
                if df[c].notnull().any():
                    base_cols.append(c)
                else:
                    skipped_all_null.append(c)

        if skipped_all_null:
            preview = ", ".join(skipped_all_null[:8])
            suffix = "" if len(skipped_all_null) <= 8 else f" ... (+{len(skipped_all_null) - 8} more)"
            logging.warning(
                "SequenceFeatureBuilder: Skipping %d all-null base columns: %s%s",
                len(skipped_all_null),
                preview,
                suffix,
            )

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
        if not target_df.empty:
            target_df = target_df.copy()
            if "timestamp" in target_df.columns:
                target_df["timestamp"] = pd.to_datetime(target_df["timestamp"], utc=True, errors="coerce")
            merge_candidates = [
                "timestamp",
                "token_id",
                "condition_id",
                "outcome_side",
                "trader_wallet",
                "market_title",
            ]
            merge_keys = [column for column in merge_candidates if column in combined.columns and column in target_df.columns]
            target_label_columns = [
                column
                for column in ["forward_return_15m", "tp_before_sl_60m", "target_up", "mfe_60m", "mae_60m", "entry_price", "anchor_timestamp"]
                if column in target_df.columns
            ]
            if merge_keys and target_label_columns:
                target_view = target_df[merge_keys + target_label_columns].drop_duplicates(subset=merge_keys, keep="last")
                combined = combined.merge(target_view, on=merge_keys, how="left", suffixes=("", "_target"))
                for column in target_label_columns:
                    target_col = f"{column}_target"
                    if target_col not in combined.columns:
                        continue
                    if column in combined.columns:
                        combined[column] = combined[column].fillna(combined[target_col])
                        combined = combined.drop(columns=[target_col])
                    else:
                        combined = combined.rename(columns={target_col: column})
        default_min_lag_feature_count = max(1, min(len(lag_cols), max(3, len(lags) * 2))) if lag_cols else 0
        requested_min_lag_feature_count = int(
            os.getenv("SEQUENCE_MIN_LAG_FEATURE_COUNT", str(default_min_lag_feature_count)) or default_min_lag_feature_count
        )
        min_lag_feature_count = max(0, min(len(lag_cols), requested_min_lag_feature_count))
        lag_feature_count = combined[lag_cols].notna().sum(axis=1) if lag_cols else pd.Series(0, index=combined.index)
        result = combined[lag_feature_count >= min_lag_feature_count].reset_index(drop=True)
        label_cols = [column for column in ["forward_return_15m", "tp_before_sl_60m", "target_up"] if column in result.columns]
        if label_cols:
            labeled_mask = result[label_cols].notna().any(axis=1)
            result = result[labeled_mask].reset_index(drop=True)
        if self.brain_context is not None and not result.empty:
            result = filter_frame_for_brain(result, self.brain_context)
        logging.info(
            "SequenceFeatureBuilder: Generated %s sequence rows using %s base features (min_lag_feature_count=%s).",
            len(result),
            len(base_cols),
            min_lag_feature_count,
        )
        return result

    def write(self):
        df = self.build()
        # Always rewrite the output file so stale sequence datasets do not
        # survive cycles where no valid sequence rows were produced.
        if df is None:
            df = pd.DataFrame()
        df.to_csv(self.output_file, index=False)
        return df
