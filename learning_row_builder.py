"""Canonical learning-row builder.

Produces one learning row per executed trade, keyed by
``(candidate_id, cycle_id, token_id, timestamp)``.

Sources are joined in priority order so every field has a single
authoritative origin:

1. **candidate_decisions** (supervisor)  – entry context, gate decisions
2. **positions / closed_positions** (trade_manager)  – trade outcome, P&L
3. **entry_signal_snapshot_json** (trade_lifecycle)  – full feature snapshot
4. **contract_targets** (contract_target_builder)  – TP/SL labels, MFE/MAE
5. **historical_dataset** (historical_dataset_builder)  – enriched features
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LEARNING_KEY_COLS = ["candidate_id", "cycle_id", "token_id", "timestamp"]


# ── helpers ─────────────────────────────────────────────────────────

def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _parse_ts(series: pd.Series) -> pd.Series:
    local_tz = os.getenv("BOT_LOG_LOCAL_TIMEZONE", "Europe/Lisbon")
    # Use utc=True to handle mixed-timezone strings without raising
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except Exception:
        pass
    # Fallback: localize naive timestamps then convert
    raw = pd.to_datetime(series, errors="coerce", format="mixed")
    if not pd.api.types.is_datetime64_any_dtype(raw):
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    try:
        tz = getattr(raw.dt, "tz", None)
    except AttributeError:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    if tz is None:
        try:
            return raw.dt.tz_localize(
                local_tz, nonexistent="shift_forward", ambiguous="NaT"
            ).dt.tz_convert("UTC")
        except Exception:
            return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    return raw.dt.tz_convert("UTC")


def _parse_snapshot_json(value) -> dict:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _explode_snapshot_column(df: pd.DataFrame, json_col: str) -> pd.DataFrame:
    """Expand a JSON column into individual feature columns."""
    if json_col not in df.columns:
        return df
    records = df[json_col].apply(_parse_snapshot_json)
    flat = pd.json_normalize(records)
    if flat.empty:
        return df
    # avoid overwriting existing columns
    new_cols = [c for c in flat.columns if c not in df.columns]
    if not new_cols:
        return df
    return pd.concat([df.reset_index(drop=True), flat[new_cols].reset_index(drop=True)], axis=1)


def _coalesce(df: pd.DataFrame, *col_candidates: str) -> pd.Series:
    """Return the first non-null column among *col_candidates*."""
    result = pd.Series(pd.NA, index=df.index)
    for col in col_candidates:
        if col in df.columns:
            result = result.fillna(df[col])
    return result


def _series_or_default(df: pd.DataFrame, column: str, default=0.0) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        num = float(value)
        return float(default) if not np.isfinite(num) else num
    except Exception:
        return float(default)


def _truthy(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_merge_asof(left: pd.DataFrame, right: pd.DataFrame, on: str, by: Sequence[str] | None = None, **kwargs) -> pd.DataFrame:
    if left.empty or right.empty or on not in left.columns or on not in right.columns:
        return left
    left = left.copy()
    right = right.copy()
    left[on] = _parse_ts(left[on]) if not pd.api.types.is_datetime64_any_dtype(left[on]) else left[on]
    right[on] = _parse_ts(right[on]) if not pd.api.types.is_datetime64_any_dtype(right[on]) else right[on]
    left_mask = left[on].notna()
    right_mask = right[on].notna()
    if not left_mask.any() or not right_mask.any():
        return left
    valid_left = left[left_mask].sort_values(on)
    valid_right = right[right_mask].sort_values(on)
    merged = pd.merge_asof(valid_left, valid_right, on=on, by=by, direction="backward", **kwargs)
    if left_mask.all():
        return merged
    return pd.concat([merged, left[~left_mask]], ignore_index=True, sort=False)


# ── entry context from candidate_decisions ──────────────────────────

_ENTRY_CONTEXT_COLS = [
    "cycle_id",
    "candidate_id",
    "token_id",
    "condition_id",
    "outcome_side",
    "market",
    "market_slug",
    "trader_wallet",
    "entry_intent",
    "model_action",
    "final_decision",
    "reject_reason",
    "reject_category",
    "gate",
    "confidence",
    "p_tp_before_sl",
    "expected_return",
    "edge_score",
    "calibrated_edge",
    "calibrated_baseline",
    "proposed_size_usdc",
    "final_size_usdc",
    "available_balance",
    "details_json",
    "created_at",
]


def _load_entry_context(logs_dir: Path) -> pd.DataFrame:
    df = _safe_read(logs_dir / "candidate_decisions.csv")
    if df.empty:
        return df
    keep = [c for c in _ENTRY_CONTEXT_COLS if c in df.columns]
    df = df[keep].copy()
    if "created_at" in df.columns:
        df["timestamp"] = _parse_ts(df["created_at"])
    return df


# ── trade outcome from positions + closed_positions ────────────────

_OUTCOME_COLS = [
    "token_id",
    "condition_id",
    "outcome_side",
    "entry_price",
    "exit_price",
    "size_usdc",
    "shares",
    "realized_pnl",
    "net_realized_pnl",
    "opened_at",
    "closed_at",
    "close_reason",
    "exit_reason_family",
    "status",
    "confidence_at_entry",
    "signal_label",
    "entry_signal_snapshot_json",
    "entry_context_complete",
    "learning_eligible",
    "market_family",
    "horizon_bucket",
    "liquidity_bucket",
    "volatility_bucket",
    "technical_regime_bucket",
    "entry_model_family",
    "entry_model_version",
    "performance_governor_level",
    "actual_execution_path",
    "operational_close_flag",
    "reconciliation_close_flag",
]


def _load_trade_outcomes(logs_dir: Path) -> pd.DataFrame:
    closed = _safe_read(logs_dir / "closed_positions.csv")
    positions = _safe_read(logs_dir / "positions.csv")
    frames = []
    for df in (closed, positions):
        if df.empty:
            continue
        keep = [c for c in _OUTCOME_COLS if c in df.columns]
        frames.append(df[keep].copy())
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        subset=[c for c in ["token_id", "outcome_side", "opened_at"] if c in combined.columns],
        keep="last",
    )
    if "opened_at" in combined.columns:
        combined["timestamp"] = _parse_ts(combined["opened_at"])
    return combined


def _load_historical_features(logs_dir: Path) -> pd.DataFrame:
    df = _safe_read(logs_dir / "historical_dataset.csv")
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df["timestamp"] = _parse_ts(df["timestamp"])
    return df.loc[:, ~df.columns.duplicated()].copy()


def _load_btc_trade_attribution(logs_dir: Path) -> pd.DataFrame:
    df = _safe_read(logs_dir / "btc_trade_attribution.csv")
    if df.empty:
        return df
    time_col = "closed_at" if "closed_at" in df.columns else "timestamp" if "timestamp" in df.columns else None
    if time_col:
        df["timestamp"] = _parse_ts(df[time_col])
    return df.loc[:, ~df.columns.duplicated()].copy()


# ── main builder ────────────────────────────────────────────────────

class LearningRowBuilder:
    """Build one canonical learning row per executed trade.

    Writes three CSV files:
    - learning_dataset.csv          (all families, for backward compat)
    - btc/learning_dataset.csv      (BTC family only)
    - weather_temperature/learning_dataset.csv  (weather family only)
    """

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.output_file = self.logs_dir / "learning_dataset.csv"
        self.clean_output_file = self.logs_dir / "learning_dataset_clean.csv"
        self.btc_output_file = self.logs_dir / "btc" / "learning_dataset.csv"
        self.btc_clean_output_file = self.logs_dir / "btc" / "learning_dataset_clean.csv"
        self.weather_output_file = self.logs_dir / "weather_temperature" / "learning_dataset.csv"
        self.weather_clean_output_file = self.logs_dir / "weather_temperature" / "learning_dataset_clean.csv"

    def _derive_learning_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()

        if "entry_price" in out.columns and "exit_price" in out.columns:
            entry_px = pd.to_numeric(out["entry_price"], errors="coerce")
            exit_px = pd.to_numeric(out["exit_price"], errors="coerce")
            price_return = ((exit_px - entry_px) / entry_px.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            if "price_return" not in out.columns:
                out["price_return"] = price_return
            else:
                out["price_return"] = pd.to_numeric(out["price_return"], errors="coerce").fillna(price_return)
        else:
            out["price_return"] = pd.to_numeric(_series_or_default(out, "price_return", 0.0), errors="coerce").fillna(0.0)

        realized_pnl = pd.to_numeric(
            _coalesce(out, "net_realized_pnl", "realized_pnl"),
            errors="coerce",
        ).fillna(0.0)
        size_usdc = pd.to_numeric(_series_or_default(out, "size_usdc", 0.0), errors="coerce").fillna(0.0)
        out["entry_edge_realized"] = (realized_pnl / size_usdc.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out["trade_outcome_label"] = np.where(
            realized_pnl > 0,
            "win",
            np.where(realized_pnl < 0, "loss", "flat"),
        )
        out["entry_quality"] = np.where(out["entry_edge_realized"] > 0, 1, 0)

        # exit_regret = mfe_60m - price_return is only valid for BTC trades.
        # mfe_60m is a price-space path metric (% move from entry over 60m window).
        # Applying it to weather rows (probability-space contracts) produces
        # meaningless labels — their exit quality must be assessed separately.
        _family_col = _coalesce(out, "market_family", "").astype(str).str.lower()
        _is_btc = ~_family_col.str.startswith("weather")
        mfe = pd.to_numeric(out.get("mfe_60m", np.nan), errors="coerce")
        _exit_regret_btc = (mfe - out["price_return"]).replace([np.inf, -np.inf], np.nan)
        out["exit_regret"] = np.where(_is_btc, _exit_regret_btc, np.nan)
        out["exit_regret"] = pd.to_numeric(out["exit_regret"], errors="coerce").fillna(0.0)
        out["exit_quality"] = np.where(
            _is_btc, np.where(out["exit_regret"] <= 0.01, 1, 0), np.nan
        )

        predicted_slippage = pd.to_numeric(
            _coalesce(out, "exec_expected_slippage", "predicted_slippage"),
            errors="coerce",
        ).fillna(0.0)
        actual_slippage = pd.to_numeric(_series_or_default(out, "actual_slippage", np.nan), errors="coerce")
        if "exit_realized_slippage_bps" in out.columns:
            exit_slip = pd.to_numeric(out["exit_realized_slippage_bps"], errors="coerce").div(10000.0)
            actual_slippage = actual_slippage.fillna(exit_slip)
        actual_slippage = actual_slippage.fillna(0.0)
        out["slippage_error"] = (actual_slippage - predicted_slippage).fillna(0.0)
        out["execution_quality"] = np.where(out["slippage_error"] <= 0.0, 1, 0)

        strat_group = (
            _coalesce(out, "market_family", "technical_regime_bucket", "signal_label")
            .astype(str)
            .fillna("unknown")
        )
        strat_mean = out.groupby(strat_group)["entry_edge_realized"].transform("mean")
        out["strategy_quality"] = np.where(strat_mean > 0, 1, 0)

        close_reason = _coalesce(out, "close_reason", "exit_reason_family").astype(str).str.lower()

        # Infer actual_execution_path from close_reason where it was never written.
        # Any trade closed with a named alpha/policy exit necessarily executed as a
        # live fill — the realised PnL proves the order hit the CLOB.
        # external_manual_close rows are already caught by reconciliation_artifact_flag,
        # so labelling them here doesn't inflate the clean set.
        _LIVE_FILL_REASONS = {
            "take_profit_roi", "stop_loss", "hard_emergency_stop",
            "trajectory_panic_exit", "trajectory_liquidity_stress", "rl_exit",
            "time_stop", "policy_exit",
        }
        if "actual_execution_path" not in out.columns:
            out["actual_execution_path"] = ""
        _raw_path = out["actual_execution_path"].astype(str).str.strip()
        _path_missing = _raw_path.isin(["", "nan", "None", "NaN"])
        _inferred_path = close_reason.map(
            lambda r: (
                "live_exit_filled" if r in _LIVE_FILL_REASONS
                else "external_manual_close" if "external_manual_close" in r
                else ""
            )
        )
        out.loc[_path_missing, "actual_execution_path"] = _inferred_path[_path_missing]

        execution_path = out["actual_execution_path"].astype(str).str.lower()
        out["reconciliation_artifact_flag"] = (
            _series_or_default(out, "reconciliation_close_flag", False).apply(_truthy)
            | close_reason.str.contains("reconciliation|external_manual_close", regex=True, na=False)
        )
        out["operational_close_flag"] = _series_or_default(out, "operational_close_flag", False).apply(_truthy)
        out["dead_orderbook_artifact_flag"] = (
            close_reason.str.contains("dead_orderbook", regex=False, na=False)
            | execution_path.str.contains("dead_orderbook", regex=False, na=False)
        )
        out["stale_price_artifact_flag"] = close_reason.str.contains("stale", regex=False, na=False)
        out["missing_execution_path_flag"] = execution_path.eq("") | execution_path.eq("nan")
        out["malformed_fill_flag"] = (
            size_usdc <= 0
        ) | (
            pd.to_numeric(_series_or_default(out, "entry_price", 0.0), errors="coerce").fillna(0.0) <= 0
        )
        out["execution_path_valid"] = ~out["missing_execution_path_flag"]
        out["learning_eligible"] = ~(
            out["reconciliation_artifact_flag"]
            | out["operational_close_flag"]
            | out["dead_orderbook_artifact_flag"]
            | out["stale_price_artifact_flag"]
            | out["malformed_fill_flag"]
            | out["missing_execution_path_flag"]
        )
        out["contaminated_learning_row"] = ~out["learning_eligible"]
        out["learning_exclusion_reason"] = np.select(
            [
                out["reconciliation_artifact_flag"],
                out["operational_close_flag"],
                out["dead_orderbook_artifact_flag"],
                out["stale_price_artifact_flag"],
                out["malformed_fill_flag"],
                out["missing_execution_path_flag"],
            ],
            [
                "reconciliation_artifact",
                "operational_close",
                "dead_orderbook",
                "stale_price",
                "malformed_fill",
                "missing_execution_path",
            ],
            default="clean",
        )
        return out

    def build(self) -> pd.DataFrame:
        # 1. trade outcomes are the spine
        outcomes = _load_trade_outcomes(self.logs_dir)
        if outcomes.empty:
            logger.warning("No trade outcomes found — learning dataset empty.")
            return pd.DataFrame()

        # 2. explode entry_signal_snapshot_json into feature columns
        outcomes = _explode_snapshot_column(outcomes, "entry_signal_snapshot_json")

        # 3. join entry context from candidate_decisions
        entry_ctx = _load_entry_context(self.logs_dir)
        if not entry_ctx.empty:
            join_keys = [c for c in ["token_id", "outcome_side"] if c in outcomes.columns and c in entry_ctx.columns]
            if join_keys:
                # keep only accepted entries
                accepted = entry_ctx[
                    entry_ctx.get("final_decision", pd.Series("", dtype=str)).isin(
                        ["ENTRY_FILLED", "PAPER_OPENED"]
                    )
                ].copy()
                if not accepted.empty:
                    outcomes = outcomes.merge(
                        accepted,
                        on=join_keys,
                        how="left",
                        suffixes=("", "_ctx"),
                    )
                    # coalesce cycle_id and candidate_id
                    for col in ["cycle_id", "candidate_id"]:
                        if f"{col}_ctx" in outcomes.columns:
                            if col not in outcomes.columns:
                                outcomes[col] = outcomes[f"{col}_ctx"]
                            else:
                                outcomes[col] = outcomes[col].fillna(outcomes[f"{col}_ctx"])
                    drop = [c for c in outcomes.columns if c.endswith("_ctx")]
                    outcomes = outcomes.drop(columns=drop)

        # 4. join contract targets (TP/SL labels)
        targets = _safe_read(self.logs_dir / "contract_targets.csv")
        if not targets.empty:
            target_cols = [
                c
                for c in [
                    "token_id",
                    "tp_before_sl_60m",
                    "mfe_60m",
                    "mae_60m",
                    "target_up",
                    "forward_return_15m",
                ]
                if c in targets.columns
            ]
            if "token_id" in target_cols:
                existing_target_cols = [c for c in target_cols if c != "token_id" and c in outcomes.columns]
                outcomes = outcomes.merge(
                    targets[target_cols].drop_duplicates(subset=["token_id"], keep="last"),
                    on="token_id",
                    how="left",
                    suffixes=("", "_tgt"),
                )
                for col in existing_target_cols:
                    tgt_col = f"{col}_tgt"
                    if tgt_col in outcomes.columns:
                        outcomes[col] = outcomes[col].fillna(outcomes[tgt_col])
                drop = [c for c in outcomes.columns if c.endswith("_tgt")]
                outcomes = outcomes.drop(columns=drop)

        # 4b. join enriched historical features by token/timestamp so the
        # canonical learning dataset carries the feature context used by
        # downstream training tasks.
        historical = _load_historical_features(self.logs_dir)
        if not historical.empty and "timestamp" in outcomes.columns:
            hist_join_keys = [c for c in ["token_id", "outcome_side"] if c in outcomes.columns and c in historical.columns]
            hist_feature_cols = [c for c in historical.columns if c not in {"candidate_id", "cycle_id"}]
            historical = historical[hist_feature_cols].copy()
            if hist_join_keys:
                existing_hist_cols = [c for c in historical.columns if c not in {"timestamp", *hist_join_keys} and c in outcomes.columns]
                outcomes = _safe_merge_asof(
                    outcomes,
                    historical,
                    on="timestamp",
                    by=hist_join_keys,
                    tolerance=pd.Timedelta("12h"),
                    suffixes=("", "_hist"),
                )
                for col in existing_hist_cols:
                    hist_col = f"{col}_hist"
                    if hist_col in outcomes.columns:
                        outcomes[col] = outcomes[col].fillna(outcomes[hist_col])
                outcomes = outcomes.drop(columns=[c for c in outcomes.columns if c.endswith("_hist")])

        # 4c. join BTC attribution data when present; this preserves execution
        # and reconciliation context inside the canonical learning row.
        btc_attr = _load_btc_trade_attribution(self.logs_dir)
        if not btc_attr.empty and "timestamp" in outcomes.columns:
            attr_join_keys = [c for c in ["token_id", "outcome_side"] if c in outcomes.columns and c in btc_attr.columns]
            attr_cols = [c for c in btc_attr.columns if c not in {"cycle_id", "candidate_id"}]
            btc_attr = btc_attr[attr_cols].copy()
            if attr_join_keys:
                existing_attr_cols = [c for c in btc_attr.columns if c not in {"timestamp", *attr_join_keys} and c in outcomes.columns]
                outcomes["_attr_join_ts"] = _coalesce(outcomes, "closed_at", "timestamp")
                btc_attr["_attr_join_ts"] = btc_attr["timestamp"]
                outcomes = _safe_merge_asof(
                    outcomes,
                    btc_attr,
                    on="_attr_join_ts",
                    by=attr_join_keys,
                    tolerance=pd.Timedelta("7d"),
                    suffixes=("", "_attr"),
                )
                for col in existing_attr_cols:
                    attr_col = f"{col}_attr"
                    if attr_col in outcomes.columns:
                        outcomes[col] = outcomes[col].fillna(outcomes[attr_col])
                outcomes = outcomes.drop(columns=[c for c in outcomes.columns if c.endswith("_attr")] + ["_attr_join_ts"])

        # 5. compute hold_time_minutes from opened_at / closed_at
        if "opened_at" in outcomes.columns and "closed_at" in outcomes.columns:
            opened = _parse_ts(outcomes["opened_at"])
            closed = _parse_ts(outcomes["closed_at"])
            outcomes["hold_time_minutes"] = (closed - opened).dt.total_seconds().div(60)

        # 6. assign canonical timestamp
        outcomes["timestamp"] = _coalesce(outcomes, "timestamp", "opened_at", "created_at")
        if not pd.api.types.is_datetime64_any_dtype(outcomes["timestamp"]):
            outcomes["timestamp"] = _parse_ts(outcomes["timestamp"])

        # 7. dedup on canonical key
        dedup_cols = [c for c in _LEARNING_KEY_COLS if c in outcomes.columns]
        if dedup_cols:
            outcomes = outcomes.drop_duplicates(subset=dedup_cols, keep="last")

        outcomes = self._derive_learning_labels(outcomes)

        outcomes = outcomes.loc[:, ~outcomes.columns.duplicated()].copy()
        logger.info(
            "Built %d learning rows (%d columns).",
            len(outcomes),
            len(outcomes.columns),
        )
        return outcomes

    def write(self) -> pd.DataFrame:
        from model_feature_safety import clean_dataframe_for_training

        df = self.build()
        if df.empty:
            return df
        df = clean_dataframe_for_training(df, context="learning_dataset")

        # Write combined file (all families)
        df.to_csv(self.output_file, index=False)
        logger.info("Saved learning dataset to %s", self.output_file)
        clean_df = df[df.get("learning_eligible", True).fillna(False).astype(bool)].copy()
        clean_df.to_csv(self.clean_output_file, index=False)
        logger.info("Saved clean learning dataset to %s (%d rows)", self.clean_output_file, len(clean_df))

        # Write per-family split files
        family_col = "market_family" if "market_family" in df.columns else None
        if family_col:
            _weather_mask = df[family_col].astype(str).str.startswith("weather_temperature")
            _btc_df = df[~_weather_mask].copy()
            _weather_df = df[_weather_mask].copy()
        else:
            _btc_df = df.copy()
            _weather_df = pd.DataFrame()

        if not _btc_df.empty:
            self.btc_output_file.parent.mkdir(parents=True, exist_ok=True)
            _btc_df.to_csv(self.btc_output_file, index=False)
            logger.info("Saved BTC learning dataset to %s (%d rows)", self.btc_output_file, len(_btc_df))
            _btc_clean_df = _btc_df[_btc_df.get("learning_eligible", True).fillna(False).astype(bool)].copy()
            _btc_clean_df.to_csv(self.btc_clean_output_file, index=False)
            logger.info("Saved clean BTC learning dataset to %s (%d rows)", self.btc_clean_output_file, len(_btc_clean_df))

        if not _weather_df.empty:
            self.weather_output_file.parent.mkdir(parents=True, exist_ok=True)
            _weather_df.to_csv(self.weather_output_file, index=False)
            logger.info("Saved weather learning dataset to %s (%d rows)", self.weather_output_file, len(_weather_df))
            _weather_clean_df = _weather_df[_weather_df.get("learning_eligible", True).fillna(False).astype(bool)].copy()
            _weather_clean_df.to_csv(self.weather_clean_output_file, index=False)
            logger.info("Saved clean weather learning dataset to %s (%d rows)", self.weather_clean_output_file, len(_weather_clean_df))

        return df
