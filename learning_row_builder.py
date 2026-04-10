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
    raw = pd.to_datetime(series, errors="coerce", utc=False, format="mixed")
    if getattr(raw.dt, "tz", None) is None:
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


# ── main builder ────────────────────────────────────────────────────

class LearningRowBuilder:
    """Build one canonical learning row per executed trade."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.output_file = self.logs_dir / "learning_dataset.csv"

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

        outcomes = outcomes.loc[:, ~outcomes.columns.duplicated()].copy()
        logger.info(
            "Built %d learning rows (%d columns).",
            len(outcomes),
            len(outcomes.columns),
        )
        return outcomes

    def write(self) -> pd.DataFrame:
        df = self.build()
        if df.empty:
            return df
        df.to_csv(self.output_file, index=False)
        logger.info("Saved learning dataset to %s", self.output_file)
        return df
