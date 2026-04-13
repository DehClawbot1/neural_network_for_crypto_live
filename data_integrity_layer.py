"""
data_integrity_layer.py
Phase 3 — Data Cleaning Before Learning

Before any model retraining the system passes all feedback rows through
this layer, which:

  3A  Excludes contaminated rows entirely (reliability = 0.0)
  3B  Down-weights rows with uncertain provenance  (reliability = 0.5)
  3C  Scores every row 0.0 / 0.1 / 0.5 / 1.0

Phase 14 anti-failure rules are hard-coded here:
  - Never learn alpha from reconciliation / API / stale / manual / broken fills
  - sample_weight = 0 for any excluded row (they are dropped entirely)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard exclusion reasons (cannot be overridden by config)
# ---------------------------------------------------------------------------
_HARD_EXCLUDE_CLOSE_REASONS: set[str] = {
    "reconciliation",
    "external_manual_close",
    "api_error",
    "api_rejected",
    "stale_data",
    "stale_context",
    "broken_fill",
    "manual_override",
    "manual_exit",
}

# Flags from label_builder.py that reduce reliability
# These map to the missing_data flag columns written into contract_targets.csv
_ZERO_RELIABILITY_FLAGS: set[str] = {
    "reconciliation_close_flag",
    "operational_close_flag",
    "weather_resolution_unavailable",   # unresolved weather = not usable as alpha truth
}
_HALF_RELIABILITY_FLAGS: set[str] = {
    "entry_price_from_fallback",         # CLOB anchor was missing
    "path_horizon_short",               # 15m window was empty
    "btc_implied_prob_missing",         # no orderbook snapshot at entry
    "weather_implied_prob_missing",     # same, weather family
    "exec_cost_unavailable",            # no execution log row for this trade
}

FamilyType = Literal["btc", "weather_temperature", "execution", "meta", "all"]


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        num = float(value)
        return float(default) if not np.isfinite(num) else num
    except Exception:
        return float(default)


# ---------------------------------------------------------------------------
# DataIntegrityLayer
# ---------------------------------------------------------------------------
class DataIntegrityLayer:
    """
    Reads feedback CSVs, applies contamination exclusion and reliability
    weighting, and returns cleaned DataFrames ready for model training.

    Usage:
        dil = DataIntegrityLayer(logs_dir="logs")
        df = dil.build_clean_training_set("btc")   # reliability > 0
        # df has a "sample_weight" column for use with sample_weight= param
    """

    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = Path(logs_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reliability_score(self, row: dict) -> float:
        """
        Compute a reliability score for a single feedback row.
        Returns 0.0 (exclude), 0.1 (almost unusable), 0.5 (usable noisy),
        or 1.0 (clean truth).
        """
        # Hard-zero from new label flags
        for flag in _ZERO_RELIABILITY_FLAGS:
            if _safe_bool(row.get(flag)):
                return 0.0

        # Phase 14 hard exclusions
        close_reason = str(row.get("close_reason") or "").strip().lower()
        if close_reason in _HARD_EXCLUDE_CLOSE_REASONS:
            return 0.0
        entry_price = _safe_float(row.get("entry_price"))
        exit_price = _safe_float(row.get("exit_price"))
        if entry_price <= 0 or exit_price <= 0:
            return 0.0

        # Phase 3B soft down-weights from new label flags
        score = 1.0
        for flag in _HALF_RELIABILITY_FLAGS:
            if _safe_bool(row.get(flag)):
                score = min(score, 0.5)

        if _safe_bool(row.get("fill_quality_poor")):
            score = min(score, 0.5)
        if _safe_bool(row.get("forecast_disagreement_extreme")):
            score = min(score, 0.5)
        if _safe_bool(row.get("execution_uncertainty_high")):
            score = min(score, 0.5)
        # Policy-forced exit (not market-driven)
        if str(row.get("close_reason") or "").strip().lower() in {
            "time_stop", "portfolio_cap_force_close", "governor_force_close"
        }:
            score = min(score, 0.5)
        # Incomplete entry context
        if not _safe_bool(row.get("entry_context_complete", True)):
            score = min(score, 0.1)
        # Learning eligibility flag set to False
        if "learning_eligible" in row and not _safe_bool(row.get("learning_eligible")):
            score = min(score, 0.0)

        return score

    def clean_alpha_feedback(
        self,
        df: pd.DataFrame,
        *,
        min_reliability: float = 0.1,
    ) -> pd.DataFrame:
        """
        Apply Phase 3 cleaning to alpha feedback rows.
        Adds `sample_weight` column. Drops rows with reliability = 0.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        work = df.copy()

        # Score each row
        if "reliability_score" not in work.columns:
            work["reliability_score"] = work.apply(
                lambda row: self.reliability_score(row.to_dict()), axis=1
            )

        # Hard exclude
        before = len(work)
        work = work[work["reliability_score"] > 0].copy()
        work = work[work["reliability_score"] >= min_reliability].copy()
        excluded = before - len(work)
        if excluded > 0:
            logger.info("DataIntegrityLayer: excluded %d/%d contaminated rows", excluded, before)

        if work.empty:
            return work

        work["sample_weight"] = work["reliability_score"].clip(lower=0.0, upper=1.0)
        return work

    def build_clean_training_set(
        self,
        family: FamilyType = "all",
        *,
        min_reliability: float = 0.1,
    ) -> pd.DataFrame:
        """
        Build a clean, reliability-weighted training dataset for the
        given market family from the alpha_feedback_clean.csv sink.

        Returns DataFrame with `sample_weight` column.
        """
        alpha_csv = self.logs_dir / "alpha_feedback_clean.csv"
        df = self._safe_read(alpha_csv)
        if df.empty:
            return df

        if family != "all" and "market_family" in df.columns:
            family_series = df["market_family"].fillna("").astype(str).str.lower()
            df = df[family_series.str.startswith(str(family).lower())].copy()

        return self.clean_alpha_feedback(df, min_reliability=min_reliability)

    def build_execution_training_set(self, *, min_reliability: float = 0.1) -> pd.DataFrame:
        """Build cleaned execution feedback dataset."""
        csv_path = self.logs_dir / "execution_feedback.csv"
        df = self._safe_read(csv_path)
        if df.empty:
            return df
        work = df.copy()
        if "reliability_score" not in work.columns:
            work["reliability_score"] = 1.0
        work = work[work["reliability_score"] >= min_reliability].copy()
        work["sample_weight"] = work["reliability_score"].clip(lower=0.0, upper=1.0)
        return work

    def build_policy_training_set(self, *, min_reliability: float = 0.1) -> pd.DataFrame:
        """Build cleaned policy feedback dataset."""
        csv_path = self.logs_dir / "policy_feedback.csv"
        df = self._safe_read(csv_path)
        if df.empty:
            return df
        work = df.copy()
        if "reliability_score" not in work.columns:
            work["reliability_score"] = 1.0
        work = work[work["reliability_score"] >= min_reliability].copy()
        work["sample_weight"] = work["reliability_score"].clip(lower=0.0, upper=1.0)
        return work

    def operational_failure_count(self, *, lookback: int = 200) -> dict[str, int]:
        """Return counts of each operational failure type from recent rows."""
        csv_path = self.logs_dir / "operational_failures.csv"
        df = self._safe_read(csv_path)
        if df.empty or "failure_type" not in df.columns:
            return {}
        recent = df.tail(lookback).copy()
        counts = recent["failure_type"].value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def data_quality_summary(self) -> dict:
        """Return a summary dict suitable for dashboard / logging."""
        alpha_csv = self.logs_dir / "alpha_feedback_clean.csv"
        exec_csv = self.logs_dir / "execution_feedback.csv"
        policy_csv = self.logs_dir / "policy_feedback.csv"
        ops_csv = self.logs_dir / "operational_failures.csv"

        def _row_count(path: Path) -> int:
            df = self._safe_read(path)
            return len(df)

        alpha_df = self._safe_read(alpha_csv)
        if not alpha_df.empty and "reliability_score" in alpha_df.columns:
            clean_alpha = int((alpha_df["reliability_score"] >= 1.0).sum())
            noisy_alpha = int(((alpha_df["reliability_score"] > 0) & (alpha_df["reliability_score"] < 1.0)).sum())
        else:
            clean_alpha = _row_count(alpha_csv)
            noisy_alpha = 0

        return {
            "alpha_feedback_total": _row_count(alpha_csv),
            "alpha_feedback_clean": clean_alpha,
            "alpha_feedback_noisy": noisy_alpha,
            "execution_feedback_total": _row_count(exec_csv),
            "policy_feedback_total": _row_count(policy_csv),
            "operational_failures_total": _row_count(ops_csv),
            "operational_failure_counts": self.operational_failure_count(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _safe_read(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()
