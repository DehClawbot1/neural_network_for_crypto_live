"""Phase 2 training table builder.

Reads the family-split learning datasets produced by LearningRowBuilder:
  logs/btc/learning_dataset.csv
  logs/weather_temperature/learning_dataset.csv

and builds four task-specific supervised training tables for EACH family:

  {family}/entry_training.csv      — was entering this trade a good decision?
  {family}/exit_training.csv       — was exiting at this point optimal?
  {family}/execution_training.csv  — how accurately was the order executed?
  {family}/strategy_training.csv   — which strategy / regime configs have edge?

BTC and weather trades are NEVER mixed:
  - BTC labels live in price/return space (entry_edge_realized, exit_regret,
    slippage_error, path_tp_hit_60m, mfe/mae in % terms).
  - Weather labels live in probability space (contract resolution, market edge).
    Price-return thresholds are meaningless for binary outcome contracts.

Each table:
  - Uses the family-specific source file, not the combined dataset
  - Applies the base learning_eligible filter (artifact flags clean)
  - Applies model-specific additional exclusion rules on top
  - Selects feature columns valid for that family's market structure
  - Includes only realized-outcome labels (no look-ahead, no broad heuristics)
  - Writes a companion _exclusion_report.csv documenting every dropped row

Output layout:
  logs/training_tables/btc/entry_training.csv
  logs/training_tables/btc/entry_exclusion_report.csv
  logs/training_tables/btc/exit_training.csv
  ... (same for exit, execution, strategy)
  logs/training_tables/weather_temperature/entry_training.csv
  ... (same four tables)
  logs/training_tables/build_summary.csv

Run directly:
  python training_table_builder.py [--validate]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── identity columns always carried in every table ────────────────────────────

_IDENTITY_COLS = [
    "candidate_id",
    "cycle_id",
    "token_id",
    "timestamp",
    "market_family",
    "close_reason",
    "actual_execution_path",
]

# All artifact flag columns written by LearningRowBuilder._derive_learning_labels
_ALL_ARTIFACT_FLAGS = [
    "reconciliation_artifact_flag",
    "operational_close_flag",
    "dead_orderbook_artifact_flag",
    "stale_price_artifact_flag",
    "malformed_fill_flag",
    "missing_execution_path_flag",
]

# ── BTC feature / label sets ──────────────────────────────────────────────────
# BTC trades are CLOB price-direction bets.  Labels live in return / price space.

_BTC_ENTRY_FEATURES = [
    # Supervisor signal
    "confidence",
    "confidence_at_entry",
    "edge_score",
    "calibrated_edge",
    "calibrated_baseline",
    "p_tp_before_sl",
    "expected_return",
    "signal_label",
    "entry_intent",
    "model_action",
    "gate",
    "reject_reason",
    "reject_category",
    "entry_model_family",
    "entry_model_version",
    "performance_governor_level",
    # Market regime at entry
    "horizon_bucket",
    "liquidity_bucket",
    "volatility_bucket",
    "technical_regime_bucket",
    "btc_trend_bias",
    "btc_volatility_regime",
    "btc_price_at_entry",
    # Microstructure
    "spread_at_entry",
    "depth_at_entry",
    "volume_24h",
    "market_implied_prob",
    # Sizing
    "proposed_size_usdc",
    "final_size_usdc",
    "available_balance",
    # Snapshot model outputs (exploded from entry_signal_snapshot_json)
    "btc_prob_up_15m",
    "btc_prob_up_60m",
    "forward_return_15m",
    "target_up",
]

_BTC_ENTRY_LABELS = [
    "trade_outcome_label",   # win / loss / flat
    "entry_edge_realized",   # realized_pnl / size_usdc
    "entry_quality",         # 1 if entry_edge_realized > 0
]

_BTC_EXIT_FEATURES = [
    "hold_time_minutes",
    "price_return",          # actual % return captured
    "mfe_60m",               # max upside available in 60m window
    "mae_60m",               # max drawdown in 60m window
    "tp_before_sl_60m",      # TP(+4%) hit before SL(-3%) in 60m
    "exit_reason_family",
    "entry_price",
    "exit_price",
    "size_usdc",
    "technical_regime_bucket",
    "volatility_bucket",
    "horizon_bucket",
    "btc_trend_bias",
    "btc_volatility_regime",
    "performance_governor_level",
    "hour_of_day",
    "day_of_week",
]

_BTC_EXIT_LABELS = [
    "exit_regret",    # mfe_60m - price_return  (missed upside, in return space)
    "exit_quality",   # 1 if exit_regret <= 0.01
]

_BTC_EXECUTION_FEATURES = [
    "size_usdc",
    "shares",
    "entry_price",
    "spread_at_entry",
    "depth_at_entry",
    "volume_24h",
    "liquidity_bucket",
    "volatility_bucket",
    "technical_regime_bucket",
    "exit_fill_latency_seconds",
    "exit_realized_slippage_bps",
    "exec_expected_slippage",
    "predicted_slippage",
    "btc_volatility_regime",
    "hour_of_day",
]

_BTC_EXECUTION_LABELS = [
    "slippage_error",      # actual_slippage - predicted_slippage
    "execution_quality",   # 1 if slippage_error <= 0
]

_BTC_STRATEGY_FEATURES = [
    "market_family",
    "entry_model_family",
    "entry_model_version",
    "signal_label",
    "technical_regime_bucket",
    "volatility_bucket",
    "horizon_bucket",
    "liquidity_bucket",
    "performance_governor_level",
    "btc_trend_bias",
    "btc_volatility_regime",
    "gate",
]

_BTC_STRATEGY_LABELS = [
    "entry_edge_realized",   # per-trade edge captured
    "strategy_quality",      # 1 if strategy group mean edge > 0
    "trade_outcome_label",   # win / loss / flat
]

# ── Weather feature / label sets ──────────────────────────────────────────────
# Weather trades are binary outcome contracts (YES/NO resolution).
# Labels live in PROBABILITY space — never mix with BTC price-return columns.
# TP/SL % thresholds and MFE/MAE in return space are meaningless here.

_WEATHER_ENTRY_FEATURES = [
    # Supervisor signal
    "confidence",
    "confidence_at_entry",
    "edge_score",
    "calibrated_edge",
    "calibrated_baseline",
    "expected_return",
    "signal_label",
    "entry_intent",
    "model_action",
    "gate",
    "reject_reason",
    "reject_category",
    "entry_model_family",
    "entry_model_version",
    "performance_governor_level",
    # Contract / market structure
    "market_implied_prob",    # CLOB mid-price = implied YES probability
    "outcome_side",           # YES or NO
    "horizon_bucket",
    "liquidity_bucket",
    "spread_at_entry",
    "depth_at_entry",
    "volume_24h",
    # Sizing
    "proposed_size_usdc",
    "final_size_usdc",
    "available_balance",
    # Snapshot model outputs
    "weather_forecast_prob",       # model's P(YES)
    "weather_market_edge_at_entry",  # model_prob - market_implied_prob
]

_WEATHER_ENTRY_LABELS = [
    "trade_outcome_label",            # win / loss / flat (based on realized_pnl)
    "entry_edge_realized",            # realized_pnl / size_usdc
    "entry_quality",                  # 1 if entry_edge_realized > 0
    "weather_contract_resolved_yes",  # 1 if contract actually resolved YES
    "weather_market_edge",            # resolution - market_implied_prob at entry
]

_WEATHER_EXIT_FEATURES = [
    "hold_time_minutes",
    "price_return",          # CLOB price change (probability drift, not physical price)
    "exit_reason_family",
    "entry_price",           # implied prob at entry
    "exit_price",            # implied prob at exit
    "size_usdc",
    "horizon_bucket",
    "liquidity_bucket",
    "performance_governor_level",
    "hour_of_day",
    "day_of_week",
    # No mfe_60m / mae_60m / tp_before_sl_60m — those use BTC % thresholds
]

_WEATHER_EXIT_LABELS = [
    "exit_regret",    # mfe_60m - price_return (reused field, but in prob-drift units here)
    "exit_quality",   # 1 if exit_regret <= 0.01
]

_WEATHER_EXECUTION_FEATURES = [
    "size_usdc",
    "shares",
    "entry_price",
    "spread_at_entry",
    "depth_at_entry",
    "volume_24h",
    "liquidity_bucket",
    "exit_fill_latency_seconds",
    "exit_realized_slippage_bps",
    "exec_expected_slippage",
    "predicted_slippage",
    "hour_of_day",
]

_WEATHER_EXECUTION_LABELS = [
    "slippage_error",
    "execution_quality",
]

_WEATHER_STRATEGY_FEATURES = [
    "market_family",
    "entry_model_family",
    "entry_model_version",
    "signal_label",
    "outcome_side",
    "horizon_bucket",
    "liquidity_bucket",
    "performance_governor_level",
    "gate",
]

_WEATHER_STRATEGY_LABELS = [
    "entry_edge_realized",
    "strategy_quality",
    "trade_outcome_label",
    "weather_contract_resolved_yes",  # ground truth for strategy evaluation
]

# ── per-family config registry ────────────────────────────────────────────────

@dataclass
class FamilyConfig:
    name: str             # "btc" or "weather_temperature"
    source_subdir: str    # relative to logs_dir
    out_subdir: str       # relative to training_tables/
    entry_features: list[str]
    entry_labels: list[str]
    exit_features: list[str]
    exit_labels: list[str]
    execution_features: list[str]
    execution_labels: list[str]
    strategy_features: list[str]
    strategy_labels: list[str]


_FAMILY_CONFIGS: list[FamilyConfig] = [
    FamilyConfig(
        name="btc",
        source_subdir="btc",
        out_subdir="btc",
        entry_features=_BTC_ENTRY_FEATURES,
        entry_labels=_BTC_ENTRY_LABELS,
        exit_features=_BTC_EXIT_FEATURES,
        exit_labels=_BTC_EXIT_LABELS,
        execution_features=_BTC_EXECUTION_FEATURES,
        execution_labels=_BTC_EXECUTION_LABELS,
        strategy_features=_BTC_STRATEGY_FEATURES,
        strategy_labels=_BTC_STRATEGY_LABELS,
    ),
    FamilyConfig(
        name="weather_temperature",
        source_subdir="weather_temperature",
        out_subdir="weather_temperature",
        entry_features=_WEATHER_ENTRY_FEATURES,
        entry_labels=_WEATHER_ENTRY_LABELS,
        exit_features=_WEATHER_EXIT_FEATURES,
        exit_labels=_WEATHER_EXIT_LABELS,
        execution_features=_WEATHER_EXECUTION_FEATURES,
        execution_labels=_WEATHER_EXECUTION_LABELS,
        strategy_features=_WEATHER_STRATEGY_FEATURES,
        strategy_labels=_WEATHER_STRATEGY_LABELS,
    ),
]

# ── exclusion rules ──────────────────────────────────────────────────────────

@dataclass
class ExclusionRule:
    name: str
    description: str
    flags: list[str] = field(default_factory=list)
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None


def _flag_mask(df: pd.DataFrame, flags: list[str]) -> pd.Series:
    """True for rows where ANY of the listed artifact flag columns is truthy."""
    mask = pd.Series(False, index=df.index)
    for flag in flags:
        if flag not in df.columns:
            continue
        col = df[flag]
        if col.dtype == object:
            col = col.astype(str).str.lower().isin({"true", "1", "yes"})
        else:
            col = col.fillna(False).astype(bool)
        mask = mask | col
    return mask


_ENTRY_EXCLUSION_RULES: list[ExclusionRule] = [
    ExclusionRule(
        name="reconciliation_artifact",
        description="Trade closed by reconciliation — outcome does not reflect alpha signal quality",
        flags=["reconciliation_artifact_flag"],
    ),
    ExclusionRule(
        name="operational_close",
        description="Trade closed for operational reasons — exit distorts entry outcome label",
        flags=["operational_close_flag"],
    ),
    ExclusionRule(
        name="malformed_fill",
        description="Fill is missing size or entry price — entry labels cannot be computed reliably",
        flags=["malformed_fill_flag"],
    ),
    ExclusionRule(
        name="missing_execution_path",
        description="No execution path recorded — entry outcome attribution is ambiguous",
        flags=["missing_execution_path_flag"],
    ),
    ExclusionRule(
        name="missing_entry_edge",
        description="entry_edge_realized is NaN — primary target cannot be computed",
        predicate=lambda df: pd.to_numeric(
            df.get("entry_edge_realized", pd.Series(dtype=float)), errors="coerce"
        ).isna(),
    ),
]

_EXIT_EXCLUSION_RULES: list[ExclusionRule] = [
    ExclusionRule(
        name="reconciliation_artifact",
        description="Exit was forced by reconciliation — not a model exit decision",
        flags=["reconciliation_artifact_flag"],
    ),
    ExclusionRule(
        name="operational_close",
        description="Exit was forced for operational reasons — exit_regret label would be misleading",
        flags=["operational_close_flag"],
    ),
    ExclusionRule(
        name="dead_orderbook",
        description="Exit filled against dead orderbook — price_return is pathological",
        flags=["dead_orderbook_artifact_flag"],
    ),
    ExclusionRule(
        name="stale_price",
        description="Exit used stale price — price_return and exit_regret labels are unreliable",
        flags=["stale_price_artifact_flag"],
    ),
]

_EXECUTION_EXCLUSION_RULES: list[ExclusionRule] = [
    ExclusionRule(
        name="dead_orderbook",
        description="Fill against dead orderbook — slippage is not representative of normal execution",
        flags=["dead_orderbook_artifact_flag"],
    ),
    ExclusionRule(
        name="stale_price",
        description="Stale price at fill — actual_slippage measurement is corrupted",
        flags=["stale_price_artifact_flag"],
    ),
    ExclusionRule(
        name="malformed_fill",
        description="Fill has zero/missing size or price — execution metrics cannot be trusted",
        flags=["malformed_fill_flag"],
    ),
    ExclusionRule(
        name="missing_execution_path",
        description="No execution path recorded — slippage source is unknown",
        flags=["missing_execution_path_flag"],
    ),
]

_STRATEGY_EXCLUSION_RULES: list[ExclusionRule] = [
    ExclusionRule(
        name="any_artifact",
        description="Any contamination flag — strategy quality requires clean outcome attribution",
        flags=_ALL_ARTIFACT_FLAGS,
    ),
    ExclusionRule(
        name="missing_entry_edge",
        description="entry_edge_realized is NaN — strategy edge cannot be aggregated without it",
        predicate=lambda df: pd.to_numeric(
            df.get("entry_edge_realized", pd.Series(dtype=float)), errors="coerce"
        ).isna(),
    ),
]


# ── core builder ──────────────────────────────────────────────────────────────

class TrainingTableBuilder:
    """Build task-specific training tables, split by market family."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.out_root = self.logs_dir / "training_tables"

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load_family_source(self, family: FamilyConfig) -> pd.DataFrame:
        path = self.logs_dir / family.source_subdir / "learning_dataset.csv"
        if not path.exists():
            logger.warning("[%s] source file not found: %s", family.name, path)
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
            logger.info("[%s] loaded %d rows from %s", family.name, len(df), path)
            return df
        except Exception as exc:
            logger.error("[%s] failed to load source: %s", family.name, exc)
            return pd.DataFrame()

    def _write_table(
        self,
        df: pd.DataFrame,
        exclusion_df: pd.DataFrame,
        family: FamilyConfig,
        model: str,
    ) -> None:
        out_dir = self.out_root / family.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        table_path = out_dir / f"{model}_training.csv"
        report_path = out_dir / f"{model}_exclusion_report.csv"
        df.to_csv(table_path, index=False)
        exclusion_df.to_csv(report_path, index=False)
        logger.info(
            "[%s/%s] %d clean rows → %s | %d excluded → %s",
            family.name, model, len(df), table_path, len(exclusion_df), report_path,
        )

    # ── filtering ─────────────────────────────────────────────────────────────

    def _apply_base_filter(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split on learning_eligible, or re-derive from artifact flags."""
        if "learning_eligible" in df.columns:
            col = df["learning_eligible"]
            if col.dtype == object:
                eligible_mask = col.astype(str).str.lower().isin({"true", "1", "yes"})
            else:
                eligible_mask = col.fillna(False).astype(bool)
        else:
            eligible_mask = ~_flag_mask(df, _ALL_ARTIFACT_FLAGS)

        clean = df[eligible_mask].copy()
        excluded = df[~eligible_mask].copy()
        if not excluded.empty:
            reason_col = "learning_exclusion_reason" if "learning_exclusion_reason" in excluded.columns else None
            excluded = excluded.copy()
            excluded["table_exclusion_reason"] = (
                excluded[reason_col].fillna("base_learning_ineligible")
                if reason_col
                else "base_learning_ineligible"
            )
        return clean, excluded

    def _apply_model_exclusions(
        self,
        df: pd.DataFrame,
        rules: list[ExclusionRule],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        exclusion_frames: list[pd.DataFrame] = []
        remaining = df.copy()
        for rule in rules:
            if remaining.empty:
                break
            bad = pd.Series(False, index=remaining.index)
            if rule.flags:
                bad = bad | _flag_mask(remaining, rule.flags)
            if rule.predicate is not None:
                try:
                    bad = bad | rule.predicate(remaining).reindex(remaining.index, fill_value=False)
                except Exception as exc:
                    logger.warning("Exclusion predicate '%s' failed: %s", rule.name, exc)
            if bad.any():
                dropped = remaining[bad].copy()
                dropped["table_exclusion_reason"] = rule.name
                dropped["table_exclusion_description"] = rule.description
                exclusion_frames.append(dropped)
                remaining = remaining[~bad].copy()
        exc_df = pd.concat(exclusion_frames, ignore_index=True, sort=False) if exclusion_frames else pd.DataFrame()
        return remaining, exc_df

    def _select_columns(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_cols: list[str],
    ) -> pd.DataFrame:
        identity = [c for c in _IDENTITY_COLS if c in df.columns]
        features = [c for c in feature_cols if c in df.columns]
        labels = [c for c in label_cols if c in df.columns]
        all_cols = list(dict.fromkeys(identity + features + labels))
        return df[all_cols].copy()

    def _build_exclusion_report(
        self,
        base_exc: pd.DataFrame,
        model_exc: pd.DataFrame,
    ) -> pd.DataFrame:
        frames = []
        report_cols = _IDENTITY_COLS + [
            "table_exclusion_reason",
            "table_exclusion_description",
            "learning_exclusion_reason",
        ]
        for exc_df in (base_exc, model_exc):
            if not exc_df.empty:
                keep = [c for c in report_cols if c in exc_df.columns]
                frames.append(exc_df[keep])
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    # ── per-model builders ────────────────────────────────────────────────────

    def _build_model(
        self,
        df: pd.DataFrame,
        features: list[str],
        labels: list[str],
        rules: list[ExclusionRule],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        clean, base_exc = self._apply_base_filter(df)
        clean, model_exc = self._apply_model_exclusions(clean, rules)
        table = self._select_columns(clean, features, labels)
        report = self._build_exclusion_report(base_exc, model_exc)
        return table, report

    # ── family build ──────────────────────────────────────────────────────────

    def _build_family(self, family: FamilyConfig) -> list[dict]:
        df = self._load_family_source(family)
        if df.empty:
            logger.warning("[%s] skipping — no source data", family.name)
            return []

        total = len(df)
        summary_rows = []

        models = [
            ("entry",     family.entry_features,     family.entry_labels,     _ENTRY_EXCLUSION_RULES),
            ("exit",      family.exit_features,       family.exit_labels,      _EXIT_EXCLUSION_RULES),
            ("execution", family.execution_features,  family.execution_labels, _EXECUTION_EXCLUSION_RULES),
            ("strategy",  family.strategy_features,   family.strategy_labels,  _STRATEGY_EXCLUSION_RULES),
        ]

        for model, features, labels, rules in models:
            table, report = self._build_model(df, features, labels, rules)
            self._write_table(table, report, family, model)

            reason_counts = (
                report["table_exclusion_reason"].value_counts().to_dict()
                if not report.empty and "table_exclusion_reason" in report.columns
                else {}
            )
            summary_rows.append({
                "family": family.name,
                "model": model,
                "source_rows": total,
                "clean_rows": len(table),
                "excluded_rows": total - len(table),
                "pct_clean": round(100 * len(table) / total, 1) if total > 0 else 0.0,
                "feature_cols": len([c for c in features if c in table.columns]),
                "label_cols": len([c for c in labels if c in table.columns]),
                "exclusion_reasons": str(reason_counts) if reason_counts else "{}",
            })

        return summary_rows

    # ── main entry point ──────────────────────────────────────────────────────

    def build_all(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Build all tables for all families. Returns {family: {model: DataFrame}}."""
        all_summary: list[dict] = []
        results: dict[str, dict[str, pd.DataFrame]] = {}

        for family in _FAMILY_CONFIGS:
            rows = self._build_family(family)
            all_summary.extend(rows)
            # Re-load written tables into results for validation
            if rows:
                results[family.name] = {}
                for model in ("entry", "exit", "execution", "strategy"):
                    path = self.out_root / family.out_subdir / f"{model}_training.csv"
                    if path.exists():
                        try:
                            results[family.name][model] = pd.read_csv(path, low_memory=False)
                        except Exception:
                            pass

        if all_summary:
            summary_df = pd.DataFrame(all_summary)
            summary_path = self.out_root / "build_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            _print_summary(summary_df)

        return results


# ── label integrity checks ────────────────────────────────────────────────────

def validate_labels(df: pd.DataFrame, family: str, model: str) -> list[str]:
    """Return integrity warnings for one training table."""
    warns: list[str] = []
    tag = f"[{family}/{model}]"

    # No artifact rows should survive into any table
    for flag in _ALL_ARTIFACT_FLAGS:
        if flag in df.columns:
            n = _flag_mask(df, [flag]).sum()
            if n > 0:
                warns.append(f"{tag} {n} rows with {flag}=True leaked through — exclusion broken")

    if model == "entry":
        if "entry_edge_realized" in df.columns:
            vals = pd.to_numeric(df["entry_edge_realized"], errors="coerce")
            if vals.isna().all():
                warns.append(f"{tag} entry_edge_realized: all NaN — no usable labels")
            elif vals.nunique() == 1:
                warns.append(f"{tag} entry_edge_realized: degenerate (single value)")
        if "trade_outcome_label" in df.columns:
            dist = df["trade_outcome_label"].value_counts(normalize=True)
            if dist.max() > 0.95:
                warns.append(f"{tag} trade_outcome_label: >95% one class ({dist.idxmax()}) — check data quality")

    elif model == "exit":
        if "exit_regret" in df.columns:
            vals = pd.to_numeric(df["exit_regret"], errors="coerce")
            if vals.isna().mean() > 0.5:
                warns.append(f"{tag} exit_regret: >50% NaN — mfe_60m may be missing upstream")
        # BTC-specific: mfe_60m / mae_60m should not appear in weather exit tables
        if family.startswith("weather") and "mfe_60m" in df.columns:
            warns.append(f"{tag} mfe_60m present in weather exit table — price-space column leaked in")

    elif model == "execution":
        if "slippage_error" in df.columns:
            vals = pd.to_numeric(df["slippage_error"], errors="coerce")
            if vals.isna().all():
                warns.append(f"{tag} slippage_error: all NaN — execution feedback data missing upstream")

    elif model == "strategy":
        if "strategy_quality" in df.columns:
            dist = df["strategy_quality"].value_counts(normalize=True)
            if dist.max() > 0.95:
                warns.append(f"{tag} strategy_quality: >95% one class — strategies uniformly losing")
        # BTC price-space columns must not appear in weather strategy table
        btc_only = {"btc_trend_bias", "btc_volatility_regime", "mfe_60m", "mae_60m", "tp_before_sl_60m"}
        leaked = btc_only & set(df.columns)
        if leaked and family.startswith("weather"):
            warns.append(f"{tag} BTC-only columns in weather table: {sorted(leaked)}")

    return warns


# ── print ─────────────────────────────────────────────────────────────────────

def _print_summary(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 75)
    print("PHASE 2 TRAINING TABLE BUILD SUMMARY")
    print("=" * 75)
    current_family = None
    for _, row in summary.iterrows():
        if row["family"] != current_family:
            current_family = row["family"]
            print(f"\n  {current_family.upper()}")
        print(
            f"    {row['model']:12s}  "
            f"clean={row['clean_rows']:5d}/{row['source_rows']}  "
            f"({row['pct_clean']:.1f}%)  "
            f"features={row['feature_cols']}  labels={row['label_cols']}"
        )
        if row["exclusion_reasons"] and row["exclusion_reasons"] != "{}":
            print(f"                 exclusions: {row['exclusion_reasons']}")
    print("\n" + "=" * 75 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Build phase-2 family-split training tables.")
    parser.add_argument("--logs-dir", default="logs", help="Path to logs directory (default: logs)")
    parser.add_argument("--validate", action="store_true", help="Run label integrity checks after build")
    args = parser.parse_args()

    builder = TrainingTableBuilder(logs_dir=args.logs_dir)
    results = builder.build_all()

    if args.validate and results:
        print("LABEL INTEGRITY CHECKS")
        print("-" * 55)
        any_warning = False
        for fam_name, model_tables in results.items():
            for model_name, table in model_tables.items():
                warns = validate_labels(table, fam_name, model_name)
                if warns:
                    any_warning = True
                    for w in warns:
                        print(f"  WARN  {w}")
                else:
                    print(f"  OK    [{fam_name}/{model_name}]")
        if not any_warning:
            print("All tables passed integrity checks.")
        print()
