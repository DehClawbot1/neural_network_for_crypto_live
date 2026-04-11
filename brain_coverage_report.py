from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from brain_paths import BTC_FAMILY, BrainContext, resolve_brain_context


CORE_LIVE_SOURCE_FEATURES = [
    "btc_live_price_kalman",
    "btc_live_index_price_kalman",
    "btc_live_return_15m_kalman",
]

CORE_REGIME_SOURCE_FEATURES = [
    "btc_market_regime_score",
    "btc_market_regime_weight_stage1",
]

CORE_TRAINING_FEATURES = [
    *CORE_LIVE_SOURCE_FEATURES,
    *CORE_REGIME_SOURCE_FEATURES,
]

OPTIONAL_CONTEXT_FEATURES = [
    "open_positions_count",
    "open_positions_unrealized_pnl_pct_total",
    "execution_quality_score",
]

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "captured_at",
    "recorded_at",
    "observed_at",
    "created_at",
    "updated_at",
    "closed_at",
]

DEFAULT_MIN_POST_ROLLOUT_CONTRACT_ROWS = 200
DEFAULT_MIN_POST_ROLLOUT_SEQUENCE_ROWS = 30
DEFAULT_MIN_CORE_FEATURE_COVERAGE_RATIO = 0.80


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _extract_timestamp_series(df: pd.DataFrame) -> pd.Series:
    for column in TIMESTAMP_CANDIDATES:
        if column in df.columns:
            return pd.to_datetime(df[column], errors="coerce", utc=True)
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")


def _first_seen_times(df: pd.DataFrame, feature_names: list[str]) -> dict[str, pd.Timestamp]:
    if df.empty:
        return {}
    timestamps = _extract_timestamp_series(df)
    seen: dict[str, pd.Timestamp] = {}
    for feature in feature_names:
        if feature not in df.columns:
            continue
        mask = df[feature].notna() & timestamps.notna()
        if not bool(mask.any()):
            continue
        seen[feature] = timestamps.loc[mask].min()
    return seen


def _group_rollout_time(first_seen: dict[str, pd.Timestamp], required_features: list[str]) -> pd.Timestamp | None:
    values = [first_seen.get(feature) for feature in required_features]
    if any(value is None or pd.isna(value) for value in values):
        return None
    return max(values)


def _slice_post_rollout(df: pd.DataFrame, rollout_from: pd.Timestamp | None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if rollout_from is None or pd.isna(rollout_from):
        return df.iloc[0:0].copy()
    timestamps = _extract_timestamp_series(df)
    if timestamps.isna().all():
        return df.iloc[0:0].copy()
    return df.loc[timestamps >= rollout_from].copy()


def _coverage_ratio(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return float(df[column].notna().mean())


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _iso_or_blank(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _feature_report_rows(
    contract_post_rollout: pd.DataFrame,
    *,
    live_first_seen: dict[str, pd.Timestamp],
    regime_first_seen: dict[str, pd.Timestamp],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in CORE_TRAINING_FEATURES + OPTIONAL_CONTEXT_FEATURES:
        source_group = (
            "live_source"
            if feature in CORE_LIVE_SOURCE_FEATURES
            else "regime_source"
            if feature in CORE_REGIME_SOURCE_FEATURES
            else "optional_context"
        )
        first_seen = live_first_seen.get(feature) or regime_first_seen.get(feature)
        total_rows = int(len(contract_post_rollout.index))
        non_null_rows = int(contract_post_rollout[feature].notna().sum()) if feature in contract_post_rollout.columns else 0
        coverage_ratio = (float(non_null_rows / total_rows) if total_rows else 0.0)
        status = "healthy" if coverage_ratio >= 0.80 else "sparse" if coverage_ratio > 0 else "missing"
        rows.append(
            {
                "feature": feature,
                "source_group": source_group,
                "first_seen_at": _iso_or_blank(first_seen),
                "post_rollout_rows": total_rows,
                "post_rollout_non_null_rows": non_null_rows,
                "coverage_ratio": round(coverage_ratio, 4),
                "status": status,
            }
        )
    return rows


def build_btc_brain_coverage_report(
    *,
    brain_context: BrainContext | None = None,
    shared_logs_dir: str | Path = "logs",
    shared_weights_dir: str | Path = "weights",
    persist: bool = True,
) -> dict[str, Any]:
    context = brain_context or resolve_brain_context(
        BTC_FAMILY,
        shared_logs_dir=shared_logs_dir,
        shared_weights_dir=shared_weights_dir,
    )
    if context.market_family != BTC_FAMILY:
        raise ValueError("build_btc_brain_coverage_report only supports the BTC brain.")

    shared_logs_dir = Path(context.shared_logs_dir)
    contract_targets = _safe_read_csv(context.logs_dir / "contract_targets.csv")
    historical_dataset = _safe_read_csv(context.logs_dir / "historical_dataset.csv")
    sequence_dataset = _safe_read_csv(context.logs_dir / "sequence_dataset.csv")
    btc_live_snapshot = _safe_read_csv(shared_logs_dir / "btc_live_snapshot.csv")
    technical_regime_snapshot = _safe_read_csv(shared_logs_dir / "technical_regime_snapshot.csv")

    live_first_seen = _first_seen_times(btc_live_snapshot, CORE_LIVE_SOURCE_FEATURES)
    regime_first_seen = _first_seen_times(technical_regime_snapshot, CORE_REGIME_SOURCE_FEATURES)

    live_rollout_from = _group_rollout_time(live_first_seen, CORE_LIVE_SOURCE_FEATURES)
    regime_rollout_from = _group_rollout_time(regime_first_seen, CORE_REGIME_SOURCE_FEATURES)
    rollout_groups_ready = live_rollout_from is not None and regime_rollout_from is not None
    rollout_ready_from = (
        max(live_rollout_from, regime_rollout_from) if rollout_groups_ready else None
    )

    contract_post_rollout = _slice_post_rollout(contract_targets, rollout_ready_from)
    historical_post_rollout = _slice_post_rollout(historical_dataset, rollout_ready_from)
    sequence_post_rollout = _slice_post_rollout(sequence_dataset, rollout_ready_from)

    min_contract_rows = max(1, _env_int("BTC_BRAIN_MIN_POST_ROLLOUT_CONTRACT_ROWS", DEFAULT_MIN_POST_ROLLOUT_CONTRACT_ROWS))
    min_sequence_rows = max(1, _env_int("BTC_BRAIN_MIN_POST_ROLLOUT_SEQUENCE_ROWS", DEFAULT_MIN_POST_ROLLOUT_SEQUENCE_ROWS))
    min_core_coverage = min(1.0, max(0.0, _env_float("BTC_BRAIN_MIN_CORE_FEATURE_COVERAGE_RATIO", DEFAULT_MIN_CORE_FEATURE_COVERAGE_RATIO)))

    core_feature_coverages = {
        feature: _coverage_ratio(contract_post_rollout, feature)
        for feature in CORE_TRAINING_FEATURES
    }
    optional_feature_coverages = {
        feature: _coverage_ratio(contract_post_rollout, feature)
        for feature in OPTIONAL_CONTEXT_FEATURES
    }

    core_feature_coverage_ratio = (
        float(sum(core_feature_coverages.values()) / len(core_feature_coverages))
        if core_feature_coverages
        else 0.0
    )
    optional_context_coverage_ratio = (
        float(sum(optional_feature_coverages.values()) / len(optional_feature_coverages))
        if optional_feature_coverages
        else 0.0
    )

    missing_groups: list[str] = []
    if live_rollout_from is None:
        missing_groups.append("live_kalman_source")
    if regime_rollout_from is None:
        missing_groups.append("market_regime_source")

    not_ready_reasons: list[str] = []
    if missing_groups:
        not_ready_reasons.append(f"awaiting source activation: {', '.join(missing_groups)}")

    contract_gap = max(0, min_contract_rows - int(len(contract_post_rollout.index)))
    if contract_gap > 0:
        not_ready_reasons.append(f"collect {contract_gap} more post-rollout contract rows")

    sequence_gap = max(0, min_sequence_rows - int(len(sequence_post_rollout.index)))
    if sequence_gap > 0:
        not_ready_reasons.append(f"collect {sequence_gap} more post-rollout sequence rows")

    if core_feature_coverage_ratio < min_core_coverage:
        not_ready_reasons.append(
            f"core feature coverage {core_feature_coverage_ratio:.0%} < required {min_core_coverage:.0%}"
        )

    retrain_confident_ready = len(not_ready_reasons) == 0
    readiness_reason = (
        "btc brain has enough post-rollout coverage"
        if retrain_confident_ready
        else " | ".join(not_ready_reasons)
    )

    feature_rows = _feature_report_rows(
        contract_post_rollout,
        live_first_seen=live_first_seen,
        regime_first_seen=regime_first_seen,
    )
    feature_report = pd.DataFrame(feature_rows)

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "brain_id": context.brain_id,
        "market_family": context.market_family,
        "rollout_groups_ready": rollout_groups_ready,
        "live_rollout_from": _iso_or_blank(live_rollout_from),
        "regime_rollout_from": _iso_or_blank(regime_rollout_from),
        "rollout_ready_from": _iso_or_blank(rollout_ready_from),
        "post_rollout_contract_rows": int(len(contract_post_rollout.index)),
        "post_rollout_historical_rows": int(len(historical_post_rollout.index)),
        "post_rollout_sequence_rows": int(len(sequence_post_rollout.index)),
        "min_post_rollout_contract_rows": int(min_contract_rows),
        "min_post_rollout_sequence_rows": int(min_sequence_rows),
        "core_feature_coverage_ratio": round(core_feature_coverage_ratio, 4),
        "min_core_feature_coverage_ratio": round(min_core_coverage, 4),
        "optional_context_coverage_ratio": round(optional_context_coverage_ratio, 4),
        "retrain_confident_ready": retrain_confident_ready,
        "readiness_reason": readiness_reason,
        "missing_source_groups": ",".join(missing_groups),
    }

    if persist:
        context.logs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(context.logs_dir / "brain_coverage_report.csv", index=False)
        feature_report.to_csv(context.logs_dir / "brain_coverage_feature_report.csv", index=False)
        (context.logs_dir / "brain_coverage_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    return {
        **summary,
        "feature_report": feature_report,
        "core_feature_coverages": core_feature_coverages,
        "optional_feature_coverages": optional_feature_coverages,
    }


def format_btc_brain_coverage_line(summary: dict[str, Any]) -> str:
    rollout = summary.get("rollout_ready_from") or "not-active"
    contract_rows = int(summary.get("post_rollout_contract_rows", 0) or 0)
    contract_target = int(summary.get("min_post_rollout_contract_rows", 0) or 0)
    sequence_rows = int(summary.get("post_rollout_sequence_rows", 0) or 0)
    sequence_target = int(summary.get("min_post_rollout_sequence_rows", 0) or 0)
    coverage = float(summary.get("core_feature_coverage_ratio", 0.0) or 0.0)
    coverage_target = float(summary.get("min_core_feature_coverage_ratio", 0.0) or 0.0)
    ready = "yes" if bool(summary.get("retrain_confident_ready")) else "no"
    reason = str(summary.get("readiness_reason") or "").strip()
    reason_suffix = f" | reason={reason}" if reason and ready == "no" else ""
    return (
        "BTC brain coverage: "
        f"rollout_from={rollout} | "
        f"contract_rows={contract_rows}/{contract_target} | "
        f"sequence_rows={sequence_rows}/{sequence_target} | "
        f"core_coverage={coverage:.0%}/{coverage_target:.0%} | "
        f"ready={ready}"
        f"{reason_suffix}"
    )
