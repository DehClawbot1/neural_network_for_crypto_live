from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping

import pandas as pd


def safe_float(value, default=0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def normalize_close_reason(reason) -> str:
    return str(reason or "").strip().lower()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _looks_like_hex_identifier(value) -> bool:
    text = str(value or "").strip().lower()
    return bool(re.fullmatch(r"0x[a-f0-9]{16,}", text))


def _normalize_outcome_side(value) -> str:
    side = str(value or "").strip().upper()
    if side.startswith("YES"):
        return "YES"
    if side.startswith("NO"):
        return "NO"
    if side.startswith("UP"):
        return "UP"
    if side.startswith("DOWN"):
        return "DOWN"
    return side


def _is_blank_signal_label(value) -> bool:
    return str(value or "").strip().upper() in {"", "UNKNOWN", "NAN", "NONE"}


def _is_numeric_signal_label(value) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text))


def _normalize_signal_label(value) -> str:
    return str(value or "").strip().upper()


def _confidence_value(source: Mapping | None) -> float:
    source = source or {}
    return safe_float(source.get("confidence_at_entry", source.get("confidence")), default=0.0)


def is_reconciliation_close(source: Mapping | None) -> bool:
    source = source or {}
    explicit = str(source.get("reconciliation_close_flag", source.get("is_reconciliation_close", ""))).strip().lower()
    if explicit in {"true", "1", "yes"}:
        return True
    reason_norm = normalize_close_reason(source.get("close_reason"))
    if reason_norm in {"external_manual_close", "exchange_reconciliation_empty_streak"}:
        return True
    lifecycle_source = str(source.get("lifecycle_source", "") or "").strip().lower()
    if "reconcile" in lifecycle_source or "dead_orderbook" in lifecycle_source:
        return True
    return False


def _build_source_lookup(source_df: pd.DataFrame, timestamp_col: str) -> dict:
    if source_df.empty:
        return {}
    work = source_df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    work["outcome_norm"] = work.get("outcome_side", pd.Series("", index=work.index)).map(_normalize_outcome_side)
    lookup = {}
    for key, group in work.groupby(["token_id", "condition_id", "outcome_norm"], dropna=False):
        lookup[tuple("" if pd.isna(v) else str(v) for v in key)] = group.sort_values(timestamp_col).reset_index(drop=True)
    return lookup


def _pick_nearest_entry(group: pd.DataFrame, opened_at, timestamp_col: str) -> Mapping | None:
    if group is None or group.empty:
        return None
    if pd.isna(opened_at) or timestamp_col not in group.columns:
        return group.iloc[-1].to_dict()
    timed = group.copy()
    timed["_delta_seconds"] = (timed[timestamp_col] - opened_at).abs().dt.total_seconds()
    timed = timed.sort_values(["_delta_seconds", timestamp_col], na_position="last")
    if timed.empty:
        return None
    return timed.iloc[0].to_dict()


def repair_closed_positions_frame(frame: pd.DataFrame | None, logs_dir="logs") -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()

    work = frame.copy()
    logs_path = Path(logs_dir)
    if "market_slug" not in work.columns:
        work["market_slug"] = None
    if "market_title" not in work.columns:
        work["market_title"] = work.get("market")
    if "confidence" not in work.columns:
        work["confidence"] = 0.0
    if "confidence_at_entry" not in work.columns:
        work["confidence_at_entry"] = work.get("confidence", 0.0)

    markets_df = _safe_read_csv(logs_path / "markets.csv")
    market_map = {}
    if not markets_df.empty and "condition_id" in markets_df.columns:
        for _, row in markets_df.iterrows():
            condition_id = str(row.get("condition_id") or "").strip()
            if not condition_id:
                continue
            title = str(row.get("market_title") or row.get("question") or "").strip()
            slug = str(row.get("slug") or "").strip()
            if title or slug:
                market_map[condition_id] = {"title": title, "slug": slug}

    work["opened_at"] = work.get("opened_at")
    opened_at_ts = pd.to_datetime(work.get("opened_at"), utc=True, errors="coerce")
    work["outcome_norm"] = work.get("outcome_side", pd.Series("", index=work.index)).map(_normalize_outcome_side)

    signals_lookup = _build_source_lookup(_safe_read_csv(logs_path / "signals.csv"), "timestamp")
    execution_lookup = _build_source_lookup(_safe_read_csv(logs_path / "execution_log.csv"), "timestamp")

    for idx, row in work.iterrows():
        condition_id = str(row.get("condition_id") or "").strip()
        market_meta = market_map.get(condition_id, {})
        current_market = str(row.get("market") or "").strip()
        current_title = str(row.get("market_title") or "").strip()
        if market_meta.get("title"):
            if not current_title or _looks_like_hex_identifier(current_title):
                work.at[idx, "market_title"] = market_meta["title"]
            if not current_market or _looks_like_hex_identifier(current_market):
                work.at[idx, "market"] = market_meta["title"]
        if market_meta.get("slug") and not str(row.get("market_slug") or "").strip():
            work.at[idx, "market_slug"] = market_meta["slug"]

        lookup_key = (
            str(row.get("token_id") or ""),
            str(row.get("condition_id") or ""),
            str(row.get("outcome_norm") or ""),
        )
        opened_at = opened_at_ts.iloc[idx] if idx < len(opened_at_ts.index) else pd.NaT
        signal_entry = _pick_nearest_entry(signals_lookup.get(lookup_key), opened_at, "timestamp")
        execution_entry = _pick_nearest_entry(execution_lookup.get(lookup_key), opened_at, "timestamp")

        if _is_blank_signal_label(row.get("signal_label")):
            for source in (signal_entry, execution_entry):
                if source and not _is_blank_signal_label(source.get("signal_label")):
                    work.at[idx, "signal_label"] = source.get("signal_label")
                    break

        if _confidence_value(row.to_dict()) <= 0:
            for source in (signal_entry, execution_entry):
                confidence = _confidence_value(source)
                if confidence > 0:
                    work.at[idx, "confidence"] = confidence
                    work.at[idx, "confidence_at_entry"] = confidence
                    break

        for source in (signal_entry, execution_entry):
            if not source:
                continue
            source_title = str(source.get("market_title") or source.get("market") or "").strip()
            if source_title:
                if not str(work.at[idx, "market_title"] or "").strip() or _looks_like_hex_identifier(work.at[idx, "market_title"]):
                    work.at[idx, "market_title"] = source_title
                if not str(work.at[idx, "market"] or "").strip() or _looks_like_hex_identifier(work.at[idx, "market"]):
                    work.at[idx, "market"] = source_title
            source_slug = str(source.get("market_slug") or "").strip()
            if source_slug and not str(work.at[idx, "market_slug"] or "").strip():
                work.at[idx, "market_slug"] = source_slug

    return work.drop(columns=["outcome_norm"], errors="ignore")


def infer_signal_label(source: Mapping | None) -> str:
    source = source or {}
    signal_label = _normalize_signal_label(source.get("signal_label", ""))
    if signal_label not in {"", "UNKNOWN", "NAN", "NONE"} and not _is_numeric_signal_label(signal_label):
        return signal_label
    market_family = infer_market_family(source.get("market_slug"), source.get("market", source.get("market_title")))
    horizon_bucket = infer_horizon_bucket(source.get("market_slug"), source.get("market", source.get("market_title")))
    if market_family == "other" and horizon_bucket == "unknown":
        return "UNKNOWN"
    family_label = market_family.upper()
    horizon_label = horizon_bucket.upper()
    return f"LEGACY_{family_label}_{horizon_label}"


def resolve_entry_signal_label(source: Mapping | None) -> str:
    source = source or {}
    signal_label = _normalize_signal_label(source.get("signal_label", ""))
    if signal_label not in {"", "UNKNOWN", "NAN", "NONE"} and not _is_numeric_signal_label(signal_label):
        return signal_label
    market_family = infer_market_family(source.get("market_slug"), source.get("market", source.get("market_title")))
    horizon_bucket = infer_horizon_bucket(source.get("market_slug"), source.get("market", source.get("market_title")))
    if market_family == "other" and horizon_bucket == "unknown":
        return "UNKNOWN"
    family_label = market_family.upper()
    horizon_label = horizon_bucket.upper()
    return f"CONTEXT_{family_label}_{horizon_label}"


def classify_exit_reason_family(reason) -> str:
    reason_norm = normalize_close_reason(reason)
    if not reason_norm:
        return "unknown"
    if reason_norm in {"external_manual_close", "exchange_reconciliation_empty_streak", "local_dust_pruned"}:
        return "operational"
    if "manual" in reason_norm or "reconciliation" in reason_norm:
        return "operational"
    if reason_norm in {"stop_loss", "trajectory_panic_exit"}:
        return "hard_stop"
    if reason_norm.startswith("technical_") or reason_norm in {"trajectory_reversal_exit", "ai_close_long"}:
        return "technical_invalidation"
    if reason_norm == "time_stop":
        return "time_stop"
    if reason_norm in {"take_profit_roi", "take_profit_price_move", "take_profit_model_target", "trailing_stop", "trajectory_profit_lock", "trajectory_liquidity_stress"}:
        return "profit_take"
    if reason_norm.startswith("rl_"):
        return "rl_discretionary"
    return "other"


def is_operational_close_reason(reason) -> bool:
    return classify_exit_reason_family(reason) == "operational"


def infer_market_family(market_slug=None, market=None) -> str:
    slug = str(market_slug or "").strip().lower()
    title = str(market or "").strip().lower()
    combined = f"{slug} {title}"
    if "btc-updown" in combined or "bitcoin up or down" in combined:
        return "btc_directional_intraday"
    if "will the price of bitcoin be above" in combined or "bitcoin-above" in combined:
        return "btc_price_threshold"
    if "dip to" in combined:
        return "btc_downside_threshold"
    if "bitcoin" in combined or "btc" in combined:
        return "btc_other"
    return "other"


def infer_horizon_bucket(market_slug=None, market=None) -> str:
    slug = str(market_slug or "").strip().lower()
    title = str(market or "").strip().lower()
    for marker, bucket in (
        ("5m", "intraday_5m"),
        ("15m", "intraday_15m"),
        ("30m", "intraday_30m"),
        ("1h", "intraday_1h"),
        ("4h", "intraday_4h"),
    ):
        if marker in slug or marker in title:
            return bucket
    if "april" in title or re.search(r"\bon [a-z]+ \d{1,2}\b", title):
        return "dated_daily"
    return "unknown"


def infer_liquidity_bucket(source: Mapping | None) -> str:
    source = source or {}
    liquidity_score = safe_float(
        source.get("liquidity_score", source.get("liquidity_depth_score", source.get("market_liquidity_score"))),
        default=float("nan"),
    )
    if math.isfinite(liquidity_score):
        if liquidity_score >= 0.75:
            return "high"
        if liquidity_score >= 0.45:
            return "medium"
        return "low"
    size_usdc = safe_float(source.get("size_usdc", source.get("proposed_size_usdc")), default=0.0)
    if size_usdc >= 25:
        return "high"
    if size_usdc >= 5:
        return "medium"
    return "low"


def infer_volatility_bucket(source: Mapping | None) -> str:
    source = source or {}
    volatility_risk = safe_float(source.get("volatility_risk", source.get("volatility_short")), default=float("nan"))
    if math.isfinite(volatility_risk):
        if volatility_risk >= 0.66:
            return "high"
        if volatility_risk >= 0.33:
            return "medium"
        return "low"
    return "unknown"


def infer_technical_regime_bucket(source: Mapping | None) -> str:
    source = source or {}
    bias = str(source.get("btc_trend_bias", source.get("entry_btc_trend_bias", "NEUTRAL")) or "NEUTRAL").strip().upper()
    alligator = str(source.get("alligator_alignment", source.get("entry_alligator_alignment", "NEUTRAL")) or "NEUTRAL").strip().upper()
    adx_value = safe_float(source.get("adx_value", source.get("entry_adx_value")), default=0.0)
    adx_threshold = safe_float(source.get("adx_threshold", source.get("entry_adx_threshold")), default=0.0)
    trending = adx_value >= max(0.0, adx_threshold)
    if bias == "LONG" and alligator == "BULLISH":
        return "bull_trending" if trending else "bull_soft"
    if bias == "SHORT" and alligator == "BEARISH":
        return "bear_trending" if trending else "bear_soft"
    if alligator == "BULLISH":
        return "bull_mixed"
    if alligator == "BEARISH":
        return "bear_mixed"
    return "neutral"


def entry_context_complete(source: Mapping | None) -> bool:
    source = source or {}
    signal_label = infer_signal_label(source)
    market_family = infer_market_family(source.get("market_slug"), source.get("market", source.get("market_title")))
    horizon_bucket = infer_horizon_bucket(source.get("market_slug"), source.get("market", source.get("market_title")))
    model_family = str(source.get("entry_model_family", "") or "").strip()
    has_trade_identity = any(
        str(source.get(key, "") or "").strip()
        for key in ("market", "market_title", "market_slug", "token_id", "condition_id")
    )
    has_signal_context = signal_label not in {"", "UNKNOWN"}
    has_market_context = market_family != "other" or horizon_bucket != "unknown"
    has_model_context = bool(model_family)
    return bool(has_trade_identity and (has_signal_context or has_market_context or has_model_context))


def learning_eligible(source: Mapping | None) -> bool:
    source = source or {}
    if str(source.get("learning_eligible", "")).strip().lower() in {"true", "1", "yes"}:
        return True
    if is_reconciliation_close(source):
        return False
    if str(source.get("operational_close_flag", "")).strip().lower() in {"true", "1", "yes"}:
        return False
    reason = source.get("close_reason")
    return (not is_operational_close_reason(reason)) and entry_context_complete(source)


def build_quality_context(source: Mapping | None) -> dict:
    source = source or {}
    family = classify_exit_reason_family(source.get("close_reason"))
    reconciliation_close_flag = is_reconciliation_close(source)
    context = {
        "signal_label": infer_signal_label(source),
        "market_family": infer_market_family(source.get("market_slug"), source.get("market", source.get("market_title"))),
        "horizon_bucket": infer_horizon_bucket(source.get("market_slug"), source.get("market", source.get("market_title"))),
        "liquidity_bucket": infer_liquidity_bucket(source),
        "volatility_bucket": infer_volatility_bucket(source),
        "technical_regime_bucket": infer_technical_regime_bucket(source),
        "exit_reason_family": family,
        "operational_close_flag": (family == "operational") and not reconciliation_close_flag,
        "reconciliation_close_flag": reconciliation_close_flag,
    }
    context["entry_context_complete"] = entry_context_complete({**source, **context})
    context["learning_eligible"] = (
        (not context["operational_close_flag"])
        and (not context["reconciliation_close_flag"])
        and context["entry_context_complete"]
    )
    return context


def enrich_quality_frame(frame: pd.DataFrame | None, logs_dir=None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()

    work = repair_closed_positions_frame(frame, logs_dir=logs_dir) if logs_dir else frame.copy()
    for field, default in (
        ("market_family", None),
        ("horizon_bucket", None),
        ("liquidity_bucket", None),
        ("volatility_bucket", None),
        ("technical_regime_bucket", None),
        ("exit_reason_family", None),
        ("operational_close_flag", None),
        ("reconciliation_close_flag", None),
        ("entry_context_complete", None),
        ("learning_eligible", None),
    ):
        if field not in work.columns:
            work[field] = default
        work[field] = work[field].astype("object")

    for idx, row in work.iterrows():
        enriched = build_quality_context(row.to_dict())
        for key, value in enriched.items():
            current = work.at[idx, key]
            is_blank = pd.isna(current) or current in [None, "", "nan", "None", False]
            if key == "signal_label":
                is_blank = is_blank or str(current).strip().upper() == "UNKNOWN"
            if is_blank:
                work.at[idx, key] = value

    bool_fields = ("operational_close_flag", "reconciliation_close_flag", "entry_context_complete", "learning_eligible")
    for field in bool_fields:
        work[field] = (
            work[field]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
    return work
