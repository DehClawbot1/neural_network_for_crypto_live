from __future__ import annotations

import math
import re
from typing import Mapping


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
    signal_label = str(source.get("signal_label", "") or "").strip().upper()
    confidence = safe_float(source.get("confidence_at_entry", source.get("confidence")), default=0.0)
    model_family = str(source.get("entry_model_family", "") or "").strip()
    model_version = str(source.get("entry_model_version", "") or "").strip()
    return signal_label not in {"", "UNKNOWN"} and confidence > 0 and bool(model_family) and bool(model_version)


def learning_eligible(source: Mapping | None) -> bool:
    source = source or {}
    if str(source.get("learning_eligible", "")).strip().lower() in {"true", "1", "yes"}:
        return True
    if str(source.get("operational_close_flag", "")).strip().lower() in {"true", "1", "yes"}:
        return False
    reason = source.get("close_reason")
    return (not is_operational_close_reason(reason)) and entry_context_complete(source)


def build_quality_context(source: Mapping | None) -> dict:
    source = source or {}
    family = classify_exit_reason_family(source.get("close_reason"))
    context = {
        "market_family": infer_market_family(source.get("market_slug"), source.get("market", source.get("market_title"))),
        "horizon_bucket": infer_horizon_bucket(source.get("market_slug"), source.get("market", source.get("market_title"))),
        "liquidity_bucket": infer_liquidity_bucket(source),
        "volatility_bucket": infer_volatility_bucket(source),
        "technical_regime_bucket": infer_technical_regime_bucket(source),
        "exit_reason_family": family,
        "operational_close_flag": family == "operational",
    }
    context["entry_context_complete"] = entry_context_complete({**source, **context})
    context["learning_eligible"] = (not context["operational_close_flag"]) and context["entry_context_complete"]
    return context
