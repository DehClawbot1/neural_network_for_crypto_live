"""Feature lineage report generator.

Produces ``logs/feature_lineage_report.csv`` listing every feature with:
  source -> runtime_use -> training_use -> target_use

This makes it trivial to audit which features flow where.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from model_feature_catalog import (
    DEFAULT_TABULAR_FEATURE_COLUMNS,
    SEQUENCE_BASE_COLUMNS,
    TRAINING_FEATURE_FAMILIES,
)
from feature_treatment_policy import FEATURE_TREATMENT, get_treatment

logger = logging.getLogger(__name__)


# ── source mapping ──────────────────────────────────────────────────

_SOURCE_MAP = {
    # wallet_copy
    "trader_win_rate": "feature_builder",
    "wallet_trade_count_30d": "feature_builder",
    "wallet_alpha_30d": "feature_builder / wallet_alpha_history.csv",
    "wallet_avg_forward_return_15m": "feature_builder / wallet_alpha_history.csv",
    "wallet_signal_precision_tp": "feature_builder",
    "wallet_recent_streak": "feature_builder",
    "normalized_trade_size": "feature_builder",
    "whale_pressure": "feature_builder",
    # market_microstructure
    "current_price": "feature_builder / signal",
    "spread": "feature_builder / market_monitor",
    "time_left": "feature_builder / market_monitor",
    "liquidity_score": "feature_builder / market_monitor",
    "volume_score": "feature_builder / market_monitor",
    "probability_momentum": "feature_builder",
    "volatility_score": "feature_builder",
    "market_structure_score": "feature_builder",
    # onchain_network
    "btc_fee_pressure_score": "btc_network_monitor",
    "btc_mempool_congestion_score": "btc_network_monitor",
    "btc_network_activity_score": "btc_network_monitor",
    "btc_network_stress_score": "btc_network_monitor",
    # btc_spot_regime
    "trend_score": "technical_analyzer",
    "btc_spot_return_5m": "btc_live_price_tracker",
    "btc_spot_return_15m": "btc_live_price_tracker",
    "btc_realized_vol_15m": "technical_analyzer",
    "btc_volume_proxy": "btc_live_price_tracker",
    "btc_atr_pct_15m": "technical_analyzer",
    "btc_realized_vol_1h": "technical_analyzer",
    "btc_realized_vol_4h": "technical_analyzer",
    "btc_volatility_regime_score": "technical_analyzer",
    "btc_trend_persistence": "technical_analyzer",
    # btc_momentum_quality
    "btc_rsi_14": "technical_analyzer",
    "btc_rsi_distance_mid": "technical_analyzer",
    "btc_rsi_divergence_score": "technical_analyzer",
    "btc_macd": "technical_analyzer",
    "btc_macd_signal": "technical_analyzer",
    "btc_macd_hist": "technical_analyzer",
    "btc_macd_hist_slope": "technical_analyzer",
    "btc_momentum_confluence": "technical_analyzer",
    # btc_live_index — all from btc_live_price_tracker / kalman_feature_smoother
    **{f: "btc_live_price_tracker" for f in [
        "btc_live_price", "btc_live_spot_price", "btc_live_index_price",
        "btc_live_mark_price", "btc_live_price_kalman", "btc_live_spot_price_kalman",
        "btc_live_index_price_kalman", "btc_live_mark_price_kalman",
        "btc_live_funding_rate", "btc_live_source_quality_score",
        "btc_live_source_divergence_bps",
        "btc_live_spot_index_basis_bps", "btc_live_mark_index_basis_bps",
        "btc_live_mark_spot_basis_bps",
        "btc_live_spot_index_basis_bps_kalman", "btc_live_mark_index_basis_bps_kalman",
        "btc_live_mark_spot_basis_bps_kalman",
        "btc_live_return_1m", "btc_live_return_5m", "btc_live_return_15m",
        "btc_live_return_1h", "btc_live_return_1m_kalman", "btc_live_return_5m_kalman",
        "btc_live_return_15m_kalman", "btc_live_return_1h_kalman",
        "btc_live_volatility_proxy", "btc_live_confluence", "btc_live_confluence_kalman",
        "btc_live_index_ready", "btc_live_index_feed_available",
        "btc_live_mark_feed_available",
    ]},
    # btc_market_regime
    **{f: "btc_regime_router" for f in [
        "btc_market_regime_score", "btc_market_regime_trend_score",
        "btc_market_regime_volatility_score", "btc_market_regime_chaos_score",
        "btc_market_regime_stability_score", "btc_market_regime_is_calm",
        "btc_market_regime_is_trend", "btc_market_regime_is_volatile",
        "btc_market_regime_is_chaotic", "btc_market_regime_confidence_multiplier",
        "btc_market_regime_weight_legacy", "btc_market_regime_weight_stage1",
        "btc_market_regime_weight_stage2",
    ]},
    # btc_sentiment
    **{f: "sentiment_aggregator" for f in [
        "sentiment_score", "btc_funding_rate", "is_overheated_long",
        "fgi_value", "fgi_normalized", "fgi_extreme_fear", "fgi_extreme_greed",
        "fgi_contrarian", "fgi_momentum", "fgi_momentum_3d",
        "gtrends_bitcoin", "gtrends_zscore", "gtrends_spike", "gtrends_momentum",
        "twitter_sentiment", "twitter_post_count", "twitter_sentiment_pos",
        "twitter_sentiment_neg", "twitter_engagement_proxy", "twitter_sentiment_zscore",
        "twitter_bullish", "twitter_bearish", "twitter_sentiment_momentum",
        "reddit_sentiment", "reddit_post_count", "reddit_sentiment_pos",
        "reddit_sentiment_neg", "reddit_sentiment_zscore",
        "reddit_bullish", "reddit_bearish", "reddit_sentiment_momentum",
    ]},
    # portfolio_context
    **{f: "live_position_book" for f in [
        "open_positions_count", "open_positions_negotiated_value_total",
        "open_positions_max_payout_total", "open_positions_current_value_total",
        "open_positions_unrealized_pnl_total", "open_positions_unrealized_pnl_pct_total",
        "open_positions_avg_to_now_price_change_pct_mean",
        "open_positions_avg_to_now_price_change_pct_min",
        "open_positions_avg_to_now_price_change_pct_max",
        "open_positions_winner_count", "open_positions_loser_count",
    ]},
    # weather families
    **{f: "weather_signal_evaluator" for f in [
        "wallet_temp_hit_rate_90d", "wallet_temp_realized_pnl_90d",
        "wallet_region_score", "wallet_temp_range_skill", "wallet_temp_threshold_skill",
        "wallet_quality_score", "wallet_state_confidence", "wallet_state_freshness_score",
        "wallet_size_change_score", "wallet_agreement_score",
        "execution_quality_score",
        "forecast_p_hit_interval", "forecast_margin_to_lower_c",
        "forecast_margin_to_upper_c", "forecast_uncertainty_c", "forecast_drift_c",
        "weather_fair_probability_yes", "weather_fair_probability_side",
        "weather_market_probability", "weather_forecast_edge",
        "weather_forecast_margin_score", "weather_forecast_stability_score",
    ]},
}


def _runtime_use(feature: str) -> str:
    """Classify how a feature is used at runtime."""
    t = get_treatment(feature)
    if t.kind == "boolean":
        return "binary_flag"
    if t.scope == "tree":
        return "stage1_tree_input"
    if t.scope == "nn":
        return "stage2_nn_input"
    return "all_model_input"


def _training_use(feature: str) -> str:
    """Classify how a feature is used during training."""
    t = get_treatment(feature)
    parts = []
    if t.scope in ("all", "tree"):
        parts.append("stage1_tabular")
    if t.scope in ("all", "nn"):
        parts.append("stage2_temporal")
    if feature in SEQUENCE_BASE_COLUMNS:
        parts.append("sequence_base")
    return " + ".join(parts) if parts else "unused"


def _target_use(feature: str) -> str:
    """Classify if a feature is also used as a target or derived from targets."""
    targets = {
        "tp_before_sl_60m": "classification_target",
        "forward_return_15m": "regression_target",
        "target_up": "binary_target",
        "mfe_60m": "auxiliary_target",
        "mae_60m": "auxiliary_target",
    }
    return targets.get(feature, "feature_only")


def build_lineage_report() -> pd.DataFrame:
    """Build the feature lineage report as a DataFrame."""
    rows = []
    for family_name, features in TRAINING_FEATURE_FAMILIES.items():
        for feature in features:
            t = get_treatment(feature)
            rows.append({
                "feature": feature,
                "family": family_name,
                "source": _SOURCE_MAP.get(feature, "unknown"),
                "treatment_kind": t.kind,
                "treatment_scope": t.scope,
                "runtime_use": _runtime_use(feature),
                "training_use": _training_use(feature),
                "target_use": _target_use(feature),
            })
    return pd.DataFrame(rows)


def write_lineage_report(logs_dir: str = "logs") -> Path:
    """Write ``logs/feature_lineage_report.csv``."""
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    out = logs_path / "feature_lineage_report.csv"
    df = build_lineage_report()
    df.to_csv(out, index=False)
    logger.info("Feature lineage report written to %s (%d features)", out, len(df))
    return out
