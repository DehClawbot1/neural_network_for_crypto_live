from __future__ import annotations

from collections import OrderedDict

WALLET_FEATURES = [
    "trader_win_rate",
    "wallet_trade_count_30d",
    "wallet_alpha_30d",
    "wallet_avg_forward_return_15m",
    "wallet_signal_precision_tp",
    "wallet_recent_streak",
    "normalized_trade_size",
    "whale_pressure",
]

MARKET_STRUCTURE_FEATURES = [
    "current_price",
    "spread",
    "time_left",
    "liquidity_score",
    "volume_score",
    "probability_momentum",
    "volatility_score",
    "market_structure_score",
]

ONCHAIN_FEATURES = [
    "btc_fee_pressure_score",
    "btc_mempool_congestion_score",
    "btc_network_activity_score",
    "btc_network_stress_score",
]

BTC_SPOT_REGIME_FEATURES = [
    "trend_score",
    "btc_spot_return_5m",
    "btc_spot_return_15m",
    "btc_realized_vol_15m",
    "btc_volume_proxy",
    "btc_atr_pct_15m",
    "btc_realized_vol_1h",
    "btc_realized_vol_4h",
    "btc_volatility_regime_score",
    "btc_trend_persistence",
]

BTC_MOMENTUM_QUALITY_FEATURES = [
    "btc_rsi_14",
    "btc_rsi_distance_mid",
    "btc_rsi_divergence_score",
    "btc_macd",
    "btc_macd_signal",
    "btc_macd_hist",
    "btc_macd_hist_slope",
    "btc_momentum_confluence",
]

BTC_LIVE_INDEX_FEATURES = [
    "btc_live_price",
    "btc_live_spot_price",
    "btc_live_index_price",
    "btc_live_mark_price",
    "btc_live_price_kalman",
    "btc_live_spot_price_kalman",
    "btc_live_index_price_kalman",
    "btc_live_mark_price_kalman",
    "btc_live_funding_rate",
    "btc_live_source_quality_score",
    "btc_live_source_divergence_bps",
    "btc_live_spot_index_basis_bps",
    "btc_live_mark_index_basis_bps",
    "btc_live_mark_spot_basis_bps",
    "btc_live_spot_index_basis_bps_kalman",
    "btc_live_mark_index_basis_bps_kalman",
    "btc_live_mark_spot_basis_bps_kalman",
    "btc_live_return_1m",
    "btc_live_return_5m",
    "btc_live_return_15m",
    "btc_live_return_1h",
    "btc_live_return_1m_kalman",
    "btc_live_return_5m_kalman",
    "btc_live_return_15m_kalman",
    "btc_live_return_1h_kalman",
    "btc_live_volatility_proxy",
    "btc_live_confluence",
    "btc_live_confluence_kalman",
    "btc_live_index_ready",
    "btc_live_index_feed_available",
    "btc_live_mark_feed_available",
]

BTC_MARKET_REGIME_FEATURES = [
    "btc_market_regime_score",
    "btc_market_regime_trend_score",
    "btc_market_regime_volatility_score",
    "btc_market_regime_chaos_score",
    "btc_market_regime_stability_score",
    "btc_market_regime_is_calm",
    "btc_market_regime_is_trend",
    "btc_market_regime_is_volatile",
    "btc_market_regime_is_chaotic",
    "btc_market_regime_confidence_multiplier",
    "btc_market_regime_weight_legacy",
    "btc_market_regime_weight_stage1",
    "btc_market_regime_weight_stage2",
]

BTC_SENTIMENT_FEATURES = [
    "sentiment_score",
    "btc_funding_rate",
    "is_overheated_long",
    "fgi_value",
    "fgi_normalized",
    "fgi_extreme_fear",
    "fgi_extreme_greed",
    "fgi_contrarian",
    "fgi_momentum",
    "fgi_momentum_3d",
    "gtrends_bitcoin",
    "gtrends_zscore",
    "gtrends_spike",
    "gtrends_momentum",
    "twitter_sentiment",
    "twitter_post_count",
    "twitter_sentiment_pos",
    "twitter_sentiment_neg",
    "twitter_engagement_proxy",
    "twitter_sentiment_zscore",
    "twitter_bullish",
    "twitter_bearish",
    "twitter_sentiment_momentum",
    "reddit_sentiment",
    "reddit_post_count",
    "reddit_sentiment_pos",
    "reddit_sentiment_neg",
    "reddit_sentiment_zscore",
    "reddit_bullish",
    "reddit_bearish",
    "reddit_sentiment_momentum",
]

PORTFOLIO_CONTEXT_FEATURES = [
    "open_positions_count",
    "open_positions_negotiated_value_total",
    "open_positions_max_payout_total",
    "open_positions_current_value_total",
    "open_positions_unrealized_pnl_total",
    "open_positions_unrealized_pnl_pct_total",
    "open_positions_avg_to_now_price_change_pct_mean",
    "open_positions_avg_to_now_price_change_pct_min",
    "open_positions_avg_to_now_price_change_pct_max",
    "open_positions_winner_count",
    "open_positions_loser_count",
]

WEATHER_WALLET_COPY_FEATURES = [
    "wallet_temp_hit_rate_90d",
    "wallet_temp_realized_pnl_90d",
    "wallet_region_score",
    "wallet_temp_range_skill",
    "wallet_temp_threshold_skill",
    "wallet_quality_score",
    "wallet_state_confidence",
    "wallet_state_freshness_score",
    "wallet_size_change_score",
    "wallet_agreement_score",
]

WEATHER_MARKET_STRUCTURE_FEATURES = [
    "current_price",
    "spread",
    "time_left",
    "liquidity_score",
    "volume_score",
    "market_structure_score",
    "execution_quality_score",
]

WEATHER_FORECAST_EDGE_FEATURES = [
    "forecast_p_hit_interval",
    "forecast_margin_to_lower_c",
    "forecast_margin_to_upper_c",
    "forecast_uncertainty_c",
    "forecast_drift_c",
    "weather_fair_probability_yes",
    "weather_fair_probability_side",
    "weather_market_probability",
    "weather_forecast_edge",
    "weather_forecast_margin_score",
    "weather_forecast_stability_score",
]

TRAINING_FEATURE_FAMILIES = OrderedDict(
    [
        ("wallet_copy", WALLET_FEATURES),
        ("market_microstructure", MARKET_STRUCTURE_FEATURES),
        ("onchain_network", ONCHAIN_FEATURES),
        ("btc_spot_regime", BTC_SPOT_REGIME_FEATURES),
        ("btc_momentum_quality", BTC_MOMENTUM_QUALITY_FEATURES),
        ("btc_live_index", BTC_LIVE_INDEX_FEATURES),
        ("btc_market_regime", BTC_MARKET_REGIME_FEATURES),
        ("btc_sentiment", BTC_SENTIMENT_FEATURES),
        ("portfolio_context", PORTFOLIO_CONTEXT_FEATURES),
        ("weather_wallet_copy", WEATHER_WALLET_COPY_FEATURES),
        ("weather_market_structure", WEATHER_MARKET_STRUCTURE_FEATURES),
        ("weather_forecast_edge", WEATHER_FORECAST_EDGE_FEATURES),
    ]
)

DEFAULT_TABULAR_FEATURE_COLUMNS = [
    feature
    for family in TRAINING_FEATURE_FAMILIES.values()
    for feature in family
]

SEQUENCE_BASE_COLUMNS = [
    "entry_price",
    "wallet_trade_count_30d",
    "wallet_alpha_30d",
    "wallet_signal_precision_tp",
    "btc_fee_pressure_score",
    "btc_mempool_congestion_score",
    "btc_network_activity_score",
    "btc_network_stress_score",
    "btc_spot_return_5m",
    "btc_spot_return_15m",
    "btc_atr_pct_15m",
    "btc_realized_vol_1h",
    "btc_realized_vol_4h",
    "btc_volatility_regime_score",
    "btc_trend_persistence",
    "btc_rsi_14",
    "btc_rsi_distance_mid",
    "btc_rsi_divergence_score",
    "btc_macd_hist",
    "btc_macd_hist_slope",
    "btc_momentum_confluence",
    "btc_live_source_quality_score",
    "btc_live_source_divergence_bps",
    "btc_live_index_price",
    "btc_live_mark_price",
    "btc_live_index_price_kalman",
    "btc_live_mark_price_kalman",
    "btc_live_mark_index_basis_bps",
    "btc_live_mark_index_basis_bps_kalman",
    "btc_live_return_1m",
    "btc_live_return_5m",
    "btc_live_return_15m",
    "btc_live_return_1h",
    "btc_live_return_5m_kalman",
    "btc_live_return_15m_kalman",
    "btc_live_confluence",
    "btc_live_confluence_kalman",
    "btc_market_regime_score",
    "btc_market_regime_trend_score",
    "btc_market_regime_volatility_score",
    "btc_market_regime_chaos_score",
    "btc_market_regime_stability_score",
    "btc_market_regime_confidence_multiplier",
    "sentiment_score",
    "fgi_value",
    "twitter_sentiment",
    "reddit_sentiment",
    "open_positions_count",
    "open_positions_unrealized_pnl_pct_total",
    "spread",
    "current_price",
    "normalized_trade_size",
]
