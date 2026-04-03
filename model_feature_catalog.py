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
    "btc_live_funding_rate",
    "btc_live_source_quality_score",
    "btc_live_source_divergence_bps",
    "btc_live_spot_index_basis_bps",
    "btc_live_mark_index_basis_bps",
    "btc_live_mark_spot_basis_bps",
    "btc_live_return_1m",
    "btc_live_return_5m",
    "btc_live_return_15m",
    "btc_live_return_1h",
    "btc_live_volatility_proxy",
    "btc_live_confluence",
]

TRAINING_FEATURE_FAMILIES = OrderedDict(
    [
        ("wallet_copy", WALLET_FEATURES),
        ("market_microstructure", MARKET_STRUCTURE_FEATURES),
        ("onchain_network", ONCHAIN_FEATURES),
        ("btc_spot_regime", BTC_SPOT_REGIME_FEATURES),
        ("btc_momentum_quality", BTC_MOMENTUM_QUALITY_FEATURES),
        ("btc_live_index", BTC_LIVE_INDEX_FEATURES),
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
    "btc_live_mark_index_basis_bps",
    "btc_live_return_1m",
    "btc_live_return_5m",
    "btc_live_return_15m",
    "btc_live_return_1h",
    "btc_live_confluence",
    "spread",
    "current_price",
    "normalized_trade_size",
]
