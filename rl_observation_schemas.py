import numpy as np


def prepare_entry_observation(feature_row, legacy=False):
    """Build the entry-policy observation vector."""
    if legacy:
        return np.array(
            [
                float(feature_row.get("trader_win_rate", 0.5)),
                float(feature_row.get("normalized_trade_size", 0.5)),
                float(feature_row.get("current_price", 0.5)),
                float(feature_row.get("time_left", 0.5)),
            ],
            dtype=np.float32,
        )

    return np.array(
        [
            float(feature_row.get("trader_win_rate", 0.5)),
            float(feature_row.get("normalized_trade_size", 0.5)),
            float(feature_row.get("current_price", 0.5)),
            float(feature_row.get("time_left", 0.5)),
            float(feature_row.get("liquidity_score", 0.5)),
            float(feature_row.get("volume_score", 0.5)),
            float(feature_row.get("probability_momentum", 0.5)),
            float(feature_row.get("volatility_score", 0.5)),
            float(feature_row.get("whale_pressure", 0.5)),
            float(feature_row.get("market_structure_score", 0.5)),
        ],
        dtype=np.float32,
    )


def prepare_position_observation(position_row):
    """Build the position-management observation vector."""
    return np.array(
        [
            float(position_row.get("confidence", 0.5)),
            float(position_row.get("shares", 0.0)),
            float(position_row.get("current_price", 0.5)),
            float(position_row.get("entry_price", 0.5)),
            float(position_row.get("market_value", 0.0)),
            float(position_row.get("unrealized_pnl", 0.0)),
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        dtype=np.float32,
    )
