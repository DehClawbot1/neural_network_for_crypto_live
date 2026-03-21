import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class FeatureBuilder:
    """
    Converts raw wallet/activity + market-monitor data into normalized features
    for research scoring and paper-trading only.
    """

    def __init__(self):
        self.wallet_stats = {}

    def update_wallet_history(self, trades_df: pd.DataFrame):
        if trades_df is None or trades_df.empty:
            return

        grouped = trades_df.groupby("trader_wallet")
        for wallet, group in grouped:
            avg_size = _safe_float(group["size"].mean(), 1.0)
            count = int(group["size"].count())
            self.wallet_stats[wallet] = {
                "avg_size": max(avg_size, 1.0),
                "trade_count": count,
                # placeholder until real outcome-resolution tracking exists
                "win_rate": 0.55 if count >= 5 else 0.50,
            }

    def _normalized_trade_size(self, wallet: str, size: float) -> float:
        wallet_info = self.wallet_stats.get(wallet, {"avg_size": 1.0})
        avg_size = max(_safe_float(wallet_info.get("avg_size", 1.0), 1.0), 1.0)
        ratio = size / avg_size
        return float(np.clip(ratio / 3.0, 0.0, 1.0))

    def _wallet_win_rate(self, wallet: str) -> float:
        return float(np.clip(self.wallet_stats.get(wallet, {}).get("win_rate", 0.50), 0.0, 1.0))

    def _time_left_feature(self, end_date) -> float:
        if not end_date:
            return 0.5

        try:
            end_dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            seconds_left = max((end_dt - now).total_seconds(), 0)
            # squash to 0..1 over ~7 day window
            return float(np.clip(seconds_left / (7 * 24 * 3600), 0.0, 1.0))
        except Exception:
            return 0.5

    def build_feature_row(self, signal: dict, market_row: dict | None = None):
        wallet = signal.get("trader_wallet")
        size = _safe_float(signal.get("size", 0), 0.0)
        price = _safe_float(signal.get("price", 0.5), 0.5)

        market_row = market_row or {}
        time_left = self._time_left_feature(market_row.get("end_date"))

        feature_row = {
            "trader_wallet": wallet,
            "market_title": signal.get("market_title"),
            "condition_id": signal.get("condition_id"),
            "side": signal.get("side"),
            "trader_win_rate": self._wallet_win_rate(wallet),
            "normalized_trade_size": self._normalized_trade_size(wallet, size),
            "current_price": float(np.clip(price, 0.0, 1.0)),
            "time_left": time_left,
            "raw_size": size,
            "market_liquidity": _safe_float(market_row.get("liquidity", 0), 0.0),
            "market_volume": _safe_float(market_row.get("volume", 0), 0.0),
            "market_slug": market_row.get("slug"),
            "market_url": market_row.get("url"),
        }
        return feature_row

    def build_features(self, signals_df: pd.DataFrame, markets_df: pd.DataFrame | None = None):
        if signals_df is None or signals_df.empty:
            return pd.DataFrame()

        self.update_wallet_history(signals_df)

        market_lookup = {}
        if markets_df is not None and not markets_df.empty:
            for _, row in markets_df.iterrows():
                market_lookup[str(row.get("question", "")).lower()] = row.to_dict()

        rows = []
        for _, signal_row in signals_df.iterrows():
            signal = signal_row.to_dict()
            title_key = str(signal.get("market_title", "")).lower()
            matched_market = market_lookup.get(title_key)
            rows.append(self.build_feature_row(signal, matched_market))

        features_df = pd.DataFrame(rows)
        logging.info("Built %s feature rows.", len(features_df))
        return features_df
