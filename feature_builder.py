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


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


class FeatureBuilder:
    """
    Converts raw wallet/activity + market-monitor data into grouped, normalized features
    for research scoring and paper-trading only.
    """

    def __init__(self):
        self.wallet_stats = {}

    def update_wallet_history(self, trades_df: pd.DataFrame):
        if trades_df is None or trades_df.empty:
            return

        time_col = "timestamp" if "timestamp" in trades_df.columns else None
        size_col = "size" if "size" in trades_df.columns else "size_usdc" if "size_usdc" in trades_df.columns else None
        wallet_col = "trader_wallet" if "trader_wallet" in trades_df.columns else "wallet_copied" if "wallet_copied" in trades_df.columns else None
        if wallet_col is None or size_col is None:
            return

        df = trades_df.copy()
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
            df = df.sort_values(time_col)

        for wallet, group in df.groupby(wallet_col):
            avg_size = _safe_float(group[size_col].mean(), 1.0)
            count = int(group[size_col].count())
            stats = {
                "avg_size": max(avg_size, 1.0),
                "trade_count": count,
                "win_rate": np.nan,
                "alpha_30d": np.nan,
                "avg_forward_return_15m": np.nan,
                "tp_precision": np.nan,
                "recent_streak": 0,
                "same_market_history": 0,
            }

            if "future_return" in group.columns:
                stats["avg_forward_return_15m"] = _safe_float(group["future_return"].mean(), np.nan)
                stats["win_rate"] = _safe_float((group["future_return"] > 0).mean(), np.nan)
            if "tp_before_sl" in group.columns:
                stats["tp_precision"] = _safe_float(group["tp_before_sl"].mean(), np.nan)
            if "alpha_30d" in group.columns:
                stats["alpha_30d"] = _safe_float(group["alpha_30d"].iloc[-1], np.nan)

            self.wallet_stats[wallet] = stats

    def _normalized_trade_size(self, wallet: str, size: float) -> float:
        wallet_info = self.wallet_stats.get(wallet, {"avg_size": 1.0})
        avg_size = max(_safe_float(wallet_info.get("avg_size", 1.0), 1.0), 1.0)
        ratio = size / avg_size
        return _clip01(ratio / 3.0)

    def _wallet_win_rate(self, wallet: str) -> float:
        value = self.wallet_stats.get(wallet, {}).get("win_rate", np.nan)
        if pd.isna(value):
            return 0.5
        return _clip01(value)

    def _time_left_feature(self, end_date) -> float:
        if not end_date:
            return 0.5

        try:
            end_dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            seconds_left = max((end_dt - now).total_seconds(), 0)
            return _clip01(seconds_left / (7 * 24 * 3600))
        except Exception:
            return 0.5

    def _liquidity_score(self, liquidity: float) -> float:
        return _clip01(_safe_float(liquidity, 0.0) / 100000.0)

    def _volume_score(self, volume: float) -> float:
        return _clip01(_safe_float(volume, 0.0) / 250000.0)

    def _probability_momentum_proxy(self, signal_price: float, market_last_trade_price: float) -> float:
        move = abs(signal_price - market_last_trade_price)
        return _clip01(move / 0.20)

    def _volatility_proxy(self, signal_price: float, market_last_trade_price: float, volume: float) -> float:
        move_component = abs(signal_price - market_last_trade_price) / 0.15
        volume_component = _safe_float(volume, 0.0) / 250000.0
        return _clip01((move_component * 0.7) + (volume_component * 0.3))

    def _whale_consensus_score(self, size_score: float, win_rate: float) -> float:
        return _clip01((size_score * 0.6) + (win_rate * 0.4))

    def build_feature_row(self, signal: dict, market_row: dict | None = None):
        wallet = signal.get("trader_wallet")
        size = _safe_float(signal.get("size", 0), 0.0)
        signal_price = _clip01(_safe_float(signal.get("price", 0.5), 0.5))

        market_row = market_row or {}
        time_left = self._time_left_feature(market_row.get("end_date"))
        liquidity = _safe_float(market_row.get("liquidity", 0), 0.0)
        volume = _safe_float(market_row.get("volume", 0), 0.0)
        last_trade_price = _clip01(_safe_float(market_row.get("last_trade_price", signal_price), signal_price))

        trader_win_rate = self._wallet_win_rate(wallet)
        normalized_trade_size = self._normalized_trade_size(wallet, size)
        liquidity_score = self._liquidity_score(liquidity)
        volume_score = self._volume_score(volume)
        probability_momentum = self._probability_momentum_proxy(signal_price, last_trade_price)
        volatility_score = self._volatility_proxy(signal_price, last_trade_price, volume)
        whale_consensus_score = self._whale_consensus_score(normalized_trade_size, trader_win_rate)

        whale_pressure = _clip01((trader_win_rate * 0.4) + (normalized_trade_size * 0.35) + (whale_consensus_score * 0.25))
        market_structure_score = _clip01((signal_price * 0.2) + (liquidity_score * 0.3) + (volume_score * 0.25) + (probability_momentum * 0.25))
        volatility_risk = volatility_score
        time_decay_score = _clip01(1.0 - time_left)

        wallet_info = self.wallet_stats.get(wallet, {})
        feature_row = {
            "timestamp": signal.get("timestamp"),
            "trader_wallet": wallet,
            "market_title": signal.get("market_title"),
            "condition_id": signal.get("condition_id"),
            "market_slug": market_row.get("slug"),
            "side": signal.get("side"),
            "entry_price": signal_price,
            "best_bid": _safe_float(market_row.get("best_bid", signal_price), signal_price),
            "best_ask": _safe_float(market_row.get("best_ask", signal_price), signal_price),
            "spread": abs(_safe_float(market_row.get("best_ask", signal_price), signal_price) - _safe_float(market_row.get("best_bid", signal_price), signal_price)),
            # core normalized inputs
            "trader_win_rate": trader_win_rate,
            "wallet_trade_count_30d": wallet_info.get("trade_count", 0),
            "wallet_avg_size_30d": wallet_info.get("avg_size", np.nan),
            "wallet_winrate_30d": wallet_info.get("win_rate", np.nan),
            "wallet_alpha_30d": wallet_info.get("alpha_30d", np.nan),
            "wallet_avg_forward_return_15m": wallet_info.get("avg_forward_return_15m", np.nan),
            "wallet_signal_precision_tp": wallet_info.get("tp_precision", np.nan),
            "wallet_recent_streak": wallet_info.get("recent_streak", 0),
            "wallet_same_market_history": wallet_info.get("same_market_history", 0),
            "normalized_trade_size": normalized_trade_size,
            "current_price": signal_price,
            "time_left": time_left,
            # expanded microstructure/context
            "market_liquidity": liquidity,
            "market_volume": volume,
            "liquidity_score": liquidity_score,
            "volume_score": volume_score,
            "market_last_trade_price": last_trade_price,
            "probability_momentum": probability_momentum,
            "volatility_score": volatility_score,
            "whale_consensus_score": whale_consensus_score,
            # grouped sub-scores
            "whale_pressure": whale_pressure,
            "market_structure_score": market_structure_score,
            "volatility_risk": volatility_risk,
            "time_decay_score": time_decay_score,
            # metadata
            "raw_size": size,
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
        logging.info("Built %s grouped feature rows.", len(features_df))
        return features_df
