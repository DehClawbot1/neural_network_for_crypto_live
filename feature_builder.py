import logging
import threading
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from token_utils import normalize_token_id

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
        self._seen_signal_keys = set()
        self._seen_signal_queue = deque()
        self._max_seen_signal_keys = 50000
        self._seen_lock = threading.Lock()

    def _remember_signal_key(self, event_key: str) -> bool:
        """
        Remember an event key with bounded memory.
        Returns False when key is already known (duplicate), True otherwise.
        """
        with self._seen_lock:
            if event_key in self._seen_signal_keys:
                return False
            self._seen_signal_keys.add(event_key)
            self._seen_signal_queue.append(event_key)
            while len(self._seen_signal_queue) > self._max_seen_signal_keys:
                old_key = self._seen_signal_queue.popleft()
                self._seen_signal_keys.discard(old_key)
            return True

    def update_wallet_history(self, trade_row: dict):
        wallet_col = "trader_wallet" if "trader_wallet" in trade_row else "wallet_copied" if "wallet_copied" in trade_row else None
        size_col = "size" if "size" in trade_row else "size_usdc" if "size_usdc" in trade_row else None
        if wallet_col is None or size_col is None:
            return

        wallet = trade_row.get(wallet_col)
        if not wallet:
            return

        event_key_parts = [
            trade_row.get("trade_id"),
            trade_row.get("tx_hash"),
            trade_row.get(wallet_col),
            trade_row.get("token_id"),
            trade_row.get("condition_id"),
            trade_row.get("outcome_side", trade_row.get("side")),
            trade_row.get("timestamp"),
            trade_row.get("price"),
            trade_row.get(size_col),
        ]
        event_key = "|".join(
            "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
            for value in event_key_parts
        )
        if not self._remember_signal_key(event_key):
            return

        prior = self.wallet_stats.get(wallet, {"sizes": [], "forward_returns": [], "tp_labels": [], "market_counts": {}})
        sizes = prior.get("sizes", [])
        sizes.append(_safe_float(trade_row.get(size_col, 0.0), 0.0))
        forward_returns = prior.get("forward_returns", [])
        return_value = None
        if "forward_return_15m" in trade_row and pd.notna(trade_row.get("forward_return_15m")):
            return_value = trade_row.get("forward_return_15m")
        elif "future_return" in trade_row and pd.notna(trade_row.get("future_return")):
            return_value = trade_row.get("future_return")
        if return_value is not None:
            forward_returns.append(_safe_float(return_value, 0.0))
        tp_labels = prior.get("tp_labels", [])
        if "tp_before_sl_60m" in trade_row and pd.notna(trade_row.get("tp_before_sl_60m")):
            tp_labels.append(int(trade_row.get("tp_before_sl_60m")))
        market_counts = prior.get("market_counts", {})
        market_title = trade_row.get("market_title") or trade_row.get("market")
        if market_title:
            market_counts[market_title] = market_counts.get(market_title, 0) + 1

        recent_forward = forward_returns[-200:]
        recent_tp = tp_labels[-200:]
        self.wallet_stats[wallet] = {
            "sizes": sizes,
            "forward_returns": forward_returns,
            "tp_labels": tp_labels,
            "market_counts": market_counts,
            "avg_size": max(_safe_float(np.mean(sizes), 1.0), 1.0),
            "trade_count": len(sizes),
            "win_rate": _safe_float(np.mean([1 if x > 0 else 0 for x in recent_forward]), np.nan) if recent_forward else np.nan,
            "alpha_30d": _safe_float(np.mean(recent_forward), np.nan) if recent_forward else np.nan,
            "avg_forward_return_15m": _safe_float(np.mean(recent_forward), np.nan) if recent_forward else np.nan,
            "tp_precision": _safe_float(np.mean(recent_tp), np.nan) if recent_tp else np.nan,
            "recent_streak": int(sum(1 for x in recent_forward[-5:] if x > 0)) if recent_forward else 0,
            "same_market_history": market_counts.get(market_title, 0) if market_title else 0,
        }

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

    def _time_left_feature(self, end_date, reference_time=None) -> float:
        if not end_date:
            return 0.5

        try:
            end_dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
            ref_dt = pd.to_datetime(reference_time, utc=True, errors="coerce") if reference_time is not None else datetime.now(timezone.utc)
            if pd.isna(ref_dt):
                ref_dt = datetime.now(timezone.utc)
            seconds_left = max((end_dt - ref_dt).total_seconds(), 0)
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
        time_left = self._time_left_feature(market_row.get("end_date"), signal.get("timestamp"))
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
        wallet_quality_score = _clip01(_safe_float(signal.get("wallet_quality_score", trader_win_rate), trader_win_rate))
        wallet_agreement_score = _clip01(_safe_float(signal.get("wallet_agreement_score", 0.5), 0.5))
        wallet_state_confidence = _clip01(_safe_float(signal.get("source_wallet_direction_confidence", 0.0), 0.0))
        wallet_state_freshness_score = _clip01(_safe_float(signal.get("source_wallet_freshness_score", 0.0), 0.0))
        wallet_size_change_score = _clip01(_safe_float(signal.get("source_wallet_size_delta_ratio", normalized_trade_size), normalized_trade_size))
        wallet_distance_from_market = abs(signal_price - last_trade_price)
        wallet_distance_score = _clip01(1.0 - min(wallet_distance_from_market / 0.20, 1.0))

        whale_pressure = _clip01((trader_win_rate * 0.4) + (normalized_trade_size * 0.35) + (whale_consensus_score * 0.25))
        market_structure_score = _clip01((signal_price * 0.2) + (liquidity_score * 0.3) + (volume_score * 0.25) + (probability_momentum * 0.25))
        volatility_risk = volatility_score
        time_decay_score = _clip01(1.0 - time_left)
        btc_fee_pressure_score = _clip01(_safe_float(signal.get("btc_fee_pressure_score", 0.5), 0.5))
        btc_mempool_congestion_score = _clip01(_safe_float(signal.get("btc_mempool_congestion_score", 0.5), 0.5))
        btc_network_activity_score = _clip01(_safe_float(signal.get("btc_network_activity_score", 0.5), 0.5))
        btc_network_stress_score = _clip01(_safe_float(signal.get("btc_network_stress_score", 0.5), 0.5))

        outcome_side = str(signal.get("outcome_side", signal.get("side", ""))).upper()
        token_id = normalize_token_id(signal.get("token_id"))
        if (token_id in [None, ""] or pd.isna(token_id)) and market_row:
            if outcome_side in {"YES", "UP"}:
                token_id = normalize_token_id(market_row.get("yes_token_id"))
            elif outcome_side in {"NO", "DOWN"}:
                token_id = normalize_token_id(market_row.get("no_token_id"))
        token_id = normalize_token_id(token_id)
        condition_id = signal.get("condition_id") or market_row.get("condition_id")

        wallet_info = self.wallet_stats.get(wallet, {})
        feature_row = {
            "timestamp": signal.get("timestamp"),
            "last_trade_timestamp": signal.get("last_trade_timestamp", signal.get("timestamp")),
            "signal_observed_at": signal.get("signal_observed_at"),
            "market_data_timestamp": signal.get("market_data_timestamp", market_row.get("timestamp")),
            "trader_wallet": wallet,
            "market_title": signal.get("market_title", signal.get("market")),
            "condition_id": condition_id,
            "token_id": token_id,
            "market_slug": market_row.get("slug", signal.get("market_slug")),
            "order_side": signal.get("order_side", signal.get("trade_side")),
            "trade_side": signal.get("trade_side", signal.get("order_side")),
            "outcome_side": signal.get("outcome_side", signal.get("side")),
            "entry_intent": signal.get("entry_intent", "OPEN_LONG"),
            "side": signal.get("outcome_side", signal.get("side")),
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
            "wallet_quality_score": wallet_quality_score,
            "wallet_watchlist_approved": bool(signal.get("wallet_watchlist_approved", True)),
            "wallet_agreement_score": wallet_agreement_score,
            "wallet_conflict_with_stronger": bool(signal.get("wallet_conflict_with_stronger", False)),
            "wallet_stronger_conflict_score": _safe_float(signal.get("wallet_stronger_conflict_score"), 0.0),
            "wallet_support_strength": _safe_float(signal.get("wallet_support_strength"), 0.0),
            "wallet_state_gate_pass": bool(signal.get("wallet_state_gate_pass", True)),
            "wallet_state_gate_reason": signal.get("wallet_state_gate_reason"),
            "normalized_trade_size": normalized_trade_size,
            "current_price": signal_price,
            "time_left": time_left,
            # expanded microstructure/context
            "market_liquidity": liquidity,
            "market_volume": volume,
            "liquidity_score": liquidity_score,
            "volume_score": volume_score,
            "market_last_trade_price": last_trade_price,
            "market_timestamp": market_row.get("timestamp"),
            "probability_momentum": probability_momentum,
            "volatility_score": volatility_score,
            "wallet_state_confidence": wallet_state_confidence,
            "wallet_state_freshness_score": wallet_state_freshness_score,
            "wallet_size_change_score": wallet_size_change_score,
            "wallet_distance_from_market": wallet_distance_from_market,
            "wallet_distance_score": wallet_distance_score,
            "open_positions_count": int(_safe_float(signal.get("open_positions_count"), 0.0)),
            "open_positions_negotiated_value_total": _safe_float(signal.get("open_positions_negotiated_value_total"), 0.0),
            "open_positions_max_payout_total": _safe_float(signal.get("open_positions_max_payout_total"), 0.0),
            "open_positions_current_value_total": _safe_float(signal.get("open_positions_current_value_total"), 0.0),
            "open_positions_unrealized_pnl_total": _safe_float(signal.get("open_positions_unrealized_pnl_total"), 0.0),
            "open_positions_unrealized_pnl_pct_total": _safe_float(signal.get("open_positions_unrealized_pnl_pct_total"), 0.0),
            "open_positions_avg_to_now_price_change_pct_mean": _safe_float(signal.get("open_positions_avg_to_now_price_change_pct_mean"), 0.0),
            "open_positions_avg_to_now_price_change_pct_min": _safe_float(signal.get("open_positions_avg_to_now_price_change_pct_min"), 0.0),
            "open_positions_avg_to_now_price_change_pct_max": _safe_float(signal.get("open_positions_avg_to_now_price_change_pct_max"), 0.0),
            "open_positions_winner_count": int(_safe_float(signal.get("open_positions_winner_count"), 0.0)),
            "open_positions_loser_count": int(_safe_float(signal.get("open_positions_loser_count"), 0.0)),
            "source_wallet_direction_confidence": _safe_float(signal.get("source_wallet_direction_confidence"), 0.0),
            "source_wallet_position_event": signal.get("source_wallet_position_event", ""),
            "source_wallet_net_position_increased": bool(signal.get("source_wallet_net_position_increased", False)),
            "source_wallet_current_net_exposure": _safe_float(signal.get("source_wallet_current_net_exposure"), 0.0),
            "source_wallet_average_entry": _safe_float(signal.get("source_wallet_average_entry"), np.nan),
            "source_wallet_last_add": signal.get("source_wallet_last_add"),
            "source_wallet_last_reduce": signal.get("source_wallet_last_reduce"),
            "source_wallet_last_close": signal.get("source_wallet_last_close"),
            "source_wallet_current_direction": signal.get("source_wallet_current_direction", "FLAT"),
            "source_wallet_reduce_fraction": _safe_float(signal.get("source_wallet_reduce_fraction"), 0.0),
            "source_wallet_state_freshness_seconds": _safe_float(signal.get("source_wallet_state_freshness_seconds"), 0.0),
            "source_wallet_fresh": bool(signal.get("source_wallet_fresh", False)),
            "source_wallet_exit_signal": bool(signal.get("source_wallet_exit_signal", False)),
            "source_wallet_reduce_signal": bool(signal.get("source_wallet_reduce_signal", False)),
            "source_wallet_reversal_signal": bool(signal.get("source_wallet_reversal_signal", False)),
            "btc_fee_fastest_satvb": _safe_float(signal.get("btc_fee_fastest_satvb"), np.nan),
            "btc_fee_hour_satvb": _safe_float(signal.get("btc_fee_hour_satvb"), np.nan),
            "btc_difficulty_change_pct": _safe_float(signal.get("btc_difficulty_change_pct"), np.nan),
            "btc_mempool_tx_count": _safe_float(signal.get("btc_mempool_tx_count"), np.nan),
            "btc_mempool_vsize": _safe_float(signal.get("btc_mempool_vsize"), np.nan),
            "btc_fee_pressure_score": btc_fee_pressure_score,
            "btc_mempool_congestion_score": btc_mempool_congestion_score,
            "btc_network_activity_score": btc_network_activity_score,
            "btc_network_stress_score": btc_network_stress_score,
            "btc_live_price": _safe_float(signal.get("btc_live_price"), np.nan),
            "btc_live_spot_price": _safe_float(signal.get("btc_live_spot_price"), np.nan),
            "btc_live_index_price": _safe_float(signal.get("btc_live_index_price"), np.nan),
            "btc_live_mark_price": _safe_float(signal.get("btc_live_mark_price"), np.nan),
            "btc_live_funding_rate": _safe_float(signal.get("btc_live_funding_rate"), np.nan),
            "btc_live_source_quality": signal.get("btc_live_source_quality", "LOW"),
            "btc_live_source_quality_score": _safe_float(signal.get("btc_live_source_quality_score"), 0.0),
            "btc_live_source_divergence_bps": _safe_float(signal.get("btc_live_source_divergence_bps"), 0.0),
            "btc_live_spot_index_basis_bps": _safe_float(signal.get("btc_live_spot_index_basis_bps"), 0.0),
            "btc_live_mark_index_basis_bps": _safe_float(signal.get("btc_live_mark_index_basis_bps"), 0.0),
            "btc_live_mark_spot_basis_bps": _safe_float(signal.get("btc_live_mark_spot_basis_bps"), 0.0),
            "btc_live_return_1m": _safe_float(signal.get("btc_live_return_1m"), 0.0),
            "btc_live_return_5m": _safe_float(signal.get("btc_live_return_5m"), 0.0),
            "btc_live_return_15m": _safe_float(signal.get("btc_live_return_15m"), 0.0),
            "btc_live_return_1h": _safe_float(signal.get("btc_live_return_1h"), 0.0),
            "btc_live_volatility_proxy": _safe_float(signal.get("btc_live_volatility_proxy"), 0.0),
            "btc_live_bias": signal.get("btc_live_bias", "NEUTRAL"),
            "btc_live_confluence": _safe_float(signal.get("btc_live_confluence"), 0.0),
            "btc_live_index_ready": bool(signal.get("btc_live_index_ready", False)),
            "btc_live_index_feed_available": bool(signal.get("btc_live_index_feed_available", False)),
            "btc_live_mark_feed_available": bool(signal.get("btc_live_mark_feed_available", False)),
            "sentiment_score": _safe_float(signal.get("sentiment_score"), np.nan),
            "btc_funding_rate": _safe_float(signal.get("btc_funding_rate"), np.nan),
            "is_overheated_long": bool(signal.get("is_overheated_long", False)),
            "fgi_value": _safe_float(signal.get("fgi_value"), np.nan),
            "fgi_normalized": _safe_float(signal.get("fgi_normalized"), np.nan),
            "fgi_extreme_fear": _safe_float(signal.get("fgi_extreme_fear"), np.nan),
            "fgi_extreme_greed": _safe_float(signal.get("fgi_extreme_greed"), np.nan),
            "fgi_contrarian": _safe_float(signal.get("fgi_contrarian"), np.nan),
            "fgi_momentum": _safe_float(signal.get("fgi_momentum"), np.nan),
            "fgi_momentum_3d": _safe_float(signal.get("fgi_momentum_3d"), np.nan),
            "gtrends_bitcoin": _safe_float(signal.get("gtrends_bitcoin"), np.nan),
            "gtrends_zscore": _safe_float(signal.get("gtrends_zscore"), np.nan),
            "gtrends_spike": _safe_float(signal.get("gtrends_spike"), np.nan),
            "gtrends_momentum": _safe_float(signal.get("gtrends_momentum"), np.nan),
            "twitter_sentiment": _safe_float(signal.get("twitter_sentiment"), np.nan),
            "twitter_post_count": _safe_float(signal.get("twitter_post_count"), np.nan),
            "twitter_sentiment_pos": _safe_float(signal.get("twitter_sentiment_pos"), np.nan),
            "twitter_sentiment_neg": _safe_float(signal.get("twitter_sentiment_neg"), np.nan),
            "twitter_engagement_proxy": _safe_float(signal.get("twitter_engagement_proxy"), np.nan),
            "twitter_sentiment_zscore": _safe_float(signal.get("twitter_sentiment_zscore"), np.nan),
            "twitter_bullish": _safe_float(signal.get("twitter_bullish"), np.nan),
            "twitter_bearish": _safe_float(signal.get("twitter_bearish"), np.nan),
            "twitter_sentiment_momentum": _safe_float(signal.get("twitter_sentiment_momentum"), np.nan),
            "reddit_sentiment": _safe_float(signal.get("reddit_sentiment"), np.nan),
            "reddit_post_count": _safe_float(signal.get("reddit_post_count"), np.nan),
            "reddit_sentiment_pos": _safe_float(signal.get("reddit_sentiment_pos"), np.nan),
            "reddit_sentiment_neg": _safe_float(signal.get("reddit_sentiment_neg"), np.nan),
            "reddit_sentiment_zscore": _safe_float(signal.get("reddit_sentiment_zscore"), np.nan),
            "reddit_bullish": _safe_float(signal.get("reddit_bullish"), np.nan),
            "reddit_bearish": _safe_float(signal.get("reddit_bearish"), np.nan),
            "reddit_sentiment_momentum": _safe_float(signal.get("reddit_sentiment_momentum"), np.nan),
            "onchain_network_health": signal.get("onchain_network_health", "UNKNOWN"),
            "market_structure": signal.get("market_structure", "UNKNOWN"),
            "trend_score": _safe_float(signal.get("trend_score"), 0.5),
            "alligator_jaw": _safe_float(signal.get("alligator_jaw"), np.nan),
            "alligator_teeth": _safe_float(signal.get("alligator_teeth"), np.nan),
            "alligator_lips": _safe_float(signal.get("alligator_lips"), np.nan),
            "alligator_alignment": signal.get("alligator_alignment", "NEUTRAL"),
            "alligator_bullish": bool(signal.get("alligator_bullish", False)),
            "alligator_bearish": bool(signal.get("alligator_bearish", False)),
            "adx_value": _safe_float(signal.get("adx_value"), np.nan),
            "adx_threshold": _safe_float(signal.get("adx_threshold"), np.nan),
            "adx_trending": bool(signal.get("adx_trending", False)),
            "anchored_vwap": _safe_float(signal.get("anchored_vwap"), np.nan),
            "anchored_vwap_anchor": signal.get("anchored_vwap_anchor", "utc_session_open"),
            "price_vs_anchored_vwap": _safe_float(signal.get("price_vs_anchored_vwap"), 0.0),
            "price_above_anchored_vwap": bool(signal.get("price_above_anchored_vwap", False)),
            "price_below_anchored_vwap": bool(signal.get("price_below_anchored_vwap", False)),
            "btc_trend_bias": signal.get("btc_trend_bias", "NEUTRAL"),
            "btc_trend_confluence": _safe_float(signal.get("btc_trend_confluence"), 0.0),
            "btc_atr_pct_15m": _safe_float(signal.get("btc_atr_pct_15m"), 0.0),
            "btc_realized_vol_1h": _safe_float(signal.get("btc_realized_vol_1h"), 0.0),
            "btc_realized_vol_4h": _safe_float(signal.get("btc_realized_vol_4h"), 0.0),
            "btc_volatility_regime": signal.get("btc_volatility_regime", "NORMAL"),
            "btc_volatility_regime_score": _safe_float(signal.get("btc_volatility_regime_score"), 0.0),
            "btc_trend_persistence": _safe_float(signal.get("btc_trend_persistence"), 0.0),
            "btc_rsi_14": _safe_float(signal.get("btc_rsi_14"), 50.0),
            "btc_rsi_distance_mid": _safe_float(signal.get("btc_rsi_distance_mid"), 0.0),
            "btc_rsi_divergence_score": _safe_float(signal.get("btc_rsi_divergence_score"), 0.0),
            "btc_macd": _safe_float(signal.get("btc_macd"), 0.0),
            "btc_macd_signal": _safe_float(signal.get("btc_macd_signal"), 0.0),
            "btc_macd_hist": _safe_float(signal.get("btc_macd_hist"), 0.0),
            "btc_macd_hist_slope": _safe_float(signal.get("btc_macd_hist_slope"), 0.0),
            "btc_momentum_regime": signal.get("btc_momentum_regime", "NEUTRAL"),
            "btc_momentum_confluence": _safe_float(signal.get("btc_momentum_confluence"), 0.0),
            "latest_bullish_fractal": _safe_float(signal.get("latest_bullish_fractal"), np.nan),
            "latest_bearish_fractal": _safe_float(signal.get("latest_bearish_fractal"), np.nan),
            "long_fractal_breakout": bool(signal.get("long_fractal_breakout", False)),
            "short_fractal_breakout": bool(signal.get("short_fractal_breakout", False)),
            "fractal_trigger_direction": signal.get("fractal_trigger_direction", "NEUTRAL"),
            "fractal_entry_ready": bool(signal.get("fractal_entry_ready", False)),
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

        market_lookup_cond = {}
        market_lookup_slug = {}
        if markets_df is not None and not markets_df.empty:
            for _, row in markets_df.iterrows():
                row_dict = row.to_dict()
                cond = str(row.get("condition_id", "")).strip()
                if cond and cond != "nan":
                    market_lookup_cond[cond] = row_dict
                slug = str(row.get("slug", "")).strip()
                if slug and slug != "nan":
                    market_lookup_slug[slug] = row_dict

        rows = []
        work_df = signals_df.copy()
        if "timestamp" in work_df.columns:
            numeric_ts = pd.to_numeric(work_df["timestamp"], errors="coerce")
            if numeric_ts.notna().any():
                sample = numeric_ts.dropna().iloc[0]
                if sample > 1e17:
                    work_df["timestamp"] = pd.to_datetime(numeric_ts, utc=True, errors="coerce", unit="ns")
                elif sample > 1e14:
                    work_df["timestamp"] = pd.to_datetime(numeric_ts, utc=True, errors="coerce", unit="us")
                elif sample > 1e11:
                    work_df["timestamp"] = pd.to_datetime(numeric_ts, utc=True, errors="coerce", unit="ms")
                elif sample > 1e9:
                    work_df["timestamp"] = pd.to_datetime(numeric_ts, utc=True, errors="coerce", unit="s")
                else:
                    work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True, errors="coerce")
            else:
                work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True, errors="coerce")
            work_df = work_df.sort_values("timestamp")

        for _, signal_row in work_df.iterrows():
            signal = signal_row.to_dict()
            cond_id = str(signal.get("condition_id", "")).strip()
            slug_id = str(signal.get("market_slug", "")).strip()

            matched_market = market_lookup_cond.get(cond_id)
            if not matched_market:
                matched_market = market_lookup_slug.get(slug_id)

            rows.append(self.build_feature_row(signal, matched_market))
            self.update_wallet_history(signal)

        features_df = pd.DataFrame(rows)
        logging.info("Built %s grouped feature rows.", len(features_df))
        return features_df

