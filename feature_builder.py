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
        self._seen_signal_keys = set()

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
        if event_key in self._seen_signal_keys:
            return
        self._seen_signal_keys.add(event_key)

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

        whale_pressure = _clip01((trader_win_rate * 0.4) + (normalized_trade_size * 0.35) + (whale_consensus_score * 0.25))
        market_structure_score = _clip01((signal_price * 0.2) + (liquidity_score * 0.3) + (volume_score * 0.25) + (probability_momentum * 0.25))
        volatility_risk = volatility_score
        time_decay_score = _clip01(1.0 - time_left)

        outcome_side = str(signal.get("outcome_side", signal.get("side", ""))).upper()
        token_id = signal.get("token_id")
        if (token_id in [None, ""] or pd.isna(token_id)) and market_row:
            if outcome_side in {"YES", "UP"}:
                token_id = market_row.get("yes_token_id")
            elif outcome_side in {"NO", "DOWN"}:
                token_id = market_row.get("no_token_id")
        condition_id = signal.get("condition_id") or market_row.get("condition_id")

        wallet_info = self.wallet_stats.get(wallet, {})
        feature_row = {
            "timestamp": signal.get("timestamp"),
            "trader_wallet": wallet,
            "market_title": signal.get("market_title", signal.get("market")),
            "condition_id": condition_id,
            "token_id": token_id,
            "market_slug": market_row.get("slug"),
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

