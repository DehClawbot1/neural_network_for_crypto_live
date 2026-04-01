import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from token_utils import normalize_token_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class OrderFlowAnalyzer:
    """
    Analyzes raw recent trades (taker flow) and market order book depths (maker flow)
    to generate autonomous trading signals based purely on volume and market sentiment
    imbalances, removing the reliance purely on copying top wallets.
    """

    def __init__(self, min_usd_volume=500.0, volume_imbalance_threshold=0.75, min_trades_count=3):
        """
        :param min_usd_volume: Minimum total volume required to generate a signal.
        :param volume_imbalance_threshold: % of volume moving in one direction required to trigger a signal.
        :param min_trades_count: Minimum number of recent trades required.
        """
        self.min_usd_volume = min_usd_volume
        self.volume_imbalance_threshold = volume_imbalance_threshold
        self.min_trades_count = min_trades_count

    def _safe_float(self, value, default=0.0):
        try:
            val = float(value)
            return default if pd.isna(val) else val
        except Exception:
            return default

    def analyze(self, signals_df: pd.DataFrame, markets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes the flow of trades over the recent cycle to detect large volume imbalances.
        Returns a DataFrame of new synthetic signals to append to the pipeline.
        """
        if signals_df is None or signals_df.empty or markets_df is None or markets_df.empty:
            return pd.DataFrame()

        # Step 1: Compute metrics from market orderbook (Maker Flow / Liquidity)
        market_stats = {}
        if "slug" in markets_df.columns:
            for _, mrow in markets_df.iterrows():
                slug = str(mrow.get("slug", ""))
                if not slug or pd.isna(slug):
                    continue
                
                bid_size = self._safe_float(mrow.get("bid_size", 0.0))
                ask_size = self._safe_float(mrow.get("ask_size", 0.0))
                
                # Measure resting liquidity sentiment
                total_depth = bid_size + ask_size
                liquidity_imbalance = 0.5
                if total_depth > 0:
                    liquidity_imbalance = bid_size / total_depth
                    
                market_stats[slug] = {
                    "liquidity_imbalance": liquidity_imbalance,
                    "best_bid": self._safe_float(mrow.get("best_bid")),
                    "best_ask": self._safe_float(mrow.get("best_ask")),
                    "current_price": self._safe_float(mrow.get("current_price", 0.5)),
                    "yes_token_id": mrow.get("yes_token_id"),
                    "no_token_id": mrow.get("no_token_id"),
                    "condition_id": mrow.get("condition_id"),
                    "market_title": mrow.get("market_title", mrow.get("question", slug)),
                }

        # Step 2: Aggregate recent taker trades by market slug
        if "market_slug" not in signals_df.columns or "outcome_side" not in signals_df.columns:
            return pd.DataFrame()
            
        # Only process actual trades, not synthetic signals
        if "signal_source" in signals_df.columns:
            trade_signals = signals_df[~(signals_df["signal_source"].astype(str).str.lower() == "always_on_market")].copy()
        else:
            trade_signals = signals_df.copy()
            
        if trade_signals.empty:
            return pd.DataFrame()

        trade_signals["size"] = pd.to_numeric(trade_signals["size"], errors="coerce").fillna(0.0)
        trade_signals["price"] = pd.to_numeric(trade_signals["price"], errors="coerce").fillna(0.0)
        trade_signals["notional"] = trade_signals["size"] * trade_signals["price"]
        
        # Group by market and side
        agg_df = trade_signals.groupby(["market_slug", "outcome_side"]).agg(
            total_notional=("notional", "sum"),
            trade_count=("trade_id", "count"),
            avg_price=("price", "mean")
        ).unstack(fill_value=0)

        new_signals = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for slug in agg_df.index:
            try:
                # Retrieve stats (handles unstacked MultiIndex safely)
                yes_vol = self._safe_float(agg_df.loc[slug, ("total_notional", "YES")]) if ("total_notional", "YES") in agg_df.columns else 0.0
                no_vol = self._safe_float(agg_df.loc[slug, ("total_notional", "NO")]) if ("total_notional", "NO") in agg_df.columns else 0.0
                
                yes_count = int(agg_df.loc[slug, ("trade_count", "YES")]) if ("trade_count", "YES") in agg_df.columns else 0
                no_count = int(agg_df.loc[slug, ("trade_count", "NO")]) if ("trade_count", "NO") in agg_df.columns else 0

                yes_price = self._safe_float(agg_df.loc[slug, ("avg_price", "YES")]) if ("avg_price", "YES") in agg_df.columns else 0.0
                no_price = self._safe_float(agg_df.loc[slug, ("avg_price", "NO")]) if ("avg_price", "NO") in agg_df.columns else 0.0

            except Exception as e:
                logging.warning(f"OrderFlowAnalyzer: Error extracting stats for slug {slug}: {e}")
                continue

            total_vol = yes_vol + no_vol
            total_count = yes_count + no_count

            if total_vol < self.min_usd_volume or total_count < self.min_trades_count:
                continue

            yes_pct = yes_vol / total_vol
            no_pct = no_vol / total_vol

            # Determine Sentiment Side
            signal_side = None
            imbalance_strength = 0.0
            avg_fill = 0.0

            if yes_pct >= self.volume_imbalance_threshold:
                signal_side = "YES"
                imbalance_strength = yes_pct
                avg_fill = yes_price
            elif no_pct >= self.volume_imbalance_threshold:
                signal_side = "NO"
                imbalance_strength = no_pct
                avg_fill = no_price

            # Check maker liquidity confirming the taker flow (if we have market orderbook depth)
            m_stats = market_stats.get(slug)
            if signal_side and m_stats:
                liquidity_imbalance = m_stats["liquidity_imbalance"]
                # If taker buys YES, we prefer if the maker side is also skewed towards bidding YES, 
                # but strong taker flow alone is often sufficient. We will use maker flow to boost confidence.
                maker_confirm = (signal_side == "YES" and liquidity_imbalance > 0.4) or \
                                (signal_side == "NO" and liquidity_imbalance < 0.6)
                
                if maker_confirm:
                    imbalance_strength += 0.1 # Boost strength if orderbook supports the taker flow

                imbalance_strength = min(1.0, imbalance_strength)
                
                # We have a valid order flow sentiment signal!
                token_id = m_stats["yes_token_id"] if signal_side == "YES" else m_stats["no_token_id"]
                if not token_id:
                    continue
                    
                # Fix up the price if avg_fill is broken
                signal_price = avg_fill if avg_fill > 0 else m_stats["current_price"]
                
                synthetic_signal = {
                    "trade_id": f"oflow_{slug}_{int(time.time())}",
                    "tx_hash": None,
                    "trader_wallet": "system_order_flow",
                    "market_title": m_stats["market_title"],
                    "market_slug": slug,
                    "token_id": str(token_id),
                    "condition_id": m_stats["condition_id"],
                    "order_side": "BUY",
                    "trade_side": "BUY",
                    "outcome_side": signal_side,
                    "entry_intent": "OPEN_LONG",
                    "side": signal_side,
                    "price": signal_price,
                    "size": total_vol * imbalance_strength, # Give it a "size" equivalent to its confidence weight
                    "timestamp": now_iso,
                    "signal_source": "order_flow_sentiment",
                    "force_candidate": 0, # Don't force it blindly, let the model/rules evaluate it
                    "confidence": imbalance_strength, # Boost its initial confidence score pre-RL model
                }
                
                new_signals.append(synthetic_signal)
                logging.info(
                    f"OrderFlowAnalyzer: Identified strong {signal_side} flow on '{slug}'. "
                    f"Vol: ${total_vol:.0f} | Ratio: {imbalance_strength:.0%} | Trades: {total_count}"
                )

        if new_signals:
            logging.info(f"OrderFlowAnalyzer: Injected {len(new_signals)} synthetic order flow signals.")
            return pd.DataFrame(new_signals)
            
        return pd.DataFrame()
