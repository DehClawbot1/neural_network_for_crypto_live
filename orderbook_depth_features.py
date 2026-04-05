"""
Order Book Depth Feature Analyzer

Captures microstructure signals from BTC order book depth:
  1. Bid/ask depth imbalance at multiple levels (5, 10, 20 deep)
  2. Order book slope — how quickly liquidity thins out
  3. Large order detection (whale walls) — big resting orders
  4. Weighted midpoint — volume-weighted fair price
  5. Cumulative depth curves — total liquidity available at each price level

These features capture INTENT OF LARGE PLAYERS and LIQUIDITY STRUCTURE
which pure price/volume technicals cannot see.

Data sources:
  - Binance Futures order book (BTCUSDT) via REST API
  - Polymarket order book via ClobClient (for Polymarket markets)

Architecture:
  - BinanceBookFetcher: fetches L2 snapshots from Binance
  - OrderBookDepthAnalyzer: computes features from any L2 book
  - fetch_btc_depth_snapshot(): convenience for BTC prediction pipeline
  - Integrates into supervisor.py as Pillar 7
"""

from __future__ import annotations

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def _safe_merge_asof(left, right, on, **kwargs):
    """merge_asof that tolerates NaT/null in the left merge key."""
    if left.empty or right.empty or on not in left.columns or on not in right.columns:
        return left
    mask = left[on].notna()
    if not mask.any():
        return left
    valid = left[mask].copy().sort_values(on)
    work_right = right.copy().sort_values(on)
    merged = pd.merge_asof(valid, work_right, on=on, **kwargs)
    if mask.all():
        return merged
    return pd.concat([merged, left[~mask]], ignore_index=True)


# ---------------------------------------------------------------------------
# Binance order book fetcher
# ---------------------------------------------------------------------------

BINANCE_DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth"
BINANCE_SPOT_DEPTH_URL = "https://api.binance.com/api/v3/depth"


class BinanceBookFetcher:
    """
    Fetches L2 order book snapshots from Binance (Futures or Spot).

    Binance depth API returns:
      {"lastUpdateId": ..., "bids": [["price","qty"],...], "asks": [["price","qty"],...]}

    Limits: 5, 10, 20, 50, 100, 500, 1000
    Weight: 5-50 depending on limit (1000 = 50 weight, 100 = 10 weight)
    """

    def __init__(self, symbol: str = "BTCUSDT", use_futures: bool = True):
        self.symbol = symbol
        self.use_futures = use_futures
        self._session = requests.Session()

    def fetch_depth(self, limit: int = 100) -> dict | None:
        """
        Fetch order book depth snapshot.

        Returns dict with:
          bids: list of [price, qty] (highest first)
          asks: list of [price, qty] (lowest first)
          timestamp: UTC datetime
        """
        url = BINANCE_DEPTH_URL if self.use_futures else BINANCE_SPOT_DEPTH_URL
        try:
            resp = self._session.get(
                url,
                params={"symbol": self.symbol, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
            asks = [[float(p), float(q)] for p, q in data.get("asks", [])]

            return {
                "bids": bids,  # [[price, qty], ...] highest first
                "asks": asks,  # [[price, qty], ...] lowest first
                "timestamp": datetime.now(timezone.utc),
                "last_update_id": data.get("lastUpdateId"),
            }
        except Exception as exc:
            logger.warning("Binance depth fetch failed for %s: %s", self.symbol, exc)
            return None


# ---------------------------------------------------------------------------
# Order Book Depth Analyzer
# ---------------------------------------------------------------------------

class OrderBookDepthAnalyzer:
    """
    Computes advanced features from L2 order book data.

    Features computed:
      1. Depth imbalance at levels 5, 10, 20 (bid vol vs ask vol)
      2. Cumulative depth ratios
      3. Order book slope (how fast liquidity decays)
      4. Whale wall detection (large resting orders)
      5. Weighted midpoint (volume-weighted fair price)
      6. Spread and spread volatility
      7. Depth concentration (how much volume is at top levels)
    """

    # Cache for rate limiting
    _cache: dict = {}
    _cache_lock = threading.Lock()
    _cache_ttl = 5.0  # seconds

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self._fetcher = BinanceBookFetcher(symbol=symbol, use_futures=True)

    # ------------------------------------------------------------------
    # Core: Fetch + Analyze
    # ------------------------------------------------------------------

    def analyze(self, depth_limit: int = 100) -> dict:
        """
        Fetch BTC order book and compute all depth features.
        Uses cache to avoid hammering the API.

        Returns dict with feature names as keys (suitable for macro_context).
        """
        # Check cache
        cache_key = f"{self.symbol}_{depth_limit}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and (time.time() - cached["_ts"]) < self._cache_ttl:
                return {k: v for k, v in cached.items() if not k.startswith("_")}

        snapshot = self._fetcher.fetch_depth(limit=depth_limit)
        if snapshot is None:
            return self._default_features()

        bids = snapshot["bids"]
        asks = snapshot["asks"]

        if not bids or not asks:
            return self._default_features()

        features = self._compute_features(bids, asks)

        # Cache result
        with self._cache_lock:
            self._cache[cache_key] = {**features, "_ts": time.time()}

        return features

    def _compute_features(self, bids: list, asks: list) -> dict:
        """
        Compute all order book depth features from bid/ask arrays.

        Args:
            bids: [[price, qty], ...] sorted highest first
            asks: [[price, qty], ...] sorted lowest first
        """
        features = {}

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        midpoint = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        features["ob_best_bid"] = best_bid
        features["ob_best_ask"] = best_ask
        features["ob_midpoint"] = midpoint
        features["ob_spread"] = spread
        features["ob_spread_bps"] = (spread / midpoint * 10000) if midpoint > 0 else 0

        # --- 1. Depth imbalance at multiple levels ---
        for n in [5, 10, 20]:
            bid_vol = sum(q for _, q in bids[:n])
            ask_vol = sum(q for _, q in asks[:n])
            total = bid_vol + ask_vol
            imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
            features[f"ob_imbalance_{n}"] = round(imbalance, 6)
            features[f"ob_bid_vol_{n}"] = round(bid_vol, 4)
            features[f"ob_ask_vol_{n}"] = round(ask_vol, 4)

        # --- 2. Cumulative depth in USD at various distances from mid ---
        for bps_distance in [10, 25, 50, 100]:
            price_range = midpoint * bps_distance / 10000
            bid_cum = sum(q for p, q in bids if p >= (midpoint - price_range))
            ask_cum = sum(q for p, q in asks if p <= (midpoint + price_range))
            total_cum = bid_cum + ask_cum
            cum_imbalance = (bid_cum - ask_cum) / total_cum if total_cum > 0 else 0.0
            features[f"ob_cum_imbalance_{bps_distance}bps"] = round(cum_imbalance, 6)
            features[f"ob_cum_depth_{bps_distance}bps_btc"] = round(total_cum, 4)
            features[f"ob_cum_depth_{bps_distance}bps_usd"] = round(total_cum * midpoint, 2)

        # --- 3. Order book slope (how quickly liquidity thins) ---
        bid_slope = self._compute_slope(bids, midpoint, side="bid")
        ask_slope = self._compute_slope(asks, midpoint, side="ask")
        features["ob_bid_slope"] = round(bid_slope, 6)
        features["ob_ask_slope"] = round(ask_slope, 6)
        features["ob_slope_imbalance"] = round(bid_slope - ask_slope, 6)

        # --- 4. Whale wall detection ---
        whale_features = self._detect_whale_walls(bids, asks, midpoint)
        features.update(whale_features)

        # --- 5. Weighted midpoint (VWAP of top levels) ---
        wmid = self._weighted_midpoint(bids[:20], asks[:20])
        features["ob_weighted_midpoint"] = round(wmid, 2)
        features["ob_wmid_vs_mid"] = round((wmid - midpoint) / midpoint * 10000, 4) if midpoint > 0 else 0  # in bps

        # --- 6. Depth concentration (what % of total vol is in top 5 levels) ---
        top5_bid = sum(q for _, q in bids[:5])
        top5_ask = sum(q for _, q in asks[:5])
        total_bid = sum(q for _, q in bids) or 1.0
        total_ask = sum(q for _, q in asks) or 1.0
        features["ob_bid_concentration"] = round(top5_bid / total_bid, 4)
        features["ob_ask_concentration"] = round(top5_ask / total_ask, 4)

        # --- 7. Price level density (how many price levels per bps range) ---
        bid_prices = [p for p, _ in bids if p >= midpoint * 0.999]  # within 10bps
        ask_prices = [p for p, _ in asks if p <= midpoint * 1.001]
        features["ob_bid_density_10bps"] = len(bid_prices)
        features["ob_ask_density_10bps"] = len(ask_prices)

        features["ob_ready"] = True
        return features

    # ------------------------------------------------------------------
    # Sub-computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_slope(levels: list, midpoint: float, side: str) -> float:
        """
        Compute the "slope" of the order book — how fast liquidity decays
        as we move away from the best price.

        A steep slope means liquidity is concentrated near the top (thin book).
        A flat slope means deep liquidity at all levels (thick book).

        Returns slope as qty_decay_per_bps (higher = steeper = thinner deep book).
        """
        if len(levels) < 3 or midpoint <= 0:
            return 0.0

        # Compute cumulative volume at increasing distances from midpoint
        distances = []  # in bps
        cum_vols = []
        cum = 0.0

        for price, qty in levels[:50]:
            dist_bps = abs(price - midpoint) / midpoint * 10000
            cum += qty
            distances.append(dist_bps)
            cum_vols.append(cum)

        if len(distances) < 2 or max(distances) == 0:
            return 0.0

        # Fit linear regression: cum_vol = slope * distance + intercept
        # Higher slope = more volume added per bps = deeper book
        distances = np.array(distances)
        cum_vols = np.array(cum_vols)

        # Normalize
        if distances[-1] > 0:
            return float(cum_vols[-1] / distances[-1])  # avg vol per bps
        return 0.0

    @staticmethod
    def _detect_whale_walls(
        bids: list, asks: list, midpoint: float,
        wall_threshold_mult: float = 5.0,
        max_distance_bps: float = 100,
    ) -> dict:
        """
        Detect large resting orders ("whale walls") in the order book.

        A whale wall is a price level with size > threshold_mult * median size
        within max_distance_bps of the midpoint.

        Returns:
          ob_whale_bid_wall: 1.0 if large bid wall detected, else 0.0
          ob_whale_ask_wall: 1.0 if large ask wall detected, else 0.0
          ob_whale_bid_wall_size: size of largest bid wall (BTC)
          ob_whale_ask_wall_size: size of largest ask wall (BTC)
          ob_whale_bid_wall_dist_bps: distance of bid wall from mid (bps)
          ob_whale_ask_wall_dist_bps: distance of ask wall from mid (bps)
          ob_whale_wall_bias: +1 if bid wall (support), -1 if ask wall (resistance), 0 if none/both
        """
        result = {
            "ob_whale_bid_wall": 0.0,
            "ob_whale_ask_wall": 0.0,
            "ob_whale_bid_wall_size": 0.0,
            "ob_whale_ask_wall_size": 0.0,
            "ob_whale_bid_wall_dist_bps": 0.0,
            "ob_whale_ask_wall_dist_bps": 0.0,
            "ob_whale_wall_bias": 0.0,
        }

        if midpoint <= 0:
            return result

        def _find_wall(levels, side):
            nearby = []
            for price, qty in levels:
                dist_bps = abs(price - midpoint) / midpoint * 10000
                if dist_bps <= max_distance_bps:
                    nearby.append((price, qty, dist_bps))

            if len(nearby) < 3:
                return None, 0, 0

            sizes = [q for _, q, _ in nearby]
            median_size = float(np.median(sizes))
            threshold = median_size * wall_threshold_mult

            # Find largest wall
            best = max(nearby, key=lambda x: x[1])
            if best[1] >= threshold and best[1] > 0:
                return best  # (price, qty, dist_bps)
            return None, 0, 0

        bid_wall = _find_wall(bids, "bid")
        ask_wall = _find_wall(asks, "ask")

        if bid_wall[0] is not None:
            result["ob_whale_bid_wall"] = 1.0
            result["ob_whale_bid_wall_size"] = round(bid_wall[1], 4)
            result["ob_whale_bid_wall_dist_bps"] = round(bid_wall[2], 2)

        if ask_wall[0] is not None:
            result["ob_whale_ask_wall"] = 1.0
            result["ob_whale_ask_wall_size"] = round(ask_wall[1], 4)
            result["ob_whale_ask_wall_dist_bps"] = round(ask_wall[2], 2)

        # Bias: bid wall = support = bullish, ask wall = resistance = bearish
        if result["ob_whale_bid_wall"] and not result["ob_whale_ask_wall"]:
            result["ob_whale_wall_bias"] = 1.0  # bullish support
        elif result["ob_whale_ask_wall"] and not result["ob_whale_bid_wall"]:
            result["ob_whale_wall_bias"] = -1.0  # bearish resistance
        elif result["ob_whale_bid_wall"] and result["ob_whale_ask_wall"]:
            # Both walls — bias toward larger one
            if result["ob_whale_bid_wall_size"] > result["ob_whale_ask_wall_size"]:
                result["ob_whale_wall_bias"] = 0.5
            else:
                result["ob_whale_wall_bias"] = -0.5

        return result

    @staticmethod
    def _weighted_midpoint(bids: list, asks: list) -> float:
        """
        Compute volume-weighted midpoint (VWAP of top bid/ask levels).

        This gives a fairer "true price" that accounts for where the liquidity
        actually sits, not just the best bid/ask.
        """
        if not bids or not asks:
            return 0.0

        bid_vwap_num = sum(p * q for p, q in bids)
        bid_vwap_den = sum(q for _, q in bids)
        ask_vwap_num = sum(p * q for p, q in asks)
        ask_vwap_den = sum(q for _, q in asks)

        if bid_vwap_den == 0 or ask_vwap_den == 0:
            return (bids[0][0] + asks[0][0]) / 2.0 if bids and asks else 0.0

        bid_vwap = bid_vwap_num / bid_vwap_den
        ask_vwap = ask_vwap_num / ask_vwap_den

        # Weight by total volume on each side
        total = bid_vwap_den + ask_vwap_den
        return (bid_vwap * ask_vwap_den + ask_vwap * bid_vwap_den) / total

    # ------------------------------------------------------------------
    # Historical snapshots for training
    # ------------------------------------------------------------------

    def collect_depth_timeseries(
        self,
        interval_seconds: int = 60,
        duration_seconds: int = 3600,
        depth_limit: int = 50,
    ) -> pd.DataFrame:
        """
        Collect a time series of order book snapshots for training.

        Runs for `duration_seconds`, sampling every `interval_seconds`.
        Returns DataFrame with one row per snapshot, all features as columns.

        Usage:
            analyzer = OrderBookDepthAnalyzer()
            df = analyzer.collect_depth_timeseries(interval_seconds=60, duration_seconds=3600)
            df.to_csv("data/btc_depth_features_1h.csv", index=False)
        """
        snapshots = []
        n_samples = duration_seconds // interval_seconds
        logger.info(
            "Collecting %d depth snapshots over %ds (every %ds)...",
            n_samples, duration_seconds, interval_seconds,
        )

        for i in range(n_samples):
            features = self.analyze(depth_limit=depth_limit)
            features["timestamp"] = datetime.now(timezone.utc)
            snapshots.append(features)

            if (i + 1) % 10 == 0:
                logger.info("  ... collected %d/%d snapshots", i + 1, n_samples)

            if i < n_samples - 1:
                time.sleep(interval_seconds)

        if not snapshots:
            return pd.DataFrame()

        df = pd.DataFrame(snapshots)
        logger.info("Collected %d depth snapshots with %d features", len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Merge with candle data for training
    # ------------------------------------------------------------------

    def merge_depth_with_candles(
        self,
        candle_df: pd.DataFrame,
        depth_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge pre-collected depth snapshots with candle data using merge_asof.

        Args:
            candle_df: OHLCV DataFrame with 'timestamp' column
            depth_df: depth features DataFrame from collect_depth_timeseries()

        Returns:
            Enriched candle DataFrame with depth features.
        """
        df = candle_df.copy()
        if "timestamp" not in df.columns or "timestamp" not in depth_df.columns:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        depth_df = depth_df.copy()
        depth_df["timestamp"] = pd.to_datetime(depth_df["timestamp"], errors="coerce", utc=True)

        # Select only numeric feature columns for merge
        feature_cols = [c for c in depth_df.columns if c.startswith("ob_") and c != "ob_ready"]
        merge_cols = ["timestamp"] + feature_cols

        df = df.sort_values("timestamp")
        depth_sorted = depth_df[merge_cols].sort_values("timestamp")

        df = _safe_merge_asof(df, depth_sorted, on="timestamp", direction="backward")

        logger.info("Merged %d depth snapshots into %d candles (%d new features)",
                     len(depth_df), len(df), len(feature_cols))
        return df

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _default_features() -> dict:
        return {
            "ob_best_bid": 0.0,
            "ob_best_ask": 0.0,
            "ob_midpoint": 0.0,
            "ob_spread": 0.0,
            "ob_spread_bps": 0.0,
            "ob_imbalance_5": 0.0,
            "ob_imbalance_10": 0.0,
            "ob_imbalance_20": 0.0,
            "ob_bid_vol_5": 0.0,
            "ob_ask_vol_5": 0.0,
            "ob_bid_vol_10": 0.0,
            "ob_ask_vol_10": 0.0,
            "ob_bid_vol_20": 0.0,
            "ob_ask_vol_20": 0.0,
            "ob_cum_imbalance_10bps": 0.0,
            "ob_cum_imbalance_25bps": 0.0,
            "ob_cum_imbalance_50bps": 0.0,
            "ob_cum_imbalance_100bps": 0.0,
            "ob_cum_depth_10bps_btc": 0.0,
            "ob_cum_depth_25bps_btc": 0.0,
            "ob_cum_depth_50bps_btc": 0.0,
            "ob_cum_depth_100bps_btc": 0.0,
            "ob_cum_depth_10bps_usd": 0.0,
            "ob_cum_depth_25bps_usd": 0.0,
            "ob_cum_depth_50bps_usd": 0.0,
            "ob_cum_depth_100bps_usd": 0.0,
            "ob_bid_slope": 0.0,
            "ob_ask_slope": 0.0,
            "ob_slope_imbalance": 0.0,
            "ob_whale_bid_wall": 0.0,
            "ob_whale_ask_wall": 0.0,
            "ob_whale_bid_wall_size": 0.0,
            "ob_whale_ask_wall_size": 0.0,
            "ob_whale_bid_wall_dist_bps": 0.0,
            "ob_whale_ask_wall_dist_bps": 0.0,
            "ob_whale_wall_bias": 0.0,
            "ob_weighted_midpoint": 0.0,
            "ob_wmid_vs_mid": 0.0,
            "ob_bid_concentration": 0.0,
            "ob_ask_concentration": 0.0,
            "ob_bid_density_10bps": 0,
            "ob_ask_density_10bps": 0,
            "ob_ready": False,
        }


# ---------------------------------------------------------------------------
# Convenience function for BTC prediction pipeline
# ---------------------------------------------------------------------------

def fetch_btc_depth_snapshot(symbol: str = "BTCUSDT") -> dict:
    """
    One-shot convenience: fetch BTC order book and return features dict.
    Suitable for injecting into macro_context in supervisor.py.
    """
    analyzer = OrderBookDepthAnalyzer(symbol=symbol)
    return analyzer.analyze(depth_limit=100)
