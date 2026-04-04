"""
BTC On-Chain & Derivatives Feature Fetcher

Fetches alpha-generating features that pure OHLCV technicals miss:
  1. Funding rate (derivatives positioning bias)
  2. Open interest (leverage in the system)
  3. Long/short ratio (crowd positioning)
  4. Taker buy/sell volume ratio (aggressive order flow)
  5. Liquidation data (forced cascades)

These features measure INTENT and POSITIONING — not just past price.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_FUTURES_BASE = "https://fapi.binance.com"


class BTCDerivativesFeatures:
    """
    Fetch derivatives/on-chain features for BTC from Binance Futures API.
    Designed to be called during dataset building to enrich OHLCV data.
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_funding_rate_history(self, limit: int = 1000) -> pd.DataFrame:
        """Fetch historical funding rates (8h intervals)."""
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
                params={"symbol": self.symbol, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            df["fundingRate"] = df["fundingRate"].astype(float)
            df.rename(columns={"fundingTime": "timestamp"}, inplace=True)
            return df[["timestamp", "fundingRate"]].sort_values("timestamp").reset_index(drop=True)
        except Exception as exc:
            logger.warning("Failed to fetch funding rates: %s", exc)
            return pd.DataFrame(columns=["timestamp", "fundingRate"])

    def fetch_open_interest_history(self, period: str = "15m", limit: int = 500) -> pd.DataFrame:
        """Fetch historical open interest."""
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist",
                params={"symbol": self.symbol, "period": period, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
            df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
            return df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].sort_values("timestamp").reset_index(drop=True)
        except Exception as exc:
            logger.warning("Failed to fetch open interest: %s", exc)
            return pd.DataFrame(columns=["timestamp", "sumOpenInterest", "sumOpenInterestValue"])

    def fetch_long_short_ratio(self, period: str = "15m", limit: int = 500) -> pd.DataFrame:
        """Fetch top trader long/short account ratio."""
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES_BASE}/futures/data/topLongShortAccountRatio",
                params={"symbol": self.symbol, "period": period, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["longShortRatio"] = df["longShortRatio"].astype(float)
            df["longAccount"] = df["longAccount"].astype(float)
            df["shortAccount"] = df["shortAccount"].astype(float)
            return df[["timestamp", "longShortRatio", "longAccount", "shortAccount"]].sort_values("timestamp").reset_index(drop=True)
        except Exception as exc:
            logger.warning("Failed to fetch long/short ratio: %s", exc)
            return pd.DataFrame(columns=["timestamp", "longShortRatio", "longAccount", "shortAccount"])

    def fetch_taker_buy_sell_volume(self, period: str = "15m", limit: int = 500) -> pd.DataFrame:
        """Fetch taker buy/sell volume ratio."""
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES_BASE}/futures/data/takerlongshortRatio",
                params={"symbol": self.symbol, "period": period, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["buySellRatio"] = df["buySellRatio"].astype(float)
            df["buyVol"] = df["buyVol"].astype(float)
            df["sellVol"] = df["sellVol"].astype(float)
            return df[["timestamp", "buySellRatio", "buyVol", "sellVol"]].sort_values("timestamp").reset_index(drop=True)
        except Exception as exc:
            logger.warning("Failed to fetch taker volume: %s", exc)
            return pd.DataFrame(columns=["timestamp", "buySellRatio", "buyVol", "sellVol"])

    def fetch_all_and_merge(self, candle_df: pd.DataFrame, period: str = "15m") -> pd.DataFrame:
        """
        Fetch all derivatives features and merge into candle DataFrame.
        Uses forward-fill to align 8h funding rate to 15m candles.
        """
        df = candle_df.copy()
        if "timestamp" not in df.columns:
            logger.warning("BTCDerivativesFeatures: no timestamp column, skipping")
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

        # Fetch all data sources
        funding_df = self.fetch_funding_rate_history(limit=1000)
        oi_df = self.fetch_open_interest_history(period=period, limit=500)
        ls_df = self.fetch_long_short_ratio(period=period, limit=500)
        taker_df = self.fetch_taker_buy_sell_volume(period=period, limit=500)

        # Merge using asof (nearest timestamp, backward)
        df = df.sort_values("timestamp")

        if not funding_df.empty:
            df = pd.merge_asof(df, funding_df, on="timestamp", direction="backward")
        else:
            df["fundingRate"] = np.nan

        if not oi_df.empty:
            df = pd.merge_asof(df, oi_df, on="timestamp", direction="backward")
        else:
            df["sumOpenInterest"] = np.nan
            df["sumOpenInterestValue"] = np.nan

        if not ls_df.empty:
            df = pd.merge_asof(df, ls_df, on="timestamp", direction="backward")
        else:
            df["longShortRatio"] = np.nan
            df["longAccount"] = np.nan
            df["shortAccount"] = np.nan

        if not taker_df.empty:
            df = pd.merge_asof(df, taker_df, on="timestamp", direction="backward")
        else:
            df["buySellRatio"] = np.nan
            df["buyVol"] = np.nan
            df["sellVol"] = np.nan

        # Compute derived features
        if "sumOpenInterest" in df.columns:
            df["oi_change_pct"] = df["sumOpenInterest"].pct_change()
            df["oi_change_pct_5"] = df["sumOpenInterest"].pct_change(5)

        if "fundingRate" in df.columns:
            df["funding_rate_zscore"] = (
                (df["fundingRate"] - df["fundingRate"].rolling(50).mean())
                / df["fundingRate"].rolling(50).std().replace(0, np.nan)
            )
            # Extreme funding = contrarian signal
            df["funding_extreme"] = (df["fundingRate"].abs() > df["fundingRate"].abs().rolling(100).quantile(0.9)).astype(float)

        if "longShortRatio" in df.columns:
            df["ls_ratio_change"] = df["longShortRatio"].diff()
            df["crowd_contrarian"] = (df["longShortRatio"] - 1.0).abs()

        if "buySellRatio" in df.columns:
            df["taker_imbalance"] = df["buySellRatio"] - 1.0
            df["taker_imbalance_sma_5"] = df["taker_imbalance"].rolling(5).mean()

        logger.info(
            "BTCDerivativesFeatures: merged %d funding, %d OI, %d L/S, %d taker rows into %d candles",
            len(funding_df), len(oi_df), len(ls_df), len(taker_df), len(df),
        )
        return df
