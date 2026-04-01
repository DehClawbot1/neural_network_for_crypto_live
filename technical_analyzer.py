import logging
import time
import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TechnicalAnalyzer:
    """
    Pillar 3: Technical Analysis (The Trend)
    Fetches raw BTC/USDT daily candles from the Binance Public API to determine
    the macro trend configuration (e.g. above/below the 200-day SMA).
    """

    def __init__(self, cache_ttl_seconds=3600):
        # Cache daily TA context for an hour since the 200 SMA doesn't move drastically intra-day
        self.cache_ttl = cache_ttl_seconds
        self._cached_context = None
        self._last_fetch_time = 0

    def _safe_float(self, value, default=0.0):
        try:
            val = float(value)
            return default if pd.isna(val) else val
        except Exception:
            return default

    def analyze(self) -> dict:
        """
        Fetches the latest daily candles and calculates the 200 SMA and 21 EMA.
        Returns a dictionary expressing the current technical posture.
        """
        now = time.time()
        if self._cached_context and (now - self._last_fetch_time) < self.cache_ttl:
            return self._cached_context

        # Safe defaults if API fails
        context = {
            "btc_price": None,
            "sma_200": None,
            "ema_21": None,
            "market_structure": "UNKNOWN",
            "distance_to_sma_200": 0.0,
            "trend_score": 0.5, # 0.0 (Bearish) to 1.0 (Bullish)
        }

        url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1d&limit=250"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if len(data) < 200:
                logging.warning("TechnicalAnalyzer: Not enough daily candles returned to calculate 200 SMA.")
                return context

            # Binance Kline format: [Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore]
            # Extract just the Close prices
            closes = [self._safe_float(candle[4]) for candle in data]
            df = pd.DataFrame({"close": closes})

            # Calculate Moving Averages
            df["sma_200"] = df["close"].rolling(window=200).mean()
            df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

            latest = df.iloc[-1]
            current_price = latest["close"]
            sma_200 = latest["sma_200"]
            ema_21 = latest["ema_21"]

            if pd.isna(sma_200):
                return context

            distance = (current_price - sma_200) / sma_200
            
            # Formulate Market Structure Trend
            if current_price > sma_200 and ema_21 > sma_200:
                structure = "BULLISH"
                trend_score = min(1.0, 0.7 + (distance * 2)) # Caps at 1.0 (very bullish if far above SMA)
            elif current_price < sma_200 and ema_21 < sma_200:
                structure = "BEARISH"
                trend_score = max(0.0, 0.3 + (distance * 2)) # distance is negative here, approaches 0
            else:
                structure = "MIXED"
                trend_score = 0.5
                
            context.update({
                "btc_price": current_price,
                "sma_200": round(sma_200, 2),
                "ema_21": round(ema_21, 2),
                "market_structure": structure,
                "distance_to_sma_200": round(distance, 4),
                "trend_score": round(trend_score, 3),
            })
            
            logging.info(
                f"TechnicalAnalyzer Evaluated: Price=${current_price:.0f} | 200SMA=${sma_200:.0f} | "
                f"Trend: {structure} | Distance to 200SMA: {distance:.2%}"
            )

            self._cached_context = context
            self._last_fetch_time = now
            return context

        except Exception as e:
            logging.warning(f"TechnicalAnalyzer: Failed to fetch Binance klines: {e}")
            return context
