import logging
import os
import time
import pandas as pd
import numpy as np
from candle_data_service import CandleDataService
from btc_live_price_tracker import BTCLivePriceTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TechnicalAnalyzer:
    """
    Pillar 3: Technical Analysis (The Trend)
    Fetches raw BTC/USDT daily candles from the Binance Public API to determine
    the macro trend configuration (e.g. above/below the 200-day SMA).
    """

    def __init__(
        self,
        cache_ttl_seconds=300,
        candle_data_service: CandleDataService | None = None,
        btc_live_tracker: BTCLivePriceTracker | None = None,
    ):
        # Cache TA context briefly so intraday trend signals stay relevant.
        self.cache_ttl = cache_ttl_seconds
        self._cached_context = None
        self._last_fetch_time = 0
        self.candle_data_service = candle_data_service or CandleDataService(symbol="BTCUSDT")
        self.btc_live_tracker = btc_live_tracker or BTCLivePriceTracker(
            candle_data_service=self.candle_data_service,
        )

    def _safe_float(self, value, default=0.0):
        try:
            val = float(value)
            return default if pd.isna(val) else val
        except Exception:
            return default

    def _smma(self, series: pd.Series, period: int) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").astype(float)
        result = pd.Series(np.nan, index=values.index, dtype=float)
        if len(values) < period:
            return result
        seed = values.iloc[:period].mean()
        result.iloc[period - 1] = seed
        for idx in range(period, len(values)):
            prev = result.iloc[idx - 1]
            curr = values.iloc[idx]
            if pd.isna(prev) or pd.isna(curr):
                continue
            result.iloc[idx] = ((prev * (period - 1)) + curr) / period
        return result

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = pd.to_numeric(df["high"], errors="coerce").astype(float)
        low = pd.to_numeric(df["low"], errors="coerce").astype(float)
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
            dtype=float,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
            dtype=float,
        )

        tr_components = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = self._smma(true_range, period)
        plus_di = 100.0 * self._smma(plus_dm, period) / atr.replace(0, np.nan)
        minus_di = 100.0 * self._smma(minus_dm, period) / atr.replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return self._smma(dx, period)

    def _latest_confirmed_fractals(self, df: pd.DataFrame) -> tuple[float, float]:
        if df is None or len(df) < 5:
            return np.nan, np.nan

        highs = pd.to_numeric(df["high"], errors="coerce").astype(float)
        lows = pd.to_numeric(df["low"], errors="coerce").astype(float)
        bullish = pd.Series(False, index=df.index, dtype=bool)
        bearish = pd.Series(False, index=df.index, dtype=bool)

        for idx in range(2, len(df) - 2):
            center_high = highs.iloc[idx]
            center_low = lows.iloc[idx]
            if not np.isfinite(center_high) or not np.isfinite(center_low):
                continue
            prior_highs = highs.iloc[idx - 2:idx]
            next_highs = highs.iloc[idx + 1:idx + 3]
            prior_lows = lows.iloc[idx - 2:idx]
            next_lows = lows.iloc[idx + 1:idx + 3]
            if center_high > prior_highs.max() and center_high > next_highs.max():
                bullish.iloc[idx] = True
            if center_low < prior_lows.min() and center_low < next_lows.min():
                bearish.iloc[idx] = True

        confirmed_bullish = highs[bullish].dropna()
        confirmed_bearish = lows[bearish].dropna()
        latest_bullish = float(confirmed_bullish.iloc[-1]) if not confirmed_bullish.empty else np.nan
        latest_bearish = float(confirmed_bearish.iloc[-1]) if not confirmed_bearish.empty else np.nan
        return latest_bullish, latest_bearish

    def _compute_intraday_trend_context(self) -> dict:
        adx_threshold = max(1.0, self._safe_float(os.getenv("TECHNICAL_TREND_ADX_MIN", "18"), 18.0))
        context = {
            "alligator_jaw": None,
            "alligator_teeth": None,
            "alligator_lips": None,
            "alligator_alignment": "NEUTRAL",
            "alligator_bullish": False,
            "alligator_bearish": False,
            "adx_value": None,
            "adx_threshold": round(adx_threshold, 2),
            "adx_trending": False,
            "anchored_vwap": None,
            "anchored_vwap_anchor": "utc_session_open",
            "price_vs_anchored_vwap": 0.0,
            "price_above_anchored_vwap": False,
            "price_below_anchored_vwap": False,
            "btc_trend_bias": "NEUTRAL",
            "btc_trend_confluence": 0.0,
            "latest_bullish_fractal": None,
            "latest_bearish_fractal": None,
            "long_fractal_breakout": False,
            "short_fractal_breakout": False,
            "fractal_trigger_direction": "NEUTRAL",
            "fractal_entry_ready": False,
        }

        intraday = self.candle_data_service.refresh_latest_closed_candles(
            "15m",
            limit=400,
            timezone_name="UTC",
        )
        if intraday is None or intraday.empty or len(intraday) < 80:
            return context

        intraday["median_price"] = (intraday["high"] + intraday["low"]) / 2.0
        intraday["jaw"] = self._smma(intraday["median_price"], 13)
        intraday["teeth"] = self._smma(intraday["median_price"], 8)
        intraday["lips"] = self._smma(intraday["median_price"], 5)
        intraday["adx"] = self._compute_adx(intraday, period=14)
        intraday["typical_price"] = (intraday["high"] + intraday["low"] + intraday["close"]) / 3.0
        latest_bullish_fractal, latest_bearish_fractal = self._latest_confirmed_fractals(intraday)

        latest = intraday.iloc[-1]
        current_price = self._safe_float(latest["close"], np.nan)
        jaw = self._safe_float(latest["jaw"], np.nan)
        teeth = self._safe_float(latest["teeth"], np.nan)
        lips = self._safe_float(latest["lips"], np.nan)
        adx_value = self._safe_float(latest["adx"], np.nan)

        today_start = pd.Timestamp.now(tz="UTC").normalize()
        anchored = intraday[intraday["open_time"] >= today_start].copy()
        if anchored.empty:
            anchored = intraday.tail(96).copy()
        pv = anchored["typical_price"] * anchored["volume"].fillna(0.0)
        vv = anchored["volume"].fillna(0.0)
        denom = vv.sum()
        anchored_vwap = float(pv.sum() / denom) if denom > 0 else np.nan

        alligator_alignment = "NEUTRAL"
        if all(np.isfinite([jaw, teeth, lips, current_price])):
            if lips > teeth > jaw and current_price >= lips:
                alligator_alignment = "BULLISH"
            elif lips < teeth < jaw and current_price <= lips:
                alligator_alignment = "BEARISH"

        price_vs_vwap = 0.0
        price_above_vwap = False
        price_below_vwap = False
        if np.isfinite(current_price) and np.isfinite(anchored_vwap) and anchored_vwap > 0:
            price_vs_vwap = (current_price - anchored_vwap) / anchored_vwap
            price_above_vwap = current_price > anchored_vwap
            price_below_vwap = current_price < anchored_vwap

        adx_trending = bool(np.isfinite(adx_value) and adx_value >= adx_threshold)
        trend_bias = "NEUTRAL"
        if alligator_alignment == "BULLISH" and adx_trending and price_above_vwap:
            trend_bias = "LONG"
        elif alligator_alignment == "BEARISH" and adx_trending and price_below_vwap:
            trend_bias = "SHORT"

        long_fractal_breakout = bool(np.isfinite(latest_bullish_fractal) and np.isfinite(current_price) and current_price > latest_bullish_fractal)
        short_fractal_breakout = bool(np.isfinite(latest_bearish_fractal) and np.isfinite(current_price) and current_price < latest_bearish_fractal)
        fractal_trigger_direction = "NEUTRAL"
        fractal_entry_ready = False
        if trend_bias == "LONG" and long_fractal_breakout:
            fractal_trigger_direction = "LONG"
            fractal_entry_ready = True
        elif trend_bias == "SHORT" and short_fractal_breakout:
            fractal_trigger_direction = "SHORT"
            fractal_entry_ready = True

        confluence = 0.0
        confluence += 0.4 if alligator_alignment != "NEUTRAL" else 0.0
        confluence += 0.3 if adx_trending else 0.0
        confluence += 0.3 if (price_above_vwap or price_below_vwap) else 0.0
        confluence += 0.1 if fractal_entry_ready else 0.0

        context.update(
            {
                "alligator_jaw": round(jaw, 2) if np.isfinite(jaw) else None,
                "alligator_teeth": round(teeth, 2) if np.isfinite(teeth) else None,
                "alligator_lips": round(lips, 2) if np.isfinite(lips) else None,
                "alligator_alignment": alligator_alignment,
                "alligator_bullish": alligator_alignment == "BULLISH",
                "alligator_bearish": alligator_alignment == "BEARISH",
                "adx_value": round(adx_value, 2) if np.isfinite(adx_value) else None,
                "adx_trending": adx_trending,
                "anchored_vwap": round(anchored_vwap, 2) if np.isfinite(anchored_vwap) else None,
                "price_vs_anchored_vwap": round(price_vs_vwap, 4),
                "price_above_anchored_vwap": price_above_vwap,
                "price_below_anchored_vwap": price_below_vwap,
                "btc_trend_bias": trend_bias,
                "btc_trend_confluence": round(confluence, 3),
                "latest_bullish_fractal": round(latest_bullish_fractal, 2) if np.isfinite(latest_bullish_fractal) else None,
                "latest_bearish_fractal": round(latest_bearish_fractal, 2) if np.isfinite(latest_bearish_fractal) else None,
                "long_fractal_breakout": long_fractal_breakout,
                "short_fractal_breakout": short_fractal_breakout,
                "fractal_trigger_direction": fractal_trigger_direction,
                "fractal_entry_ready": fractal_entry_ready,
            }
        )
        return context

    def analyze(self) -> dict:
        """
        Fetches the latest daily candles and calculates the 200 SMA and 21 EMA.
        Returns a dictionary expressing the current technical posture.
        """
        now = time.time()
        live_context = self.btc_live_tracker.analyze()
        if self._cached_context and (now - self._last_fetch_time) < self.cache_ttl:
            cached = dict(self._cached_context)
            cached.update(live_context)
            return cached

        # Safe defaults if API fails
        context = {
            "btc_price": None,
            "sma_200": None,
            "ema_21": None,
            "market_structure": "UNKNOWN",
            "distance_to_sma_200": 0.0,
            "trend_score": 0.5, # 0.0 (Bearish) to 1.0 (Bullish)
        }
        context.update(self._compute_intraday_trend_context())

        try:
            daily_df = self.candle_data_service.refresh_latest_closed_candles(
                "1d",
                limit=250,
                timezone_name="UTC",
            )
            if daily_df is None or daily_df.empty or len(daily_df) < 200:
                logging.warning("TechnicalAnalyzer: Not enough daily candles returned to calculate 200 SMA.")
                context.update(live_context)
                self._cached_context = dict(context)
                self._last_fetch_time = now
                return context

            df = daily_df[["close"]].copy()

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
                f"Trend: {structure} | Distance to 200SMA: {distance:.2%} | "
                f"Alligator={context.get('alligator_alignment')} | ADX={context.get('adx_value')} | "
                f"AVWAP={context.get('anchored_vwap')} | Bias={context.get('btc_trend_bias')} | "
                f"FractalReady={context.get('fractal_entry_ready')}"
            )

            context.update(live_context)
            self._cached_context = dict(context)
            self._last_fetch_time = now
            return context

        except Exception as e:
            logging.warning(f"TechnicalAnalyzer: Failed to fetch closed BTC candles: {e}")
            context.update(live_context)
            return context
