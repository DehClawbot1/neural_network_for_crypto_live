import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from btc_live_price_tracker import BTCLivePriceTracker
from candle_data_service import CandleDataService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Pillar 3: Technical Analysis (The Trend)
    Fetches BTC candle context and derives both legacy trend posture and
    newer Phase 6 volatility / momentum quality indicators.
    """

    def __init__(
        self,
        cache_ttl_seconds=300,
        candle_data_service: CandleDataService | None = None,
        btc_live_tracker: BTCLivePriceTracker | None = None,
        logs_dir: str = "logs",
    ):
        self.cache_ttl = cache_ttl_seconds
        self._cached_context = None
        self._last_fetch_time = 0
        self.candle_data_service = candle_data_service or CandleDataService(symbol="BTCUSDT")
        self.btc_live_tracker = btc_live_tracker or BTCLivePriceTracker(
            candle_data_service=self.candle_data_service,
        )
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_file = self.logs_dir / "technical_regime_snapshot.csv"

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

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").astype(float)
        return values.ewm(span=period, adjust=False).mean()

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

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = pd.to_numeric(df["high"], errors="coerce").astype(float)
        low = pd.to_numeric(df["low"], errors="coerce").astype(float)
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        tr_components = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        return self._smma(true_range, period)

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        close = pd.to_numeric(series, errors="coerce").astype(float)
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        avg_gain = self._smma(gains, period)
        avg_loss = self._smma(losses, period)
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def _compute_macd(self, series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        close = pd.to_numeric(series, errors="coerce").astype(float)
        fast = self._ema(close, 12)
        slow = self._ema(close, 26)
        macd_line = fast - slow
        signal_line = self._ema(macd_line, 9)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

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

    def _classify_volatility_regime(self, atr_pct: float, realized_vol_1h: float, realized_vol_4h: float) -> tuple[str, float]:
        atr_pct = max(0.0, self._safe_float(atr_pct, 0.0))
        realized_vol_1h = max(0.0, self._safe_float(realized_vol_1h, 0.0))
        realized_vol_4h = max(0.0, self._safe_float(realized_vol_4h, 0.0))
        score = min(1.0, (atr_pct / 0.012) * 0.45 + (realized_vol_1h / 0.010) * 0.35 + (realized_vol_4h / 0.020) * 0.20)
        if score >= 0.90:
            return "EXTREME", score
        if score >= 0.65:
            return "HIGH", score
        if score <= 0.25:
            return "LOW", score
        return "NORMAL", score

    def _compute_rsi_divergence_score(self, closes: pd.Series, rsi: pd.Series, lookback: int = 12) -> float:
        if closes is None or rsi is None or len(closes) < lookback + 2 or len(rsi) < lookback + 2:
            return 0.0
        recent_close = pd.to_numeric(closes.iloc[-(lookback + 1):], errors="coerce").astype(float)
        recent_rsi = pd.to_numeric(rsi.iloc[-(lookback + 1):], errors="coerce").astype(float)
        current_close = self._safe_float(recent_close.iloc[-1], None)
        current_rsi = self._safe_float(recent_rsi.iloc[-1], None)
        prior_close = recent_close.iloc[:-1]
        prior_rsi = recent_rsi.iloc[:-1]
        if current_close is None or current_rsi is None or prior_close.empty or prior_rsi.empty:
            return 0.0
        bearish_idx = prior_close.idxmax()
        bullish_idx = prior_close.idxmin()
        bearish_close = self._safe_float(prior_close.loc[bearish_idx], None)
        bullish_close = self._safe_float(prior_close.loc[bullish_idx], None)
        bearish_rsi = self._safe_float(prior_rsi.loc[bearish_idx], None)
        bullish_rsi = self._safe_float(prior_rsi.loc[bullish_idx], None)
        if None in (bearish_close, bullish_close, bearish_rsi, bullish_rsi):
            return 0.0
        score = 0.0
        if current_close >= bearish_close * 0.998 and current_rsi < bearish_rsi - 2.0:
            score -= min(1.0, (bearish_rsi - current_rsi) / 12.0)
        if current_close <= bullish_close * 1.002 and current_rsi > bullish_rsi + 2.0:
            score += min(1.0, (current_rsi - bullish_rsi) / 12.0)
        return float(np.clip(score, -1.0, 1.0))

    def _classify_momentum_regime(
        self,
        *,
        rsi_value: float,
        macd_hist: float,
        macd_hist_slope: float,
        rsi_divergence_score: float,
        trend_persistence: float,
    ) -> tuple[str, float]:
        rsi_value = self._safe_float(rsi_value, 50.0)
        macd_hist = self._safe_float(macd_hist, 0.0)
        macd_hist_slope = self._safe_float(macd_hist_slope, 0.0)
        rsi_divergence_score = self._safe_float(rsi_divergence_score, 0.0)
        trend_persistence = self._safe_float(trend_persistence, 0.0)

        long_score = 0.0
        short_score = 0.0
        if rsi_value >= 55:
            long_score += min(0.25, (rsi_value - 50.0) / 50.0)
        if rsi_value <= 45:
            short_score += min(0.25, (50.0 - rsi_value) / 50.0)
        if macd_hist > 0:
            long_score += min(0.25, abs(macd_hist) * 40.0)
        elif macd_hist < 0:
            short_score += min(0.25, abs(macd_hist) * 40.0)
        if macd_hist_slope > 0:
            long_score += min(0.15, abs(macd_hist_slope) * 120.0)
        elif macd_hist_slope < 0:
            short_score += min(0.15, abs(macd_hist_slope) * 120.0)
        if rsi_divergence_score > 0:
            long_score += min(0.20, abs(rsi_divergence_score) * 0.20)
        elif rsi_divergence_score < 0:
            short_score += min(0.20, abs(rsi_divergence_score) * 0.20)
        long_score += min(0.15, trend_persistence * 0.15)
        short_score += min(0.15, (1.0 - trend_persistence) * 0.15)

        if rsi_value >= 72 and rsi_divergence_score < -0.20:
            return "OVERBOUGHT_EXHAUSTION", float(np.clip(short_score, 0.0, 1.0))
        if rsi_value <= 28 and rsi_divergence_score > 0.20:
            return "OVERSOLD_EXHAUSTION", float(np.clip(long_score, 0.0, 1.0))
        if long_score >= short_score + 0.12 and long_score >= 0.35:
            return "BULLISH", float(np.clip(long_score, 0.0, 1.0))
        if short_score >= long_score + 0.12 and short_score >= 0.35:
            return "BEARISH", float(np.clip(short_score, 0.0, 1.0))
        return "NEUTRAL", float(np.clip(max(long_score, short_score) * 0.5, 0.0, 1.0))

    def _write_snapshot(self, context: dict):
        snapshot_keys = [
            "technical_timestamp",
            "btc_price",
            "market_structure",
            "trend_score",
            "btc_atr_pct_15m",
            "btc_realized_vol_1h",
            "btc_realized_vol_4h",
            "btc_volatility_regime",
            "btc_volatility_regime_score",
            "btc_trend_persistence",
            "btc_rsi_14",
            "btc_rsi_distance_mid",
            "btc_rsi_divergence_score",
            "btc_macd",
            "btc_macd_signal",
            "btc_macd_hist",
            "btc_macd_hist_slope",
            "btc_momentum_regime",
            "btc_momentum_confluence",
        ]
        payload = {key: context.get(key) for key in snapshot_keys}
        try:
            pd.DataFrame([payload]).to_csv(
                self.snapshot_file,
                mode="a",
                header=not self.snapshot_file.exists(),
                index=False,
            )
        except Exception as exc:
            logger.debug("TechnicalAnalyzer: Snapshot write failed: %s", exc)

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
            "btc_atr_pct_15m": 0.0,
            "btc_realized_vol_1h": 0.0,
            "btc_realized_vol_4h": 0.0,
            "btc_volatility_regime": "NORMAL",
            "btc_volatility_regime_score": 0.0,
            "btc_trend_persistence": 0.0,
            "btc_rsi_14": 50.0,
            "btc_rsi_distance_mid": 0.0,
            "btc_rsi_divergence_score": 0.0,
            "btc_macd": 0.0,
            "btc_macd_signal": 0.0,
            "btc_macd_hist": 0.0,
            "btc_macd_hist_slope": 0.0,
            "btc_momentum_regime": "NEUTRAL",
            "btc_momentum_confluence": 0.0,
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
        intraday["atr"] = self._compute_atr(intraday, period=14)
        intraday["rsi_14"] = self._compute_rsi(intraday["close"], period=14)
        intraday["macd_line"], intraday["macd_signal"], intraday["macd_hist"] = self._compute_macd(intraday["close"])
        intraday["macd_hist_slope"] = intraday["macd_hist"].diff(3)
        intraday["close_return"] = pd.to_numeric(intraday["close"], errors="coerce").astype(float).pct_change()
        intraday["realized_vol_1h"] = intraday["close_return"].rolling(4).std()
        intraday["realized_vol_4h"] = intraday["close_return"].rolling(16).std()
        intraday["ema_20_intraday"] = self._ema(intraday["close"], 20)
        intraday["typical_price"] = (intraday["high"] + intraday["low"] + intraday["close"]) / 3.0
        latest_bullish_fractal, latest_bearish_fractal = self._latest_confirmed_fractals(intraday)

        latest = intraday.iloc[-1]
        current_price = self._safe_float(latest["close"], np.nan)
        jaw = self._safe_float(latest["jaw"], np.nan)
        teeth = self._safe_float(latest["teeth"], np.nan)
        lips = self._safe_float(latest["lips"], np.nan)
        adx_value = self._safe_float(latest["adx"], np.nan)
        atr_value = self._safe_float(latest["atr"], np.nan)
        atr_pct = float(atr_value / current_price) if np.isfinite(atr_value) and np.isfinite(current_price) and current_price > 0 else 0.0
        realized_vol_1h = self._safe_float(latest.get("realized_vol_1h"), 0.0)
        realized_vol_4h = self._safe_float(latest.get("realized_vol_4h"), 0.0)
        rsi_value = self._safe_float(latest.get("rsi_14"), 50.0)
        rsi_distance_mid = float((rsi_value - 50.0) / 50.0)
        macd_value = self._safe_float(latest.get("macd_line"), 0.0)
        macd_signal = self._safe_float(latest.get("macd_signal"), 0.0)
        macd_hist = self._safe_float(latest.get("macd_hist"), 0.0)
        macd_hist_slope = self._safe_float(latest.get("macd_hist_slope"), 0.0)
        recent_trend_window = intraday.tail(8).copy()
        trend_persistence = 0.0
        if not recent_trend_window.empty:
            trend_persistence = float(
                np.clip(
                    (
                        recent_trend_window["close"].astype(float)
                        > recent_trend_window["ema_20_intraday"].astype(float)
                    ).mean(),
                    0.0,
                    1.0,
                )
            )
        rsi_divergence_score = self._compute_rsi_divergence_score(intraday["close"], intraday["rsi_14"], lookback=12)
        volatility_regime, volatility_regime_score = self._classify_volatility_regime(atr_pct, realized_vol_1h, realized_vol_4h)
        momentum_regime, momentum_confluence = self._classify_momentum_regime(
            rsi_value=rsi_value,
            macd_hist=macd_hist,
            macd_hist_slope=macd_hist_slope,
            rsi_divergence_score=rsi_divergence_score,
            trend_persistence=trend_persistence,
        )

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
                "btc_atr_pct_15m": round(atr_pct, 6),
                "btc_realized_vol_1h": round(realized_vol_1h, 6),
                "btc_realized_vol_4h": round(realized_vol_4h, 6),
                "btc_volatility_regime": volatility_regime,
                "btc_volatility_regime_score": round(volatility_regime_score, 4),
                "btc_trend_persistence": round(trend_persistence, 4),
                "btc_rsi_14": round(rsi_value, 2),
                "btc_rsi_distance_mid": round(rsi_distance_mid, 4),
                "btc_rsi_divergence_score": round(rsi_divergence_score, 4),
                "btc_macd": round(macd_value, 6),
                "btc_macd_signal": round(macd_signal, 6),
                "btc_macd_hist": round(macd_hist, 6),
                "btc_macd_hist_slope": round(macd_hist_slope, 6),
                "btc_momentum_regime": momentum_regime,
                "btc_momentum_confluence": round(momentum_confluence, 4),
            }
        )
        return context

    def analyze(self) -> dict:
        now = time.time()
        live_context = self.btc_live_tracker.analyze()
        if self._cached_context and (now - self._last_fetch_time) < self.cache_ttl:
            cached = dict(self._cached_context)
            cached.update(live_context)
            return cached

        context = {
            "btc_price": None,
            "sma_200": None,
            "ema_21": None,
            "market_structure": "UNKNOWN",
            "distance_to_sma_200": 0.0,
            "trend_score": 0.5,
            "technical_timestamp": datetime.now(timezone.utc).isoformat(),
            "btc_atr_pct_15m": 0.0,
            "btc_realized_vol_1h": 0.0,
            "btc_realized_vol_4h": 0.0,
            "btc_volatility_regime": "NORMAL",
            "btc_volatility_regime_score": 0.0,
            "btc_trend_persistence": 0.0,
            "btc_rsi_14": 50.0,
            "btc_rsi_distance_mid": 0.0,
            "btc_rsi_divergence_score": 0.0,
            "btc_macd": 0.0,
            "btc_macd_signal": 0.0,
            "btc_macd_hist": 0.0,
            "btc_macd_hist_slope": 0.0,
            "btc_momentum_regime": "NEUTRAL",
            "btc_momentum_confluence": 0.0,
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
                self._write_snapshot(context)
                self._cached_context = dict(context)
                self._last_fetch_time = now
                return context

            df = daily_df[["close"]].copy()
            df["sma_200"] = df["close"].rolling(window=200).mean()
            df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

            latest = df.iloc[-1]
            current_price = latest["close"]
            sma_200 = latest["sma_200"]
            ema_21 = latest["ema_21"]

            if pd.isna(sma_200):
                context.update(live_context)
                self._write_snapshot(context)
                self._cached_context = dict(context)
                self._last_fetch_time = now
                return context

            distance = (current_price - sma_200) / sma_200
            if current_price > sma_200 and ema_21 > sma_200:
                structure = "BULLISH"
                trend_score = min(1.0, 0.7 + (distance * 2))
            elif current_price < sma_200 and ema_21 < sma_200:
                structure = "BEARISH"
                trend_score = max(0.0, 0.3 + (distance * 2))
            else:
                structure = "MIXED"
                trend_score = 0.5

            context.update(
                {
                    "btc_price": current_price,
                    "sma_200": round(sma_200, 2),
                    "ema_21": round(ema_21, 2),
                    "market_structure": structure,
                    "distance_to_sma_200": round(distance, 4),
                    "trend_score": round(trend_score, 3),
                    "technical_timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            logging.info(
                f"TechnicalAnalyzer Evaluated: Price=${current_price:.0f} | 200SMA=${sma_200:.0f} | "
                f"Trend: {structure} | Distance to 200SMA: {distance:.2%} | "
                f"Alligator={context.get('alligator_alignment')} | ADX={context.get('adx_value')} | "
                f"AVWAP={context.get('anchored_vwap')} | Bias={context.get('btc_trend_bias')} | "
                f"FractalReady={context.get('fractal_entry_ready')} | VolRegime={context.get('btc_volatility_regime')} | "
                f"Momentum={context.get('btc_momentum_regime')} | RSI={context.get('btc_rsi_14')}"
            )

            context.update(live_context)
            self._write_snapshot(context)
            self._cached_context = dict(context)
            self._last_fetch_time = now
            return context

        except Exception as e:
            logging.warning(f"TechnicalAnalyzer: Failed to fetch closed BTC candles: {e}")
            context.update(live_context)
            self._write_snapshot(context)
            return context
