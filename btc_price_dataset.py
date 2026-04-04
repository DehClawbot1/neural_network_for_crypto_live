"""
BTC Price Dataset Builder

Builds labelled datasets for BTC price prediction from:
  1. Live candle data (via CandleDataService / Binance API)
  2. Downloaded historical CSVs (see download_btc_dataset.py)

Produces feature rows with technical indicators + forward-return labels
suitable for training btc_forecast_model.py.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from csv_utils import safe_csv_append

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_LOGS_DIR = "logs"
_DATASET_FILENAME = "btc_price_dataset.csv"
_MIN_ROWS_FOR_FEATURES = 210  # need 200-SMA + some look-ahead


class BTCPriceDatasetBuilder:
    """
    Converts raw BTC OHLCV candle data into a feature-rich, labelled dataset
    for price-direction and return-magnitude prediction.

    Call flow:
        builder = BTCPriceDatasetBuilder()
        df = builder.build_from_candles(candle_df)      # in-memory
        builder.append_to_disk(df)                       # persist incrementally
    """

    # Prediction horizons (in number of rows / candles)
    HORIZONS = {
        "5": 5,
        "15": 15,
        "60": 60,
    }

    def __init__(self, logs_dir: str = _DEFAULT_LOGS_DIR):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.logs_dir / _DATASET_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accept a candle DataFrame with columns:
            open, high, low, close, volume  (and optionally 'timestamp')

        Returns a feature+label DataFrame ready for training.
        """
        df = self._normalise_candle_df(df)
        if df is None or len(df) < _MIN_ROWS_FOR_FEATURES:
            logger.warning(
                "BTCPriceDatasetBuilder: need >= %d candle rows, got %d",
                _MIN_ROWS_FOR_FEATURES,
                0 if df is None else len(df),
            )
            return pd.DataFrame()

        features = self._compute_features(df)
        labels = self._compute_labels(df)
        result = pd.concat([features, labels], axis=1)
        # Drop rows where features or labels are NaN (edges of rolling windows)
        result.dropna(inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    def build_from_csv(self, csv_path: str | Path) -> pd.DataFrame:
        """Build dataset from a downloaded CSV file (e.g. from Binance or Kaggle)."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            logger.error("BTCPriceDatasetBuilder: CSV not found: %s", csv_path)
            return pd.DataFrame()
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        return self.build_from_candles(df)

    def append_to_disk(self, df: pd.DataFrame) -> int:
        """Append new labelled rows to the persistent dataset CSV."""
        if df is None or df.empty:
            return 0
        safe_csv_append(self.dataset_path, df)
        logger.info("BTCPriceDatasetBuilder: appended %d rows to %s", len(df), self.dataset_path)
        return len(df)

    def load_dataset(self) -> pd.DataFrame:
        """Load the full persisted dataset."""
        if not self.dataset_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_path, engine="python", on_bad_lines="skip")
        except Exception as exc:
            logger.warning("BTCPriceDatasetBuilder: failed to load dataset: %s", exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _normalise_candle_df(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Ensure standard column names and sort by time."""
        df = df.copy()
        # Handle common column name variants
        col_map = {}
        targets_claimed = set()
        for col in df.columns:
            lc = str(col).strip().lower()
            if lc in ("open", "open_price") and "open" not in targets_claimed:
                col_map[col] = "open"
                targets_claimed.add("open")
            elif lc in ("high", "high_price") and "high" not in targets_claimed:
                col_map[col] = "high"
                targets_claimed.add("high")
            elif lc in ("low", "low_price") and "low" not in targets_claimed:
                col_map[col] = "low"
                targets_claimed.add("low")
            elif lc in ("close", "close_price", "price") and "close" not in targets_claimed:
                col_map[col] = "close"
                targets_claimed.add("close")
            elif lc in ("volume", "vol") and "volume" not in targets_claimed:
                col_map[col] = "volume"
                targets_claimed.add("volume")
            elif lc in ("timestamp", "date", "datetime", "open_time") and "timestamp" not in targets_claimed:
                col_map[col] = "timestamp"
                targets_claimed.add("timestamp")
        df.rename(columns=col_map, inplace=True)

        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            logger.error("BTCPriceDatasetBuilder: missing columns %s", required - set(df.columns))
            return None

        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        else:
            df["volume"] = 0.0

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df.sort_values("timestamp", inplace=True)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicator features from OHLCV."""
        f = pd.DataFrame(index=df.index)

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        if "timestamp" in df.columns:
            f["timestamp"] = df["timestamp"]

        # --- Returns ---
        f["return_1"] = close.pct_change(1)
        f["return_5"] = close.pct_change(5)
        f["return_15"] = close.pct_change(15)

        # --- Moving averages ---
        f["sma_10"] = close.rolling(10).mean()
        f["sma_20"] = close.rolling(20).mean()
        f["sma_50"] = close.rolling(50).mean()
        f["sma_200"] = close.rolling(200).mean()
        f["ema_9"] = close.ewm(span=9, adjust=False).mean()
        f["ema_21"] = close.ewm(span=21, adjust=False).mean()

        # Price relative to MAs (normalised distances)
        f["close_to_sma_20"] = (close - f["sma_20"]) / f["sma_20"]
        f["close_to_sma_50"] = (close - f["sma_50"]) / f["sma_50"]
        f["close_to_sma_200"] = (close - f["sma_200"]) / f["sma_200"]
        f["ema_9_21_cross"] = (f["ema_9"] - f["ema_21"]) / close

        # --- RSI (14) ---
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        f["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # --- MACD ---
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        f["macd"] = ema_12 - ema_26
        f["macd_signal"] = f["macd"].ewm(span=9, adjust=False).mean()
        f["macd_hist"] = f["macd"] - f["macd_signal"]

        # --- ATR (14) ---
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        f["atr_14"] = tr.rolling(14).mean()
        f["atr_pct"] = f["atr_14"] / close

        # --- Bollinger Bands ---
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        f["bb_upper"] = bb_sma + 2 * bb_std
        f["bb_lower"] = bb_sma - 2 * bb_std
        bb_width = f["bb_upper"] - f["bb_lower"]
        f["bb_position"] = (close - f["bb_lower"]) / bb_width.replace(0, np.nan)
        f["bb_width_pct"] = bb_width / close

        # --- ADX (14) ---
        f["adx"] = self._compute_adx(high, low, close, period=14)

        # --- Stochastic RSI ---
        rsi_series = f["rsi_14"]
        rsi_min = rsi_series.rolling(14).min()
        rsi_max = rsi_series.rolling(14).max()
        rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
        f["stoch_rsi_k"] = ((rsi_series - rsi_min) / rsi_range) * 100
        f["stoch_rsi_d"] = f["stoch_rsi_k"].rolling(3).mean()

        # --- Volume features ---
        f["volume_sma_20"] = volume.rolling(20).mean()
        vol_sma = f["volume_sma_20"].replace(0, np.nan)
        f["volume_ratio"] = volume / vol_sma
        f["obv"] = (np.sign(close.diff()) * volume).cumsum()
        f["obv_sma_10"] = f["obv"].rolling(10).mean()

        # --- Volatility ---
        f["realized_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(20)
        f["realized_vol_60"] = close.pct_change().rolling(60).std() * np.sqrt(60)

        # --- Candle patterns (simple) ---
        body = (close - df["open"].astype(float)).abs()
        wick_upper = high - pd.concat([close, df["open"].astype(float)], axis=1).max(axis=1)
        wick_lower = pd.concat([close, df["open"].astype(float)], axis=1).min(axis=1) - low
        candle_range = (high - low).replace(0, np.nan)
        f["body_ratio"] = body / candle_range
        f["upper_wick_ratio"] = wick_upper / candle_range
        f["lower_wick_ratio"] = wick_lower / candle_range

        # --- Price ---
        f["close"] = close

        # ============================================================
        # ADVANCED FEATURES (Phase 2 — accuracy improvement)
        # ============================================================

        # --- Lag features: let the model see how indicators evolved ---
        lag_cols = ["rsi_14", "macd_hist", "adx", "bb_position", "volume_ratio", "atr_pct"]
        for col in lag_cols:
            if col in f.columns:
                for lag in [1, 2, 4, 8]:
                    f[f"{col}_lag_{lag}"] = f[col].shift(lag)
                # Rate of change of the indicator itself
                f[f"{col}_roc_4"] = f[col] - f[col].shift(4)

        # --- Return sequences (momentum signature) ---
        returns = close.pct_change()
        for w in [5, 10, 20]:
            f[f"return_mean_{w}"] = returns.rolling(w).mean()
            f[f"return_std_{w}"] = returns.rolling(w).std()
            f[f"return_skew_{w}"] = returns.rolling(w).skew()
            f[f"return_kurt_{w}"] = returns.rolling(w).kurt()

        # --- Trend strength / persistence ---
        f["up_streak"] = self._compute_streak(returns > 0)
        f["down_streak"] = self._compute_streak(returns < 0)
        f["trend_consistency_10"] = returns.rolling(10).apply(
            lambda x: (x > 0).sum() / len(x), raw=True
        )
        f["trend_consistency_20"] = returns.rolling(20).apply(
            lambda x: (x > 0).sum() / len(x), raw=True
        )

        # --- Price relative to range (Donchian position) ---
        for w in [20, 50]:
            high_w = high.rolling(w).max()
            low_w = low.rolling(w).min()
            range_w = (high_w - low_w).replace(0, np.nan)
            f[f"donchian_pos_{w}"] = (close - low_w) / range_w

        # --- Williams %R ---
        for w in [14, 28]:
            hh = high.rolling(w).max()
            ll = low.rolling(w).min()
            f[f"williams_r_{w}"] = -100 * (hh - close) / (hh - ll).replace(0, np.nan)

        # --- CCI (Commodity Channel Index) ---
        typical_price = (high + low + close) / 3
        tp_sma = typical_price.rolling(20).mean()
        tp_mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        f["cci_20"] = (typical_price - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

        # --- MFI (Money Flow Index) — volume-weighted RSI ---
        mf_raw = typical_price * volume
        mf_pos = mf_raw.where(typical_price > typical_price.shift(1), 0.0)
        mf_neg = mf_raw.where(typical_price < typical_price.shift(1), 0.0)
        mf_ratio = mf_pos.rolling(14).sum() / mf_neg.rolling(14).sum().replace(0, np.nan)
        f["mfi_14"] = 100.0 - (100.0 / (1.0 + mf_ratio))

        # --- VWAP deviation ---
        cumvol = volume.cumsum()
        cum_tp_vol = (typical_price * volume).cumsum()
        vwap = cum_tp_vol / cumvol.replace(0, np.nan)
        f["vwap_deviation"] = (close - vwap) / vwap.replace(0, np.nan)

        # --- Volume microstructure ---
        f["volume_delta"] = volume - volume.shift(1)
        f["volume_acceleration"] = f["volume_delta"] - f["volume_delta"].shift(1)
        f["buy_volume_pct"] = (close - low) / (high - low).replace(0, np.nan)
        f["sell_volume_pct"] = (high - close) / (high - low).replace(0, np.nan)
        f["volume_force"] = returns * volume
        f["volume_force_sma_10"] = f["volume_force"].rolling(10).mean()

        # --- Volatility regime features ---
        vol_20 = f["realized_vol_20"]
        vol_60 = f["realized_vol_60"]
        f["vol_ratio_20_60"] = vol_20 / vol_60.replace(0, np.nan)
        f["vol_zscore_20"] = (vol_20 - vol_20.rolling(100).mean()) / vol_20.rolling(100).std().replace(0, np.nan)
        f["atr_zscore"] = (f["atr_pct"] - f["atr_pct"].rolling(100).mean()) / f["atr_pct"].rolling(100).std().replace(0, np.nan)

        # --- Garman-Klass volatility estimator ---
        log_hl = np.log(high / low.replace(0, np.nan)) ** 2
        log_co = np.log(close / df["open"].astype(float).replace(0, np.nan)) ** 2
        gk_vol = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
        f["gk_volatility_20"] = gk_vol.rolling(20).mean().apply(np.sqrt)

        # --- Cross-timeframe synthetic features ---
        # Simulate higher timeframes by resampling within the 15m data
        for mult, name in [(4, "1h"), (16, "4h")]:
            close_htf = close.rolling(mult).apply(lambda x: x.iloc[-1], raw=False)
            high_htf = high.rolling(mult).max()
            low_htf = low.rolling(mult).min()
            f[f"return_{name}"] = close_htf.pct_change(mult)
            f[f"range_pct_{name}"] = (high_htf - low_htf) / close.replace(0, np.nan)
            f[f"rsi_{name}"] = self._compute_rsi_series(close_htf, period=14)

        # --- Hour-of-day cyclical encoding (if timestamp available) ---
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            hour = ts.dt.hour + ts.dt.minute / 60.0
            f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
            dow = ts.dt.dayofweek.astype(float)
            f["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            f["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        return f

    def _compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute forward-return labels at multiple horizons."""
        labels = pd.DataFrame(index=df.index)
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        for name, periods in self.HORIZONS.items():
            future_close = close.shift(-periods)
            # Continuous: forward return
            labels[f"fwd_return_{name}"] = (future_close - close) / close
            # Binary: price goes up?
            labels[f"fwd_up_{name}"] = (future_close > close).astype(int)
            # Ternary: direction with dead zone (< 0.1% = neutral)
            pct = labels[f"fwd_return_{name}"]
            labels[f"fwd_direction_{name}"] = np.where(
                pct > 0.001, 1, np.where(pct < -0.001, -1, 0)
            )

            # --- Advanced labels ---
            # Volatility-normalised return (better target for varying regimes)
            rolling_vol = close.pct_change().rolling(20).std().replace(0, np.nan)
            labels[f"fwd_return_vol_adj_{name}"] = labels[f"fwd_return_{name}"] / rolling_vol

            # Max adverse excursion — worst drawdown before reaching the horizon
            if periods <= 60:
                future_lows = pd.Series(np.nan, index=df.index)
                future_highs = pd.Series(np.nan, index=df.index)
                for i in range(len(df) - periods):
                    future_lows.iloc[i] = low.iloc[i + 1: i + 1 + periods].min()
                    future_highs.iloc[i] = high.iloc[i + 1: i + 1 + periods].max()
                labels[f"fwd_max_drawdown_{name}"] = (future_lows - close) / close
                labels[f"fwd_max_runup_{name}"] = (future_highs - close) / close

        return labels

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_streak(condition: pd.Series) -> pd.Series:
        """Count consecutive True values (resets on False)."""
        groups = (~condition).cumsum()
        return condition.groupby(groups).cumsum().astype(float)

    @staticmethod
    def _compute_rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI from any price series."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))

        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return adx
