"""
btc_dataset_refresh_scheduler.py

Fix 2: Auto-refresh of BTC historical candle dataset.

Why this matters:
  - btc_price_dataset.csv is 325 MB and NOT automatically refreshed
  - Regime features (RSI, MACD, SMA200) become stale during live trading
  - Forecast models can drift away from the current BTC market state

Solution:
  - BTCDatasetRefreshScheduler checks file age on every call to maybe_refresh()
  - If the file is older than BTC_DATASET_REFRESH_HOURS (default 4h), it
    fetches the latest candles from Binance and appends them
  - Designed to be called from brain_training_orchestrator or supervisor
  - Non-blocking: logs a warning on failure but does not crash

Usage:
    scheduler = BTCDatasetRefreshScheduler(logs_dir="logs")
    new_rows = scheduler.maybe_refresh()   # returns 0 if skipped
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_REFRESH_HOURS = 4
_DEFAULT_CANDLE_LOOKBACK_DAYS = 30   # fetch most recent N days of 1h candles
_DEFAULT_CANDLE_INTERVAL = "1h"
_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


class BTCDatasetRefreshScheduler:
    """
    Keeps btc_price_dataset.csv fresh by appending new candles
    from Binance whenever the file is stale.

    Architecture:
      1. Checks age of btc_price_dataset.csv
      2. If older than refresh_hours: fetches new candles from Binance
      3. Converts candles → feature rows via BTCPriceDatasetBuilder
      4. Deduplicates by timestamp and appends only new rows
    """

    def __init__(
        self,
        logs_dir: str = "logs",
        *,
        refresh_hours: float | None = None,
        candle_lookback_days: int | None = None,
        candle_interval: str | None = None,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.dataset_path = self.logs_dir / "btc_price_dataset.csv"
        self.refresh_hours = float(
            refresh_hours
            or os.getenv("BTC_DATASET_REFRESH_HOURS", str(_DEFAULT_REFRESH_HOURS))
        )
        self.candle_lookback_days = int(
            candle_lookback_days
            or os.getenv("BTC_DATASET_LOOKBACK_DAYS", str(_DEFAULT_CANDLE_LOOKBACK_DAYS))
        )
        self.candle_interval = str(
            candle_interval
            or os.getenv("BTC_DATASET_CANDLE_INTERVAL", _DEFAULT_CANDLE_INTERVAL)
        )
        self._last_refresh_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_stale(self) -> bool:
        """Return True if dataset file is missing or older than refresh_hours."""
        if not self.dataset_path.exists():
            return True
        try:
            age_hours = (time.time() - self.dataset_path.stat().st_mtime) / 3600.0
            return age_hours > self.refresh_hours
        except Exception:
            return True

    def maybe_refresh(self) -> int:
        """
        Refresh BTCpriceDataset if stale.
        Returns number of new rows appended (0 if skipped or failed).
        """
        if not self.is_stale():
            return 0

        logger.info(
            "BTCDatasetRefreshScheduler: dataset is stale (refresh_hours=%.1f). Fetching candles...",
            self.refresh_hours,
        )
        try:
            candle_df = self._fetch_binance_candles()
            if candle_df is None or candle_df.empty:
                logger.warning("BTCDatasetRefreshScheduler: candle fetch returned empty data; skipping refresh")
                return 0

            # Pass through BTCPriceDatasetBuilder
            try:
                from btc_price_dataset import BTCPriceDatasetBuilder
                builder = BTCPriceDatasetBuilder(logs_dir=str(self.logs_dir))
                feature_df = builder.build_from_candles(candle_df)
            except Exception as exc:
                logger.warning("BTCDatasetRefreshScheduler: feature build failed: %s", exc)
                return 0

            if feature_df.empty:
                logger.warning("BTCDatasetRefreshScheduler: feature build produced 0 rows; skipping")
                return 0

            new_rows = self._dedupe_and_append(feature_df)
            self._last_refresh_ts = time.time()
            logger.info(
                "BTCDatasetRefreshScheduler: refresh complete — %d new rows appended to %s",
                new_rows, self.dataset_path,
            )
            return new_rows

        except Exception as exc:
            logger.warning("BTCDatasetRefreshScheduler: refresh failed (non-blocking): %s", exc)
            return 0

    def force_refresh(self) -> int:
        """Force a refresh regardless of staleness."""
        if self.dataset_path.exists():
            self.dataset_path.stat()  # touch mtime check
        import time as _time
        _original_mtime = self.dataset_path.stat().st_mtime if self.dataset_path.exists() else 0
        # Temporarily set threshold to 0 to force refresh
        original_hours = self.refresh_hours
        self.refresh_hours = 0.0
        result = self.maybe_refresh()
        self.refresh_hours = original_hours
        return result

    # ------------------------------------------------------------------
    # Private — Binance candle fetch
    # ------------------------------------------------------------------

    def _fetch_binance_candles(self) -> Optional[pd.DataFrame]:
        """Fetch recent 1h BTCUSDT candles from Binance REST API."""
        try:
            import requests
            # Calculate startTime as N days back
            lookback_ms = int(self.candle_lookback_days * 24 * 3600 * 1000)
            start_ms = int(time.time() * 1000) - lookback_ms
            params = {
                "symbol": "BTCUSDT",
                "interval": self.candle_interval,
                "startTime": start_ms,
                "limit": 1000,
            }
            resp = requests.get(_BINANCE_KLINES_URL, params=params, timeout=20)
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                return None

            # Binance kline columns:
            # 0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume, ...
            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "n_trades",
                "taker_buy_vol", "taker_buy_quote_vol", "_ignore",
            ])
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
            logger.info(
                "BTCDatasetRefreshScheduler: fetched %d candles (interval=%s, lookback=%dd)",
                len(df), self.candle_interval, self.candle_lookback_days,
            )
            return df
        except Exception as exc:
            logger.warning("BTCDatasetRefreshScheduler: Binance fetch failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Private — dedup + append
    # ------------------------------------------------------------------

    def _dedupe_and_append(self, new_df: pd.DataFrame) -> int:
        """
        Merge new feature rows with existing dataset.
        Deduplicates on 'timestamp', keeping the newest version.
        Returns number of genuinely new rows added.
        """
        if not self.dataset_path.exists():
            try:
                new_df.to_csv(self.dataset_path, index=False)
                return len(new_df)
            except Exception as exc:
                logger.warning("BTCDatasetRefreshScheduler: initial write failed: %s", exc)
                return 0

        try:
            existing = pd.read_csv(self.dataset_path, engine="python", on_bad_lines="skip")
        except Exception as exc:
            logger.warning("BTCDatasetRefreshScheduler: cannot read existing dataset: %s", exc)
            return 0

        if "timestamp" not in new_df.columns or "timestamp" not in existing.columns:
            # No timestamp — just append
            try:
                from csv_utils import safe_csv_append
                safe_csv_append(self.dataset_path, new_df)
                return len(new_df)
            except Exception:
                return 0

        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce")
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True, errors="coerce")

        existing_ts = set(existing["timestamp"].dropna().astype(str))
        truly_new = new_df[~new_df["timestamp"].astype(str).isin(existing_ts)]
        if truly_new.empty:
            logger.info("BTCDatasetRefreshScheduler: all %d fetched rows already in dataset", len(new_df))
            return 0

        try:
            combined = pd.concat([existing, truly_new], ignore_index=True)
            combined.to_csv(self.dataset_path, index=False)
            return len(truly_new)
        except Exception as exc:
            logger.warning("BTCDatasetRefreshScheduler: append failed: %s", exc)
            return 0
