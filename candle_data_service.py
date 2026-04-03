import logging
import time
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandleRequest:
    interval: str
    limit: int
    closed_only: bool = True
    timezone: str = "UTC"


class CandleDataService:
    """
    Central BTC OHLCV data layer.

    Guarantees:
    - timezone-aware timestamps
    - closed candles only by default
    - cleaned, sorted, deduplicated historical data
    - live refreshes update the cached candle history without exposing unfinished bars
    """

    BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
    _INTERVAL_MAP = {
        "1m": timedelta(minutes=1),
        "3m": timedelta(minutes=3),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "2h": timedelta(hours=2),
        "4h": timedelta(hours=4),
        "6h": timedelta(hours=6),
        "8h": timedelta(hours=8),
        "12h": timedelta(hours=12),
        "1d": timedelta(days=1),
    }

    def __init__(self, symbol: str = "BTCUSDT", cache_ttl_seconds: int = 20):
        self.symbol = symbol
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds or 20))
        self._cache: dict[tuple[str, int, bool, str], tuple[float, pd.DataFrame]] = {}

    def _interval_to_timedelta(self, interval: str) -> timedelta:
        if interval not in self._INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}")
        return self._INTERVAL_MAP[interval]

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            val = float(value)
            if pd.isna(val):
                return default
            return val
        except Exception:
            return default

    def _fetch_raw_klines(self, interval: str, limit: int) -> list:
        response = requests.get(
            self.BASE_URL,
            params={"symbol": self.symbol, "interval": interval, "limit": int(limit)},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected kline payload for {self.symbol} {interval}")
        return payload

    def _klines_to_frame(self, klines: list) -> pd.DataFrame:
        if not klines:
            return pd.DataFrame(
                columns=[
                    "open_time",
                    "close_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_volume",
                    "trade_count",
                    "taker_buy_base_volume",
                    "taker_buy_quote_volume",
                ]
            )

        frame = pd.DataFrame(
            {
                "open_time": pd.to_datetime([row[0] for row in klines], unit="ms", utc=True, errors="coerce"),
                "open": [self._safe_float(row[1], None) for row in klines],
                "high": [self._safe_float(row[2], None) for row in klines],
                "low": [self._safe_float(row[3], None) for row in klines],
                "close": [self._safe_float(row[4], None) for row in klines],
                "volume": [self._safe_float(row[5], None) for row in klines],
                "close_time": pd.to_datetime([row[6] for row in klines], unit="ms", utc=True, errors="coerce"),
                "quote_volume": [self._safe_float(row[7], None) for row in klines],
                "trade_count": [int(self._safe_float(row[8], 0)) for row in klines],
                "taker_buy_base_volume": [self._safe_float(row[9], None) for row in klines],
                "taker_buy_quote_volume": [self._safe_float(row[10], None) for row in klines],
            }
        )
        return self._clean_frame(frame)

    def _clean_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=list(frame.columns) if frame is not None else [])

        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
        work = frame.copy()
        for col in numeric_cols:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        work = work.dropna(subset=["open_time", "open", "high", "low", "close", "volume"])
        work = work.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
        work = work[(work["high"] >= work["low"]) & (work["volume"] >= 0)]
        return work.reset_index(drop=True)

    def _drop_unfinished_candles(
        self,
        frame: pd.DataFrame,
        interval: str,
        *,
        now_utc: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame if frame is not None else pd.DataFrame()

        interval_delta = self._interval_to_timedelta(interval)
        current_time = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
        current_time = pd.Timestamp(current_time)
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize("UTC")
        else:
            current_time = current_time.tz_convert("UTC")

        closed_mask = (frame["open_time"] + interval_delta) <= current_time
        return frame.loc[closed_mask].copy().reset_index(drop=True)

    def _convert_timezone(self, frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame if frame is not None else pd.DataFrame()
        if not timezone_name or timezone_name.upper() == "UTC":
            return frame
        work = frame.copy()
        work["open_time"] = work["open_time"].dt.tz_convert(timezone_name)
        work["close_time"] = work["close_time"].dt.tz_convert(timezone_name)
        return work

    def get_candles(
        self,
        interval: str,
        limit: int,
        *,
        closed_only: bool = True,
        timezone_name: str = "UTC",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_key = (interval, int(limit), bool(closed_only), timezone_name)
        now = time.time()
        if use_cache and cache_key in self._cache:
            cached_at, cached_df = self._cache[cache_key]
            if (now - cached_at) < self.cache_ttl_seconds:
                return cached_df.copy()

        raw = self._fetch_raw_klines(interval, limit)
        frame = self._klines_to_frame(raw)
        if closed_only:
            frame = self._drop_unfinished_candles(frame, interval)
        frame = self._convert_timezone(frame, timezone_name)
        self._cache[cache_key] = (now, frame.copy())
        return frame

    def refresh_latest_closed_candles(
        self,
        interval: str,
        *,
        limit: int = 400,
        timezone_name: str = "UTC",
    ) -> pd.DataFrame:
        # Live refresh path: intentionally bypass short cache and rebuild from the latest
        # exchange snapshot, still enforcing closed-candle only semantics.
        frame = self.get_candles(
            interval,
            limit,
            closed_only=True,
            timezone_name=timezone_name,
            use_cache=False,
        )
        logger.debug(
            "Refreshed %s closed %s candles for %s (rows=%s)",
            self.symbol,
            interval,
            timezone_name,
            len(frame),
        )
        return frame
