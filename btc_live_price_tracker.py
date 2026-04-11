import logging
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from candle_data_service import CandleDataService
from brain_log_routing import append_csv_with_brain_mirrors
from brain_paths import BTC_FAMILY
from kalman_feature_smoother import AdaptiveScalarKalmanFilter

logger = logging.getLogger(__name__)


class BTCLivePriceTracker:
    """
    Public BTC live-price/index tracker for decision support.

    Produces a short-lived snapshot combining:
    - spot references from multiple venues
    - Binance futures mark/index pricing
    - live-vs-closed-candle returns across several horizons
    - source quality / divergence diagnostics
    """

    BINANCE_SPOT_URL = "https://api.binance.com/api/v3/ticker/price"
    BINANCE_PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
    COINBASE_TICKER_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
    KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker"

    def __init__(
        self,
        *,
        cache_ttl_seconds: int = 12,
        candle_data_service: CandleDataService | None = None,
        logs_dir: str = "logs",
    ):
        self.cache_ttl_seconds = max(2, int(cache_ttl_seconds or 12))
        self.candle_data_service = candle_data_service or CandleDataService(symbol="BTCUSDT")
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_file = self.logs_dir / "btc_live_snapshot.csv"
        self._cached_context: dict | None = None
        self._last_fetch_time = 0.0
        self._ctx_lock = threading.Lock()
        self._kalman_filters = {
            "btc_live_price_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.0007,
                measurement_noise_ratio=0.0030,
                min_scale=100.0,
            ),
            "btc_live_spot_price_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.0007,
                measurement_noise_ratio=0.0030,
                min_scale=100.0,
            ),
            "btc_live_index_price_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.0007,
                measurement_noise_ratio=0.0032,
                min_scale=100.0,
            ),
            "btc_live_mark_price_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.0008,
                measurement_noise_ratio=0.0034,
                min_scale=100.0,
            ),
            "btc_live_spot_index_basis_bps_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.050,
                measurement_noise_ratio=0.180,
                min_scale=1.0,
            ),
            "btc_live_mark_index_basis_bps_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.060,
                measurement_noise_ratio=0.200,
                min_scale=1.0,
            ),
            "btc_live_mark_spot_basis_bps_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.060,
                measurement_noise_ratio=0.200,
                min_scale=1.0,
            ),
            "btc_live_return_1m_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.120,
                measurement_noise_ratio=0.300,
                min_scale=0.001,
            ),
            "btc_live_return_5m_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.100,
                measurement_noise_ratio=0.250,
                min_scale=0.001,
            ),
            "btc_live_return_15m_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.090,
                measurement_noise_ratio=0.220,
                min_scale=0.001,
            ),
            "btc_live_return_1h_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.080,
                measurement_noise_ratio=0.200,
                min_scale=0.001,
            ),
            "btc_live_confluence_kalman": AdaptiveScalarKalmanFilter(
                process_noise_ratio=0.120,
                measurement_noise_ratio=0.220,
                min_scale=0.05,
            ),
        }

    @staticmethod
    def _safe_float(value, default=None):
        try:
            num = float(value)
        except Exception:
            return default
        if not np.isfinite(num):
            return default
        return float(num)

    @staticmethod
    def _basis_bps(numerator, denominator) -> float:
        num = BTCLivePriceTracker._safe_float(numerator, default=None)
        den = BTCLivePriceTracker._safe_float(denominator, default=None)
        if num is None or den in (None, 0.0):
            return 0.0
        return float(((num - den) / den) * 10000.0)

    @staticmethod
    def _clip01(value) -> float:
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except Exception:
            return 0.0

    def _fetch_json(self, url: str, *, params: dict | None = None, timeout: int = 8):
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _fetch_binance_spot_price(self):
        try:
            payload = self._fetch_json(self.BINANCE_SPOT_URL, params={"symbol": "BTCUSDT"})
            return self._safe_float(payload.get("price"), default=None)
        except Exception as exc:
            logger.debug("BTCLivePriceTracker: Binance spot fetch failed: %s", exc)
            return None

    def _fetch_binance_premium_index(self) -> dict:
        try:
            payload = self._fetch_json(self.BINANCE_PREMIUM_INDEX_URL, params={"symbol": "BTCUSDT"})
            return {
                "mark_price": self._safe_float(payload.get("markPrice"), default=None),
                "index_price": self._safe_float(payload.get("indexPrice"), default=None),
                "funding_rate": self._safe_float(payload.get("lastFundingRate"), default=0.0),
            }
        except Exception as exc:
            logger.debug("BTCLivePriceTracker: Binance premiumIndex fetch failed: %s", exc)
            return {"mark_price": None, "index_price": None, "funding_rate": 0.0}

    def _fetch_coinbase_spot_price(self):
        try:
            payload = self._fetch_json(self.COINBASE_TICKER_URL)
            return self._safe_float(payload.get("price"), default=None)
        except Exception as exc:
            logger.debug("BTCLivePriceTracker: Coinbase spot fetch failed: %s", exc)
            return None

    def _fetch_kraken_spot_price(self):
        try:
            payload = self._fetch_json(self.KRAKEN_TICKER_URL, params={"pair": "XBTUSD"})
            result = payload.get("result") or {}
            pair_payload = next(iter(result.values()), {})
            price_list = pair_payload.get("c") or []
            price = price_list[0] if isinstance(price_list, (list, tuple)) and price_list else None
            return self._safe_float(price, default=None)
        except Exception as exc:
            logger.debug("BTCLivePriceTracker: Kraken spot fetch failed: %s", exc)
            return None

    def _compute_live_return(self, interval: str, live_price: float) -> float:
        if live_price in (None, 0.0):
            return 0.0
        try:
            candles = self.candle_data_service.refresh_latest_closed_candles(
                interval,
                limit=2,
                timezone_name="UTC",
            )
            if candles is None or candles.empty:
                return 0.0
            reference = self._safe_float(candles["close"].iloc[-1], default=None)
            if reference in (None, 0.0):
                return 0.0
            return float((live_price - reference) / reference)
        except Exception as exc:
            logger.debug("BTCLivePriceTracker: Failed computing %s live return: %s", interval, exc)
            return 0.0

    def _classify_live_bias(
        self,
        ret_1m: float,
        ret_5m: float,
        ret_15m: float,
        ret_1h: float,
        *,
        mark_index_basis_bps: float,
        source_quality_score: float,
    ) -> tuple[str, float]:
        directional_votes = 0.0
        if ret_1m > 0:
            directional_votes += 0.10
        elif ret_1m < 0:
            directional_votes -= 0.10
        if ret_5m > 0:
            directional_votes += 0.30
        elif ret_5m < 0:
            directional_votes -= 0.30
        if ret_15m > 0:
            directional_votes += 0.40
        elif ret_15m < 0:
            directional_votes -= 0.40
        if ret_1h > 0:
            directional_votes += 0.20
        elif ret_1h < 0:
            directional_votes -= 0.20

        basis_confirmation = np.clip(mark_index_basis_bps / 25.0, -0.20, 0.20)
        directional_votes += float(basis_confirmation)

        confluence = self._clip01((abs(directional_votes) * 0.85) + (source_quality_score * 0.35))
        if source_quality_score < 0.20:
            return "NEUTRAL", 0.0
        if directional_votes >= 0.35:
            return "LONG", confluence
        if directional_votes <= -0.35:
            return "SHORT", confluence
        return "NEUTRAL", confluence * 0.5

    def _write_snapshot(self, context: dict):
        try:
            append_csv_with_brain_mirrors(
                self.snapshot_file,
                pd.DataFrame([context]),
                family_hint=BTC_FAMILY,
                shared_logs_dir=self.logs_dir,
                include_shared=True,
            )
        except Exception as exc:
            logger.debug("BTCLivePriceTracker: Snapshot write failed: %s", exc)

    def _apply_kalman_smoothing(self, context: dict) -> dict:
        raw_to_smoothed = {
            "btc_live_price": "btc_live_price_kalman",
            "btc_live_spot_price": "btc_live_spot_price_kalman",
            "btc_live_index_price": "btc_live_index_price_kalman",
            "btc_live_mark_price": "btc_live_mark_price_kalman",
            "btc_live_spot_index_basis_bps": "btc_live_spot_index_basis_bps_kalman",
            "btc_live_mark_index_basis_bps": "btc_live_mark_index_basis_bps_kalman",
            "btc_live_mark_spot_basis_bps": "btc_live_mark_spot_basis_bps_kalman",
            "btc_live_return_1m": "btc_live_return_1m_kalman",
            "btc_live_return_5m": "btc_live_return_5m_kalman",
            "btc_live_return_15m": "btc_live_return_15m_kalman",
            "btc_live_return_1h": "btc_live_return_1h_kalman",
            "btc_live_confluence": "btc_live_confluence_kalman",
        }
        smoothed = {}
        for raw_key, smooth_key in raw_to_smoothed.items():
            filter_obj = self._kalman_filters.get(smooth_key)
            if filter_obj is None:
                continue
            filtered_value = filter_obj.update(context.get(raw_key))
            if filtered_value is None:
                smoothed[smooth_key] = context.get(raw_key)
            elif "basis_bps" in smooth_key:
                smoothed[smooth_key] = round(float(filtered_value), 4)
            elif "return" in smooth_key or "confluence" in smooth_key:
                smoothed[smooth_key] = round(float(filtered_value), 6)
            else:
                smoothed[smooth_key] = round(float(filtered_value), 6)
        return smoothed

    def analyze(self) -> dict:
        now = time.time()
        with self._ctx_lock:
            if self._cached_context and (now - self._last_fetch_time) < self.cache_ttl_seconds:
                return dict(self._cached_context)

        premium = self._fetch_binance_premium_index()
        raw_index_price = premium.get("index_price")
        raw_mark_price = premium.get("mark_price")
        spot_sources = {
            "binance_spot": self._fetch_binance_spot_price(),
            "coinbase_spot": self._fetch_coinbase_spot_price(),
            "kraken_spot": self._fetch_kraken_spot_price(),
        }
        valid_spot_prices = [price for price in spot_sources.values() if price not in (None, 0.0)]
        spot_price = float(np.median(valid_spot_prices)) if valid_spot_prices else None
        index_price = raw_index_price or spot_price
        mark_price = raw_mark_price or raw_index_price or spot_price
        live_price = spot_price or index_price or mark_price

        source_divergence_bps = 0.0
        if len(valid_spot_prices) >= 2 and spot_price not in (None, 0.0):
            source_divergence_bps = float(((max(valid_spot_prices) - min(valid_spot_prices)) / spot_price) * 10000.0)

        source_quality_score = 0.15
        source_quality = "LOW"
        if len(valid_spot_prices) >= 3 and source_divergence_bps <= 8:
            source_quality_score = 0.95
            source_quality = "HIGH"
        elif len(valid_spot_prices) >= 2 and source_divergence_bps <= 15:
            source_quality_score = 0.75
            source_quality = "HIGH"
        elif len(valid_spot_prices) >= 2 and source_divergence_bps <= 30:
            source_quality_score = 0.55
            source_quality = "MEDIUM"
        elif len(valid_spot_prices) >= 1:
            source_quality_score = 0.35
            source_quality = "MEDIUM" if source_divergence_bps <= 40 else "LOW"

        ret_1m = self._compute_live_return("1m", live_price)
        ret_5m = self._compute_live_return("5m", live_price)
        ret_15m = self._compute_live_return("15m", live_price)
        ret_1h = self._compute_live_return("1h", live_price)

        spot_index_basis_bps = self._basis_bps(spot_price, index_price)
        mark_index_basis_bps = self._basis_bps(mark_price, index_price)
        mark_spot_basis_bps = self._basis_bps(mark_price, spot_price)

        live_bias, live_confluence = self._classify_live_bias(
            ret_1m,
            ret_5m,
            ret_15m,
            ret_1h,
            mark_index_basis_bps=mark_index_basis_bps,
            source_quality_score=source_quality_score,
        )

        volatility_proxy = self._clip01(
            (
                abs(ret_1m) * 4.0
                + abs(ret_5m) * 2.5
                + abs(ret_15m) * 1.5
                + abs(mark_index_basis_bps) / 40.0
            )
        )

        context = {
            "btc_live_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "btc_live_price": live_price,
            "btc_live_spot_price": spot_price,
            "btc_live_index_price": index_price,
            "btc_live_mark_price": mark_price,
            "btc_live_funding_rate": premium.get("funding_rate", 0.0),
            "btc_live_spot_source_count": len(valid_spot_prices),
            "btc_live_source_quality": source_quality,
            "btc_live_source_quality_score": round(source_quality_score, 4),
            "btc_live_source_divergence_bps": round(source_divergence_bps, 4),
            "btc_live_spot_index_basis_bps": round(spot_index_basis_bps, 4),
            "btc_live_mark_index_basis_bps": round(mark_index_basis_bps, 4),
            "btc_live_mark_spot_basis_bps": round(mark_spot_basis_bps, 4),
            "btc_live_return_1m": round(ret_1m, 6),
            "btc_live_return_5m": round(ret_5m, 6),
            "btc_live_return_15m": round(ret_15m, 6),
            "btc_live_return_1h": round(ret_1h, 6),
            "btc_live_volatility_proxy": round(volatility_proxy, 6),
            "btc_live_bias": live_bias,
            "btc_live_confluence": round(live_confluence, 6),
            "btc_live_price_source": "spot_median" if spot_price not in (None, 0.0) else "binance_index",
            "btc_live_index_ready": bool(raw_index_price not in (None, 0.0) and raw_mark_price not in (None, 0.0)),
            "btc_live_index_feed_available": bool(raw_index_price not in (None, 0.0)),
            "btc_live_mark_feed_available": bool(raw_mark_price not in (None, 0.0)),
            "btc_live_sources_json": str({k: round(v, 2) for k, v in spot_sources.items() if v not in (None, 0.0)}),
        }
        context.update(self._apply_kalman_smoothing(context))
        with self._ctx_lock:
            self._cached_context = dict(context)
            self._last_fetch_time = now
        self._write_snapshot(context)
        logger.info(
            "BTCLivePriceTracker: live=%s index=%s mark=%s bias=%s confluence=%.2f divergence_bps=%.2f",
            round(live_price, 2) if live_price not in (None, 0.0) else None,
            round(index_price, 2) if index_price not in (None, 0.0) else None,
            round(mark_price, 2) if mark_price not in (None, 0.0) else None,
            live_bias,
            context["btc_live_confluence"],
            context["btc_live_source_divergence_bps"],
        )
        return dict(context)
