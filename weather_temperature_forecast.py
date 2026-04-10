from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from statistics import NormalDist

import pandas as pd
import requests


logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _safe_float(value, default=None):
    try:
        parsed = float(value)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


class WeatherForecastService:
    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.session = requests.Session()
        self._geocode_cache: dict[tuple[str, str], dict] = {}
        self._forecast_cache: dict[tuple[str, str], dict] = {}
        self._forecast_memory: dict[tuple[str, str], dict] = {}
        self.stale_minutes = max(5, int(os.getenv("WEATHER_FORECAST_STALE_MINUTES", "90") or 90))
        self.default_uncertainty_c = max(0.4, float(os.getenv("WEATHER_FORECAST_BASE_UNCERTAINTY_C", "1.6") or 1.6))

    def geocode_location(self, location: str, country_hint: str | None = None) -> dict | None:
        normalized_location = str(location or "").strip()
        normalized_country = str(country_hint or "").strip().upper()
        cache_key = (normalized_location.lower(), normalized_country)
        if cache_key in self._geocode_cache:
            return dict(self._geocode_cache[cache_key])

        if not normalized_location:
            return None

        params = {
            "name": normalized_location,
            "count": 5,
            "language": "en",
            "format": "json",
        }
        response = self.session.get(GEOCODE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            return None

        chosen = results[0]
        if normalized_country:
            for result in results:
                if str(result.get("country_code") or "").strip().upper() == normalized_country:
                    chosen = result
                    break
        cleaned = {
            "latitude": _safe_float(chosen.get("latitude")),
            "longitude": _safe_float(chosen.get("longitude")),
            "timezone": str(chosen.get("timezone") or "UTC"),
            "name": str(chosen.get("name") or normalized_location),
            "country": str(chosen.get("country") or ""),
            "country_code": str(chosen.get("country_code") or ""),
            "admin1": str(chosen.get("admin1") or ""),
        }
        self._geocode_cache[cache_key] = cleaned
        return dict(cleaned)

    def _resolve_uncertainty(self, days_out: int) -> float:
        return float(max(0.4, self.default_uncertainty_c + (max(0, days_out) * 0.15)))

    def _p_interval_hit(self, *, max_temp_c: float, lower_c: float | None, upper_c: float | None, uncertainty_c: float) -> float:
        dist = NormalDist(mu=float(max_temp_c), sigma=max(float(uncertainty_c), 0.2))
        if lower_c is None and upper_c is None:
            return 0.5
        if upper_c is None:
            return max(0.0, min(1.0, 1.0 - dist.cdf(float(lower_c))))
        if lower_c is None:
            return max(0.0, min(1.0, dist.cdf(float(upper_c))))
        lower = float(lower_c)
        upper = float(upper_c)
        if upper < lower:
            lower, upper = upper, lower
        return max(0.0, min(1.0, dist.cdf(upper) - dist.cdf(lower)))

    def fetch_market_forecast(self, market_row: dict) -> dict:
        market_row = dict(market_row or {})
        location = str(market_row.get("weather_location") or "").strip()
        event_date_local = str(market_row.get("weather_event_date_local") or "").strip()
        lower_c = _safe_float(market_row.get("weather_lower_c"), None)
        upper_c = _safe_float(market_row.get("weather_upper_c"), None)
        if not location or not event_date_local:
            return {
                "forecast_ready": False,
                "forecast_missing_reason": "missing_location_or_event_date",
                "forecast_stale": True,
                "forecast_last_update_ts": None,
            }

        cache_key = (location.lower(), event_date_local)
        now_utc = datetime.now(timezone.utc)
        cached = self._forecast_cache.get(cache_key)
        if cached is not None:
            last_update = pd.to_datetime(cached.get("forecast_last_update_ts"), utc=True, errors="coerce")
            if pd.notna(last_update):
                age_minutes = max(0.0, (now_utc - last_update.to_pydatetime()).total_seconds() / 60.0)
                if age_minutes <= self.stale_minutes:
                    return dict(cached)

        try:
            geo = self.geocode_location(location, country_hint=market_row.get("weather_country"))
        except Exception as exc:
            logger.warning("Weather geocode failed for %s: %s", location, exc)
            geo = None
        if not geo:
            return {
                "forecast_ready": False,
                "forecast_missing_reason": "geocode_failed",
                "forecast_stale": True,
                "forecast_last_update_ts": None,
            }

        params = {
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "timezone": geo.get("timezone") or "auto",
            "daily": "temperature_2m_max",
            "forecast_days": 16,
        }
        try:
            response = self.session.get(FORECAST_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Weather forecast request failed for %s on %s: %s", location, event_date_local, exc)
            if cached is not None:
                cached_out = dict(cached)
                cached_out["forecast_stale"] = True
                cached_out["forecast_missing_reason"] = "forecast_request_failed"
                return cached_out
            return {
                "forecast_ready": False,
                "forecast_missing_reason": "forecast_request_failed",
                "forecast_stale": True,
                "forecast_last_update_ts": None,
            }

        daily = payload.get("daily") or {}
        dates = daily.get("time") or []
        temps = daily.get("temperature_2m_max") or []
        temp_by_date = {
            str(date_value): _safe_float(temp_value)
            for date_value, temp_value in zip(dates, temps)
        }
        forecast_max_temp_c = temp_by_date.get(event_date_local)
        if forecast_max_temp_c is None:
            return {
                "forecast_ready": False,
                "forecast_missing_reason": "event_date_outside_forecast_horizon",
                "forecast_stale": True,
                "forecast_last_update_ts": None,
                "weather_country": geo.get("country"),
                "weather_resolution_timezone": geo.get("timezone"),
            }

        event_date = pd.to_datetime(event_date_local, errors="coerce")
        days_out = 0
        if pd.notna(event_date):
            days_out = int((event_date.date() - now_utc.date()).days)
        uncertainty_c = self._resolve_uncertainty(days_out)
        p_hit = self._p_interval_hit(
            max_temp_c=forecast_max_temp_c,
            lower_c=lower_c,
            upper_c=upper_c,
            uncertainty_c=uncertainty_c,
        )
        previous = self._forecast_memory.get(cache_key)
        previous_value = _safe_float(previous.get("forecast_max_temp_c"), None) if previous else None
        forecast_drift_c = 0.0 if previous_value is None else float(forecast_max_temp_c - previous_value)
        out = {
            "weather_country": geo.get("country"),
            "weather_resolution_timezone": geo.get("timezone"),
            "forecast_ready": True,
            "forecast_source": "open_meteo",
            "forecast_stale": False,
            "forecast_missing_reason": None,
            "forecast_max_temp_c": round(float(forecast_max_temp_c), 4),
            "forecast_p_hit_interval": round(float(p_hit), 6),
            "forecast_margin_to_lower_c": None if lower_c is None else round(float(forecast_max_temp_c - lower_c), 4),
            "forecast_margin_to_upper_c": None if upper_c is None else round(float(upper_c - forecast_max_temp_c), 4),
            "forecast_uncertainty_c": round(float(uncertainty_c), 4),
            "forecast_last_update_ts": now_utc.isoformat(),
            "forecast_drift_c": round(float(forecast_drift_c), 4),
        }
        self._forecast_memory[cache_key] = out
        self._forecast_cache[cache_key] = out
        return dict(out)
