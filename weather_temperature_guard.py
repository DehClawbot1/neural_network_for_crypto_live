from __future__ import annotations

import math
import os
import re
from typing import Iterable


def _safe_float(value, default=None):
    try:
        parsed = float(value)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _normalize_location(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def weather_city_date_cluster_key(source: dict | None) -> str:
    source = source or {}
    location = _normalize_location(source.get("weather_location"))
    event_date = str(source.get("weather_event_date_local") or "").strip()
    if not location or not event_date:
        return ""
    return f"{location}|{event_date}"


def _intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    left_low, left_high = left
    right_low, right_high = right
    return max(left_low, right_low) <= min(left_high, right_high)


def weather_payout_intervals(source: dict | None) -> list[tuple[float, float]]:
    source = source or {}
    family = str(source.get("market_family") or "").strip().lower()
    side = str(source.get("outcome_side", source.get("side")) or "").strip().upper()
    lower_c = _safe_float(source.get("weather_lower_c"), None)
    upper_c = _safe_float(source.get("weather_upper_c"), None)
    inf = float("inf")
    if family == "weather_temperature_threshold":
        if lower_c is None:
            return []
        if side == "YES":
            return [(lower_c, inf)]
        if side == "NO":
            return [(-inf, lower_c)]
        return []
    if family == "weather_temperature_range":
        if lower_c is None or upper_c is None:
            return []
        if upper_c < lower_c:
            lower_c, upper_c = upper_c, lower_c
        if side == "YES":
            return [(lower_c, upper_c)]
        if side == "NO":
            return [(-inf, lower_c), (upper_c, inf)]
        return []
    return []


def weather_positions_conflict(candidate: dict | None, existing: dict | None) -> bool:
    candidate_intervals = weather_payout_intervals(candidate)
    existing_intervals = weather_payout_intervals(existing)
    if not candidate_intervals or not existing_intervals:
        return False
    for left in candidate_intervals:
        for right in existing_intervals:
            if _intervals_overlap(left, right):
                return False
    return True


def find_conflicting_weather_temperature_position(
    candidate_row: dict,
    active_trades: Iterable,
    *,
    cluster_cap: int | None = None,
):
    cluster_cap = cluster_cap if cluster_cap is not None else max(
        1,
        int(os.getenv("WEATHER_CITY_DATE_CLUSTER_CAP", "1") or 1),
    )
    candidate_cluster = weather_city_date_cluster_key(candidate_row)
    if not candidate_cluster:
        return None

    same_cluster = []
    for trade in active_trades or []:
        trade_row = getattr(trade, "__dict__", {})
        trade_family = str(trade_row.get("market_family") or "").strip().lower()
        if not trade_family.startswith("weather_temperature"):
            continue
        trade_cluster = weather_city_date_cluster_key(trade_row)
        if trade_cluster != candidate_cluster:
            continue
        same_cluster.append(trade)
        if weather_positions_conflict(candidate_row, trade_row):
            return {
                "reason": "weather_temperature_interval_conflict",
                "trade": trade,
                "cluster_key": candidate_cluster,
            }

    if len(same_cluster) >= max(1, int(cluster_cap)):
        return {
            "reason": "weather_temperature_cluster_cap_reached",
            "trade": same_cluster[0],
            "cluster_key": candidate_cluster,
            "cluster_size": len(same_cluster),
            "cluster_cap": int(cluster_cap),
        }
    return None
