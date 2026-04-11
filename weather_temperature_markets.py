from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests

from token_utils import parse_token_id_list


logger = logging.getLogger(__name__)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
_TEMP_DEGREE_RE = r"(?:[^0-9A-Za-z\s]{1,2})?"
_DASH_RE = r"(?:-|–|to)"

_THRESHOLD_RE = re.compile(
    rf"will\s+the\s+(?:highest|high|maximum|max(?:imum)?)\s+temperature\s+in\s+(?P<location>.+?)\s+be\s+"
    rf"(?P<threshold>-?\d+(?:\.\d+)?)\s*{_TEMP_DEGREE_RE}\s*(?P<unit>[fc])(?:\s+or\s+higher|\+)?\s+on\s+(?P<date>.+?)\??$",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    rf"will\s+the\s+(?:highest|high|maximum|max(?:imum)?)\s+temperature\s+in\s+(?P<location>.+?)\s+be\s+between\s+"
    rf"(?P<lower>-?\d+(?:\.\d+)?)\s*{_DASH_RE}\s*(?P<upper>-?\d+(?:\.\d+)?)\s*{_TEMP_DEGREE_RE}\s*(?P<unit>[fc])\s+on\s+(?P<date>.+?)\??$",
    re.IGNORECASE,
)


def _safe_float(value, default=None):
    try:
        parsed = float(value)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _f_to_c(temp_f: float) -> float:
    return (float(temp_f) - 32.0) * (5.0 / 9.0)


def _normalize_location_text(value) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip("?.!,;:")
    return text


def _normalize_unit(unit_value) -> str:
    unit = str(unit_value or "").strip().upper()
    return "F" if unit == "F" else "C"


def _to_celsius(value, unit: str) -> float:
    unit = _normalize_unit(unit)
    numeric = _safe_float(value, default=0.0) or 0.0
    return _f_to_c(numeric) if unit == "F" else float(numeric)


def _parse_date_without_year(raw_date, reference_date: datetime | None = None) -> str | None:
    text = str(raw_date or "").strip().rstrip("?")
    if not text:
        return None

    ref = reference_date.astimezone(timezone.utc) if reference_date is not None else datetime.now(timezone.utc)
    direct = pd.to_datetime(text, errors="coerce")
    if pd.notna(direct):
        try:
            return direct.date().isoformat()
        except Exception:
            return None

    candidates = []
    for year in (ref.year, ref.year + 1):
        parsed = pd.to_datetime(f"{text} {year}", errors="coerce")
        if pd.notna(parsed):
            candidates.append(parsed)
    if not candidates:
        return None

    chosen = candidates[0]
    for candidate in candidates:
        if candidate.date() >= ref.date():
            chosen = candidate
            break
    return chosen.date().isoformat()


def is_weather_temperature_market(market: dict | None) -> bool:
    market = market or {}
    question = str(market.get("question") or market.get("title") or "").strip().lower()
    if not question:
        return False
    return any(
        phrase in question
        for phrase in (
            "highest temperature",
            "high temperature",
            "maximum temperature",
            "max temperature",
        )
    )


def parse_weather_temperature_market_text(
    market_title: str | None,
    *,
    market_slug: str | None = None,
    reference_date: datetime | None = None,
) -> dict:
    text = str(market_title or "").strip()
    parsed = {
        "market_family": "weather_temperature_other",
        "weather_parseable": False,
        "weather_parse_error": "not_weather_temperature_market",
        "weather_location": None,
        "weather_country": None,
        "weather_event_date_local": None,
        "weather_resolution_timezone": None,
        "weather_question_type": None,
        "weather_temp_unit": None,
        "weather_lower_c": None,
        "weather_upper_c": None,
        "weather_interval_width_c": None,
    }
    if not text:
        return parsed

    match = _RANGE_RE.match(text)
    if match:
        unit = _normalize_unit(match.group("unit"))
        lower_c = _to_celsius(match.group("lower"), unit)
        upper_c = _to_celsius(match.group("upper"), unit)
        if upper_c < lower_c:
            lower_c, upper_c = upper_c, lower_c
        parsed.update(
            {
                "market_family": "weather_temperature_range",
                "weather_parseable": True,
                "weather_parse_error": None,
                "weather_location": _normalize_location_text(match.group("location")),
                "weather_event_date_local": _parse_date_without_year(match.group("date"), reference_date=reference_date),
                "weather_resolution_timezone": None,
                "weather_question_type": "range",
                "weather_temp_unit": unit,
                "weather_lower_c": round(lower_c, 4),
                "weather_upper_c": round(upper_c, 4),
                "weather_interval_width_c": round(max(0.0, upper_c - lower_c), 4),
            }
        )
        return parsed

    match = _THRESHOLD_RE.match(text)
    if match:
        unit = _normalize_unit(match.group("unit"))
        threshold_c = _to_celsius(match.group("threshold"), unit)
        parsed.update(
            {
                "market_family": "weather_temperature_threshold",
                "weather_parseable": True,
                "weather_parse_error": None,
                "weather_location": _normalize_location_text(match.group("location")),
                "weather_event_date_local": _parse_date_without_year(match.group("date"), reference_date=reference_date),
                "weather_resolution_timezone": None,
                "weather_question_type": "threshold",
                "weather_temp_unit": unit,
                "weather_lower_c": round(threshold_c, 4),
                "weather_upper_c": None,
                "weather_interval_width_c": None,
            }
        )
        return parsed

    parsed["weather_parse_error"] = "unrecognized_temperature_question"
    if str(market_slug or "").strip():
        parsed["market_slug"] = market_slug
    return parsed


def enrich_weather_temperature_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()

    work = frame.copy()
    rows = []
    for _, row in work.iterrows():
        base = row.to_dict()
        parsed = parse_weather_temperature_market_text(
            base.get("market_title", base.get("question", base.get("title"))),
            market_slug=base.get("market_slug", base.get("slug")),
        )
        base.update(parsed)
        rows.append(base)
    return pd.DataFrame(rows)


def _market_to_weather_row(market: dict) -> dict:
    question = str(market.get("question") or market.get("title") or "").strip()
    clob_token_ids = parse_token_id_list(market.get("clobTokenIds") or market.get("clob_token_ids") or [])
    yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
    no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None
    best_bid = market.get("bestBid") if market.get("bestBid") is not None else market.get("best_bid", market.get("bid"))
    best_ask = market.get("bestAsk") if market.get("bestAsk") is not None else market.get("best_ask", market.get("ask"))
    parsed = parse_weather_temperature_market_text(question, market_slug=market.get("slug"))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": market.get("id") or market.get("market_id"),
        "condition_id": market.get("conditionId") or market.get("condition_id"),
        "question": question,
        "market_title": question,
        "active": market.get("active"),
        "closed": market.get("closed"),
        "liquidity": _safe_float(market.get("liquidity"), 0.0),
        "volume": _safe_float(market.get("volume"), 0.0),
        "last_trade_price": _safe_float(market.get("lastTradePrice") or market.get("last_trade_price"), 0.0),
        "current_price": _safe_float(market.get("lastTradePrice") or market.get("last_trade_price"), 0.0),
        "best_bid": _safe_float(best_bid),
        "best_ask": _safe_float(best_ask),
        "spread": abs((_safe_float(best_ask, 0.0) or 0.0) - (_safe_float(best_bid, 0.0) or 0.0)),
        "end_date": market.get("endDate") or market.get("end_date"),
        "slug": market.get("slug"),
        "market_slug": market.get("slug"),
        "clob_token_ids": clob_token_ids,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "url": f"https://polymarket.com/event/{market.get('slug')}" if market.get("slug") else None,
        "market_family": parsed.get("market_family"),
        "weather_parseable": parsed.get("weather_parseable"),
        "weather_parse_error": parsed.get("weather_parse_error"),
        "weather_location": parsed.get("weather_location"),
        "weather_country": parsed.get("weather_country"),
        "weather_event_date_local": parsed.get("weather_event_date_local"),
        "weather_resolution_timezone": parsed.get("weather_resolution_timezone"),
        "weather_question_type": parsed.get("weather_question_type"),
        "weather_temp_unit": parsed.get("weather_temp_unit"),
        "weather_lower_c": parsed.get("weather_lower_c"),
        "weather_upper_c": parsed.get("weather_upper_c"),
        "weather_interval_width_c": parsed.get("weather_interval_width_c"),
    }


def _fetch_page(session: requests.Session, *, closed: bool, limit: int, offset: int = 0):
    response = session.get(
        GAMMA_MARKETS_URL,
        params={"limit": limit, "closed": str(bool(closed)).lower(), "offset": offset},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def fetch_weather_temperature_markets(limit: int = 500, closed: bool = False, max_offset: int = 0) -> pd.DataFrame:
    if max_offset is None:
        try:
            max_offset = int(os.getenv("WEATHER_MARKETS_MAX_OFFSET", "5000") or 5000)
        except Exception:
            max_offset = 5000
    max_offset = max(0, int(max_offset))
    session = requests.Session()
    offsets = [0]
    if max_offset:
        step = max(50, min(int(limit), 500))
        offsets = list(range(0, int(max_offset) + 1, step))

    rows = []
    seen_ids = set()
    scanned_markets = 0
    for offset in offsets:
        try:
            markets = _fetch_page(session, closed=closed, limit=limit, offset=offset)
        except Exception as exc:
            logger.warning("Weather market fetch failed at offset=%s: %s", offset, exc)
            break
        if not markets:
            break
        scanned_markets += len(markets)
        for market in markets:
            if not is_weather_temperature_market(market):
                continue
            row = _market_to_weather_row(market)
            market_id = str(row.get("market_id") or "")
            if market_id and market_id in seen_ids:
                continue
            if market_id:
                seen_ids.add(market_id)
            rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty and "market_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["market_id"], keep="last")
    logger.info(
        "Fetched %s weather temperature markets (closed=%s, scanned=%s Gamma markets, max_offset=%s).",
        len(frame),
        closed,
        scanned_markets,
        max_offset,
    )
    return frame
