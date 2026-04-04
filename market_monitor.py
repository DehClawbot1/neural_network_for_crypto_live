import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from token_utils import parse_token_id_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

# Slug rotation intervals for btc-updown markets (prefix → seconds between rotations)
_ROTATING_MARKET_INTERVALS = {
    "btc-updown-5m-": 300,      # 5 minutes
    "btc-updown-15m-": 900,     # 15 minutes
    "btc-updown-4h-": 14400,    # 4 hours
}

# Daily BTC market slug pattern (date-based, not timestamp-based)
_DAILY_BTC_PREFIX = "bitcoin-up-or-down-on-"
BTC_KEYWORDS = [
    "bitcoin",
    "btc",
    "bitcoin above",
    "bitcoin below",
    "bitcoin price",
    "bitcoin up or down",
    "bitcoin para cima ou para baixo",
    "para cima ou para baixo",
    "$btc",
]


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _is_btc_market(market: dict) -> bool:
    question = str(market.get("question", ""))
    title = str(market.get("title", ""))
    text_blob = f"{question} {title}".lower()
    return any(keyword in text_blob for keyword in BTC_KEYWORDS)


def _market_to_row(market: dict) -> dict:
    question = str(market.get("question", "") or market.get("title", ""))
    clob_token_ids = parse_token_id_list(market.get("clobTokenIds") or market.get("clob_token_ids") or [])
    yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
    no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None
    best_bid = market.get("bestBid") if market.get("bestBid") is not None else market.get("best_bid", market.get("bid"))
    best_ask = market.get("bestAsk") if market.get("bestAsk") is not None else market.get("best_ask", market.get("ask")) # BUG FIX 10: Do not drop explicit 0.0s
    midpoint = None
    spread = None
    if best_bid is not None and best_ask is not None:
        try:
            bid_f = _safe_float(best_bid, None)
            ask_f = _safe_float(best_ask, None)
            if bid_f is not None and ask_f is not None and (bid_f + ask_f) > 0:
                midpoint = (bid_f + ask_f) / 2.0
                spread = abs(ask_f - bid_f)
        except Exception:
            midpoint = None
            spread = None
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
        "midpoint": midpoint,
        "spread": spread,
        "bid_size": _safe_float(market.get("bidSize") or market.get("bid_size")),
        "ask_size": _safe_float(market.get("askSize") or market.get("ask_size")),
        "end_date": market.get("endDate") or market.get("end_date"),
        "slug": market.get("slug"),
        "clob_token_ids": clob_token_ids,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "url": f"https://polymarket.com/event/{market.get('slug')}" if market.get("slug") else None,
    }


def _fetch_page(session: requests.Session, *, closed: bool, limit: int, offset: int = 0, slug: str | None = None):
    params = {"limit": limit, "closed": str(bool(closed)).lower(), "offset": offset}
    if slug:
        params["slug"] = slug
    response = session.get(GAMMA_MARKETS_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def _generate_candidate_slugs(prefix: str, look_back: int = 6, look_ahead: int = 3) -> list[str]:
    """
    Generate candidate rotating-market slugs around the current time.

    For btc-updown-5m- markets, timestamps are Unix epochs at 300-second boundaries.
    For btc-updown-4h- markets, timestamps are at 14400-second boundaries.

    look_back/look_ahead = how many intervals in each direction to check.
    """
    interval = None
    for known_prefix, secs in _ROTATING_MARKET_INTERVALS.items():
        if prefix.startswith(known_prefix) or known_prefix.startswith(prefix):
            interval = secs
            prefix = known_prefix  # normalise
            break
    if interval is None:
        return []

    import time
    now = int(time.time())
    # Align to interval boundary
    base = (now // interval) * interval
    slugs = []
    for i in range(-look_back, look_ahead + 1):
        ts = base + i * interval
        if ts > 0:
            slugs.append(f"{prefix}{ts}")
    return slugs


def _fetch_event_markets(session: requests.Session, event_slug: str) -> list[dict]:
    """
    Fetch markets from the Gamma events API by event slug.

    Returns the list of market dicts nested under the event, or [].
    The events endpoint returns markets that do NOT appear in bulk /markets listings.
    """
    try:
        resp = session.get(GAMMA_EVENTS_URL, params={"slug": event_slug}, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        # The events endpoint returns a list of event objects, each with a "markets" key
        events = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
        markets = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_markets = event.get("markets") or []
            if isinstance(event_markets, list):
                markets.extend(event_markets)
        return markets
    except Exception as exc:
        logging.debug("Events API query failed for slug=%s: %s", event_slug, exc)
        return []


def fetch_btc_updown_markets(prefix: str = "btc-updown-5m-", closed: bool = False,
                              look_back: int = 6, look_ahead: int = 3) -> pd.DataFrame:
    """
    Fetch btc-updown rotating markets by generating candidate slugs and querying
    both the events API and the direct markets API.

    These markets do NOT appear in bulk /markets pagination — they must be queried
    by exact slug or via the events endpoint.
    """
    candidate_slugs = _generate_candidate_slugs(prefix, look_back=look_back, look_ahead=look_ahead)
    if not candidate_slugs:
        logging.warning("No candidate slugs generated for prefix=%s", prefix)
        return pd.DataFrame()

    session = requests.Session()
    rows = []
    seen = set()

    for slug in candidate_slugs:
        # Try events API first (returns markets nested under event)
        event_markets = _fetch_event_markets(session, slug)
        if event_markets:
            for market in event_markets:
                if closed is False and str(market.get("closed", "")).lower() in ("true", "1", "yes"):
                    continue
                row = _market_to_row(market)
                mid = str(row.get("market_id") or "")
                if mid and mid not in seen:
                    seen.add(mid)
                    rows.append(row)

        # Also try direct market slug query as fallback
        try:
            direct_markets = _fetch_page(session, closed=closed, limit=10, offset=0, slug=slug)
            for market in direct_markets:
                row = _market_to_row(market)
                mid = str(row.get("market_id") or "")
                if mid and mid not in seen:
                    seen.add(mid)
                    rows.append(row)
        except Exception as exc:
            logging.debug("Direct slug query failed for %s: %s", slug, exc)

    df = pd.DataFrame(rows)
    if not df.empty and "slug" in df.columns:
        df = df.drop_duplicates(subset=["slug"], keep="last")
    logging.info(
        "Fetched %d btc-updown markets for prefix '%s' (candidates=%d, closed=%s).",
        len(df), prefix, len(candidate_slugs), closed,
    )
    return df


def fetch_btc_markets(limit=1000, closed=False, max_offset=0):
    """
    Fetch BTC-related public markets.

    Supervisor currently calls this with max_offset, so this function must accept it.
    The older implementation crashed on the unexpected keyword and broke the bot at runtime.
    """
    session = requests.Session()
    rows = []
    seen_market_ids = set()
    offsets = [0]
    if max_offset:
        step = max(50, min(int(limit), 500))
        offsets = list(range(0, int(max_offset) + 1, step))

    for offset in offsets:
        markets = _fetch_page(session, closed=closed, limit=limit, offset=offset)
        if not markets:
            break
        for market in markets:
            if not _is_btc_market(market):
                continue
            row = _market_to_row(market)
            market_id = str(row.get("market_id") or "")
            if market_id and market_id in seen_market_ids:
                continue
            if market_id:
                seen_market_ids.add(market_id)
            rows.append(row)
        if not markets: break # BUG FIX 4: Prevent arbitrary page-size limits from halting fetch

    df = pd.DataFrame(rows)
    if not df.empty and "market_id" in df.columns:
        df = df.drop_duplicates(subset=["market_id"], keep="last")
    logging.info("Fetched %s BTC-related markets (closed=%s).", len(df), closed)
    return df


def fetch_markets_by_slugs(slugs):
    session = requests.Session()
    rows = []
    for slug in [str(s).strip() for s in (slugs or []) if str(s).strip()]:
        try:
            for market in _fetch_page(session, closed=False, limit=50, offset=0, slug=slug):
                if market.get("slug") == slug:
                    rows.append(_market_to_row(market))
        except Exception as exc:
            logging.warning("Failed to fetch market by slug=%s: %s", slug, exc)
    df = pd.DataFrame(rows)
    if not df.empty and "slug" in df.columns:
        df = df.drop_duplicates(subset=["slug"], keep="last")
    return df


def fetch_markets_by_slug_prefix(prefix: str, limit: int = 500, max_offset: int = 2000, closed: bool = False):
    """
    Fetch markets whose slug starts with the provided prefix.

    Strategy:
    1. For known rotating markets (btc-updown-5m-, btc-updown-4h-), use the events API
       with generated candidate slugs — these markets do NOT appear in bulk pagination.
    2. Fall back to scanning paginated gamma markets for other prefixes.
    """
    slug_prefix = str(prefix or "").strip().lower()
    if not slug_prefix:
        return pd.DataFrame()

    # --- Strategy 1: Events API for known rotating markets ---
    is_rotating = any(slug_prefix.startswith(p) or p.startswith(slug_prefix)
                      for p in _ROTATING_MARKET_INTERVALS)
    if is_rotating:
        df = fetch_btc_updown_markets(prefix=slug_prefix, closed=closed)
        if not df.empty:
            return df
        logging.info(
            "Events API returned 0 results for rotating prefix '%s', trying bulk scan...",
            slug_prefix,
        )

    # --- Strategy 2: Bulk pagination scan (original approach) ---
    session = requests.Session()
    rows = []
    seen = set()
    page_size = max(50, min(int(limit), 500))
    offsets = list(range(0, int(max_offset) + 1, page_size))

    for offset in offsets:
        try:
            markets = _fetch_page(session, closed=closed, limit=page_size, offset=offset)
        except Exception as exc:
            logging.warning(
                "Failed prefix fetch for slug_prefix=%s closed=%s offset=%s: %s",
                slug_prefix,
                closed,
                offset,
                exc,
            )
            break
        if not markets:
            break
        for market in markets:
            slug = str(market.get("slug") or "").strip().lower()
            if not slug.startswith(slug_prefix):
                continue
            row = _market_to_row(market)
            dedupe_key = str(row.get("market_id") or row.get("slug") or "")
            if dedupe_key and dedupe_key in seen:
                continue
            if dedupe_key:
                seen.add(dedupe_key)
            rows.append(row)
        if len(markets) < page_size:
            break

    df = pd.DataFrame(rows)
    if not df.empty and "slug" in df.columns:
        df = df.drop_duplicates(subset=["slug"], keep="last")
    logging.info(
        "Fetched %s markets by slug prefix '%s' (closed=%s).",
        len(df),
        slug_prefix,
        closed,
    )
    return df


def save_market_snapshot(markets_df, logs_dir="logs"):
    import os as _os
    if markets_df is None or markets_df.empty:
        return

    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    output_file = logs_path / "markets.csv"

    existing = pd.DataFrame()
    if output_file.exists():
        try:
            existing = pd.read_csv(output_file, engine="python", on_bad_lines="skip")
        except Exception:
            existing = pd.DataFrame()

    merged = pd.concat([existing, markets_df], ignore_index=True)
    if "timestamp" in merged.columns:
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
        # Purge rows older than TTL — prevents stale resolved-market token IDs
        # from persisting in the scraper's market universe indefinitely.
        try:
            ttl_hours = max(1, int(_os.getenv("MARKET_SNAPSHOT_TTL_HOURS", "48")))
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=ttl_hours)
            mask = merged["timestamp"].isna() | (merged["timestamp"] >= cutoff)
            dropped = int((~mask).sum())
            merged = merged[mask]
            if dropped > 0:
                logging.info("Market snapshot: purged %d stale rows older than %dh", dropped, ttl_hours)
        except Exception as _exc:
            logging.debug("Market snapshot TTL purge skipped: %s", _exc)
    dedupe_cols = [c for c in ["market_id", "condition_id", "slug"] if c in merged.columns] # BUG FIX 2: Stop deduplicating on timestamp
    if dedupe_cols:
        merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
    merged.to_csv(output_file, index=False)
    logging.info("Saved deduplicated market snapshot to %s (%d rows)", output_file, len(merged))


if __name__ == "__main__":
    df = fetch_btc_markets()
    print(df.head())
