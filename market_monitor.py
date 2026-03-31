import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from token_utils import parse_token_id_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
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
            midpoint = (_safe_float(best_bid, 0.0) + _safe_float(best_ask, 0.0)) / 2.0
            spread = abs(_safe_float(best_ask, 0.0) - _safe_float(best_bid, 0.0)) # BUG FIX 9: Handle empty strings gracefully
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


def save_market_snapshot(markets_df, logs_dir="logs"):
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
    dedupe_cols = [c for c in ["market_id", "condition_id", "slug"] if c in merged.columns] # BUG FIX 2: Stop deduplicating on timestamp
    if dedupe_cols:
        merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
    merged.to_csv(output_file, index=False)
    logging.info("Saved deduplicated market snapshot to %s", output_file)


if __name__ == "__main__":
    df = fetch_btc_markets()
    print(df.head())
