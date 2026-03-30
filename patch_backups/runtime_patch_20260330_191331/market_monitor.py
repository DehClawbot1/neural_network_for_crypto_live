import ast
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"


def _normalize_token_list(value):
    """FIX: Tutorial shows clobTokenIds comes as a JSON string that needs parsing:
        clob_token_ids = market.get('clobTokenIds')
        clob_token_ids = json.loads(clob_token_ids)
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def parse_gamma_market(market):
    """Helper to convert raw Gamma API market object to our internal schema.

    FIX: Aligned with tutorial's market deep-dive pattern:
        market = markets[1]
        clob_token_ids = json.loads(market.get('clobTokenIds'))
        yes_token_id = clob_token_ids[0]
        no_token_id = clob_token_ids[1]
    """
    question = str(market.get("question", ""))
    title = str(market.get("title", ""))

    # FIX: Tutorial extracts tokens from clobTokenIds (JSON string)
    # This is more reliable than the tokens array for CLOB trading
    clob_token_ids = _normalize_token_list(market.get("clobTokenIds"))

    # Also check the tokens array as a fallback
    tokens = market.get("tokens") or []
    yes_token = next((t for t in tokens if str(t.get("outcome", "")).upper() == "YES"), {})
    no_token = next((t for t in tokens if str(t.get("outcome", "")).upper() == "NO"), {})

    # FIX: Prefer clobTokenIds[0]/[1] (tutorial pattern), fall back to tokens array
    yes_token_id = (clob_token_ids[0] if len(clob_token_ids) > 0 else None) or yes_token.get("token_id") or yes_token.get("id")
    no_token_id = (clob_token_ids[1] if len(clob_token_ids) > 1 else None) or no_token.get("token_id") or no_token.get("id")

    best_bid = market.get("bestBid") or market.get("best_bid") or market.get("bid")
    best_ask = market.get("bestAsk") or market.get("best_ask") or market.get("ask")

    # FIX: Tutorial shows outcomePrices as a useful field for quick price checks
    outcome_prices = market.get("outcomePrices")
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except Exception:
            outcome_prices = None

    midpoint, spread = None, None
    if best_bid is not None and best_ask is not None:
        try:
            midpoint = (float(best_bid) + float(best_ask)) / 2.0
            spread = abs(float(best_ask) - float(best_bid))
        except Exception:
            pass

    last_trade_price = market.get("lastTradePrice", 0)
    # FIX: If lastTradePrice is missing, try extracting from outcomePrices
    if (last_trade_price is None or last_trade_price == 0) and outcome_prices:
        try:
            if isinstance(outcome_prices, list) and len(outcome_prices) > 0:
                last_trade_price = float(outcome_prices[0])
            elif isinstance(outcome_prices, str):
                last_trade_price = float(json.loads(outcome_prices)[0])
        except Exception:
            pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": market.get("id"),
        "condition_id": market.get("conditionId"),
        "question": question or title,
        "active": market.get("active"),
        "closed": market.get("closed"),
        "liquidity": market.get("liquidity", 0),
        "volume": market.get("volume", 0),
        "volume24hr": market.get("volume24hr", 0),
        "last_trade_price": last_trade_price,
        "outcome_prices": json.dumps(outcome_prices) if outcome_prices else None,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "midpoint": midpoint,
        "spread": spread,
        "bid_size": market.get("bidSize") or market.get("bid_size"),
        "ask_size": market.get("askSize") or market.get("ask_size"),
        "end_date": market.get("endDate"),
        "slug": market.get("slug"),
        "clob_token_ids": clob_token_ids,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "url": f"https://polymarket.com/event/{market.get('slug')}" if market.get("slug") else None,
    }


def fetch_markets_by_condition_ids(condition_ids):
    """Legacy helper retained for compatibility; Gamma condition_id filtering is unreliable."""
    return pd.DataFrame()


def fetch_markets_by_slugs(slugs):
    """Fetch specific markets from Gamma API by their slugs to fill metadata gaps."""
    new_markets = []
    slugs_to_fetch = list(set([s for s in slugs if s and str(s) != "nan"]))
    logging.info("JIT: Fetching metadata for %s specific slugs...", len(slugs_to_fetch))
    for slug in slugs_to_fetch:
        try:
            response = requests.get(GAMMA_MARKETS_URL, params={"slug": slug}, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, list):
                for market in data:
                    if str(market.get("slug", "")) == str(slug):
                        new_markets.append(parse_gamma_market(market))
            time.sleep(0.1)
        except Exception as e:
            logging.error("Failed to fetch market slug %s: %s", slug, e)
    return pd.DataFrame(new_markets)


def fetch_active_markets_by_volume(limit=10):
    """FIX: Added tutorial-style market discovery sorted by 24h volume:
        response = requests.get(GAMMA_API + "/markets", params={
            "limit": 10, "active": True, "closed": False,
            "order": "volume24hr", "ascending": False
        })
    """
    params = {
        "limit": limit,
        "active": True,
        "closed": False,
        "order": "volume24hr",
        "ascending": False,
    }
    try:
        response = requests.get(GAMMA_MARKETS_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        markets = [parse_gamma_market(m) for m in data]
        logging.info("Fetched %s active markets sorted by 24h volume.", len(markets))
        return pd.DataFrame(markets)
    except Exception as e:
        logging.error("Failed to fetch active markets by volume: %s", e)
        return pd.DataFrame()


def fetch_btc_markets(limit_per_page=100, closed=False, max_offset=2000):
    """
    Fetch public Polymarket markets and filter for Bitcoin/BTC-related ones.
    This is for research/monitoring only.

    FIX: Uses tutorial-compatible pagination and adds volume24hr sorting.
    """
    markets = []
    closed_flag = bool(closed)
    offset = 0
    while True:
        params = {
            "limit": limit_per_page,
            "offset": offset,
            "closed": str(closed_flag).lower(),
        }
        try:
            response = requests.get(GAMMA_MARKETS_URL, params=params, timeout=20)
            response.raise_for_status()
            page_data = response.json()
        except Exception as e:
            logging.error("Failed to fetch markets at offset %d: %s", offset, e)
            break
        if not page_data:
            break
        markets.extend(page_data)
        offset += limit_per_page
        if offset > max_offset:
            break

    btc_markets = []
    for market in markets:
        question = str(market.get("question", ""))
        title = str(market.get("title", ""))
        text_blob = f"{question} {title}".lower()
        btc_keywords = [
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
        if any(keyword in text_blob for keyword in btc_keywords):
            btc_markets.append(parse_gamma_market(market))

    logging.info("Fetched %s BTC-related markets.", len(btc_markets))
    return pd.DataFrame(btc_markets)


def save_market_snapshot(markets_df, logs_dir="logs"):
    if markets_df is None or markets_df.empty:
        return

    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    output_file = logs_path / "markets.csv"

    existing_df = pd.DataFrame()
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file, engine="python", on_bad_lines="skip")
        except Exception:
            existing_df = pd.DataFrame()

    combined = pd.concat([existing_df, markets_df], ignore_index=True, sort=False)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    dedupe_cols = [c for c in ["condition_id", "market_id", "question", "slug"] if c in combined.columns]
    if dedupe_cols:
        combined = combined.sort_values("timestamp", kind="stable") if "timestamp" in combined.columns else combined
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    combined.to_csv(output_file, index=False)
    logging.info("Saved market snapshot to %s", output_file)


if __name__ == "__main__":
    # FIX: Demo now matches tutorial pattern for market discovery
    print("=== Active Markets by Volume ===")
    vol_df = fetch_active_markets_by_volume(limit=5)
    if not vol_df.empty:
        for _, m in vol_df.iterrows():
            print(f"  {m.get('question')}")
            print(f"    Volume 24h: ${float(m.get('volume24hr', 0)):,.0f}")
            print(f"    Liquidity: ${float(m.get('liquidity', 0)):,.0f}")
            print(f"    Price: {m.get('outcome_prices', 'N/A')}")
            print()

    print("=== BTC Markets ===")
    df = fetch_btc_markets()
    if df.empty:
        print("No BTC-related markets found.")
    else:
        print(df.head()[["market_id", "question", "liquidity", "volume", "url"]])
