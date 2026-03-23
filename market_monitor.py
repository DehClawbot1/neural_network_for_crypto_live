import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"


def fetch_btc_markets(limit_per_page=100, closed=False, max_offset=2000):
    """
    Fetch public Polymarket markets and filter for Bitcoin/BTC-related ones.
    This is for research/monitoring only.
    """
    markets = []
    offset = 0
    while True:
        params = {
            "limit": limit_per_page,
            "offset": offset,
            "closed": str(closed).lower(),
        }
        response = requests.get(GAMMA_MARKETS_URL, params=params, timeout=20)
        response.raise_for_status()
        page_data = response.json()
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
            clob_token_ids = market.get("clobTokenIds") or []
            yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
            no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None
            best_bid = market.get("bestBid") or market.get("best_bid") or market.get("bid")
            best_ask = market.get("bestAsk") or market.get("best_ask") or market.get("ask")
            midpoint = None
            spread = None
            if best_bid is not None and best_ask is not None:
                try:
                    midpoint = (float(best_bid) + float(best_ask)) / 2.0
                    spread = abs(float(best_ask) - float(best_bid))
                except Exception:
                    midpoint = None
                    spread = None

            btc_markets.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "market_id": market.get("id"),
                    "condition_id": market.get("conditionId"),
                    "question": question or title,
                    "active": market.get("active"),
                    "closed": market.get("closed"),
                    "liquidity": market.get("liquidity", 0),
                    "volume": market.get("volume", 0),
                    "last_trade_price": market.get("lastTradePrice", 0),
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
            )

    logging.info("Fetched %s BTC-related markets.", len(btc_markets))
    return pd.DataFrame(btc_markets)


def save_market_snapshot(markets_df, logs_dir="logs"):
    if markets_df is None or markets_df.empty:
        return

    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    output_file = logs_path / "markets.csv"
    markets_df.to_csv(output_file, mode="a", header=not output_file.exists(), index=False)
    logging.info("Saved market snapshot to %s", output_file)


if __name__ == "__main__":
    df = fetch_btc_markets()
    if df.empty:
        print("No BTC-related markets found.")
    else:
        print(df.head()[["market_id", "question", "liquidity", "volume", "url"]])

