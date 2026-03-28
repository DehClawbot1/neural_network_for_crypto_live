"""
price_tracker.py
================
Tutorial-compatible price tracker and position tracker utilities.

Based on the Polymarket Python tutorial:
  - BONUS 1: Real-time price tracking via CLOB midpoint polling
  - BONUS 2: Address position tracking via Data API

Usage:
    from price_tracker import track_price, get_user_positions

    # Track price changes in real-time
    prices = track_price(token_id, duration_seconds=30, interval=5)

    # Get a user's current positions
    positions = get_user_positions("0xWalletAddress")
"""

import time
import logging

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


def track_price(token_id, duration_seconds=30, interval=5):
    """BONUS 1: Track price changes in real-time.

    Matches tutorial pattern:
        client = ClobClient(CLOB_API)
        mid = client.get_midpoint(token_id)
        mid_price = float(mid['mid'])
    """
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        logging.error("py-clob-client required for price tracking")
        return []

    print(f"Tracking price for {duration_seconds}s...\n")

    client = ClobClient(CLOB_API)
    start_time = time.time()
    prices = []

    while time.time() - start_time < duration_seconds:
        try:
            mid = client.get_midpoint(str(token_id))
            mid_price = float(mid.get("mid", 0.0))
            timestamp = time.strftime("%H:%M:%S")
            prices.append(mid_price)

            change = ""
            if len(prices) > 1:
                diff = prices[-1] - prices[-2]
                change = f" ({'+' if diff >= 0 else ''}{diff:.4f})"

            print(f"[{timestamp}] Price: {mid_price}{change}")
        except Exception as exc:
            logging.warning("Price fetch failed: %s", exc)

        time.sleep(interval)

    if len(prices) >= 2:
        print(f"\nTotal change: {prices[-1] - prices[0]:.4f}")
    return prices


def get_user_positions(wallet_address):
    """BONUS 2: Get a user's current positions via Data API.

    Matches tutorial pattern:
        url = f"{DATA_API}/positions"
        params = {"user": wallet_address}
        response = requests.get(url, params=params)
    """
    if not wallet_address:
        return []

    url = f"{DATA_API}/positions"
    params = {"user": wallet_address}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logging.error("Failed to fetch positions for %s: %s", wallet_address[:12], exc)
        return []


def get_market_deep_dive(market, clob_client=None):
    """Tutorial-style market deep dive:
        - Extract condition_id, token_ids
        - Fetch order book for YES token
        - Get midpoint, buy/sell prices, spread
    """
    if clob_client is None:
        try:
            from py_clob_client.client import ClobClient
            clob_client = ClobClient(CLOB_API)
        except ImportError:
            return None

    result = {
        "question": market.get("question"),
        "end_date": market.get("endDate", market.get("end_date")),
        "condition_id": market.get("conditionId", market.get("condition_id")),
    }

    # Extract token IDs (tutorial pattern)
    import json
    clob_token_ids = market.get("clobTokenIds", market.get("clob_token_ids", []))
    if isinstance(clob_token_ids, str):
        try:
            clob_token_ids = json.loads(clob_token_ids)
        except Exception:
            clob_token_ids = []

    yes_token_id = clob_token_ids[0] if len(clob_token_ids) >= 1 else market.get("yes_token_id")
    no_token_id = clob_token_ids[1] if len(clob_token_ids) >= 2 else market.get("no_token_id")

    result["yes_token_id"] = yes_token_id
    result["no_token_id"] = no_token_id

    if not yes_token_id:
        return result

    # Order book analysis (tutorial pattern)
    try:
        book = clob_client.get_order_book(str(yes_token_id))
        sorted_bids = sorted(getattr(book, "bids", []), key=lambda x: float(x.price), reverse=True)
        sorted_asks = sorted(getattr(book, "asks", []), key=lambda x: float(x.price), reverse=False)

        result["top_bids"] = [{"price": float(b.price), "size": float(b.size)} for b in sorted_bids[:5]]
        result["top_asks"] = [{"price": float(a.price), "size": float(a.size)} for a in sorted_asks[:5]]
    except Exception as exc:
        logging.warning("Order book fetch failed: %s", exc)
        result["top_bids"] = []
        result["top_asks"] = []

    # Midpoint, prices, spread (tutorial pattern)
    try:
        mid = clob_client.get_midpoint(str(yes_token_id))
        result["midpoint"] = float(mid.get("mid", 0.0))
    except Exception:
        result["midpoint"] = None

    try:
        buy_price = clob_client.get_price(str(yes_token_id), "BUY")
        result["best_ask"] = float(buy_price.get("price", 0.0))
    except Exception:
        result["best_ask"] = None

    try:
        sell_price = clob_client.get_price(str(yes_token_id), "SELL")
        result["best_bid"] = float(sell_price.get("price", 0.0))
    except Exception:
        result["best_bid"] = None

    try:
        spread = clob_client.get_spread(str(yes_token_id))
        result["spread"] = float(spread.get("spread", 0.0))
    except Exception:
        result["spread"] = None

    return result


if __name__ == "__main__":
    print("=== Polymarket Price & Position Tracker ===\n")

    # Example: fetch top markets and analyze the first one
    try:
        from market_monitor import fetch_active_markets_by_volume
        markets_df = fetch_active_markets_by_volume(limit=3)
        if not markets_df.empty:
            first = markets_df.iloc[0]
            yes_token = first.get("yes_token_id")
            if yes_token:
                print(f"Analyzing: {first.get('question')}")
                print(f"YES Token: {yes_token}\n")
                prices = track_price(yes_token, duration_seconds=10, interval=2)
    except Exception as exc:
        print(f"Demo failed: {exc}")
