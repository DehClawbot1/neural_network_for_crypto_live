import time
import logging

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging for zero-intervention monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _build_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_top_crypto_traders(limit=100):
    """Fetches the top proxy wallets by PnL in the CRYPTO category for the week."""
    url = "https://data-api.polymarket.com/v1/leaderboard"
    params = {
        "category": "CRYPTO",
        "timePeriod": "WEEK",
        "orderBy": "PNL",
        "limit": limit,
    }

    try:
        session = _build_session()
        response = session.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        top_wallets = [user.get("proxyWallet") for user in data if user.get("proxyWallet")]
        logging.info(f"Successfully fetched {len(top_wallets)} top traders from the leaderboard.")
        return top_wallets
    except Exception as e:
        logging.error(f"Error fetching leaderboard: {e}")
        return []


def get_recent_btc_trades(wallet_address, limit=50):
    """Fetch recent wallet trades from the public Data API and filter for BTC markets."""
    url = "https://data-api.polymarket.com/trades"
    params = {
        "user": wallet_address,
        "limit": limit,
        "side": "BUY",
    }

    try:
        session = _build_session()
        response = session.get(url, params=params, timeout=20)
        response.raise_for_status()
        trades = response.json()

        signals = []
        for trade in trades:
            title = str(trade.get("title", ""))
            title_l = title.lower()

            if (
                "bitcoin" in title_l
                or "btc" in title_l
                or "bitcoin para cima ou para baixo" in title_l
                or "para cima ou para baixo" in title_l
            ):
                signals.append(
                    {
                        "trader_wallet": wallet_address,
                        "market_title": title,
                        "token_id": trade.get("tokenId"),
                        "condition_id": trade.get("conditionId"),
                        "trade_side": trade.get("side"),
                        "outcome_side": trade.get("outcome"),
                        "side": trade.get("outcome"),
                        "price": float(trade.get("price", 0)),
                        "size": float(trade.get("size", 0)),
                        "timestamp": trade.get("timestamp"),
                    }
                )
        return signals
    except Exception as e:
        logging.error(f"Error fetching trades for {wallet_address[:8]}...: {e}")
        return []


def run_scraper_cycle():
    """Main execution loop to pull the latest Alpha Signals."""
    logging.info("Starting Alpha Signal Scraper cycle...")
    top_traders = get_top_crypto_traders(limit=100)

    all_signals = []
    for trader in top_traders:
        logging.info(f"Scanning recent trades for wallet: {trader[:8]}...")
        trades = get_recent_btc_trades(trader, limit=15)
        all_signals.extend(trades)
        time.sleep(0.25)

    if all_signals:
        df = pd.DataFrame(all_signals)
        df = df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
        logging.info(f"Extracted {len(df)} relevant BTC trade signals.")
        return df
    else:
        logging.info("No relevant BTC trade signals found in this cycle.")
        return pd.DataFrame()


if __name__ == "__main__":
    signals_df = run_scraper_cycle()
    if not signals_df.empty:
        print("\n--- Latest Alpha Signals ---")
        print(signals_df[["trader_wallet", "side", "price", "size", "market_title"]].head())
