import requests
import pandas as pd
import time
import logging

# Configure logging for zero-intervention monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_top_crypto_traders(limit=5):
    """Fetches the top proxy wallets by PnL in the CRYPTO category for the week."""
    url = "https://data-api.polymarket.com/v1/leaderboard"
    params = {
        "category": "CRYPTO",
        "timePeriod": "WEEK",
        "orderBy": "PNL",
        "limit": limit,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # PolyMarket proxy wallets actually execute the trades
        top_wallets = [user.get("proxyWallet") for user in data if user.get("proxyWallet")]
        logging.info(f"Successfully fetched {len(top_wallets)} top traders from the leaderboard.")
        return top_wallets
    except Exception as e:
        logging.error(f"Error fetching leaderboard: {e}")
        return []


def get_recent_btc_trades(wallet_address, limit=15):
    """Fetches recent trades for a specific wallet and filters for BTC markets."""
    url = "https://data-api.polymarket.com/activity"
    params = {
        "user": wallet_address,
        "limit": limit,
        "type": "TRADE",
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        trades = response.json()

        signals = []
        for trade in trades:
            # Safely extract title and normalize to lowercase for filtering
            title = trade.get("title", "").lower()

            # Filter strictly for Bitcoin markets
            if "bitcoin" in title or "btc" in title:
                signals.append(
                    {
                        "trader_wallet": wallet_address,
                        "market_title": trade.get("title"),
                        "condition_id": trade.get("conditionId"),
                        "side": trade.get("side"),  # e.g., BUY / SELL
                        "price": float(trade.get("price", 0)),
                        "size": float(trade.get("size", 0)),
                        "timestamp": trade.get("timestamp"),
                    }
                )
        return signals
    except Exception as e:
        logging.error(f"Error fetching activity for {wallet_address[:8]}...: {e}")
        return []


def run_scraper_cycle():
    """Main execution loop to pull the latest Alpha Signals."""
    logging.info("Starting Alpha Signal Scraper cycle...")
    top_traders = get_top_crypto_traders(limit=5)

    all_signals = []
    for trader in top_traders:
        logging.info(f"Scanning recent trades for wallet: {trader[:8]}...")
        trades = get_recent_btc_trades(trader, limit=15)
        all_signals.extend(trades)

        # Respect PolyMarket's Data API rate limits (1,000 req / 10s)
        time.sleep(0.5)

    if all_signals:
        df = pd.DataFrame(all_signals)
        # Sort by most recent trades
        df = df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
        logging.info(f"Extracted {len(df)} relevant BTC trade signals.")
        return df
    else:
        logging.info("No relevant BTC trade signals found in this cycle.")
        return pd.DataFrame()
