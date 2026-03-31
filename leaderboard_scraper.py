import time
import logging
import os
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from token_utils import normalize_token_id
from urllib3.util.retry import Retry

# Configure logging for zero-intervention monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _env_int(name, default, minimum=1, maximum=10000):
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = int(default)
    value = max(int(minimum), value)
    value = min(int(maximum), value)
    return value


def _env_float(name, default, minimum=0.0, maximum=60.0):
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        value = float(default)
    value = max(float(minimum), value)
    value = min(float(maximum), value)
    return value


def _today_start_iso():
    """Use a rolling UTC lookback window instead of a hard Europe/Lisbon day boundary."""
    lookback_hours = int(os.getenv("SIGNAL_LOOKBACK_HOURS", "24"))
    lookback_hours = max(1, lookback_hours)
    return pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_hours)


def _normalize_timestamp_value(value):
    if value in [None, ""]:
        return value
    try:
        numeric = float(value)
        if numeric > 1e17:
            return pd.to_datetime(numeric, utc=True, unit="ns").isoformat()
        if numeric > 1e14:
            return pd.to_datetime(numeric, utc=True, unit="us").isoformat()
        if numeric > 1e11:
            return pd.to_datetime(numeric, utc=True, unit="ms").isoformat()
        if numeric > 1e9:
            return pd.to_datetime(numeric, utc=True, unit="s").isoformat()
    except Exception:
        pass
    try:
        return pd.to_datetime(value, utc=True).isoformat()
    except Exception:
        return value


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


def _trade_to_signal(trade, market_universe=None, wallet_fallback=None):
    market_universe = market_universe or {"condition_ids": set(), "token_ids": set(), "slugs": set()}
    cond_id = trade.get("conditionId") or trade.get("condition_id")
    token_id = trade.get("tokenId") or trade.get("token_id") or trade.get("asset")
    slug = str(trade.get("slug", trade.get("marketSlug", "")) or "")
    title = str(trade.get("title", ""))
    title_l = title.lower()
    condition_id = str(cond_id or "")
    token_id_str = normalize_token_id(token_id) or ""
    mapped_to_btc = (
        condition_id in market_universe.get("condition_ids", set())
        or token_id_str in market_universe.get("token_ids", set())
        or slug in market_universe.get("slugs", set())
    )
    keyword_fallback = (
        "bitcoin" in title_l or "btc" in title_l or "bitcoin para cima ou para baixo" in title_l or "para cima ou para baixo" in title_l
    )
    if not mapped_to_btc and not keyword_fallback:
        return None

    normalized_ts = _normalize_timestamp_value(trade.get("timestamp"))
    trade_ts = pd.to_datetime(normalized_ts, utc=True, errors="coerce")
    if pd.isna(trade_ts) or trade_ts < _today_start_iso():
        return None

    order_side = str(trade.get("side", "BUY") or "BUY").upper()
    entry_intent = "OPEN_LONG" if order_side == "BUY" else "CLOSE_LONG"
    trader_wallet = (
        trade.get("proxyWallet")
        or trade.get("trader_wallet")
        or trade.get("wallet")
        or wallet_fallback
    )

    return {
        "trade_id": trade.get("id") or trade.get("transactionHash"),
        "tx_hash": trade.get("transactionHash", trade.get("txHash")),
        "trader_wallet": trader_wallet,
        "market_title": title,
        "market_slug": slug,
        "token_id": token_id_str or None,
        "condition_id": cond_id,
        "order_side": order_side,
        "trade_side": order_side,
        "outcome_side": trade.get("outcome"),
        "entry_intent": entry_intent,
        "side": trade.get("outcome"),
        "price": float(trade.get("price", 0)),
        "size": float(trade.get("size", 0)),
        "timestamp": normalized_ts,
    }


def load_btc_market_universe(logs_dir="logs"):
    markets_file = Path(logs_dir) / "markets.csv"
    if not markets_file.exists():
        return {"condition_ids": set(), "token_ids": set(), "slugs": set()}
    try:
        df = pd.read_csv(markets_file, engine="python", on_bad_lines="skip")
    except Exception:
        return {"condition_ids": set(), "token_ids": set(), "slugs": set()}

    condition_ids = set(df.get("condition_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
    token_ids = set()
    for col in ["yes_token_id", "no_token_id", "token_id"]:
        if col in df.columns:
            for value in df[col].dropna().tolist():
                token = normalize_token_id(value)
                if token:
                    token_ids.add(token)
    slugs = set(df.get("slug", pd.Series(dtype=str)).dropna().astype(str).tolist())
    return {"condition_ids": condition_ids, "token_ids": token_ids, "slugs": slugs}


def get_recent_btc_trades(wallet_address, limit=50, market_universe=None, session=None):
    """Fetch recent wallet trades from the public Data API and keep only trades that map into the BTC universe."""
    url = "https://data-api.polymarket.com/trades"
    params = {
        "user": wallet_address,
        "limit": limit,
    }

    try:
        session = session or _build_session()
        response = session.get(url, params=params, timeout=20)
        response.raise_for_status()
        trades = response.json()

        signals = []
        for trade in trades:
            signal = _trade_to_signal(trade, market_universe=market_universe, wallet_fallback=wallet_address)
            if signal is not None:
                signals.append(signal)
        return signals
    except Exception as e:
        logging.error(f"Error fetching trades for {wallet_address[:8]}...: {e}")
        return []


def get_recent_btc_global_trades(limit=300, page_size=100, market_universe=None, session=None):
    """
    Fetch global recent trades (not tied to leaderboard wallets) and keep only BTC universe trades.
    This expands coverage beyond top-wallet activity.
    """
    url = "https://data-api.polymarket.com/trades"
    session = session or _build_session()
    market_universe = market_universe or {"condition_ids": set(), "token_ids": set(), "slugs": set()}
    limit = max(1, int(limit))
    page_size = max(1, min(int(page_size), 500))
    lookback_cutoff = _today_start_iso()

    signals = []
    fetched = 0
    for offset in range(0, limit, page_size):
        page_limit = min(page_size, limit - offset)
        params = {"limit": page_limit, "offset": offset}
        try:
            response = session.get(url, params=params, timeout=20)
            response.raise_for_status()
            trades = response.json()
        except Exception as exc:
            logging.warning("Global trade fetch failed (offset=%s): %s", offset, exc)
            break
        if not isinstance(trades, list) or not trades:
            break

        oldest_seen = None
        for trade in trades:
            normalized_ts = _normalize_timestamp_value(trade.get("timestamp"))
            trade_ts = pd.to_datetime(normalized_ts, utc=True, errors="coerce")
            if pd.notna(trade_ts):
                oldest_seen = trade_ts if oldest_seen is None else min(oldest_seen, trade_ts)
            signal = _trade_to_signal(trade, market_universe=market_universe, wallet_fallback=trade.get("proxyWallet"))
            if signal is not None:
                signals.append(signal)
                fetched += 1
        if oldest_seen is not None and oldest_seen < lookback_cutoff:
            # Endpoint is ordered newest-first; once we pass lookback, stop paging.
            break
    logging.info("Global BTC trade scan captured %s candidate signals.", fetched)
    return signals


def run_scraper_cycle():
    """Main execution loop to pull the latest Alpha Signals."""
    logging.info("Starting Alpha Signal Scraper cycle...")
    leaderboard_limit = _env_int("LEADERBOARD_TOP_TRADERS_LIMIT", 100, minimum=20, maximum=1000)
    per_wallet_limit = _env_int("TRADES_PER_WALLET_LIMIT", 50, minimum=10, maximum=500)
    wallet_sleep = _env_float("SCRAPER_WALLET_SLEEP_SECONDS", 0.25, minimum=0.0, maximum=5.0)
    global_scan_enabled = os.getenv("ENABLE_GLOBAL_BTC_TRADE_SCAN", "true").strip().lower() in {"1", "true", "yes", "on"}
    global_limit = _env_int("GLOBAL_BTC_TRADE_LIMIT", 300, minimum=50, maximum=5000)
    global_page_size = _env_int("GLOBAL_BTC_TRADE_PAGE_SIZE", 100, minimum=20, maximum=500)

    top_traders = get_top_crypto_traders(limit=leaderboard_limit)

    market_universe = load_btc_market_universe()
    all_signals = []
    shared_session = _build_session()
    for trader in top_traders:
        logging.info(f"Scanning recent trades for wallet: {trader[:8]}...")
        trades = get_recent_btc_trades(trader, limit=per_wallet_limit, market_universe=market_universe, session=shared_session)
        all_signals.extend(trades)
        if wallet_sleep > 0:
            time.sleep(wallet_sleep)

    if global_scan_enabled:
        all_signals.extend(
            get_recent_btc_global_trades(
                limit=global_limit,
                page_size=global_page_size,
                market_universe=market_universe,
                session=shared_session,
            )
        )

    if all_signals:
        df = pd.DataFrame(all_signals)
        dedupe_cols = [
            c
            for c in [
                "trade_id",
                "tx_hash",
                "trader_wallet",
                "token_id",
                "condition_id",
                "order_side",
                "outcome_side",
                "price",
                "size",
                "timestamp",
            ]
            if c in df.columns
        ]
        if dedupe_cols:
            before = len(df)
            df = df.drop_duplicates(subset=dedupe_cols, keep="first")
            removed = before - len(df)
            if removed > 0:
                logging.info("Removed %s duplicate raw wallet trades in scraper cycle.", removed)
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

