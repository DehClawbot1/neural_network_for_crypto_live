"""
Download BTC Historical OHLCV Data

Downloads historical BTC/USDT candle data from Binance public API
for training the BTC price forecast model.

Usage:
    python download_btc_dataset.py                          # default: 15m candles, 2 years
    python download_btc_dataset.py --interval 1h --days 365
    python download_btc_dataset.py --interval 5m --days 90
    python download_btc_dataset.py --build-dataset           # download + build labelled dataset

Supported intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # Binance max per request


def download_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    days: int = 730,
    output_dir: str = "data",
) -> Path:
    """
    Download historical klines from Binance and save as CSV.

    Returns path to the saved CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}_{interval}_{days}d.csv"

    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_rows = []
    current_start = start_time
    request_count = 0

    logger.info(
        "Downloading %s %s candles for last %d days from Binance...",
        symbol, interval, days,
    )

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": MAX_LIMIT,
        }

        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("Request failed (attempt will retry): %s", exc)
            time.sleep(2)
            continue

        if not data:
            break

        for row in data:
            all_rows.append({
                "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": pd.to_datetime(int(row[6]), unit="ms", utc=True),
                "quote_volume": float(row[7]),
                "trades": int(row[8]),
                "taker_buy_base": float(row[9]),
                "taker_buy_quote": float(row[10]),
            })

        # Move start to after last candle
        last_close_time = int(data[-1][6])
        if last_close_time <= current_start:
            break
        current_start = last_close_time + 1
        request_count += 1

        # Rate limiting: Binance allows 1200 req/min, be conservative
        if request_count % 10 == 0:
            logger.info("  ... downloaded %d candles so far", len(all_rows))
            time.sleep(0.5)
        else:
            time.sleep(0.1)

    if not all_rows:
        logger.error("No data downloaded!")
        return output_path

    df = pd.DataFrame(all_rows)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "Downloaded %d candles (%s to %s) → %s",
        len(df),
        df["timestamp"].iloc[0],
        df["timestamp"].iloc[-1],
        output_path,
    )
    return output_path


def build_labelled_dataset(csv_path: Path, logs_dir: str = "logs", enrich_derivatives: bool = False) -> Path:
    """Build a labelled training dataset from downloaded candles."""
    from btc_price_dataset import BTCPriceDatasetBuilder

    candle_df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    if enrich_derivatives:
        try:
            from btc_onchain_features import BTCDerivativesFeatures
            logger.info("Enriching candles with derivatives data (funding, OI, L/S ratio, taker volume)...")
            fetcher = BTCDerivativesFeatures()
            candle_df = fetcher.fetch_all_and_merge(candle_df, period="15m")
            logger.info("Derivatives enrichment complete: %d columns", len(candle_df.columns))
        except Exception as exc:
            logger.warning("Derivatives enrichment failed (continuing without): %s", exc)

    builder = BTCPriceDatasetBuilder(logs_dir=logs_dir)
    dataset = builder.build_from_candles(candle_df)
    if dataset.empty:
        logger.error("Failed to build dataset from %s", csv_path)
        return Path(logs_dir) / "btc_price_dataset.csv"

    builder.append_to_disk(dataset)
    logger.info("Built labelled dataset: %d rows, %d features", len(dataset), len(dataset.columns))
    return builder.dataset_path


def main():
    parser = argparse.ArgumentParser(description="Download BTC historical data for ML training")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--interval", default="15m", help="Candle interval: 1m,5m,15m,30m,1h,4h,1d (default: 15m)")
    parser.add_argument("--days", type=int, default=730, help="Days of history (default: 730)")
    parser.add_argument("--output-dir", default="data", help="Output directory (default: data)")
    parser.add_argument("--build-dataset", action="store_true", help="Also build labelled dataset for training")
    parser.add_argument("--train", action="store_true", help="Download + build dataset + train model")
    parser.add_argument("--enrich", action="store_true", help="Enrich with derivatives data (funding rate, OI, L/S ratio)")
    args = parser.parse_args()

    csv_path = download_binance_klines(
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        output_dir=args.output_dir,
    )

    if args.build_dataset or args.train:
        dataset_path = build_labelled_dataset(csv_path, enrich_derivatives=args.enrich)
        logger.info("Labelled dataset at: %s", dataset_path)

    if args.train:
        from btc_forecast_model import BTCForecastModel
        from btc_price_dataset import BTCPriceDatasetBuilder

        builder = BTCPriceDatasetBuilder()
        df = builder.load_dataset()
        if df.empty:
            logger.error("No dataset to train on")
            sys.exit(1)

        model = BTCForecastModel()
        metrics = model.train(df)
        logger.info("Training complete: %s", metrics)


if __name__ == "__main__":
    main()
