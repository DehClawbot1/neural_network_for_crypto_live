from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


class TargetBuilder:
    """
    Fetch BTC/USD history from a public source and prepare forward-return targets.
    Research/paper-trading only.
    """

    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.logs_dir / "btc_targets.csv"

    def fetch_btc_history(self, days=30):
        response = requests.get(self.COINGECKO_URL, params={"vs_currency": "usd", "days": days}, timeout=30)
        response.raise_for_status()
        data = response.json()
        prices = data.get("prices", [])

        rows = []
        for timestamp_ms, price in prices:
            ts = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            rows.append({"timestamp": ts.isoformat(), "btc_price": float(price)})

        return pd.DataFrame(rows)

    def build_targets(self, days=30, horizon_minutes=60):
        df = self.fetch_btc_history(days=days)
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        horizon_steps = max(1, horizon_minutes // 5)
        df["future_price"] = df["btc_price"].shift(-horizon_steps)
        df["future_return"] = (df["future_price"] - df["btc_price"]) / df["btc_price"]
        df["target_up"] = (df["future_return"] > 0).astype(int)

        df["btc_spot_return_5m"] = df["btc_price"].pct_change(1)
        df["btc_spot_return_15m"] = df["btc_price"].pct_change(3)
        df["btc_realized_vol_15m"] = df["btc_spot_return_5m"].rolling(3).std()
        df["btc_volume_proxy"] = df["btc_spot_return_5m"].abs().rolling(6).sum()
        return df.dropna().reset_index(drop=True)

    def write(self, days=30, horizon_minutes=60):
        df = self.build_targets(days=days, horizon_minutes=horizon_minutes)
        if not df.empty:
            df.to_csv(self.output_file, index=False)
        return df
