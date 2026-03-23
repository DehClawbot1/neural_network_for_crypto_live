from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

CLOB_URL = "https://clob.polymarket.com/prices-history"


class CLOBHistoryClient:
    """Public CLOB history fetcher for paper/research labels and replay only."""

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.logs_dir / "clob_price_history.csv"

    def fetch_history(self, token_id, days=7, interval="1m", fidelity=1):
        end_ts = int(datetime.now(timezone.utc).timestamp())
        start_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        response = requests.get(
            CLOB_URL,
            params={
                "market": token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "interval": interval,
                "fidelity": fidelity,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        history = payload.get("history", [])
        rows = []
        for item in history:
            rows.append(
                {
                    "token_id": token_id,
                    "timestamp": datetime.fromtimestamp(int(item.get("t", 0)), tz=timezone.utc).isoformat(),
                    "price": float(item.get("p", 0.0)),
                }
            )
        return pd.DataFrame(rows)

    def append_history(self, token_ids, days=7, interval="1m"):
        frames = []
        for token_id in token_ids:
            if not token_id:
                continue
            try:
                frames.append(self.fetch_history(token_id, days=days, interval=interval))
            except Exception:
                continue
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df.empty:
            df.to_csv(self.output_file, mode="a", header=not self.output_file.exists(), index=False)
        return df

