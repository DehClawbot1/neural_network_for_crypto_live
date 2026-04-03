import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from csv_utils import safe_csv_append

CLOB_URL = "https://clob.polymarket.com/prices-history"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CLOBHistoryClient:
    """Public CLOB history fetcher for paper/research labels and replay only."""

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.logs_dir / "clob_price_history.csv"

    def fetch_history(self, token_id, days=7, interval="1m", fidelity=10):
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
            timeout=15,  # BUG FIX: reduced from 30 to 15s per token
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
        total = len(token_ids)
        for idx, token_id in enumerate(token_ids):
            normalized_token = str(token_id or "").strip().strip('"').strip("'")
            if not normalized_token or not re.fullmatch(r"\d{8,}", normalized_token):
                logging.warning("Skipping invalid token_id for CLOB fetch: %r", token_id)
                continue
            try:
                logging.info("CLOB fetch %d/%d: %s...", idx + 1, total, normalized_token[:16])
                frames.append(self.fetch_history(normalized_token, days=days, interval=interval))
            except Exception as exc:
                # BUG FIX: Log and skip instead of silently continuing
                logging.warning("CLOB fetch failed for token %s: %s (skipping)", normalized_token[:16], exc)
                continue
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df.empty:
            safe_csv_append(self.output_file, df)
        logging.info("CLOB history: fetched %d rows across %d/%d tokens.", len(df), len(frames), total)
        return df
