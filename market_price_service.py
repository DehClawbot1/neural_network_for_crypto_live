from datetime import datetime, timedelta, timezone

import requests


class MarketPriceService:
    CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"

    def get_latest_price(self, token_id, interval="1m"):
        if not token_id:
            return None
        end_ts = int(datetime.now(timezone.utc).timestamp())
        start_ts = int((datetime.now(timezone.utc) - timedelta(hours=6)).timestamp())
        response = requests.get(
            self.CLOB_HISTORY_URL,
            params={
                "market": token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "interval": interval,
                "fidelity": 1,
            },
            timeout=20,
        )
        response.raise_for_status()
        history = response.json().get("history", [])
        if not history:
            return None
        return float(history[-1].get("p", 0.0))

    def get_latest_prices(self, token_ids):
        prices = {}
        for token_id in token_ids:
            try:
                prices[str(token_id)] = self.get_latest_price(token_id)
            except Exception:
                prices[str(token_id)] = None
        return prices
