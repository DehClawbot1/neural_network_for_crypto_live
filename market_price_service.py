from datetime import datetime, timedelta, timezone

import asyncio
import json

import requests


class MarketPriceService:
    CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
    CLOB_PRICE_URL = "https://clob.polymarket.com/price"
    CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, max_age_seconds=20):
        self.max_age_seconds = max_age_seconds
        self.cache = {}

    def _is_fresh(self, token_id):
        record = self.cache.get(str(token_id))
        if not record:
            return False
        age = (datetime.now(timezone.utc) - record["timestamp"]).total_seconds()
        return age <= self.max_age_seconds

    def _rest_price(self, token_id, side=None):
        params = {"token_id": str(token_id)}
        if side is not None:
            params["side"] = str(side).upper()
        response = requests.get(self.CLOB_PRICE_URL, params=params, timeout=10)
        if not response.ok:
            return None
        payload = response.json()
        return payload

    def _history_last_price(self, token_id, interval="1m"):
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

    def get_executable_price(self, token_id, side="SELL"):
        if not token_id:
            return None
        token_id = str(token_id)
        try:
            payload = self._rest_price(token_id, side=side)
            if payload:
                return float(payload.get("price") or payload.get("best_bid") or payload.get("best_ask"))
        except Exception:
            pass
        return self._history_last_price(token_id)

    def get_midpoint(self, token_id):
        if not token_id:
            return None
        token_id = str(token_id)
        if self._is_fresh(token_id) and "midpoint" in self.cache[token_id]:
            return self.cache[token_id]["midpoint"]
        try:
            buy_payload = self._rest_price(token_id, side="BUY")
            sell_payload = self._rest_price(token_id, side="SELL")
            best_ask = float(buy_payload.get("price") or buy_payload.get("best_ask")) if buy_payload else None
            best_bid = float(sell_payload.get("price") or sell_payload.get("best_bid")) if sell_payload else None
            if best_bid is not None and best_ask is not None:
                midpoint = (best_bid + best_ask) / 2.0
                spread = abs(best_ask - best_bid)
                self.cache[token_id] = {
                    "price": midpoint,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "midpoint": midpoint,
                    "spread": spread,
                    "timestamp": datetime.now(timezone.utc),
                }
                return midpoint
        except Exception:
            pass
        return self._history_last_price(token_id)

    def get_spread(self, token_id):
        self.get_midpoint(token_id)
        record = self.cache.get(str(token_id), {})
        return record.get("spread")

    def get_latest_price(self, token_id, interval="1m"):
        if not token_id:
            return None
        token_id = str(token_id)
        if self._is_fresh(token_id):
            return self.cache[token_id].get("midpoint") or self.cache[token_id].get("price")
        midpoint = self.get_midpoint(token_id)
        if midpoint is not None:
            return midpoint
        price = self._history_last_price(token_id, interval=interval)
        if price is not None:
            self.cache[token_id] = {"price": price, "midpoint": price, "timestamp": datetime.now(timezone.utc)}
        return price

    def get_quote(self, token_id):
        token_id = str(token_id)
        midpoint = self.get_midpoint(token_id)
        best_bid = self.get_executable_price(token_id, side="SELL")
        best_ask = self.get_executable_price(token_id, side="BUY")
        spread = abs(best_ask - best_bid) if best_bid is not None and best_ask is not None else self.get_spread(token_id)
        last_trade_price = self._history_last_price(token_id)
        quote = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "midpoint": midpoint,
            "spread": spread,
            "last_trade_price": last_trade_price,
        }
        self.cache[token_id] = {**quote, "price": midpoint or last_trade_price, "timestamp": datetime.now(timezone.utc)}
        return quote

    def get_batch_prices(self, token_ids):
        quotes = {}
        for token_id in token_ids:
            try:
                quotes[str(token_id)] = self.get_quote(token_id)
            except Exception:
                quotes[str(token_id)] = None
        return quotes

    def get_latest_prices(self, token_ids):
        quotes = self.get_batch_prices(token_ids)
        return {token_id: (quote or {}).get("price") for token_id, quote in quotes.items()}

    async def stream_prices(self, token_ids, update_callback=None):
        try:
            import websockets
        except Exception:
            return

        async with websockets.connect(self.CLOB_WS_URL) as ws:
            await ws.send(json.dumps({
                "assets_ids": [str(t) for t in token_ids if t],
                "type": "market",
            }))

            while True:
                msg = json.loads(await ws.recv())
                token_id = str(msg.get("asset_id") or msg.get("market") or "")
                if not token_id:
                    continue
                best_bid = msg.get("best_bid")
                best_ask = msg.get("best_ask")
                midpoint = msg.get("mid") or msg.get("midpoint")
                last_trade_price = msg.get("price")
                spread = None
                try:
                    if best_bid is not None and best_ask is not None:
                        spread = abs(float(best_ask) - float(best_bid))
                except Exception:
                    spread = None
                self.cache[token_id] = {
                    "price": float(midpoint or last_trade_price or 0.0),
                    "best_bid": float(best_bid) if best_bid is not None else None,
                    "best_ask": float(best_ask) if best_ask is not None else None,
                    "midpoint": float(midpoint) if midpoint is not None else None,
                    "spread": spread,
                    "last_trade_price": float(last_trade_price) if last_trade_price is not None else None,
                    "timestamp": datetime.now(timezone.utc),
                }
                if update_callback is not None:
                    update_callback(token_id, self.cache[token_id])

    def stream_prices_forever(self, token_ids, update_callback=None):
        asyncio.run(self.stream_prices(token_ids, update_callback=update_callback))

