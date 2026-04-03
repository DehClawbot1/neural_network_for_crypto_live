from datetime import datetime, timedelta, timezone

import asyncio
import json
import logging
import threading

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MarketPriceService:
    """FIX: Added tutorial-compatible order book analysis:
        book = client.get_order_book(yes_token_id)
        sorted_bids = sorted(book.bids, key=lambda x: float(x.price), reverse=True)
        sorted_asks = sorted(book.asks, key=lambda x: float(x.price), reverse=False)
    """

    CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
    CLOB_PRICE_URL = "https://clob.polymarket.com/price"
    CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, max_age_seconds=20):
        self.max_age_seconds = max_age_seconds
        self.cache = {}
        self._cache_lock = threading.Lock()
        self._clob_client = None

    def _get_clob_client(self):
        """Lazy-initialize a read-only ClobClient for order book queries."""
        if self._clob_client is None:
            try:
                from py_clob_client.client import ClobClient
                self._clob_client = ClobClient("https://clob.polymarket.com")
            except (ImportError, ModuleNotFoundError):
                self._clob_client = None
            except Exception as exc:
                logging.warning("ClobClient init failed: %s", exc)
                self._clob_client = None
        return self._clob_client

    def _cache_get(self, token_id):
        with self._cache_lock:
            return self.cache.get(str(token_id))

    def _cache_set(self, token_id, value):
        with self._cache_lock:
            self.cache[str(token_id)] = value

    def _is_fresh(self, token_id):
        record = self._cache_get(token_id)
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
                "fidelity": 10,
            },
            timeout=20,
        )
        response.raise_for_status()
        history = response.json().get("history", [])
        if not history:
            return None
        return float(history[-1].get("p", 0.0))

    def get_order_book_analysis(self, token_id, depth=5):
        """FIX: Tutorial-style order book analysis:
            book = client.get_order_book(yes_token_id)
            sorted_bids = sorted(book.bids, key=lambda x: float(x.price), reverse=True)
            sorted_asks = sorted(book.asks, key=lambda x: float(x.price), reverse=False)
        """
        client = self._get_clob_client()
        if client is None:
            return None

        try:
            book = client.get_order_book(str(token_id))
        except Exception as exc:
            logging.warning("Order book fetch failed for %s: %s", token_id, exc)
            return None

        bids = getattr(book, "bids", []) or []
        asks = getattr(book, "asks", []) or []

        sorted_bids = sorted(bids, key=lambda x: float(getattr(x, "price", 0)), reverse=True)
        sorted_asks = sorted(asks, key=lambda x: float(getattr(x, "price", 0)), reverse=False)

        top_bids = [{"price": float(b.price), "size": float(b.size)} for b in sorted_bids[:depth]]
        top_asks = [{"price": float(a.price), "size": float(a.size)} for a in sorted_asks[:depth]]

        best_bid = float(sorted_bids[0].price) if sorted_bids else None
        best_ask = float(sorted_asks[0].price) if sorted_asks else None
        midpoint = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
        spread = abs(best_ask - best_bid) if best_bid is not None and best_ask is not None else None

        bid_volume = sum(float(b.size) for b in sorted_bids[:depth])
        ask_volume = sum(float(a.size) for a in sorted_asks[:depth])
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.0

        result = {
            "token_id": str(token_id),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "midpoint": midpoint,
            "spread": spread,
            "bid_depth": len(sorted_bids),
            "ask_depth": len(sorted_asks),
            "top_bids": top_bids,
            "top_asks": top_asks,
            "bid_volume_top5": bid_volume,
            "ask_volume_top5": ask_volume,
            "order_book_imbalance": imbalance,
        }

        # Update cache
        self._cache_set(token_id, {
            **result,
            "price": midpoint or best_bid or best_ask,
            "timestamp": datetime.now(timezone.utc),
        })

        return result

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
        """FIX: Uses tutorial-compatible midpoint calculation via order book."""
        if not token_id:
            return None
        token_id = str(token_id)
        cached = self._cache_get(token_id)
        if cached and self._is_fresh(token_id) and "midpoint" in cached:
            return cached["midpoint"]

        # Try order book analysis first (tutorial pattern)
        analysis = self.get_order_book_analysis(token_id)
        if analysis and analysis.get("midpoint") is not None:
            return analysis["midpoint"]

        # Fall back to REST price endpoints
        try:
            buy_payload = self._rest_price(token_id, side="BUY")
            sell_payload = self._rest_price(token_id, side="SELL")
            best_ask = float(buy_payload.get("price") or buy_payload.get("best_ask")) if buy_payload else None
            best_bid = float(sell_payload.get("price") or sell_payload.get("best_bid")) if sell_payload else None
            if best_bid is not None and best_ask is not None:
                midpoint = (best_bid + best_ask) / 2.0
                spread = abs(best_ask - best_bid)
                self._cache_set(token_id, {
                    "price": midpoint,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "midpoint": midpoint,
                    "spread": spread,
                    "timestamp": datetime.now(timezone.utc),
                })
                return midpoint
        except Exception:
            pass
        return self._history_last_price(token_id)

    def get_spread(self, token_id):
        self.get_midpoint(token_id)
        record = self._cache_get(token_id) or {}
        return record.get("spread")

    def get_latest_price(self, token_id, interval="1m"):
        if not token_id:
            return None
        token_id = str(token_id)
        cached = self._cache_get(token_id)
        if cached and self._is_fresh(token_id):
            return cached.get("midpoint") or cached.get("price")
        midpoint = self.get_midpoint(token_id)
        if midpoint is not None:
            return midpoint
        price = self._history_last_price(token_id, interval=interval)
        if price is not None:
            self._cache_set(token_id, {"price": price, "midpoint": price, "timestamp": datetime.now(timezone.utc)})
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
        self._cache_set(token_id, {**quote, "price": midpoint or last_trade_price, "timestamp": datetime.now(timezone.utc)})
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
        except (ImportError, ModuleNotFoundError):
            logging.warning("websockets package not available — streaming disabled")
            return

        max_retries = 10
        base_delay = 2
        for attempt in range(max_retries):
            try:
                async with websockets.connect(
                    self.CLOB_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({
                        "assets_ids": [str(t) for t in token_ids if t],
                        "type": "market",
                    }))
                    logging.info("WebSocket connected (attempt %d)", attempt + 1)

                    while True:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        except asyncio.TimeoutError:
                            logging.warning("WebSocket recv timeout (60s), reconnecting...")
                            break
                        msg = json.loads(raw)
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
                        entry = {
                            "price": float(midpoint or last_trade_price or 0.0),
                            "best_bid": float(best_bid) if best_bid is not None else None,
                            "best_ask": float(best_ask) if best_ask is not None else None,
                            "midpoint": float(midpoint) if midpoint is not None else None,
                            "spread": spread,
                            "last_trade_price": float(last_trade_price) if last_trade_price is not None else None,
                            "timestamp": datetime.now(timezone.utc),
                        }
                        self._cache_set(token_id, entry)
                        if update_callback is not None:
                            update_callback(token_id, entry)
            except Exception as exc:
                delay = min(base_delay * (2 ** attempt), 60)
                logging.warning("WebSocket error (attempt %d/%d): %s — retrying in %ds",
                                attempt + 1, max_retries, exc, delay)
                await asyncio.sleep(delay)
        logging.error("WebSocket gave up after %d attempts", max_retries)

    def stream_prices_forever(self, token_ids, update_callback=None):
        asyncio.run(self.stream_prices(token_ids, update_callback=update_callback))
