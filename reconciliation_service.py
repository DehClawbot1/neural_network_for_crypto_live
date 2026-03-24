from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from db import Database


class ReconciliationService:
    def __init__(self, execution_client, logs_dir="logs"):
        self.execution_client = execution_client
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.logs_dir / "trading.db")

    def _extract_items(self, payload):
        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ["data", "items", "orders", "trades", "results"]:
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
        return []

    def _normalize_order(self, order):
        if not isinstance(order, dict):
            return None
        order_id = order.get("id") or order.get("orderID") or order.get("order_id")
        token_id = order.get("token_id") or order.get("asset_id") or order.get("asset")
        if not order_id or not token_id:
            return None
        return {
            "order_id": str(order_id),
            "token_id": str(token_id),
            "condition_id": order.get("condition_id") or order.get("market") or order.get("market_id"),
            "outcome_side": order.get("outcome_side") or order.get("side_label") or order.get("outcome"),
            "order_side": str(order.get("side") or order.get("order_side") or "BUY").upper(),
            "price": float(order.get("price") or order.get("limit_price") or 0.0),
            "size": float(order.get("size") or order.get("original_size") or order.get("amount") or 0.0),
            "status": str(order.get("status") or "OPEN").upper(),
            "created_at": order.get("created_at") or order.get("createdAt") or datetime.now(timezone.utc).isoformat(),
        }

    def _normalize_trade(self, trade):
        if not isinstance(trade, dict):
            return None
        fill_id = trade.get("id") or trade.get("tradeID") or trade.get("trade_id") or trade.get("fill_id")
        order_id = trade.get("orderID") or trade.get("order_id") or trade.get("maker_order_id") or trade.get("taker_order_id")
        token_id = trade.get("token_id") or trade.get("asset_id") or trade.get("asset")
        if not fill_id or not token_id:
            return None
        return {
            "fill_id": str(fill_id),
            "order_id": str(order_id) if order_id is not None else None,
            "token_id": str(token_id),
            "price": float(trade.get("price") or trade.get("rate") or 0.0),
            "size": float(trade.get("size") or trade.get("amount") or trade.get("matched_amount") or 0.0),
            "filled_at": trade.get("filled_at") or trade.get("created_at") or trade.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "side": str(trade.get("side") or trade.get("taker_side") or "").upper() or None,
        }

    def sync_orders_and_fills(self):
        synced_orders = 0
        synced_fills = 0

        try:
            orders_payload = self.execution_client.get_open_orders()
            for raw_order in self._extract_items(orders_payload):
                order = self._normalize_order(raw_order)
                if order is None:
                    continue
                self.db.execute(
                    "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        order["order_id"],
                        order["token_id"],
                        order["condition_id"],
                        order["outcome_side"],
                        order["order_side"],
                        order["price"],
                        order["size"],
                        order["status"],
                        order["created_at"],
                    ),
                )
                synced_orders += 1
        except Exception:
            pass

        try:
            trades_payload = self.execution_client.get_trades()
            for raw_trade in self._extract_items(trades_payload):
                trade = self._normalize_trade(raw_trade)
                if trade is None:
                    continue
                self.db.execute(
                    "INSERT OR REPLACE INTO fills (fill_id, order_id, token_id, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        trade["fill_id"],
                        trade["order_id"],
                        trade["token_id"],
                        trade["price"],
                        trade["size"],
                        trade["filled_at"],
                    ),
                )
                if trade["order_id"]:
                    self.db.execute("UPDATE orders SET status = ? WHERE order_id = ?", ("FILLED", trade["order_id"]))
                synced_fills += 1
        except Exception:
            pass

        return {"orders": synced_orders, "fills": synced_fills}
