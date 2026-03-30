from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from db import Database


class ReconciliationService:
    def __init__(self, execution_client=None, logs_dir="logs"):
        self.execution_client = execution_client
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.logs_dir / "trading.db")

    def _extract_items(self, payload):
        if payload is None:
            return []
        if isinstance(payload, str): return [] # BUG FIX 9
        if isinstance(payload, str): return [] # BUG FIX 9
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
            "condition_id": trade.get("condition_id") or trade.get("market") or trade.get("market_id"),
            "outcome_side": trade.get("outcome_side") or trade.get("side_label") or trade.get("outcome"),
            "price": float(trade.get("price") or trade.get("rate") or 0.0),
            "size": float(trade.get("size") or trade.get("amount") or trade.get("matched_amount") or 0.0),
            "filled_at": trade.get("filled_at") or trade.get("created_at") or trade.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "side": str(trade.get("side") or trade.get("taker_side") or trade.get("trade_side") or "").upper() or None,
        }

    def _safe_read_csv(self, filename):
        path = self.logs_dir / filename
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str) # BUG FIX 8
        except Exception:
            return pd.DataFrame()

    def sync_orders_and_fills(self):
        """Sync exchange orders and fills into the local SQLite database."""
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
        
        # BUG FIX 2: Canceled Order Sweep
        try:
            if orders_payload and isinstance(orders_payload, (dict, list)) and not ("error" in str(orders_payload).lower()):
                remote_open_ids = [str(self._normalize_order(o)["order_id"]) for o in self._extract_items(orders_payload) if self._normalize_order(o)]
                if remote_open_ids:
                    placeholders = ",".join("?" for _ in remote_open_ids)
                    self.db.execute(f"UPDATE orders SET status = 'CANCELED' WHERE status = 'OPEN' AND order_id NOT IN ({placeholders})", tuple(remote_open_ids))
                else:
                    self.db.execute("UPDATE orders SET status = 'CANCELED' WHERE status = 'OPEN'")
                if hasattr(self.db.conn, "commit"): self.db.conn.commit()
        except Exception:
            pass

        try:
            trades_payload = self.execution_client.get_trades()
            for raw_trade in self._extract_items(trades_payload):
                trade = self._normalize_trade(raw_trade)
                if trade is None:
                    continue
                self.db.execute(
                    "INSERT OR REPLACE INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        trade["fill_id"],
                        trade["order_id"],
                        trade["token_id"],
                        trade.get("condition_id"),
                        trade.get("outcome_side"),
                        trade.get("side"),
                        trade["price"],
                        trade["size"],
                        trade["filled_at"],
                    ),
                )
                # BUG FIX 3: Removed blind FILLED overwrite. Open order sweep will handle terminal status naturally.
                synced_fills += 1
        except Exception:
            pass

        return {"orders": synced_orders, "fills": synced_fills}

    def reconcile(self):
        """
        BUG FIX: This method was missing but expected by tests and the
        dashboard reconciliation panel.  It compares local CSV state with
        the exchange's current view and reports mismatches.

        Returns (report_dict, remote_orders_df, remote_trades_df).
        """
        try:
            local_orders = pd.read_sql_query("SELECT * FROM orders", self.db.conn)
            local_fills = pd.read_sql_query("SELECT * FROM fills", self.db.conn)
        except Exception:
            local_orders = pd.DataFrame()
            local_fills = pd.DataFrame() # BUG FIX 4: Use DB instead of split-brain CSVs

        # Fetch remote state
        remote_orders_raw = []
        remote_trades_raw = []
        try:
            remote_orders_raw = self._extract_items(self.execution_client.get_open_orders())
        except Exception:
            pass
        try:
            remote_trades_raw = self._extract_items(self.execution_client.get_trades())
        except Exception:
            pass

        remote_orders = [self._normalize_order(o) for o in remote_orders_raw]
        remote_orders = [o for o in remote_orders if o is not None]
        remote_trades = [self._normalize_trade(t) for t in remote_trades_raw]
        remote_trades = [t for t in remote_trades if t is not None]

        remote_orders_df = pd.DataFrame(remote_orders)
        remote_trades_df = pd.DataFrame(remote_trades)

        remote_order_ids = set(o["order_id"] for o in remote_orders)
        remote_trade_ids = set(t["fill_id"] for t in remote_trades)

        # Local open orders (exclude already closed/filled/canceled)
        closed_statuses = {"FILLED", "CANCELED", "REJECTED", "FAILED", "CANCELED_ALL", "CANCELED_BATCH", "CANCELED_MARKET"}
        local_open_order_ids = set()
        if not local_orders.empty and "order_id" in local_orders.columns:
            for _, row in local_orders.iterrows():
                status = str(row.get("status", "")).upper()
                oid = str(row.get("order_id", ""))
                if oid and oid != "nan" and oid != "None" and status not in closed_statuses:
                    local_open_order_ids.add(oid)

        local_trade_ids = set()
        if not local_fills.empty:
            id_col = "trade_id" if "trade_id" in local_fills.columns else "fill_id" if "fill_id" in local_fills.columns else None
            if id_col:
                local_trade_ids = set(local_fills[id_col].dropna().astype(str).tolist())

        missing_remote_orders = sorted(local_open_order_ids - remote_order_ids)
        missing_local_orders = sorted(remote_order_ids - local_open_order_ids)
        missing_remote_trades = sorted(local_trade_ids - remote_trade_ids)
        missing_local_trades = sorted(remote_trade_ids - local_trade_ids)

        # Detect status/size mismatches for orders present on both sides
        order_mismatches = []
        if not local_orders.empty and "order_id" in local_orders.columns:
            remote_order_lookup = {o["order_id"]: o for o in remote_orders}
            for _, row in local_orders.iterrows():
                oid = str(row.get("order_id", ""))
                if oid not in remote_order_lookup:
                    continue
                local_status = str(row.get("status", "")).upper()
                if local_status in closed_statuses:
                    continue
                remote = remote_order_lookup[oid]
                remote_status = str(remote.get("status", "")).upper()
                local_size = float(row.get("size", 0) or 0)
                remote_size = float(remote.get("size", 0) or 0)
                if local_status != remote_status or abs(local_size - remote_size) > 0.001:
                    order_mismatches.append({
                        "order_id": oid,
                        "local_status": local_status,
                        "remote_status": remote_status,
                        "local_size": local_size,
                        "remote_size": remote_size,
                    })

        report = {
            "local_order_rows": len(local_orders),
            "local_fill_rows": len(local_fills),
            "remote_open_orders": len(remote_orders),
            "remote_trades": len(remote_trades),
            "missing_remote_orders": missing_remote_orders,
            "missing_local_orders": missing_local_orders,
            "missing_remote_trades": missing_remote_trades,
            "missing_local_trades": missing_local_trades,
            "order_mismatches": order_mismatches,
        }

        return report, remote_orders_df, remote_trades_df
