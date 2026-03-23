from datetime import datetime
from pathlib import Path
import time

import pandas as pd

from execution_client import ExecutionClient
from live_risk_manager import LiveRiskManager
from db import Database


class OrderManager:
    """
    Live-test order manager.
    Tracks submitted orders and reconciles their local status over time.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.orders_file = self.logs_dir / "live_orders.csv"
        self.fills_file = self.logs_dir / "live_fills.csv"
        self.client = ExecutionClient()
        self.risk = LiveRiskManager()
        self.db = Database(self.logs_dir / "trading.db")

    def _append(self, path: Path, row: dict):
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

    def check_readiness(self, asset_type=None):
        try:
            return self.client.get_balance_allowance(asset_type=asset_type)
        except Exception:
            return None

    def submit_entry(self, token_id, price, size, side="BUY", condition_id=None, outcome_side=None, spread=None, open_orders=0, daily_pnl=0.0, order_type="GTC", post_only=False, execution_style="maker"):
        decision = self.risk.pre_trade_check(price=price, size=size, spread=spread, open_orders=open_orders, daily_pnl=daily_pnl)
        idempotency_key = f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M')}|{token_id}|{condition_id}|{side}|{size}|{round(float(price), 4)}"
        existing = self.list_orders()
        if not existing.empty and "idempotency_key" in existing.columns and (existing["idempotency_key"].astype(str) == idempotency_key).any():
            return {"status": "REJECTED", "reason": "duplicate_idempotency_key", "idempotency_key": idempotency_key}, None
        if not decision.allowed:
            row = {"timestamp": datetime.utcnow().isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": decision.reason}
            self._append(self.orders_file, row)
            return row, None

        readiness = self.check_readiness(asset_type="COLLATERAL")
        if not readiness:
            row = {"timestamp": datetime.utcnow().isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "missing_readiness"}
            self._append(self.orders_file, row)
            return row, None

        try:
            response = self.client.create_and_post_order(token_id=token_id, price=price, size=size, side=side, order_type=order_type, options={"post_only": bool(post_only)})
        except Exception as exc:
            self.risk.record_failed_order()
            row = {"timestamp": datetime.utcnow().isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "FAILED", "reason": str(exc)}
            self._append(self.orders_file, row)
            return row, None

        order_id = response.get("orderID") or response.get("order_id") or response.get("id")
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "order_id": order_id,
            "idempotency_key": idempotency_key,
            "token_id": token_id,
            "condition_id": condition_id,
            "outcome_side": outcome_side,
            "order_side": side,
            "price": price,
            "size": size,
            "order_type": order_type,
            "post_only": post_only,
            "execution_style": execution_style,
            "status": response.get("status", "SUBMITTED"),
            "readiness": readiness,
        }
        self._append(self.orders_file, row)
        self.db.execute(
            "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row.get("order_id"), row.get("token_id"), row.get("condition_id"), row.get("outcome_side"), row.get("order_side"), row.get("price"), row.get("size"), row.get("status"), row.get("timestamp")),
        )
        return row, response

    def get_order_status(self, order_id):
        response = self.client.get_order(order_id)
        return response

    def wait_for_fill(self, order_id, timeout_seconds=20, poll_seconds=2):
        deadline = time.time() + float(timeout_seconds)
        last_response = None
        while time.time() < deadline:
            try:
                last_response = self.get_order_status(order_id)
            except Exception:
                last_response = None
            status = str((last_response or {}).get("status", "")).upper()
            if status in ["FILLED", "EXECUTED", "MATCHED"]:
                return {"filled": True, "response": last_response}
            if status in ["CANCELED", "FAILED", "REJECTED"]:
                return {"filled": False, "response": last_response}
            time.sleep(float(poll_seconds))
        return {"filled": False, "response": last_response, "reason": "timeout_waiting_for_fill"}

    def submit_quote_order(self, token_id, price, size, side="BUY", condition_id=None, outcome_side=None):
        return self.submit_entry(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            condition_id=condition_id,
            outcome_side=outcome_side,
            order_type="GTC",
            post_only=True,
            execution_style="maker",
        )

    def submit_taker_order(self, token_id, price, size, side="BUY", condition_id=None, outcome_side=None):
        return self.submit_entry(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            condition_id=condition_id,
            outcome_side=outcome_side,
            order_type="GTC",
            post_only=False,
            execution_style="taker",
        )

    def place_target_exit_order(self, token_id, target_price, size, condition_id=None, outcome_side=None):
        row, response = self.submit_entry(
            token_id=token_id,
            price=target_price,
            size=size,
            side="SELL",
            condition_id=condition_id,
            outcome_side=outcome_side,
        )
        return row, response

    def monitor_and_trigger_exit(self, token_id, target_price, size, condition_id=None, outcome_side=None):
        quote = None
        try:
            from market_price_service import MarketPriceService
            quote = MarketPriceService().get_quote(token_id)
        except Exception:
            quote = None

        executable_sell = (quote or {}).get("best_bid")
        if executable_sell is not None and float(executable_sell) >= float(target_price):
            return self.submit_entry(
                token_id=token_id,
                price=executable_sell,
                size=size,
                side="SELL",
                condition_id=condition_id,
                outcome_side=outcome_side,
            )
        return {"status": "WAITING", "reason": "target_not_hit", "best_bid": executable_sell}, None

    def cancel_stale_order(self, order_id):
        response = self.client.cancel_order(order_id)
        self._append(self.orders_file, {"timestamp": datetime.utcnow().isoformat(), "order_id": order_id, "status": "CANCELED"})
        return response

    def list_orders(self):
        if not self.orders_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.orders_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def record_fill(self, fill_payload: dict):
        row = {"timestamp": datetime.utcnow().isoformat(), **fill_payload}
        self._append(self.fills_file, row)
        self.db.execute(
            "INSERT OR REPLACE INTO fills (fill_id, order_id, token_id, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?)",
            (row.get("trade_id") or row.get("fill_id"), row.get("order_id"), row.get("token_id"), row.get("price"), row.get("size"), row.get("timestamp")),
        )
        return fill_payload
