from datetime import datetime
from pathlib import Path

import pandas as pd

from execution_client import ExecutionClient
from live_risk_manager import LiveRiskManager


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

    def _append(self, path: Path, row: dict):
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

    def check_readiness(self, asset_type=None):
        try:
            return self.client.get_balance_allowance(asset_type=asset_type)
        except Exception:
            return None

    def submit_entry(self, token_id, price, size, side="BUY", condition_id=None, outcome_side=None, spread=None, open_orders=0, daily_pnl=0.0):
        decision = self.risk.pre_trade_check(price=price, size=size, spread=spread, open_orders=open_orders, daily_pnl=daily_pnl)
        if not decision.allowed:
            return {"status": "REJECTED", "reason": decision.reason}, None

        readiness = self.check_readiness(asset_type="COLLATERAL")
        response = self.client.create_and_post_order(token_id=token_id, price=price, size=size, side=side)
        order_id = response.get("orderID") or response.get("order_id") or response.get("id")
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "order_id": order_id,
            "token_id": token_id,
            "condition_id": condition_id,
            "outcome_side": outcome_side,
            "order_side": side,
            "price": price,
            "size": size,
            "status": response.get("status", "SUBMITTED"),
            "readiness": readiness,
        }
        self._append(self.orders_file, row)
        return row, response

    def get_order_status(self, order_id):
        response = self.client.get_order(order_id)
        return response

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

    def record_fill(self, fill_payload: dict):
        self._append(self.fills_file, {"timestamp": datetime.utcnow().isoformat(), **fill_payload})
        return fill_payload
