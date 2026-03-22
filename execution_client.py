from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd


@dataclass
class PaperOrder:
    order_id: str
    token_id: str | None
    condition_id: str | None
    outcome_side: str | None
    order_side: str
    price: float
    size: float
    status: str
    created_at: str
    note: str | None = None


class ExecutionClient:
    """
    Paper-only execution client abstraction.
    Keeps the interface shape clean without enabling live trading.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.orders_file = self.logs_dir / "paper_orders.csv"
        self.trades_file = self.logs_dir / "paper_execution_trades.csv"

    def _append(self, path: Path, row: dict):
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

    def quote_order(self, token_id=None, condition_id=None, outcome_side=None, order_side="BUY", price=0.5, size=0.0):
        return {
            "token_id": token_id,
            "condition_id": condition_id,
            "outcome_side": outcome_side,
            "order_side": order_side,
            "price": float(price),
            "size": float(size),
            "quoted_at": datetime.utcnow().isoformat(),
            "mode": "paper_only",
        }

    def simulate_post_order(self, token_id=None, condition_id=None, outcome_side=None, order_side="BUY", price=0.5, size=0.0, note=None):
        order = PaperOrder(
            order_id=str(uuid4()),
            token_id=token_id,
            condition_id=condition_id,
            outcome_side=outcome_side,
            order_side=order_side,
            price=float(price),
            size=float(size),
            status="OPEN",
            created_at=datetime.utcnow().isoformat(),
            note=note,
        )
        self._append(self.orders_file, asdict(order))
        return asdict(order)

    def simulate_fill_order(self, order_id, fill_price=None):
        orders = self.list_open_paper_orders()
        if orders.empty or "order_id" not in orders.columns:
            return None
        match = orders[orders["order_id"].astype(str) == str(order_id)]
        if match.empty:
            return None
        order = match.iloc[0].to_dict()
        fill = {
            "trade_id": str(uuid4()),
            "order_id": order_id,
            "token_id": order.get("token_id"),
            "condition_id": order.get("condition_id"),
            "outcome_side": order.get("outcome_side"),
            "order_side": order.get("order_side"),
            "price": float(fill_price if fill_price is not None else order.get("price", 0.0)),
            "size": float(order.get("size", 0.0)),
            "filled_at": datetime.utcnow().isoformat(),
            "mode": "paper_only",
        }
        self._append(self.trades_file, fill)
        return fill

    def simulate_cancel_order(self, order_id):
        return {
            "order_id": str(order_id),
            "status": "CANCELED",
            "canceled_at": datetime.utcnow().isoformat(),
            "mode": "paper_only",
        }

    def get_order(self, order_id):
        orders = self.list_open_paper_orders()
        if orders.empty or "order_id" not in orders.columns:
            return None
        match = orders[orders["order_id"].astype(str) == str(order_id)]
        return match.iloc[0].to_dict() if not match.empty else None

    def list_open_paper_orders(self):
        if not self.orders_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.orders_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def list_paper_trades(self):
        if not self.trades_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.trades_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()
