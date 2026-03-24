from datetime import datetime, timezone
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
        self.db = Database(self.logs_dir / "trading.db")
        self.risk = LiveRiskManager(db=self.db)

    def _append(self, path: Path, row: dict):
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

    def check_readiness(self, asset_type=None, token_id=None):
        try:
            if hasattr(self.client, "update_balance_allowance"):
                self.client.update_balance_allowance(asset_type=asset_type, token_id=token_id)
        except Exception:
            pass
        try:
            return self.client.get_balance_allowance(asset_type=asset_type, token_id=token_id)
        except Exception:
            return None

    def _extract_orderbook_levels(self, orderbook):
        if isinstance(orderbook, dict):
            bids = orderbook.get("bids") or []
            asks = orderbook.get("asks") or []
            return bids, asks
        bids = getattr(orderbook, "bids", None) or []
        asks = getattr(orderbook, "asks", None) or []
        return bids, asks

    def _get_market_context(self, token_id, side):
        context = {}

        try:
            orderbook = self.client.get_orderbook(token_id)
            bids, asks = self._extract_orderbook_levels(orderbook)
            context["orderbook_ok"] = True
            context["bid_levels"] = len(bids)
            context["ask_levels"] = len(asks)
            context["tradable"] = bool(bids or asks)
        except Exception as exc:
            message = str(exc)
            context["orderbook_ok"] = False
            context["tradable"] = False
            context["orderbook_error"] = message
            if "orderbook" in message.lower() and "does not exist" in message.lower():
                context["reason"] = "orderbook_not_found"
            return context

        try:
            context["quoted_price"] = self.client.get_price(token_id, side=side)
        except Exception as exc:
            context["price_error"] = str(exc)

        try:
            context["quoted_spread"] = self.client.get_spread(token_id)
        except Exception as exc:
            context["spread_error"] = str(exc)

        return context

    def submit_entry(self, token_id, price, size, side="BUY", condition_id=None, outcome_side=None, spread=None, open_orders=0, daily_pnl=0.0, order_type="GTC", post_only=False, execution_style="maker"):
        normalized_side = str(side).upper()
        try:
            price = float(price)
        except Exception:
            price = None
        try:
            requested_size = float(size)
        except Exception:
            requested_size = None

        if price is None or price <= 0.0 or price >= 1.0:
            row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "invalid_price"}
            self._append(self.orders_file, row)
            return row, None
        if requested_size is None or requested_size <= 0.0:
            row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "invalid_size"}
            self._append(self.orders_file, row)
            return row, None

        notional_usdc = requested_size if normalized_side == "BUY" else requested_size * float(price)
        order_size_shares = requested_size / max(float(price), 1e-9) if normalized_side == "BUY" else requested_size
        decision = self.risk.pre_trade_check(token_id=token_id, price=price, size=notional_usdc, spread=spread, open_orders=open_orders, daily_pnl=daily_pnl)
        idempotency_key = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M')}|{token_id}|{condition_id}|{side}|{size}|{round(float(price), 4)}"
        existing = self.list_orders()
        if not existing.empty and "idempotency_key" in existing.columns and (existing["idempotency_key"].astype(str) == idempotency_key).any():
            return {"status": "REJECTED", "reason": "duplicate_idempotency_key", "idempotency_key": idempotency_key}, None
        if not decision.allowed:
            row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": decision.reason}
            self._append(self.orders_file, row)
            return row, None

        readiness = self.check_readiness(asset_type="COLLATERAL") if normalized_side == "BUY" else self.check_readiness(asset_type="CONDITIONAL", token_id=token_id)
        available_balance = float(readiness.get("balance", readiness.get("amount", 0.0))) if isinstance(readiness, dict) else None
        if normalized_side == "BUY":
            if not readiness:
                row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "missing_readiness"}
                self._append(self.orders_file, row)
                return row, None
            if (available_balance or 0.0) < float(notional_usdc):
                row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "size_usdc": notional_usdc, "order_size_shares": order_size_shares, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "insufficient_funds", "available_balance": available_balance}
                self._append(self.orders_file, row)
                return row, None
        else:
            if not readiness:
                row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "size_usdc": notional_usdc, "order_size_shares": order_size_shares, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "missing_token_readiness"}
                self._append(self.orders_file, row)
                return row, None
            if (available_balance or 0.0) < float(order_size_shares):
                row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "size_usdc": notional_usdc, "order_size_shares": order_size_shares, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "insufficient_token_inventory", "available_token_balance": available_balance}
                self._append(self.orders_file, row)
                return row, None

        market_context = self._get_market_context(token_id=token_id, side=side)
        if not market_context.get("tradable"):
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": None,
                "idempotency_key": idempotency_key,
                "token_id": token_id,
                "condition_id": condition_id,
                "outcome_side": outcome_side,
                "order_side": side,
                "price": price,
                "size": size,
                "size_usdc": notional_usdc,
                "order_size_shares": order_size_shares,
                "order_type": order_type,
                "post_only": post_only,
                "execution_style": execution_style,
                "status": "REJECTED",
                "reason": market_context.get("reason", "non_tradable_orderbook"),
                "readiness": readiness,
                **market_context,
            }
            self._append(self.orders_file, row)
            try:
                self.db.execute(
                    "INSERT INTO risk_events (token_id, event_type, detail) VALUES (?, ?, ?)",
                    (str(token_id), row["reason"], str(market_context.get("orderbook_error") or "non-tradable orderbook")),
                )
            except Exception:
                pass
            return row, None

        try:
            if str(order_type).upper() == "GTD" and hasattr(self.client, "GTD_order"):
                expiration = int((datetime.now(timezone.utc).timestamp()) + 3600)
                response = self.client.GTD_order(token_id=token_id, price=price, size=order_size_shares, expiration=expiration, side=side, post_only=bool(post_only))
            elif bool(post_only) and hasattr(self.client, "post_only_order"):
                response = self.client.post_only_order(token_id=token_id, price=price, size=order_size_shares, side=side, order_type=order_type)
            else:
                response = self.client.create_and_post_order(token_id=token_id, price=price, size=order_size_shares, side=side, order_type=order_type, options={"post_only": bool(post_only)})
        except Exception as exc:
            self.risk.record_failed_order()
            row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "size_usdc": notional_usdc, "order_size_shares": order_size_shares, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "FAILED", "reason": str(exc), "readiness": readiness, **market_context}
            self._append(self.orders_file, row)
            return row, None

        order_id = response.get("orderID") or response.get("order_id") or response.get("id")
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "idempotency_key": idempotency_key,
            "token_id": token_id,
            "condition_id": condition_id,
            "outcome_side": outcome_side,
            "order_side": side,
            "price": price,
            "size": size,
            "size_usdc": notional_usdc,
            "order_size_shares": order_size_shares,
            "order_type": order_type,
            "post_only": post_only,
            "execution_style": execution_style,
            "status": response.get("status", "SUBMITTED"),
            "readiness": readiness,
            **market_context,
        }
        self._append(self.orders_file, row)
        self.db.execute(
            "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row.get("order_id"), row.get("token_id"), row.get("condition_id"), row.get("outcome_side"), row.get("order_side"), row.get("price"), row.get("size"), row.get("status"), row.get("timestamp")),
        )
        return row, response

    def _update_order_status(self, order_id, status, fill_price=None, fill_size=None):
        order_id = str(order_id) if order_id is not None else None
        if not order_id:
            return
        if self.orders_file.exists():
            try:
                df = pd.read_csv(self.orders_file, engine="python", on_bad_lines="skip")
                if not df.empty and "order_id" in df.columns:
                    mask = df["order_id"].astype(str) == order_id
                    if mask.any():
                        df.loc[mask, "status"] = status
                        df.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()
                        if fill_price is not None:
                            df.loc[mask, "fill_price"] = fill_price
                        if fill_size is not None:
                            df.loc[mask, "fill_size"] = fill_size
                        df.to_csv(self.orders_file, index=False)
            except Exception:
                pass
        try:
            self.db.execute(
                "UPDATE orders SET status = ? WHERE order_id = ?",
                (status, order_id),
            )
        except Exception:
            pass

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
                fill_event_time = datetime.now(timezone.utc).isoformat()
                fill_payload = {
                    "trade_id": (last_response or {}).get("id") or f"{order_id}:{fill_event_time}",
                    "order_id": order_id,
                    "token_id": (last_response or {}).get("token_id", ""),
                    "price": float((last_response or {}).get("price", 0.0) or 0.0),
                    "size": float((last_response or {}).get("size", 0.0) or 0.0),
                    "filled_at": fill_event_time,
                }
                self._update_order_status(order_id, "FILLED", fill_price=fill_payload["price"], fill_size=fill_payload["size"])
                self.record_fill(fill_payload)
                return {"filled": True, "response": last_response}
            if status in ["CANCELED", "FAILED", "REJECTED"]:
                self._update_order_status(order_id, status)
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
        self._update_order_status(order_id, "CANCELED")
        self._append(self.orders_file, {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": order_id, "status": "CANCELED"})
        return response

    def cancel_all_orders(self):
        if not hasattr(self.client, "cancel_all"):
            raise AttributeError("Execution client does not expose cancel_all")
        response = self.client.cancel_all()
        self._append(self.orders_file, {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "status": "CANCELED_ALL"})
        return response

    def cancel_orders(self, order_ids):
        if not hasattr(self.client, "cancel_orders"):
            raise AttributeError("Execution client does not expose cancel_orders")
        order_ids = [str(order_id) for order_id in (order_ids or []) if str(order_id).strip()]
        response = self.client.cancel_orders(order_ids)
        for order_id in order_ids:
            self._update_order_status(order_id, "CANCELED")
            self._append(self.orders_file, {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": order_id, "status": "CANCELED_BATCH"})
        return response

    def cancel_market_orders(self, market="", asset_id=""):
        if not hasattr(self.client, "cancel_market_orders"):
            raise AttributeError("Execution client does not expose cancel_market_orders")
        response = self.client.cancel_market_orders(market=market, asset_id=asset_id)
        self._append(self.orders_file, {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "status": "CANCELED_MARKET", "market": market, "asset_id": asset_id})
        return response

    def submit_batch_orders(self, order_specs):
        if not hasattr(self.client, "orders"):
            raise AttributeError("Execution client does not expose batch orders")
        order_specs = list(order_specs or [])
        response = self.client.orders(order_specs)
        self._append(self.orders_file, {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "status": "BATCH_SUBMITTED", "count": len(order_specs)})
        return response

    def list_orders(self):
        if not self.orders_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.orders_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def record_fill(self, fill_payload: dict):
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **fill_payload}
        fill_id = row.get("trade_id") or row.get("fill_id") or f"{row.get('order_id', 'unknown')}:{row['timestamp']}"
        row["fill_id"] = fill_id
        self._append(self.fills_file, row)
        self.db.execute(
            "INSERT OR REPLACE INTO fills (fill_id, order_id, token_id, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?)",
            (fill_id, row.get("order_id"), row.get("token_id"), row.get("price"), row.get("size"), row.get("filled_at") or row.get("timestamp")),
        )
        return fill_payload
