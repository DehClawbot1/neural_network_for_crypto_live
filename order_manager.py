import os
import math
import logging
from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd

from execution_client import ExecutionClient
from live_risk_manager import LiveRiskManager
from db import Database

try:
    from polymarket_capabilities import apply_execution_client_patch
    apply_execution_client_patch()
except Exception:
    pass


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

    def _normalize_balance(self, raw_balance):
        if raw_balance is None:
            return 0.0
        try:
            val = float(raw_balance)
        except (TypeError, ValueError):
            return 0.0
<<<<<<< HEAD
        # If balance looks like raw microdollars (>= 1_000_000 and is integer-like),
        # convert to dollars. Otherwise assume it's already in dollars.
        # Microdollar values are always >= 1,000,000 ($1 = 1M microdollars)
        # A value of 5000.0 is $5000 in real dollars, NOT $0.005 in microdollars
=======
>>>>>>> d525b3abee41ed9164457f1fea845b586b6ae699
        try:
            from config import TradingConfig
            is_micro = getattr(TradingConfig, 'BALANCE_IS_MICRODOLLARS', True)
        except ImportError:
            is_micro = True
        if is_micro and val >= 1_000_000 and val == int(val):
            return val / 1e6
        return val

    def _round_down_shares(self, shares, decimals=6):
        try:
            shares = float(shares)
        except (TypeError, ValueError):
            return 0.0
        factor = 10 ** int(decimals)
        return math.floor(max(shares, 0.0) * factor) / factor

    def _get_available_balance(self, asset_type="COLLATERAL", token_id=None):
        readiness = self.check_readiness(asset_type=asset_type, token_id=token_id)
        raw_balance = None
        if isinstance(readiness, dict):
            for key in ["balance", "available", "available_balance", "amount"]:
                if readiness.get(key) is not None:
                    raw_balance = readiness[key]
                    break

        normalized_balance = self._normalize_balance(raw_balance)
        logging.info(
            "Balance check: raw=%s normalized=$%.2f (asset_type=%s)",
            raw_balance, normalized_balance, asset_type
        )

        onchain_balance = 0.0
        if str(asset_type).upper() == "COLLATERAL":
            try:
                onchain = self.client.get_onchain_collateral_balance()
                onchain_balance = float((onchain or {}).get("total", 0.0) or 0.0)
                logging.info("On-chain balance: $%.2f", onchain_balance)
            except Exception as exc:
                logging.warning("On-chain collateral lookup failed: %s", exc)

        available = max(normalized_balance, onchain_balance)
        logging.info("Available to trade: $%.2f (CLOB=$%.2f, onchain=$%.2f)",
                      available, normalized_balance, onchain_balance)
        return available, readiness

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
            if hasattr(self.client, "get_orderbook"):
                orderbook = self.client.get_orderbook(token_id)
            else:
                raw_client = getattr(self.client, "client", None)
                if raw_client and hasattr(raw_client, "get_order_book"):
                    orderbook = raw_client.get_order_book(token_id)
                else:
                    context["orderbook_ok"] = True
                    context["tradable"] = True
                    context["reason"] = "orderbook_check_unavailable"
                    return context

            bids, asks = self._extract_orderbook_levels(orderbook)
            context["orderbook_ok"] = True
            context["bid_levels"] = len(bids)
            context["ask_levels"] = len(asks)
            context["tradable"] = bool(bids or asks)
        except Exception as exc:
            message = str(exc)
            context["orderbook_ok"] = False
            context["orderbook_error"] = message
            if "orderbook" in message.lower() and "does not exist" in message.lower():
                context["tradable"] = False
                context["reason"] = "orderbook_not_found"
            else:
                context["tradable"] = True
                context["reason"] = "orderbook_check_failed_allowing"
            return context

        try:
            if hasattr(self.client, "get_price"):
                context["quoted_price"] = self.client.get_price(token_id, side=side)
        except Exception as exc:
            context["price_error"] = str(exc)

        try:
            if hasattr(self.client, "get_spread"):
                context["quoted_spread"] = self.client.get_spread(token_id)
        except Exception as exc:
            context["spread_error"] = str(exc)

        return context

    def submit_market_entry(self, token_id, amount, side="BUY", condition_id=None, outcome_side=None, spread=None, open_orders=0, daily_pnl=0.0):
        normalized_side = str(side).upper()
        try:
            amount = float(amount)
        except Exception:
            amount = None

        if amount is None or amount <= 0.0:
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": None,
                "token_id": token_id,
                "condition_id": condition_id,
                "outcome_side": outcome_side,
                "order_side": side,
                "amount": amount,
                "order_type": "FOK",
                "execution_style": "market",
                "status": "REJECTED",
                "reason": "invalid_amount",
            }
            self._append(self.orders_file, row)
            return row, None

        decision = self.risk.pre_trade_check(
            token_id=token_id, price=0.5, size=amount,
            spread=spread, open_orders=open_orders, daily_pnl=daily_pnl
        )
        if not decision.allowed:
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": None,
                "token_id": token_id,
                "condition_id": condition_id,
                "outcome_side": outcome_side,
                "order_side": side,
                "amount": amount,
                "order_type": "FOK",
                "execution_style": "market",
                "status": "REJECTED",
                "reason": decision.reason,
            }
            self._append(self.orders_file, row)
            return row, None

        available, readiness = self._get_available_balance(asset_type="COLLATERAL")

        if available < amount:
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": None,
                "token_id": token_id,
                "condition_id": condition_id,
                "outcome_side": outcome_side,
                "order_side": side,
                "amount": amount,
                "order_type": "FOK",
                "execution_style": "market",
                "status": "REJECTED",
                "reason": "insufficient_funds",
                "available_balance": available,
                "raw_readiness": str(readiness)[:200],
            }
            self._append(self.orders_file, row)
            logging.warning(
                "Market order rejected: insufficient_funds. Want $%.2f but only $%.2f available. Raw API: %s",
                amount, available, str(readiness)[:200]
            )
            return row, None

        try:
            response = self.client.create_and_post_market_order(
                token_id=token_id,
                amount=amount,
                side=side,
                order_type="FOK",
            )
        except Exception as exc:
            self.risk.record_failed_order()
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": None,
                "token_id": token_id,
                "condition_id": condition_id,
                "outcome_side": outcome_side,
                "order_side": side,
                "amount": amount,
                "order_type": "FOK",
                "execution_style": "market",
                "status": "FAILED",
                "reason": str(exc),
            }
            self._append(self.orders_file, row)
            logging.error("Market order failed: %s", exc)
            return row, None

        order_id = response.get("orderID") or response.get("order_id") or response.get("id")
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "token_id": token_id,
            "condition_id": condition_id,
            "outcome_side": outcome_side,
            "order_side": side,
            "amount": amount,
            "order_type": "FOK",
            "execution_style": "market",
            "status": response.get("status", "SUBMITTED"),
        }
        self._append(self.orders_file, row)
        self.db.execute(
            "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (order_id, token_id, condition_id, outcome_side, side, 0.0, amount, row["status"], row["timestamp"]),
        )
        logging.info("Market order submitted: token=%s amount=$%.2f order_id=%s", token_id, amount, order_id)
        return row, response

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

        if normalized_side == "BUY":
            notional_usdc = requested_size
            order_size_shares = self._round_down_shares(requested_size / max(float(price), 1e-9))
        else:
            order_size_shares = self._round_down_shares(requested_size)
            notional_usdc = order_size_shares * float(price)

        decision = self.risk.pre_trade_check(token_id=token_id, price=price, size=notional_usdc, spread=spread, open_orders=open_orders, daily_pnl=daily_pnl)
        idempotency_key = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')}|{token_id}|{condition_id}|{side}|{size}|{round(float(price), 4)}"
        existing = self.list_orders()
        if not existing.empty and "idempotency_key" in existing.columns and (existing["idempotency_key"].astype(str) == idempotency_key).any():
            return {"status": "REJECTED", "reason": "duplicate_idempotency_key", "idempotency_key": idempotency_key}, None
        if not decision.allowed:
            row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": decision.reason}
            self._append(self.orders_file, row)
            return row, None

        if normalized_side == "BUY":
            available_balance, readiness = self._get_available_balance(asset_type="COLLATERAL")

            if readiness is None and available_balance <= 0:
                row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "missing_readiness"}
                self._append(self.orders_file, row)
                return row, None

            if available_balance < float(notional_usdc):
                row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "size_usdc": notional_usdc, "order_size_shares": order_size_shares, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "insufficient_funds", "available_balance": available_balance}
                self._append(self.orders_file, row)
                logging.warning("Limit order rejected: want $%.2f but only $%.2f available", notional_usdc, available_balance)
                return row, None
        else:
            readiness = self.check_readiness(asset_type="CONDITIONAL", token_id=token_id)
            available_balance = 0.0
            if isinstance(readiness, dict):
                for key in ["balance", "available", "available_balance", "amount"]:
                    if readiness.get(key) is not None:
                        available_balance = self._normalize_balance(readiness[key])
                        break

            if readiness is None and available_balance <= 0:
                row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "size_usdc": notional_usdc, "order_size_shares": order_size_shares, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "missing_token_readiness"}
                self._append(self.orders_file, row)
                return row, None

            available_shares = self._round_down_shares(available_balance)
            requested_sell_shares = self._round_down_shares(order_size_shares)
            max_sell_shares = self._round_down_shares(max(0.0, available_shares * 0.999))

            if requested_sell_shares > max_sell_shares > 0:
                logging.warning(
                    "Capping SELL size for %s from %.6f to %.6f (verified balance %.6f)",
                    str(token_id)[:16], requested_sell_shares, max_sell_shares, available_shares,
                )
                order_size_shares = max_sell_shares
                size = max_sell_shares
                notional_usdc = order_size_shares * float(price)
            else:
                order_size_shares = requested_sell_shares
                size = requested_sell_shares
                notional_usdc = order_size_shares * float(price)

            if available_shares <= 0 or order_size_shares <= 0:
                row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "size_usdc": notional_usdc, "order_size_shares": order_size_shares, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": "insufficient_token_inventory", "available_token_balance": available_shares}
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
                response = self.client.create_and_post_order(token_id=token_id, price=price, size=order_size_shares, side=side, order_type=order_type, options=None)
        except Exception as exc:
            self.risk.record_failed_order()
            logging.error(
                "submit_entry failed token=%s side=%s price=%.6f requested_size=%s order_shares=%.6f error=%s",
                str(token_id)[:16], side, float(price), str(size), float(order_size_shares), exc,
            )
            row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "size_usdc": notional_usdc, "order_size_shares": order_size_shares, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "FAILED", "reason": str(exc), **market_context}
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
            self.db.execute("UPDATE orders SET status = ? WHERE order_id = ?", (status, order_id))
        except Exception:
            pass

    def get_order_status(self, order_id):
        return self.client.get_order(order_id)

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
        return self.submit_entry(token_id=token_id, price=price, size=size, side=side, condition_id=condition_id, outcome_side=outcome_side, order_type="GTC", post_only=True, execution_style="maker")

    def submit_taker_order(self, token_id, price, size, side="BUY", condition_id=None, outcome_side=None):
        return self.submit_entry(token_id=token_id, price=price, size=size, side=side, condition_id=condition_id, outcome_side=outcome_side, order_type="GTC", post_only=False, execution_style="taker")

    def place_target_exit_order(self, token_id, target_price, size, condition_id=None, outcome_side=None):
        return self.submit_entry(token_id=token_id, price=target_price, size=size, side="SELL", condition_id=condition_id, outcome_side=outcome_side)

    def monitor_and_trigger_exit(self, token_id, target_price, size, condition_id=None, outcome_side=None):
        quote = None
        try:
            from market_price_service import MarketPriceService
            quote = MarketPriceService().get_quote(token_id)
        except Exception:
            quote = None

        executable_sell = (quote or {}).get("best_bid")
        if executable_sell is not None and float(executable_sell) >= float(target_price):
            return self.submit_entry(token_id=token_id, price=executable_sell, size=size, side="SELL", condition_id=condition_id, outcome_side=outcome_side)
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
