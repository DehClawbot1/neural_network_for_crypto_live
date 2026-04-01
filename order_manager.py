import os
import math
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd

from balance_normalization import normalize_allowance_balance
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
        row = dict(row)
        if path.name == "live_orders.csv":
            row.setdefault("order_source", "order_manager")
            row.setdefault("created_at", row.get("timestamp"))
        row_df = pd.DataFrame([row])
        if not path.exists():
            row_df.to_csv(path, index=False)
            return

        try:
            header_cols = pd.read_csv(path, nrows=0, engine="python", on_bad_lines="skip").columns.tolist()
        except Exception:
            header_cols = []

        if header_cols and all(column in header_cols for column in row_df.columns):
            row_df.reindex(columns=header_cols).to_csv(path, mode="a", header=False, index=False)
            return

        try:
            existing = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            existing = pd.DataFrame(columns=header_cols)

        ordered_cols = list(existing.columns)
        for column in row_df.columns:
            if column not in ordered_cols:
                ordered_cols.append(column)

        existing = existing.reindex(columns=ordered_cols)
        row_df = row_df.reindex(columns=ordered_cols)
        pd.concat([existing, row_df], ignore_index=True).to_csv(path, index=False)

    def _extract_order_id(self, payload):
        if not isinstance(payload, dict):
            return None
        for key in ("orderID", "order_id", "id"):
            value = payload.get(key)
            if value not in (None, ""):
                return str(value)
        return None

    def _is_terminal_status(self, status):
        status = str(status or "").upper()
        return status in {"FILLED", "EXECUTED", "MATCHED", "CANCELED", "CANCELLED", "FAILED", "REJECTED"}

    def _normalize_balance(self, raw_balance):
        return normalize_allowance_balance(raw_balance, asset_type="COLLATERAL")

    def _round_down_shares(self, shares, decimals=6):
        try:
            shares = float(shares)
        except (TypeError, ValueError):
            return 0.0
        factor = 10 ** int(decimals)
        return math.floor(max(shares, 0.0) * factor) / factor

    def _has_recent_dust_clear(self, token_id, condition_id=None, outcome_side=None, lookback_seconds=300):
        try:
            lookback_seconds = max(1, int(lookback_seconds))
        except Exception:
            lookback_seconds = 300
        cutoff_iso = (datetime.now(timezone.utc) - pd.Timedelta(seconds=lookback_seconds)).isoformat()
        try:
            rows = self.db.query_all(
                """
                SELECT order_id
                FROM orders
                WHERE order_id LIKE 'dust_clear_%'
                  AND token_id = ?
                  AND COALESCE(condition_id, '') = ?
                  AND COALESCE(outcome_side, '') = ?
                  AND COALESCE(created_at, '') >= ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (
                    str(token_id or ""),
                    str(condition_id or ""),
                    str(outcome_side or ""),
                    cutoff_iso,
                ),
            )
            return bool(rows)
        except Exception:
            return False

    def _get_available_balance(self, asset_type="COLLATERAL", token_id=None, use_onchain_fallback=True):
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
                logging.info("On-chain balance (diagnostic only): $%.2f", onchain_balance)
            except Exception as exc:
                logging.warning("On-chain collateral lookup failed: %s", exc)

        allow_onchain_fallback = (
            use_onchain_fallback
            and os.getenv("ALLOW_ONCHAIN_BALANCE_FALLBACK", "false").strip().lower() in {"1", "true", "yes", "on"}
        )
        available = normalized_balance
        if allow_onchain_fallback and available <= 0 and str(asset_type).upper() == "COLLATERAL":
            available = onchain_balance
            logging.warning(
                "Using on-chain fallback balance because CLOB/API balance is zero. "
                "This can still fail at order placement."
            )
        logging.info("Spendable balance: $%.2f (CLOB=$%.2f, onchain=$%.2f)",
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

        if normalized_side == "BUY" and amount < 0.99:
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
                "reason": f"below_clob_minimum_1usd (val={amount})",
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

        available, readiness = self._get_available_balance(
            asset_type="COLLATERAL",
            use_onchain_fallback=False,
        )

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

        order_id = self._extract_order_id(response)
        if not order_id:
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
                "reason": f"missing_order_id_in_response:{response}",
            }
            self._append(self.orders_file, row)
            return row, response
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
            
        # FIX V5: Enforce Polymarket CLOB $1.00 minimum order size
        notional_val = requested_size if str(side).upper() == "BUY" else (requested_size * float(price))
        if notional_val < 0.99 and normalized_side == "BUY": # BUG 3 FIX: Allow liquidating depreciated bags
            row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": f"below_clob_minimum_1usd (val={notional_val})"}
            self._append(self.orders_file, row)
            return row, None

        if normalized_side == "BUY":
            notional_usdc = requested_size
            order_size_shares = self._round_down_shares(requested_size / max(float(price), 1e-9))
        else:
            order_size_shares = self._round_down_shares(requested_size)
            notional_usdc = order_size_shares * float(price)

        decision = self.risk.pre_trade_check(token_id=token_id, price=price, size=notional_usdc, spread=spread, open_orders=open_orders, daily_pnl=daily_pnl)
        idempotency_key = f"{token_id}|{condition_id}|{side}|{size}|{round(float(price), 4)}"
        existing = self.list_orders()
        if not existing.empty and "idempotency_key" in existing.columns:
            candidates = existing[existing["idempotency_key"].astype(str) == idempotency_key].copy()
            if not candidates.empty:
                if "status" in candidates.columns:
                    candidates = candidates[~candidates["status"].astype(str).apply(self._is_terminal_status)]
                if "timestamp" in candidates.columns:
                    ts = pd.to_datetime(candidates["timestamp"], errors="coerce", utc=True)
                    cutoff = datetime.now(timezone.utc) - pd.Timedelta(minutes=2)
                    candidates = candidates[ts >= cutoff]
                if not candidates.empty:
                    return {"status": "REJECTED", "reason": "duplicate_idempotency_key_active", "idempotency_key": idempotency_key}, None
        if not decision.allowed:
            row = {"timestamp": datetime.now(timezone.utc).isoformat(), "order_id": None, "idempotency_key": idempotency_key, "token_id": token_id, "condition_id": condition_id, "outcome_side": outcome_side, "order_side": side, "price": price, "size": size, "order_type": order_type, "post_only": post_only, "execution_style": execution_style, "status": "REJECTED", "reason": decision.reason}
            self._append(self.orders_file, row)
            return row, None

        if normalized_side == "BUY":
            available_balance, readiness = self._get_available_balance(
                asset_type="COLLATERAL",
            )

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
            max_sell_shares = self._round_down_shares(max(0.0, available_shares)) # BUG 4 FIX: Clear entire bag, no dust

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

        # --- DUST PROTECTION PATCH ---
        if normalized_side == "SELL" and float(order_size_shares) * float(price) < 0.01:
            import time
            dedupe_window = int(os.getenv("DUST_CLEAR_DEDUPE_SECONDS", "300") or 300)
            if self._has_recent_dust_clear(token_id, condition_id, outcome_side, lookback_seconds=dedupe_window):
                logging.info(
                    "Skipping duplicate dust clear for %s (recent synthetic clear already recorded within %ss).",
                    str(token_id)[:16],
                    dedupe_window,
                )
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
                    "status": "SKIPPED",
                    "reason": "dust_already_cleared_recently",
                    **market_context,
                }
                self._append(self.orders_file, row)
                return row, {"status": "SKIPPED", "reason": "dust_already_cleared_recently"}
            logging.info("Silently clearing dust position for %s (Value < $0.01). Skipping API.", str(token_id)[:16])
            dummy_response = {"status": "FILLED", "orderID": "dust_clear_" + str(int(time.time()))}
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": dummy_response["orderID"],
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
                "status": "FILLED",
                **market_context,
            }
            self._append(self.orders_file, row)
            self.db.execute(
                "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row.get("order_id"), row.get("token_id"), row.get("condition_id"), row.get("outcome_side"), row.get("order_side"), row.get("price"), row.get("size"), row.get("status"), row.get("timestamp")),
            )
            # Instantly create a synthetic fill so TradeManager clears the position
            self.record_fill({
                "trade_id": "fill_" + dummy_response["orderID"],
                "order_id": dummy_response["orderID"],
                "token_id": token_id,
                "condition_id": condition_id,
                "outcome_side": outcome_side,
                "side": side,
                "price": float(price),
                "size": float(order_size_shares),
                "filled_at": row["timestamp"]
            })
            return row, dummy_response
        # -----------------------------
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

        order_id = self._extract_order_id(response)
        if not order_id:
            self.risk.record_failed_order()
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
                "status": "FAILED",
                "reason": f"missing_order_id_in_response:{response}",
                **market_context,
            }
            self._append(self.orders_file, row)
            return row, response
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
            (row.get("order_id"), row.get("token_id"), row.get("condition_id"), row.get("outcome_side"), row.get("order_side"), row.get("price"), row.get("order_size_shares"), row.get("status"), row.get("timestamp")),
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
        # Synthetic dust-clear orders are already materialized as fills in submit_entry.
        # Do not re-record them here, otherwise each poll path duplicates SELL fills
        # and corrupts live position reconstruction.
        if order_id and "dust_clear" in str(order_id):
            db_row = {}
            try:
                rows = self.db.query_all(
                    "SELECT token_id, condition_id, outcome_side, order_side, price, size FROM orders WHERE order_id = ?",
                    (str(order_id),),
                )
                if rows:
                    db_row = rows[0]
            except Exception:
                db_row = {}
            fill_payload = {
                "trade_id": f"fill_{order_id}",
                "order_id": str(order_id),
                "token_id": db_row.get("token_id", ""),
                "condition_id": db_row.get("condition_id"),
                "outcome_side": db_row.get("outcome_side"),
                "side": db_row.get("order_side"),
                "price": float(db_row.get("price", 0.0) or 0.0),
                "size": float(db_row.get("size", 0.0) or 0.0),
                "filled_at": datetime.now(timezone.utc).isoformat(),
            }
            self._update_order_status(str(order_id), "FILLED", fill_price=fill_payload["price"], fill_size=fill_payload["size"])
            return {"filled": True, "response": fill_payload, "order_status": {"status": "FILLED", "id": str(order_id)}, "synthetic": True}

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
                db_row = {}
                try:
                    rows = self.db.query_all(
                        "SELECT token_id, condition_id, outcome_side, order_side, price, size FROM orders WHERE order_id = ?",
                        (order_id,),
                    )
                    if rows:
                        db_row = rows[0]
                except Exception:
                    db_row = {}

                trade_fill = None
                try:
                    trades_payload = self.client.get_trades() if hasattr(self.client, "get_trades") else []
                    if isinstance(trades_payload, dict):
                        trade_items = []
                        for key in ["data", "items", "trades", "results"]:
                            if isinstance(trades_payload.get(key), list):
                                trade_items = trades_payload.get(key)
                                break
                        if not trade_items:
                            trade_items = [trades_payload]
                    elif isinstance(trades_payload, list):
                        trade_items = trades_payload
                    else:
                        trade_items = []
                    for item in trade_items:
                        candidate_order_id = item.get("orderID") or item.get("order_id") or item.get("maker_order_id") or item.get("taker_order_id")
                        if str(candidate_order_id or "") == str(order_id):
                            trade_fill = item
                            break
                except Exception:
                    trade_fill = None

                price_value = None
                for key in ["price", "avgPrice", "filled_price", "matched_price", "rate"]:
                    source = trade_fill if isinstance(trade_fill, dict) else (last_response or {})
                    if source.get(key) not in [None, ""]:
                        price_value = source.get(key)
                        break
                if price_value in [None, ""]:
                    price_value = db_row.get("price", 0.0)

                size_value = None
                for key in ["size", "filledSize", "matched_amount", "amount"]:
                    source = trade_fill if isinstance(trade_fill, dict) else (last_response or {})
                    if source.get(key) not in [None, ""]:
                        size_value = source.get(key)
                        break
                if size_value in [None, ""]:
                    size_value = db_row.get("size", 0.0)
                try:
                    size_value = float(size_value or 0.0)
                except Exception:
                    size_value = 0.0

                # Backward compatibility: legacy BUY orders may have persisted
                # notional USDC in `orders.size`. Convert probable notionals to shares.
                try:
                    if (
                        size_value > 0
                        and str(db_row.get("order_side", "")).upper() == "BUY"
                        and float(db_row.get("price", 0.0) or 0.0) > 0
                        and size_value <= 50.0
                    ):
                        size_value = size_value / max(float(db_row.get("price") or 1e-9), 1e-9)
                except Exception:
                    pass

                fill_payload = {
                    "trade_id": ((trade_fill or {}).get("id") if isinstance(trade_fill, dict) else None) or (last_response or {}).get("id") or f"{order_id}:{fill_event_time}",
                    "order_id": order_id,
                    "token_id": ((trade_fill or {}).get("token_id") if isinstance(trade_fill, dict) else None) or (last_response or {}).get("token_id") or db_row.get("token_id", ""),
                    "condition_id": db_row.get("condition_id"),
                    "outcome_side": db_row.get("outcome_side"),
                    "side": db_row.get("order_side"),
                    "status": "FILLED",
                    "price": float(price_value or 0.0),
                    "size": float(size_value or 0.0),
                    "filled_at": fill_event_time,
                }
                self._update_order_status(order_id, "FILLED", fill_price=fill_payload["price"], fill_size=fill_payload["size"])
                self.record_fill(fill_payload)
                return {"filled": True, "response": fill_payload, "order_status": last_response}
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
        row["trade_id"] = row.get("trade_id") or fill_id
        row["fill_id"] = fill_id
        row["fill_source"] = row.get("fill_source") or "order_manager"
        self._append(self.fills_file, row)
        self.db.execute(
            "INSERT OR REPLACE INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fill_id,
                row.get("order_id"),
                row.get("token_id"),
                row.get("condition_id"),
                row.get("outcome_side"),
                row.get("side"),
                row.get("price"),
                row.get("size"),
                row.get("filled_at") or row.get("timestamp"),
            ),
        )
        return fill_payload
