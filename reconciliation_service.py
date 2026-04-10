from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
import os
from pathlib import Path
import json

import pandas as pd

from db import Database

logger = logging.getLogger(__name__)


class ReconciliationService:
    ORDER_CSV_COLUMNS = [
        "timestamp",
        "order_id",
        "idempotency_key",
        "token_id",
        "condition_id",
        "outcome_side",
        "order_side",
        "price",
        "size",
        "size_usdc",
        "order_size_shares",
        "order_type",
        "post_only",
        "execution_style",
        "status",
        "orderbook_ok",
        "bid_levels",
        "ask_levels",
        "tradable",
        "quoted_price",
        "quoted_spread",
        "updated_at",
        "fill_price",
        "fill_size",
        "created_at",
        "order_source",
    ]
    FILL_CSV_COLUMNS = [
        "timestamp",
        "trade_id",
        "order_id",
        "token_id",
        "condition_id",
        "outcome_side",
        "side",
        "price",
        "size",
        "filled_at",
        "fill_id",
        "fill_source",
    ]

    def __init__(self, execution_client=None, logs_dir="logs"):
        self.execution_client = execution_client
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.logs_dir / "trading.db")

    def _extract_items(self, payload):
        if payload is None:
            return []
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

    def _parse_timestamp(self, value):
        if value in (None, ""):
            return None
        try:
            return pd.to_datetime(value, utc=True, errors="coerce")
        except Exception:
            return None

    def _is_synthetic_fill_id(self, fill_id):
        fid = str(fill_id or "").strip().lower()
        return (
            fid.startswith("fill_dust_clear_")
            or fid.startswith("fill_ext_sync_")
            or fid.startswith("ext_sync_")
            or "dust_clear" in fid
        )

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

    def _extract_fill_ids_from_frame(self, frame: pd.DataFrame):
        ids = set()
        if frame is None or frame.empty:
            return ids
        for column in ["fill_id", "trade_id"]:
            if column not in frame.columns:
                continue
            for value in frame[column].dropna().astype(str):
                value = value.strip()
                if value and value.lower() not in {"nan", "none"}:
                    ids.add(value)
        return ids

    def _extract_order_ids_from_frame(self, frame: pd.DataFrame):
        ids = set()
        if frame is None or frame.empty or "order_id" not in frame.columns:
            return ids
        for value in frame["order_id"].dropna().astype(str):
            value = value.strip()
            if value and value.lower() not in {"nan", "none"}:
                ids.add(value)
        return ids

    def _load_meaningful_profile_position_keys(self):
        funder = str(os.getenv("POLYMARKET_FUNDER") or os.getenv("POLYMARKET_PUBLIC_ADDRESS") or "").strip()
        if not funder:
            return set()
        try:
            from polymarket_profile_client import PolymarketProfileClient
            client = PolymarketProfileClient(timeout=20)
            positions = client.get_positions(user=funder, limit=100, offset=0, size_threshold=0, sort_by="CURRENT") or []
        except Exception:
            return set()

        min_notional = float(os.getenv("MIN_RECONCILED_POSITION_NOTIONAL_USDC", "0.01") or 0.01)
        keys = set()
        for position in positions:
            if not isinstance(position, dict):
                continue
            token_id = str(position.get("asset") or "").strip()
            condition_id = str(position.get("conditionId") or "").strip()
            outcome_side = str(position.get("outcome") or "").strip()
            try:
                shares = float(position.get("size") or 0.0)
            except Exception:
                shares = 0.0
            try:
                price = float(position.get("curPrice") or position.get("avgPrice") or 0.0)
            except Exception:
                price = 0.0
            if not token_id or not condition_id or not outcome_side:
                continue
            if shares * max(price, 0.0) < min_notional:
                continue
            keys.add((token_id, condition_id, outcome_side.lower()))
        return keys

    def _load_recent_remote_trade_history(self, cutoff, known_order_ids=None, tracked_tokens=None):
        known_order_ids = known_order_ids or set()
        tracked_tokens = tracked_tokens or set()
        fills_df = self._safe_read_csv("live_fills.csv")
        if fills_df.empty:
            return []

        work = fills_df.copy()
        if "fill_source" in work.columns:
            allowed_sources = {"exchange_sync", "external_sync"}
            source_mask = work["fill_source"].fillna("").astype(str).str.strip().str.lower().isin(allowed_sources)
            work = work[source_mask]
        if work.empty:
            return []

        remote_history = []
        for _, row in work.iterrows():
            trade = self._normalize_trade(row.to_dict())
            if trade is None:
                continue
            ts = self._parse_timestamp(trade.get("filled_at"))
            if ts is not None and not pd.isna(ts) and ts < cutoff:
                continue
            oid = str(trade.get("order_id") or "").strip()
            tid = str(trade.get("token_id") or "").strip()
            if known_order_ids and oid and oid in known_order_ids:
                remote_history.append(trade)
                continue
            if tracked_tokens and tid and tid in tracked_tokens:
                remote_history.append(trade)
                continue
            if not known_order_ids and not tracked_tokens:
                remote_history.append(trade)
        return remote_history

    def _merge_remote_trade_windows(self, *trade_lists):
        merged = {}
        for trades in trade_lists:
            for trade in trades or []:
                if not isinstance(trade, dict):
                    continue
                fill_id = str(trade.get("fill_id") or "").strip()
                if not fill_id:
                    continue
                existing = merged.get(fill_id)
                if existing is None:
                    merged[fill_id] = trade
                    continue
                # Prefer the richer / fresher record when duplicates exist.
                existing_ts = self._parse_timestamp(existing.get("filled_at"))
                trade_ts = self._parse_timestamp(trade.get("filled_at"))
                if existing_ts is None or pd.isna(existing_ts):
                    merged[fill_id] = trade
                elif trade_ts is not None and not pd.isna(trade_ts) and trade_ts >= existing_ts:
                    merged[fill_id] = trade
        return list(merged.values())

    def _normalize_live_order_csv_row(self, order_row, order_source="exchange_sync"):
        created_at = order_row.get("created_at") or order_row.get("timestamp") or datetime.now(timezone.utc).isoformat()
        try:
            price = float(order_row.get("price") or 0.0)
        except Exception:
            price = 0.0
        try:
            shares = float(order_row.get("size") or order_row.get("order_size_shares") or 0.0)
        except Exception:
            shares = 0.0
        return {
            "timestamp": order_row.get("timestamp") or created_at,
            "order_id": order_row.get("order_id"),
            "idempotency_key": order_row.get("idempotency_key"),
            "token_id": order_row.get("token_id"),
            "condition_id": order_row.get("condition_id"),
            "outcome_side": order_row.get("outcome_side"),
            "order_side": order_row.get("order_side"),
            "price": price,
            "size": shares,
            "size_usdc": order_row.get("size_usdc") if order_row.get("size_usdc") is not None else shares * price,
            "order_size_shares": order_row.get("order_size_shares") if order_row.get("order_size_shares") is not None else shares,
            "order_type": order_row.get("order_type"),
            "post_only": order_row.get("post_only"),
            "execution_style": order_row.get("execution_style"),
            "status": order_row.get("status"),
            "orderbook_ok": order_row.get("orderbook_ok"),
            "bid_levels": order_row.get("bid_levels"),
            "ask_levels": order_row.get("ask_levels"),
            "tradable": order_row.get("tradable"),
            "quoted_price": order_row.get("quoted_price"),
            "quoted_spread": order_row.get("quoted_spread"),
            "updated_at": order_row.get("updated_at") or created_at,
            "fill_price": order_row.get("fill_price"),
            "fill_size": order_row.get("fill_size"),
            "created_at": created_at,
            "order_source": order_row.get("order_source") or order_source,
        }

    def _merge_live_order_rows(self, order_rows, update_existing=True):
        path = self.logs_dir / "live_orders.csv"
        existing = self._safe_read_csv("live_orders.csv")
        ordered_cols = list(existing.columns)
        for column in self.ORDER_CSV_COLUMNS:
            if column not in ordered_cols:
                ordered_cols.append(column)

        if existing.empty:
            existing = pd.DataFrame(columns=ordered_cols)
        else:
            existing = existing.reindex(columns=ordered_cols)

        if "order_id" not in existing.columns:
            existing["order_id"] = pd.Series(dtype=str)

        existing_order_ids = self._extract_order_ids_from_frame(existing)
        appended_rows = []
        updated_rows = 0

        for raw_row in order_rows:
            normalized = self._normalize_live_order_csv_row(raw_row, order_source=raw_row.get("order_source") or "exchange_sync")
            order_id = str(normalized.get("order_id") or "").strip()
            if not order_id:
                continue
            if order_id in existing_order_ids:
                if update_existing:
                    mask = existing["order_id"].fillna("").astype(str) == order_id
                    if mask.any():
                        for column in ["status", "updated_at", "created_at"]:
                            if column in existing.columns:
                                existing[column] = existing[column].astype(object)
                                existing.loc[mask, column] = normalized.get(column)
                        if "order_source" in existing.columns:
                            existing["order_source"] = existing["order_source"].astype(object)
                            source_mask = mask & (
                                existing["order_source"].isna()
                                | existing["order_source"].astype(str).str.strip().isin(["", "nan", "None"])
                            )
                            existing.loc[source_mask, "order_source"] = normalized.get("order_source")
                        for column in ["token_id", "condition_id", "outcome_side", "order_side", "price"]:
                            if column in existing.columns:
                                if column != "price":
                                    existing[column] = existing[column].astype(object)
                                existing.loc[mask & existing[column].isna(), column] = normalized.get(column)
                        updated_rows += int(mask.sum())
                continue
            appended_rows.append(normalized)
            existing_order_ids.add(order_id)

        if appended_rows:
            append_df = pd.DataFrame(appended_rows).reindex(columns=ordered_cols)
            existing = append_df.copy() if existing.empty else pd.concat([existing, append_df], ignore_index=True)

        if appended_rows or updated_rows:
            existing.to_csv(path, index=False)

        return {"added": len(appended_rows), "updated": updated_rows}

    def backfill_live_orders_csv_from_db(self, update_existing=True):
        try:
            db_rows = self.db.query_all(
                """
                SELECT order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at
                FROM orders
                WHERE order_id IS NOT NULL
                ORDER BY COALESCE(created_at, ''), order_id
                """
            )
        except sqlite3.Error as exc:
            logger.warning("Order DB query failed: %s", exc)
            db_rows = []

        order_rows = []
        for row in db_rows:
            order_id = str(row.get("order_id") or "").strip()
            if not order_id:
                continue
            order_source = "db_backfill"
            if order_id.lower().startswith("dust_clear_"):
                order_source = "dust_clear"
            order_rows.append(self._normalize_live_order_csv_row(row, order_source=order_source))

        return self._merge_live_order_rows(order_rows, update_existing=update_existing)

    def _normalize_live_fill_csv_row(self, fill_row, fill_source="exchange_sync"):
        filled_at = fill_row.get("filled_at") or fill_row.get("timestamp") or datetime.now(timezone.utc).isoformat()
        fill_id = str(fill_row.get("fill_id") or fill_row.get("trade_id") or "").strip()
        if not fill_id:
            fill_id = f"{fill_row.get('order_id') or 'unknown'}:{filled_at}"
        return {
            "timestamp": fill_row.get("timestamp") or filled_at,
            "trade_id": str(fill_row.get("trade_id") or fill_id),
            "order_id": fill_row.get("order_id"),
            "token_id": fill_row.get("token_id"),
            "condition_id": fill_row.get("condition_id"),
            "outcome_side": fill_row.get("outcome_side"),
            "side": fill_row.get("side"),
            "price": fill_row.get("price"),
            "size": fill_row.get("size"),
            "filled_at": filled_at,
            "fill_id": fill_id,
            "fill_source": fill_row.get("fill_source") or fill_source,
        }

    def _merge_live_fill_rows(self, fill_rows):
        path = self.logs_dir / "live_fills.csv"
        if not fill_rows:
            return 0

        existing = self._safe_read_csv("live_fills.csv")
        existing_fill_ids = self._extract_fill_ids_from_frame(existing)

        normalized_rows = []
        staged_fill_ids = set()
        for row in fill_rows:
            normalized = self._normalize_live_fill_csv_row(row, fill_source=row.get("fill_source") or "exchange_sync")
            fill_id = str(normalized.get("fill_id") or "").strip()
            if not fill_id or fill_id in existing_fill_ids or fill_id in staged_fill_ids:
                continue
            normalized_rows.append(normalized)
            staged_fill_ids.add(fill_id)

        if not normalized_rows:
            return 0

        append_df = pd.DataFrame(normalized_rows)
        ordered_cols = list(existing.columns)
        for column in self.FILL_CSV_COLUMNS:
            if column not in ordered_cols:
                ordered_cols.append(column)
        for column in append_df.columns:
            if column not in ordered_cols:
                ordered_cols.append(column)

        existing = existing.reindex(columns=ordered_cols)
        append_df = append_df.reindex(columns=ordered_cols)
        merged = append_df.copy() if existing.empty else pd.concat([existing, append_df], ignore_index=True)

        if "fill_id" in merged.columns:
            dedupe_key = merged["fill_id"].fillna("")
            if "trade_id" in merged.columns:
                trade_key = merged["trade_id"].fillna("")
                dedupe_key = dedupe_key.where(dedupe_key.astype(str).str.strip() != "", trade_key)
            merged = merged.assign(_dedupe_fill_id=dedupe_key.astype(str))
            merged = merged[merged["_dedupe_fill_id"].str.strip() != ""]
            merged = merged.drop_duplicates(subset=["_dedupe_fill_id"], keep="last").drop(columns=["_dedupe_fill_id"])

        merged.to_csv(path, index=False)
        return len(normalized_rows)

    def backfill_live_fills_csv_from_db(self, include_synthetic=True):
        try:
            db_rows = self.db.query_all(
                """
                SELECT fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at
                FROM fills
                ORDER BY COALESCE(filled_at, ''), fill_id
                """
            )
        except sqlite3.Error as exc:
            logger.warning("Fill DB query failed: %s", exc)
            db_rows = []

        fill_rows = []
        for row in db_rows:
            fill_id = str(row.get("fill_id") or "").strip()
            if not fill_id:
                continue
            if not include_synthetic and self._is_synthetic_fill_id(fill_id):
                continue
            fill_source = "db_backfill"
            lowered = fill_id.lower()
            if lowered.startswith("fill_dust_clear_") or lowered.startswith("dust_clear_") or "dust_clear" in lowered:
                fill_source = "dust_clear"
            elif lowered.startswith("fill_ext_sync_") or lowered.startswith("ext_sync_"):
                fill_source = "external_sync"
            fill_rows.append(self._normalize_live_fill_csv_row({**row, "trade_id": fill_id}, fill_source=fill_source))

        return self._merge_live_fill_rows(fill_rows)

    def sync_orders_and_fills(self):
        """Sync exchange orders and fills into the local SQLite database."""
        synced_orders = 0
        synced_order_csv_rows = 0
        synced_order_csv_updates = 0
        synced_fills = 0
        synced_fill_csv_rows = 0
        known_order_ids = set()
        tracked_tokens = set()

        try:
            for row in self.db.query_all("SELECT order_id FROM orders WHERE order_id IS NOT NULL"):
                oid = str(row.get("order_id") or "").strip()
                if oid:
                    known_order_ids.add(oid)
        except Exception:
            known_order_ids = set()
        try:
            for row in self.db.query_all("SELECT token_id FROM live_positions WHERE status = 'OPEN' AND token_id IS NOT NULL"):
                token = str(row.get("token_id") or "").strip()
                if token:
                    tracked_tokens.add(token)
        except Exception:
            tracked_tokens = set()

        try:
            orders_payload = self.execution_client.get_open_orders()
            order_rows_to_mirror = []
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
                known_order_ids.add(order["order_id"])
                order_rows_to_mirror.append(
                    self._normalize_live_order_csv_row(order, order_source="exchange_sync")
                )
            order_csv_result = self._merge_live_order_rows(order_rows_to_mirror, update_existing=True)
            synced_order_csv_rows = int(order_csv_result.get("added", 0))
            synced_order_csv_updates = int(order_csv_result.get("updated", 0))
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
            order_backfill_result = self.backfill_live_orders_csv_from_db(update_existing=True)
            synced_order_csv_rows += int(order_backfill_result.get("added", 0))
            synced_order_csv_updates += int(order_backfill_result.get("updated", 0))
        except Exception:
            pass

        try:
            trades_payload = self.execution_client.get_trades()
            # Default to scoped trade sync only. Broad import of all recent remote
            # trades can mirror unrelated/manual activity into local fills without
            # corresponding local orders, which corrupts position reconstruction.
            sync_all_recent = str(os.getenv("SYNC_ALL_RECENT_REMOTE_TRADES", "false")).strip().lower() in {"1", "true", "yes", "on"}
            lookback_hours = max(1, int(os.getenv("SYNC_RECENT_TRADES_LOOKBACK_HOURS", "72") or 72))
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_hours)
            fill_rows_to_mirror = []
            for raw_trade in self._extract_items(trades_payload):
                trade = self._normalize_trade(raw_trade)
                if trade is None:
                    continue
                ts = self._parse_timestamp(trade.get("filled_at"))
                is_recent = bool(ts is not None and not pd.isna(ts) and ts >= cutoff)
                # Exchange trades are useful for reconciling known local orders and
                # already-tracked open tokens. Broad recent trade import is opt-in.
                if sync_all_recent:
                    if not is_recent and trade.get("order_id") not in known_order_ids and trade.get("token_id") not in tracked_tokens:
                        continue
                else:
                    if trade.get("order_id") not in known_order_ids and trade.get("token_id") not in tracked_tokens:
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
                fill_rows_to_mirror.append(
                    self._normalize_live_fill_csv_row(
                        {**trade, "trade_id": trade["fill_id"]},
                        fill_source="exchange_sync",
                    )
                )
                # BUG FIX 3: Removed blind FILLED overwrite. Open order sweep will handle terminal status naturally.
                synced_fills += 1
            synced_fill_csv_rows = self._merge_live_fill_rows(fill_rows_to_mirror)
        except (sqlite3.Error, IOError) as exc:
            logger.warning("Fill sync reconciliation failed: %s", exc)
        except Exception as exc:
            logger.error("Unexpected error in fill sync: %s", exc)

        return {
            "orders": synced_orders,
            "order_csv_rows_added": synced_order_csv_rows,
            "order_csv_rows_updated": synced_order_csv_updates,
            "fills": synced_fills,
            "fill_csv_rows_added": synced_fill_csv_rows,
        }

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

        # Compare against a deeper recent window so fills that already mirrored via
        # exchange sync don't keep surfacing as soft drift when the direct trades
        # endpoint is briefly shallower than local history.
        lookback_hours = max(1, int(os.getenv("RECONCILIATION_LOOKBACK_HOURS", "72") or 72))
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_hours)

        remote_orders_df = pd.DataFrame(remote_orders)
        remote_trades_df = pd.DataFrame(remote_trades)

        known_order_ids = set()
        tracked_tokens = set()
        try:
            if not local_orders.empty and "order_id" in local_orders.columns:
                known_order_ids = set(local_orders["order_id"].dropna().astype(str).tolist())
        except Exception:
            known_order_ids = set()
        try:
            rows = self.db.query_all("SELECT token_id FROM live_positions WHERE status = 'OPEN' AND token_id IS NOT NULL")
            tracked_tokens = {str(r.get("token_id") or "").strip() for r in rows if str(r.get("token_id") or "").strip()}
        except Exception:
            tracked_tokens = set()

        history_remote_trades = self._load_recent_remote_trade_history(
            cutoff,
            known_order_ids=known_order_ids,
            tracked_tokens=tracked_tokens,
        )
        remote_trades = self._merge_remote_trade_windows(remote_trades, history_remote_trades)

        filtered_remote_trades = []
        for trade in remote_trades:
            oid = str(trade.get("order_id") or "").strip()
            tid = str(trade.get("token_id") or "").strip()
            ts = self._parse_timestamp(trade.get("filled_at"))
            if ts is not None and not pd.isna(ts) and ts < cutoff:
                continue
            if oid in known_order_ids or tid in tracked_tokens:
                filtered_remote_trades.append(trade)
        remote_trades = filtered_remote_trades
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
                for _, row in local_fills.iterrows():
                    fill_id = str(row.get(id_col) or "").strip()
                    if not fill_id or fill_id.lower() in {"nan", "none"}:
                        continue
                    if self._is_synthetic_fill_id(fill_id):
                        continue
                    ts = self._parse_timestamp(row.get("filled_at"))
                    if ts is not None and not pd.isna(ts) and ts < cutoff:
                        continue
                    oid = str(row.get("order_id") or "").strip()
                    tid = str(row.get("token_id") or "").strip()
                    if (oid and oid in known_order_ids) or (tid and tid in tracked_tokens):
                        local_trade_ids.add(fill_id)

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
            "remote_trade_history_rows": len(history_remote_trades),
            "missing_remote_orders": missing_remote_orders,
            "missing_local_orders": missing_local_orders,
            "missing_remote_trades": missing_remote_trades,
            "missing_local_trades": missing_local_trades,
            "order_mismatches": order_mismatches,
        }

        return report, remote_orders_df, remote_trades_df

    def archive_and_prune_unmatched_remote_fills(self, archive_dir=None):
        query = """
            SELECT f.fill_id, f.order_id, f.token_id, f.condition_id, f.outcome_side, f.side, f.price, f.size, f.filled_at
            FROM fills f
            WHERE f.order_id IS NOT NULL
              AND TRIM(f.order_id) != ''
              AND f.order_id != 'external_manual'
              AND f.order_id NOT LIKE 'dust_clear_%'
              AND f.fill_id NOT LIKE 'ext_sync_%'
              AND f.fill_id NOT LIKE 'fill_dust_clear_%'
              AND f.order_id NOT IN (SELECT order_id FROM orders)
            ORDER BY COALESCE(f.filled_at, ''), f.fill_id
        """
        db_rows = self.db.query_all(query)
        if not db_rows:
            return {
                "archived_rows": 0,
                "deleted_db_rows": 0,
                "deleted_live_fill_rows": 0,
                "archive_csv": None,
                "archive_jsonl": None,
            }

        db_df = pd.DataFrame(db_rows)
        live_fills_path = self.logs_dir / "live_fills.csv"
        live_fills_df = self._safe_read_csv("live_fills.csv")
        if live_fills_df.empty or "fill_id" not in live_fills_df.columns:
            enriched = db_df.copy()
            enriched["fill_source"] = None
        else:
            fill_meta_cols = [c for c in ["fill_id", "fill_source", "timestamp", "filled_at"] if c in live_fills_df.columns]
            fill_meta = live_fills_df[fill_meta_cols].copy()
            rename_map = {}
            if "timestamp" in fill_meta.columns:
                rename_map["timestamp"] = "live_fill_timestamp"
            if "filled_at" in fill_meta.columns:
                rename_map["filled_at"] = "live_fill_filled_at"
            fill_meta = fill_meta.rename(columns=rename_map)
            enriched = db_df.merge(fill_meta, on="fill_id", how="left")

        allowed_sources = {"exchange_sync", "db_backfill"}
        source_series = enriched.get("fill_source")
        if source_series is None:
            source_mask = pd.Series(False, index=enriched.index)
        else:
            source_mask = source_series.fillna("").astype(str).str.strip().str.lower().isin(allowed_sources)

        meaningful_keys = self._load_meaningful_profile_position_keys()
        if meaningful_keys:
            key_mask = enriched.apply(
                lambda row: (
                    str(row.get("token_id") or "").strip(),
                    str(row.get("condition_id") or "").strip(),
                    str(row.get("outcome_side") or "").strip().lower(),
                ) in meaningful_keys,
                axis=1,
            )
        else:
            key_mask = pd.Series(False, index=enriched.index)

        prune_df = enriched[source_mask & ~key_mask].copy()
        if prune_df.empty:
            return {
                "archived_rows": 0,
                "deleted_db_rows": 0,
                "deleted_live_fill_rows": 0,
                "archive_csv": None,
                "archive_jsonl": None,
            }

        archive_root = Path(archive_dir or (self.logs_dir / "archives"))
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_csv = archive_root / f"unmatched_remote_fills_archive_{stamp}.csv"
        archive_jsonl = archive_root / f"unmatched_remote_fills_archive_{stamp}.jsonl"

        prune_df.to_csv(archive_csv, index=False)
        with archive_jsonl.open("w", encoding="utf-8") as handle:
            for row in prune_df.to_dict("records"):
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        fill_ids = [str(fill_id) for fill_id in prune_df["fill_id"].dropna().astype(str).tolist() if str(fill_id).strip()]
        deleted_db_rows = 0
        if fill_ids:
            placeholders = ",".join("?" for _ in fill_ids)
            cursor = self.db.conn.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                cursor.execute(f"DELETE FROM fills WHERE fill_id IN ({placeholders})", tuple(fill_ids))
                deleted_db_rows = int(cursor.rowcount or 0)
                self.db.conn.commit()
            except Exception:
                self.db.conn.rollback()
                raise

        deleted_live_fill_rows = 0
        if not live_fills_df.empty and "fill_id" in live_fills_df.columns and fill_ids:
            before = len(live_fills_df.index)
            keep_df = live_fills_df[~live_fills_df["fill_id"].astype(str).isin(fill_ids)].copy()
            deleted_live_fill_rows = before - len(keep_df.index)
            keep_df.to_csv(live_fills_path, index=False)

        return {
            "archived_rows": int(len(prune_df.index)),
            "deleted_db_rows": int(deleted_db_rows),
            "deleted_live_fill_rows": int(deleted_live_fill_rows),
            "archive_csv": str(archive_csv),
            "archive_jsonl": str(archive_jsonl),
        }
