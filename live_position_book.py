from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import uuid
from typing import Any

import pandas as pd

from db import Database
from balance_normalization import maybe_trace_allowance_payload

logger = logging.getLogger(__name__)


class LivePositionBook:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.logs_dir / "trading.db")
        # TTL cache for _verify_open_positions_against_exchange so we don't
        # fire N balance-allowance calls per position on every call to
        # get_open_positions() within the same cycle.
        self._verify_cache: list | None = None
        self._verify_cache_ts: float = 0.0
        self._verify_cache_token_key: str = ""
        # Cooldown for rebuild_from_db — prevents redundant heavy queries
        # when the supervisor calls rebuild multiple times per cycle.
        self._rebuild_cooldown_seconds: float = 5.0
        self._last_rebuild_ts: float = 0.0

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _load_profile_snapshot_rows(self) -> list[dict]:
        enabled = str(os.getenv("LIVE_POSITION_PROFILE_FALLBACK_ENABLED", "true")).strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return []

        funder = str(os.getenv("POLYMARKET_FUNDER") or os.getenv("POLYMARKET_PUBLIC_ADDRESS") or "").strip()
        if not funder:
            try:
                from dotenv import load_dotenv
                load_dotenv(".env")
                funder = str(os.getenv("POLYMARKET_FUNDER") or os.getenv("POLYMARKET_PUBLIC_ADDRESS") or "").strip()
            except Exception:
                funder = ""
        if not funder:
            return []

        try:
            from polymarket_profile_client import PolymarketProfileClient
        except Exception as exc:
            logger.debug("Profile snapshot fallback unavailable: %s", exc)
            return []

        try:
            profile_client = PolymarketProfileClient(timeout=20)
            positions = profile_client.get_positions(
                user=funder,
                limit=100,
                offset=0,
                size_threshold=0,
                sort_by="CURRENT",
            ) or []
        except Exception as exc:
            logger.debug("Profile snapshot position fetch failed: %s", exc)
            return []

        rows: list[dict] = []
        min_notional = self._coerce_float(os.getenv("MIN_RECONCILED_POSITION_NOTIONAL_USDC"), 0.01)
        for position in positions:
            if not isinstance(position, dict):
                continue
            shares = self._coerce_float(position.get("size"), 0.0)
            if shares <= 1e-9:
                continue
            token_id = str(position.get("asset") or "").strip()
            condition_id = str(position.get("conditionId") or "").strip()
            outcome_side = str(position.get("outcome") or "").strip()
            if not token_id or not condition_id or not outcome_side:
                continue
            avg_entry_price = self._coerce_float(position.get("avgPrice"), 0.0)
            current_price = self._coerce_float(position.get("curPrice"), avg_entry_price)
            current_value = self._coerce_float(position.get("currentValue"), shares * current_price)
            initial_value = self._coerce_float(position.get("initialValue"), shares * avg_entry_price)
            effective_notional = max(shares * max(current_price, 0.0), shares * max(avg_entry_price, 0.0))
            if effective_notional < min_notional:
                continue
            position_key = f"{token_id}|{condition_id}|{outcome_side}"
            rows.append(
                {
                    "position_key": position_key,
                    "token_id": token_id,
                    "condition_id": condition_id,
                    "outcome_side": outcome_side,
                    "shares": shares,
                    "avg_entry_price": avg_entry_price,
                    "realized_pnl": self._coerce_float(position.get("realizedPnl"), 0.0),
                    "unrealized_pnl": current_value - initial_value,
                    "current_price": current_price,
                    "market_title": position.get("title"),
                    "market": position.get("slug") or position.get("title"),
                    "last_fill_at": None,
                    "source": "profile_snapshot",
                    "status": "OPEN",
                }
            )
        return rows

    def _merge_profile_snapshot_rows(self, rows: list[dict]) -> list[dict]:
        snapshot_rows = self._load_profile_snapshot_rows()
        if not snapshot_rows:
            return rows

        merged: dict[tuple[str, str, str], dict] = {}
        for row in rows:
            token_id = str(row.get("token_id") or "").strip()
            condition_id = str(row.get("condition_id") or "").strip()
            outcome_side = str(row.get("outcome_side") or "").strip()
            merged[(token_id, condition_id, outcome_side.lower())] = dict(row)

        added = 0
        upgraded = 0
        enriched = 0
        for snapshot in snapshot_rows:
            key = (
                str(snapshot.get("token_id") or "").strip(),
                str(snapshot.get("condition_id") or "").strip(),
                str(snapshot.get("outcome_side") or "").strip().lower(),
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(snapshot)
                added += 1
                continue

            existing_shares = self._coerce_float(existing.get("shares"), 0.0)
            snapshot_shares = self._coerce_float(snapshot.get("shares"), 0.0)
            should_upgrade_shares = snapshot_shares > existing_shares + 1e-6
            missing_metadata = (
                not str(existing.get("market_title") or "").strip()
                or not str(existing.get("market") or "").strip()
                or self._coerce_float(existing.get("current_price"), 0.0) <= 0.0
            )
            if should_upgrade_shares or missing_metadata:
                updated = dict(existing)
                if should_upgrade_shares:
                    updated["shares"] = snapshot_shares
                if self._coerce_float(snapshot.get("avg_entry_price"), 0.0) > 0 and (
                    should_upgrade_shares or self._coerce_float(existing.get("avg_entry_price"), 0.0) <= 0.0
                ):
                    updated["avg_entry_price"] = snapshot.get("avg_entry_price")
                if self._coerce_float(snapshot.get("current_price"), 0.0) > 0:
                    updated["current_price"] = snapshot.get("current_price")
                if snapshot.get("market_title"):
                    updated["market_title"] = snapshot.get("market_title")
                if snapshot.get("market"):
                    updated["market"] = snapshot.get("market")
                if self._coerce_float(snapshot.get("unrealized_pnl"), 0.0) != 0.0:
                    updated["unrealized_pnl"] = snapshot.get("unrealized_pnl")
                if self._coerce_float(snapshot.get("realized_pnl"), 0.0) != 0.0:
                    updated["realized_pnl"] = snapshot.get("realized_pnl")
                if "profile_snapshot" not in str(updated.get("source") or ""):
                    updated["source"] = f"{str(updated.get('source') or 'rebuild').strip()}+profile_snapshot"
                updated["status"] = "OPEN"
                merged[key] = updated
                if should_upgrade_shares:
                    upgraded += 1
                else:
                    enriched += 1

        if added or upgraded or enriched:
            logger.info(
                "Profile snapshot recovered %d missing live positions, upgraded %d undercounted rows, and enriched %d metadata-only rows.",
                added,
                upgraded,
                enriched,
            )

        return list(merged.values())

    def _insert_external_sync_sell_fill(self, cursor, *, token_id, condition_id, outcome_side, price, shares, now):
        shares = float(shares or 0.0)
        if shares <= 0:
            return None
        fill_id = f"ext_sync_{uuid.uuid4().hex[:10]}"
        cursor.execute(
            """
            INSERT INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at)
            VALUES (?, 'external_manual', ?, ?, ?, 'SELL', ?, ?, ?)
            """,
            (
                fill_id,
                token_id,
                condition_id,
                outcome_side,
                float(price or 0.0),
                shares,
                now,
            ),
        )
        return fill_id

    def _is_external_sync_fill(self, fill: dict) -> bool:
        fill_id = str(fill.get("fill_id") or "").strip().lower()
        order_id = str(fill.get("order_id") or "").strip().lower()
        return fill_id.startswith("ext_sync_") or order_id == "external_manual"

    def _collapse_consecutive_external_sync_fills(self, fills):
        grouped = {}
        for fill in fills:
            tid = fill.get("token_id")
            token_id = "" if pd.isna(tid) else str(tid or "")
            condition_id = fill.get("condition_id")
            outcome_side = fill.get("outcome_side")
            key = f"{token_id}|{condition_id or ''}|{outcome_side or ''}"
            grouped.setdefault(key, []).append(fill)

        collapsed = []
        for key_fills in grouped.values():
            pending_external_sync = None
            for fill in key_fills:
                if self._is_external_sync_fill(fill):
                    # Keep only the latest external-sync reconciliation in a run with
                    # no intervening real exchange fill. Older synthetic entries are
                    # superseded by the most recent exchange-confirmed share count.
                    pending_external_sync = fill
                    continue
                if pending_external_sync is not None:
                    collapsed.append(pending_external_sync)
                    pending_external_sync = None
                collapsed.append(fill)
            if pending_external_sync is not None:
                collapsed.append(pending_external_sync)

        def _sort_key(fill):
            side = str(fill.get("side") or fill.get("order_side") or "").strip().upper()
            return (
                str(fill.get("filled_at") or ""),
                0 if side == "BUY" else 1,
                str(fill.get("fill_id") or ""),
            )

        return sorted(collapsed, key=_sort_key)

    def _identify_redundant_external_sync_fill_ids(self, fills):
        grouped = {}
        for fill in fills:
            tid = fill.get("token_id")
            token_id = "" if pd.isna(tid) else str(tid or "")
            condition_id = fill.get("condition_id")
            outcome_side = fill.get("outcome_side")
            key = f"{token_id}|{condition_id or ''}|{outcome_side or ''}"
            grouped.setdefault(key, []).append(fill)

        redundant_ids = []
        for key_fills in grouped.values():
            pending_external_syncs = []
            for fill in key_fills:
                if self._is_external_sync_fill(fill):
                    pending_external_syncs.append(fill)
                    continue
                if len(pending_external_syncs) > 1:
                    redundant_ids.extend(
                        str(item.get("fill_id") or "")
                        for item in pending_external_syncs[:-1]
                        if str(item.get("fill_id") or "").strip()
                    )
                pending_external_syncs = []
            if len(pending_external_syncs) > 1:
                redundant_ids.extend(
                    str(item.get("fill_id") or "")
                    for item in pending_external_syncs[:-1]
                    if str(item.get("fill_id") or "").strip()
                )
        return redundant_ids

    def archive_and_prune_redundant_external_sync_fills(self, archive_dir=None, vacuum=False):
        fills = self.db.query_all(
            """
            SELECT
                f.fill_id,
                f.order_id,
                f.token_id,
                f.condition_id,
                f.outcome_side,
                f.side,
                f.price,
                f.size,
                f.filled_at
            FROM fills f
            ORDER BY
                COALESCE(f.filled_at, ''),
                CASE WHEN UPPER(COALESCE(f.side, '')) = 'BUY' THEN 0 ELSE 1 END,
                f.fill_id
            """
        )
        redundant_ids = self._identify_redundant_external_sync_fill_ids(fills)
        if not redundant_ids:
            return {
                "archived_rows": 0,
                "deleted_rows": 0,
                "archive_csv": None,
                "archive_jsonl": None,
            }

        archive_root = Path(archive_dir or (self.logs_dir / "archives"))
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_csv = archive_root / f"ext_sync_redundant_archive_{stamp}.csv"
        archive_jsonl = archive_root / f"ext_sync_redundant_archive_{stamp}.jsonl"

        placeholders = ",".join("?" for _ in redundant_ids)
        archived_rows = self.db.query_all(
            f"""
            SELECT fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at
            FROM fills
            WHERE fill_id IN ({placeholders})
            ORDER BY filled_at, fill_id
            """,
            tuple(redundant_ids),
        )
        archived_df = pd.DataFrame(archived_rows)
        archived_df.to_csv(archive_csv, index=False)
        archived_df.to_json(archive_jsonl, orient="records", lines=True)

        cursor = self.db.conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                f"DELETE FROM fills WHERE fill_id IN ({placeholders})",
                tuple(redundant_ids),
            )
            try:
                cursor.execute(
                    f"DELETE FROM external_position_syncs WHERE fill_id IN ({placeholders})",
                    tuple(redundant_ids),
                )
            except Exception:
                pass
            self.db.conn.commit()
        except Exception:
            self.db.conn.rollback()
            raise

        self.rebuild_from_db(force=True)
        if bool(vacuum):
            try:
                self.db.conn.execute("VACUUM")
            except Exception:
                pass

        return {
            "archived_rows": int(len(archived_rows)),
            "deleted_rows": int(len(redundant_ids)),
            "archive_csv": str(archive_csv),
            "archive_jsonl": str(archive_jsonl),
        }

    def _record_external_sync_event(
        self,
        cursor,
        *,
        position_key,
        token_id,
        condition_id,
        outcome_side,
        sync_type,
        local_shares_before,
        exchange_shares,
        delta_shares,
        avg_entry_price,
        fill_id,
        observed_at,
    ):
        sync_id = f"sync_{uuid.uuid4().hex[:12]}"
        cursor.execute(
            """
            INSERT INTO external_position_syncs
            (sync_id, position_key, token_id, condition_id, outcome_side, sync_type, local_shares_before, exchange_shares, delta_shares, avg_entry_price, fill_id, observed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sync_id,
                position_key,
                token_id,
                condition_id,
                outcome_side,
                sync_type,
                float(local_shares_before or 0.0),
                float(exchange_shares or 0.0),
                float(delta_shares or 0.0),
                float(avg_entry_price or 0.0),
                fill_id,
                observed_at,
            ),
        )

    def _has_recent_external_sync(
        self,
        cursor,
        *,
        position_key,
        sync_type,
        local_shares_before,
        exchange_shares,
        delta_shares,
    ) -> bool:
        dedupe_seconds = max(1, int(os.getenv("EXTERNAL_SYNC_DEDUPE_SECONDS", "180") or 180))
        cutoff = (datetime.now(timezone.utc) - pd.Timedelta(seconds=dedupe_seconds)).isoformat()
        tol = 1e-6
        rows = cursor.execute(
            """
            SELECT sync_id
            FROM external_position_syncs
            WHERE position_key = ?
              AND sync_type = ?
              AND ABS(COALESCE(exchange_shares, 0) - ?) <= ?
              AND ABS(COALESCE(delta_shares, 0) - ?) <= ?
              AND COALESCE(observed_at, '') >= ?
            ORDER BY observed_at DESC
            LIMIT 1
            """,
            (
                position_key,
                sync_type,
                float(exchange_shares or 0.0),
                tol,
                float(delta_shares or 0.0),
                tol,
                cutoff,
            ),
        ).fetchall()
        return bool(rows)

    def _has_correcting_sell_fill(self, cursor, *, token_id, delta_shares) -> bool:
        """Return True if a correcting ext_sync SELL fill already exists for this
        token_id with a size close to delta_shares. This is used to decide whether
        rebuild_from_db() will already produce the correct share count without
        writing another fill."""
        tol = max(1e-4, abs(delta_shares) * 0.01)  # 1% tolerance
        rows = cursor.execute(
            """
            SELECT fill_id FROM fills
            WHERE token_id = ?
              AND UPPER(COALESCE(side, '')) = 'SELL'
              AND (fill_id LIKE 'ext_sync_%' OR order_id = 'external_manual')
              AND ABS(COALESCE(size, 0) - ?) <= ?
            LIMIT 1
            """,
            (token_id, float(delta_shares or 0.0), float(tol)),
        ).fetchall()
        return bool(rows)

    def _is_synthetic_fill_id(self, fill_id) -> bool:
        fill_id = str(fill_id or "").strip().lower()
        return (
            fill_id.startswith("fill_dust_clear_")
            or fill_id.startswith("fill_ext_sync_")
            or fill_id.startswith("ext_sync_")
            or "dust_clear" in fill_id
        )

    def _load_latest_external_syncs(self):
        rows = self.db.query_all(
            """
            SELECT s.position_key, s.exchange_shares, s.sync_type, s.observed_at
            FROM external_position_syncs s
            JOIN (
                SELECT position_key, MAX(observed_at) AS observed_at
                FROM external_position_syncs
                GROUP BY position_key
            ) latest
              ON latest.position_key = s.position_key
             AND latest.observed_at = s.observed_at
            """
        )
        return {str(row.get("position_key") or ""): row for row in rows}

    def _load_known_local_order_ids(self) -> set[str]:
        rows = self.db.query_all(
            """
            SELECT order_id
            FROM orders
            WHERE order_id IS NOT NULL
            """
        )
        known_ids = set()
        for row in rows:
            order_id = str(row.get("order_id") or "").strip()
            if order_id:
                known_ids.add(order_id)
        return known_ids

    def _fill_is_rebuild_eligible(self, fill: dict, known_local_order_ids: set[str]) -> bool:
        fill_id = str(fill.get("fill_id") or "").strip()
        order_id = str(fill.get("order_id") or "").strip()
        side = str(fill.get("side") or fill.get("order_side") or "").strip().upper()
        token_id = str(fill.get("token_id") or "").strip()

        if self._is_synthetic_fill_id(fill_id):
            return True
        if order_id == "external_manual":
            return True
        if order_id and order_id in known_local_order_ids:
            return True
        if not known_local_order_ids and token_id and side in {"BUY", "SELL"}:
            return True
        return False

    def _sync_should_override_local_book(self, sync: dict, row: dict) -> bool:
        if not sync:
            return False
        try:
            observed_at = pd.to_datetime(sync.get("observed_at"), utc=True, errors="coerce")
            last_buy_at = pd.to_datetime(row.get("last_buy_at"), utc=True, errors="coerce")
            exchange_shares = float(sync.get("exchange_shares") or 0.0)
        except Exception:
            return False

        sync_type = str(sync.get("sync_type") or "").strip().lower()
        local_shares = float(row.get("shares") or 0.0)
        local_avg_entry = float(row.get("avg_entry_price") or 0.0)
        local_notional = local_shares * max(local_avg_entry, 0.0)

        # Certain exchange-synced outcomes are authoritative and must survive rebuilds,
        # even if local fill timestamps were written later during reconciliation/import.
        authoritative_zero_sync = sync_type in {"dead_orderbook_close", "full_close"} and exchange_shares <= 1e-9
        authoritative_near_zero_sync = exchange_shares <= 0.01 and local_notional >= 0.01
        if authoritative_zero_sync or authoritative_near_zero_sync:
            return True

        if pd.notna(last_buy_at) and pd.notna(observed_at) and last_buy_at > observed_at:
            return False
        return exchange_shares < local_shares - 1e-6

    def rebuild_from_db(self, *, force: bool = False):
        now = time.monotonic()
        if not force and (now - self._last_rebuild_ts) < self._rebuild_cooldown_seconds:
            return getattr(self, "_last_rebuild_result", pd.DataFrame())
        self._last_rebuild_ts = now
        known_local_order_ids = self._load_known_local_order_ids()
        fills = self.db.query_all(
            """
            SELECT
                f.fill_id,
                f.order_id,
                f.token_id,
                f.price,
                f.size,
                f.filled_at,
                COALESCE(f.condition_id, o.condition_id) AS condition_id,
                COALESCE(f.outcome_side, o.outcome_side) AS outcome_side,
                COALESCE(f.side, o.order_side) AS order_side
            FROM fills f
            LEFT JOIN orders o ON o.order_id = f.order_id
            ORDER BY
                COALESCE(f.filled_at, ''),
                CASE WHEN UPPER(COALESCE(f.side, o.order_side, '')) = 'BUY' THEN 0 ELSE 1 END,
                f.fill_id
            """
        )
        fills = self._collapse_consecutive_external_sync_fills(fills)
        books = {}
        skipped_untracked_fills = 0
        for fill in fills:
            fill_id = str(fill.get("fill_id") or "")
            # Synthetic dust-clear fills are internal bookkeeping artifacts.
            # They are useful for local logs, but should not drive live position
            # reconstruction from exchange-synced fills.
            if "dust_clear" in fill_id:
                continue
            if not self._fill_is_rebuild_eligible(fill, known_local_order_ids):
                skipped_untracked_fills += 1
                continue
            tid = fill.get("token_id"); token_id = "" if pd.isna(tid) else str(tid or "") # BUG FIX 10
            condition_id = fill.get("condition_id")
            outcome_side = fill.get("outcome_side")
            side = str(fill.get("order_side") or "").upper()
            shares = float(fill.get("size") or 0.0)
            price = float(fill.get("price") or 0.0)
            filled_at = fill.get("filled_at")
            if not token_id or shares <= 0:
                continue

            key = f"{token_id}|{condition_id or ''}|{outcome_side or ''}"
            book = books.setdefault(
                key,
                {
                    "position_key": key,
                    "token_id": token_id,
                    "condition_id": condition_id,
                    "outcome_side": outcome_side,
                    "shares": 0.0,
                    "avg_entry_price": 0.0,
                    "realized_pnl": 0.0,
                    "last_fill_at": filled_at,
                    "last_buy_at": None,
                    "source": "fills_reconciled",
                    "status": "OPEN",
                },
            )
            book["last_fill_at"] = filled_at or book.get("last_fill_at")

            if side == "BUY":
                prev_shares = float(book["shares"])
                new_shares = prev_shares + shares
                if new_shares > 0:
                    book["avg_entry_price"] = ((prev_shares * float(book["avg_entry_price"])) + (shares * price)) / new_shares
                book["shares"] = new_shares
                book["last_buy_at"] = filled_at or book.get("last_buy_at")
                if filled_at is None:
                    logger.debug(
                        "rebuild_from_db: BUY fill %s for token %s has NULL filled_at — "
                        "sync-cap timestamp guard may not fire correctly for this position.",
                        fill_id, token_id,
                    )
            elif side == "SELL":
                shares_closed = min(float(book["shares"]), shares)
                book["realized_pnl"] += shares_closed * (price - float(book["avg_entry_price"]))
                book["shares"] = max(0.0, float(book["shares"]) - shares)
                if book["shares"] <= 0:
                    book["avg_entry_price"] = 0.0
            elif side not in ("BUY", "SELL") and side:
                logger.debug("rebuild_from_db: unrecognised side=%r on fill_id=%s — skipped", side, fill_id)

        if skipped_untracked_fills:
            logger.info(
                "rebuild_from_db: skipped %d fills that were not tied to local orders or synthetic syncs.",
                skipped_untracked_fills,
            )

        latest_syncs = self._load_latest_external_syncs()
        for row in books.values():
            sync = latest_syncs.get(str(row.get("position_key") or ""))
            if not sync:
                continue
            try:
                exchange_shares = float(sync.get("exchange_shares") or 0.0)
            except Exception:
                continue
            if self._sync_should_override_local_book(sync, row):
                row["shares"] = exchange_shares
                if exchange_shares <= 1e-9:
                    row["avg_entry_price"] = 0.0
                row["source"] = f"fills_reconciled_external_sync:{str(sync.get('sync_type') or 'unknown').strip().lower()}"

        # Guard: if fills produced no books, do not wipe live_positions.
        # An empty result can come from a transient DB lock or a degenerate fills
        # state — deleting all positions would silently close every open trade.
        if not books:
            logger.warning(
                "rebuild_from_db: fills query returned 0 position books — "
                "skipping live_positions wipe to prevent data loss."
            )
            result = pd.DataFrame(columns=["position_key", "token_id", "condition_id", "outcome_side", "shares", "avg_entry_price", "realized_pnl", "last_fill_at", "source", "status"])
            self._last_rebuild_result = result
            return result

        now = datetime.now(timezone.utc).isoformat()
        dust_notional_threshold = 0.01
        cursor = self.db.conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute("DELETE FROM live_positions")
            for row in books.values():
                remaining_shares = float(row.get("shares") or 0.0)
                avg_entry_price = float(row.get("avg_entry_price") or 0.0)
                remaining_notional = remaining_shares * avg_entry_price
                row["status"] = "OPEN" if (remaining_shares > 1e-5 and remaining_notional >= dust_notional_threshold) else "CLOSED"
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO live_positions
                    (position_key, token_id, condition_id, outcome_side, shares, avg_entry_price, realized_pnl, last_fill_at, source, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["position_key"],
                        row["token_id"],
                        row["condition_id"],
                        row["outcome_side"],
                        row["shares"],
                        row["avg_entry_price"],
                        row["realized_pnl"],
                        row["last_fill_at"],
                        row["source"],
                        row["status"],
                        now,
                    ),
                )
            self.db.conn.commit()
        except Exception:
            self.db.conn.rollback()
            raise

        result = pd.DataFrame(list(books.values())) if books else pd.DataFrame(columns=["position_key", "token_id", "condition_id", "outcome_side", "shares", "avg_entry_price", "realized_pnl", "last_fill_at", "source", "status"])
        self._last_rebuild_result = result
        return result

    def _extract_available_balance(self, payload, execution_client, asset_type="COLLATERAL"):
        if not isinstance(payload, dict): return None
        if "error" in payload or "message" in payload: return None # BUG FIX 1: Prevent API errors from wiping DB
        for key in ["balance", "available", "available_balance", "amount"]:
            if payload.get(key) is not None:
                try:
                    return float(execution_client._normalize_allowance_balance(payload[key], asset_type=asset_type))
                except Exception:
                    try:
                        return float(payload[key])
                    except Exception:
                        return 0.0
        return 0.0

    def _verify_open_positions_against_exchange(self, rows):
        if not rows:
            return rows

        # --- TTL Cache -------------------------------------------------------
        # Calling get_open_positions() multiple times within a single cycle
        # (pre-cycle sync, exit manager, post-cycle telemetry, …) used to fire
        # one HTTP balance-allowance request per position on every call.  The
        # cache makes the exchange verification effectively once-per-TTL.
        ttl = max(10, int(os.getenv("POSITION_VERIFY_CACHE_TTL_SECONDS", "60") or 60))
        token_key = ",".join(sorted(str(r.get("token_id") or "") for r in rows))
        now_mono = time.monotonic()
        if (
            self._verify_cache is not None
            and token_key == self._verify_cache_token_key
            and now_mono - self._verify_cache_ts < ttl
        ):
            logger.debug(
                "Position verify cache hit (%d rows, age=%.1fs)",
                len(self._verify_cache),
                now_mono - self._verify_cache_ts,
            )
            return self._verify_cache
        # --- End Cache -------------------------------------------------------

        try:
            from execution_client import ExecutionClient
            execution_client = ExecutionClient()
        except Exception as exc:
            logger.debug("Live position verification unavailable: %s", exc)
            return rows

        verified_rows = []
        now = datetime.now(timezone.utc).isoformat()
        mutated = False
        cursor = self.db.conn.cursor()

        for row in rows:
            token_id = str(row.get("token_id") or "")
            if not token_id:
                continue
            local_shares = float(row.get("shares") or 0.0)
            local_avg_entry = float(row.get("avg_entry_price") or 0.0)
            local_notional = local_shares * local_avg_entry
            if local_shares > 0 and local_notional < 0.01:
                mutated = True
                cursor.execute(
                    "UPDATE live_positions SET status = 'CLOSED', updated_at = ? WHERE position_key = ?",
                    (now, row.get("position_key")),
                )
                continue
            try:
                payload = execution_client.get_balance_allowance(asset_type="CONDITIONAL", token_id=token_id)
                available_shares = self._extract_available_balance(payload, execution_client, asset_type="CONDITIONAL")
                maybe_trace_allowance_payload(
                    logs_dir=self.logs_dir,
                    source="live_position_book.verify",
                    asset_type="CONDITIONAL",
                    token_id=token_id,
                    payload=payload,
                    normalized_balance=available_shares,
                    local_balance=local_shares,
                    note=f"position_key={row.get('position_key')}",
                )
            except Exception as exc:
                logger.debug("Conditional balance verification failed for %s: %s", token_id[:16], exc)
                verified_rows.append(row)
                continue

            position_key = str(row.get("position_key") or f"{token_id}|{row.get('condition_id') or ''}|{row.get('outcome_side') or ''}")
            if available_shares is not None and available_shares <= 1e-9: # BUG FIX 1
                mutated = True
                fill_id = None
                # Check fills table first — if a correcting SELL already exists rebuild_from_db
                # will compute shares=0 without us needing to write another fill.
                sell_already_written = self._has_correcting_sell_fill(
                    cursor, token_id=token_id, delta_shares=local_shares
                )
                if not self._has_recent_external_sync(
                    cursor,
                    position_key=position_key,
                    sync_type="full_close",
                    local_shares_before=local_shares,
                    exchange_shares=0.0,
                    delta_shares=local_shares,
                ) or not sell_already_written:
                    logger.warning(
                        "Closing stale local live position for %s because exchange conditional balance is zero.",
                        token_id,
                    )
                    if not sell_already_written:
                        # Capture avg_entry_price BEFORE it may be zeroed by rebuild — needed for
                        # correct realized PnL on the synthetic SELL fill (avoids recording $0 PnL).
                        entry_price_for_fill = float(row.get("avg_entry_price") or 0.0)
                        # Persist external/manual close as synthetic SELL fill so future
                        # rebuilds do not resurrect stale shares from historical BUY fills.
                        fill_id = self._insert_external_sync_sell_fill(
                            cursor,
                            token_id=token_id,
                            condition_id=row.get("condition_id"),
                            outcome_side=row.get("outcome_side"),
                            price=entry_price_for_fill,
                            shares=local_shares,
                            now=now,
                        )
                    self._record_external_sync_event(
                        cursor,
                        position_key=position_key,
                        token_id=token_id,
                        condition_id=row.get("condition_id"),
                        outcome_side=row.get("outcome_side"),
                        sync_type="full_close",
                        local_shares_before=local_shares,
                        exchange_shares=0.0,
                        delta_shares=local_shares,
                        avg_entry_price=row.get("avg_entry_price"),
                        fill_id=fill_id,
                        observed_at=now,
                    )
                else:
                    logger.debug(
                        "Duplicate external full-close for %s already corrected in fills (local=%.6f exchange=0) — updating position only.",
                        token_id,
                        local_shares,
                    )

                cursor.execute(
                    "UPDATE live_positions SET shares = 0, status = 'CLOSED', updated_at = ? WHERE position_key = ?",
                    (now, row.get("position_key")),
                )
                continue

            if available_shares < local_shares - 1e-5: # BUG FIX 5: Permanently save partial external sells to DB
                delta_shares = max(0.0, local_shares - float(available_shares or 0.0))
                if delta_shares > 1e-9:
                    # Check fills table first — if a matching SELL already exists, rebuild_from_db
                    # will naturally produce the correct share count without writing another fill.
                    sell_already_written = self._has_correcting_sell_fill(
                        cursor, token_id=token_id, delta_shares=delta_shares
                    )
                    if not self._has_recent_external_sync(
                        cursor,
                        position_key=position_key,
                        sync_type="partial_close",
                        local_shares_before=local_shares,
                        exchange_shares=available_shares,
                        delta_shares=delta_shares,
                    ) or not sell_already_written:
                        logger.warning(
                            "Reconciling partial external close for %s: local_shares=%.6f exchange_shares=%.6f delta=%.6f",
                            token_id,
                            local_shares,
                            float(available_shares or 0.0),
                            delta_shares,
                        )
                        if not sell_already_written:
                            fill_id = self._insert_external_sync_sell_fill(
                                cursor,
                                token_id=token_id,
                                condition_id=row.get("condition_id"),
                                outcome_side=row.get("outcome_side"),
                                price=float(row.get("avg_entry_price") or 0.0),
                                shares=delta_shares,
                                now=now,
                            )
                        else:
                            fill_id = None
                        self._record_external_sync_event(
                            cursor,
                            position_key=position_key,
                            token_id=token_id,
                            condition_id=row.get("condition_id"),
                            outcome_side=row.get("outcome_side"),
                            sync_type="partial_close",
                            local_shares_before=local_shares,
                            exchange_shares=available_shares,
                            delta_shares=delta_shares,
                            avg_entry_price=row.get("avg_entry_price"),
                            fill_id=fill_id,
                            observed_at=now,
                        )
                    else:
                        logger.debug(
                            "Duplicate partial-close for %s already corrected in fills (local=%.6f exchange=%.6f delta=%.6f) — updating position only.",
                            token_id,
                            local_shares,
                            float(available_shares or 0.0),
                            delta_shares,
                        )
                row["shares"] = available_shares
                new_status = "OPEN"
                try:
                    if float(available_shares or 0.0) * float(local_avg_entry or 0.0) < 0.01:
                        new_status = "CLOSED"
                except Exception:
                    new_status = "OPEN"
                row["status"] = new_status
                cursor.execute(
                    "UPDATE live_positions SET shares = ?, status = ?, updated_at = ? WHERE position_key = ?",
                    (available_shares, new_status, now, row.get("position_key")),
                )
                mutated = True
            else:
                row["shares"] = local_shares
                row["status"] = "OPEN"
            if str(row.get("status") or "").upper() == "OPEN":
                verified_rows.append(row)

        if mutated:
            self.db.conn.commit()
        # Update cache after a real verification run
        self._verify_cache = verified_rows
        self._verify_cache_ts = now_mono
        self._verify_cache_token_key = token_key
        return verified_rows

    def close_dead_token_positions(self, dead_tokens: set) -> int:
        """Safeguard: Force-close any open positions for tokens that are known to be dead (404/no orderbook)."""
        if not dead_tokens:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.db.conn.cursor()
        
        closed_total = 0
        for token_id in dead_tokens:
            rows = cursor.execute(
                """
                SELECT position_key, token_id, condition_id, outcome_side, shares, avg_entry_price
                FROM live_positions
                WHERE status = 'OPEN' AND token_id = ?
                """,
                (token_id,),
            ).fetchall()
            if rows:
                logger.warning("Safeguard: Auto-closing %d live positions for dead token %s (404/no orderbook)", len(rows), token_id)
                for row in rows:
                    row_data = dict(row)
                    local_shares = float(row_data.get("shares") or 0.0)
                    if local_shares > 0:
                        self._record_external_sync_event(
                            cursor,
                            position_key=row_data.get("position_key"),
                            token_id=row_data.get("token_id"),
                            condition_id=row_data.get("condition_id"),
                            outcome_side=row_data.get("outcome_side"),
                            sync_type="dead_orderbook_close",
                            local_shares_before=local_shares,
                            exchange_shares=0.0,
                            delta_shares=local_shares,
                            avg_entry_price=row_data.get("avg_entry_price"),
                            fill_id=None,
                            observed_at=now,
                        )
                cursor.execute(
                    """
                    UPDATE live_positions
                    SET shares = 0.0, status = 'CLOSED', source = 'dead_orderbook_tombstone', updated_at = ?
                    WHERE status = 'OPEN' AND token_id = ?
                    """,
                    (now, token_id),
                )
                closed_total += cursor.rowcount
                
        if closed_total > 0:
            self.db.conn.commit()
            # Invalidate the verification cache so a rebuild isn't stale
            self._verify_cache_ts = 0.0
            self._verify_cache = None
            self._verify_cache_token_key = ""
            
        return closed_total

    def get_open_positions(self):
        rebuild_on_read = str(os.getenv("LIVE_POSITION_REBUILD_ON_READ", "true")).strip().lower() in {"1", "true", "yes", "on"}
        if rebuild_on_read:
            try:
                self.rebuild_from_db()
            except Exception as exc:
                logger.debug("LivePositionBook rebuild_on_read failed: %s", exc)
        rows = self.db.query_all(
            """
            SELECT position_key, token_id, condition_id, outcome_side, shares, avg_entry_price, realized_pnl, last_fill_at, source, status, updated_at
            FROM live_positions
            WHERE status = 'OPEN' AND shares > 0
            ORDER BY COALESCE(last_fill_at, '') DESC
            """
        )
        rows = self._verify_open_positions_against_exchange(rows)
        rows = self._merge_profile_snapshot_rows(rows)
        return pd.DataFrame(rows)

    def get_enriched_open_positions(self, scored_df=None, fallback_df=None):
        live_df = self.get_open_positions()
        if live_df.empty:
            return live_df

        enriched = live_df.copy()
        if fallback_df is not None and not fallback_df.empty and "token_id" in fallback_df.columns:
            fallback = fallback_df.copy()
            fallback["token_id"] = fallback["token_id"].astype(str)
            enriched["token_id"] = enriched["token_id"].astype(str)
            enriched = enriched.merge(fallback, on="token_id", how="left", suffixes=("", "_local"))
            if "entry_price" not in enriched.columns:
                enriched["entry_price"] = enriched["avg_entry_price"]
            else:
                enriched["entry_price"] = enriched["entry_price"].fillna(enriched["avg_entry_price"])

        if scored_df is not None and not scored_df.empty and "token_id" in scored_df.columns:
            market_cols = [c for c in ["token_id", "market_title", "market", "current_price", "best_bid", "best_ask", "confidence", "spread", "bid_size"] if c in scored_df.columns]
            market_df = scored_df[market_cols].copy().drop_duplicates(subset=["token_id"])
            market_df["token_id"] = market_df["token_id"].astype(str)
            enriched["token_id"] = enriched["token_id"].astype(str)
            enriched = enriched.merge(market_df, on="token_id", how="left", suffixes=("", "_market"))

        if "entry_price" not in enriched.columns:
            enriched["entry_price"] = enriched["avg_entry_price"]
        return enriched
