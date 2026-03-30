from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from db import Database

logger = logging.getLogger(__name__)


class LivePositionBook:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.logs_dir / "trading.db")

    def rebuild_from_db(self):
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
            ORDER BY COALESCE(f.filled_at, ''), f.fill_id
            """
        )
        books = {}
        for fill in fills:
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
            elif side == "SELL":
                shares_closed = min(float(book["shares"]), shares)
                book["realized_pnl"] += shares_closed * (price - float(book["avg_entry_price"]))
                book["shares"] = max(0.0, float(book["shares"]) - shares)
                if book["shares"] <= 0:
                    book["avg_entry_price"] = 0.0

        now = datetime.now(timezone.utc).isoformat()
        cursor = self.db.conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute("DELETE FROM live_positions")
            for row in books.values():
                row["status"] = "OPEN" if float(row["shares"]) > 1e-5 else "CLOSED" # BUG FIX 6: Prevent dust re-opening
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

        return pd.DataFrame(list(books.values())) if books else pd.DataFrame(columns=["position_key", "token_id", "condition_id", "outcome_side", "shares", "avg_entry_price", "realized_pnl", "last_fill_at", "source", "status"]) # BUG FIX 4 if books else pd.DataFrame(columns=["position_key", "token_id", "condition_id", "outcome_side", "shares", "avg_entry_price", "realized_pnl", "last_fill_at", "source", "status"]) # BUG FIX 4

    def _extract_available_balance(self, payload, execution_client):
        if not isinstance(payload, dict): return None
        if "error" in payload or "message" in payload: return None # BUG FIX 1: Prevent API errors from wiping DB
        for key in ["balance", "available", "available_balance", "amount"]:
            if payload.get(key) is not None:
                try:
                    return float(execution_client._normalize_usdc_balance(payload[key]))
                except Exception:
                    try:
                        return float(payload[key])
                    except Exception:
                        return 0.0
        return 0.0

    def _verify_open_positions_against_exchange(self, rows):
        if not rows:
            return rows

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
            try:
                payload = execution_client.get_balance_allowance(asset_type="CONDITIONAL", token_id=token_id)
                available_shares = self._extract_available_balance(payload, execution_client)
            except Exception as exc:
                logger.debug("Conditional balance verification failed for %s: %s", token_id[:16], exc)
                verified_rows.append(row)
                continue

            if available_shares is not None and available_shares <= 1e-9: # BUG FIX 1
                mutated = True
                logger.warning(
                    "Closing stale local live position for %s because exchange conditional balance is zero.",
                    token_id,
                )
                # --- PATCH: Inject synthetic SELL fill to permanently prevent resurrection ---
                local_shares = float(row.get("shares") or 0.0)
                if local_shares > 0:
                    import uuid
                    fake_fill_id = f"ext_sync_{uuid.uuid4().hex[:8]}"
                    cursor.execute(
                        """
                        INSERT INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at)
                        VALUES (?, 'external_manual', ?, ?, ?, 'SELL', ?, ?, ?)
                        """,
                        (
                            fake_fill_id, 
                            token_id, 
                            row.get("condition_id"), 
                            row.get("outcome_side"), 
                            float(row.get("avg_entry_price") or 0.0), 
                            local_shares, 
                            now
                        )
                    )
                # -------------------------------------------------------------------------
                
                cursor.execute(
                    "UPDATE live_positions SET shares = 0, status = 'CLOSED', updated_at = ? WHERE position_key = ?",
                    (now, row.get("position_key")),
                )
                continue

            local_shares = float(row.get("shares") or 0.0)
            if available_shares < local_shares - 1e-5: # BUG FIX 5: Permanently save partial external sells to DB
                row["shares"] = available_shares
                cursor.execute("UPDATE live_positions SET shares = ?, updated_at = ? WHERE position_key = ?", (available_shares, now, row.get("position_key")))
                mutated = True
            else:
                row["shares"] = local_shares
            verified_rows.append(row)

        if mutated:
            self.db.conn.commit()
        return verified_rows

    def get_open_positions(self):
        rows = self.db.query_all(
            """
            SELECT position_key, token_id, condition_id, outcome_side, shares, avg_entry_price, realized_pnl, last_fill_at, source, status, updated_at
            FROM live_positions
            WHERE status = 'OPEN' AND shares > 0
            ORDER BY COALESCE(last_fill_at, '') DESC
            """
        )
        rows = self._verify_open_positions_against_exchange(rows)
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
