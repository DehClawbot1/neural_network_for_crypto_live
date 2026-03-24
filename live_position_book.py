from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from db import Database


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
                o.condition_id,
                o.outcome_side,
                o.order_side
            FROM fills f
            LEFT JOIN orders o ON o.order_id = f.order_id
            ORDER BY COALESCE(f.filled_at, ''), f.fill_id
            """
        )
        books = {}
        for fill in fills:
            token_id = str(fill.get("token_id") or "")
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

        self.db.execute("DELETE FROM live_positions")
        now = datetime.now(timezone.utc).isoformat()
        for row in books.values():
            row["status"] = "OPEN" if float(row["shares"]) > 0 else "CLOSED"
            self.db.execute(
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
        return pd.DataFrame(list(books.values()))

    def get_open_positions(self):
        rows = self.db.query_all(
            """
            SELECT token_id, condition_id, outcome_side, shares, avg_entry_price, realized_pnl, last_fill_at, source, status, updated_at
            FROM live_positions
            WHERE status = 'OPEN' AND shares > 0
            ORDER BY COALESCE(last_fill_at, '') DESC
            """
        )
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
