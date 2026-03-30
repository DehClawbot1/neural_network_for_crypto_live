from __future__ import annotations

import pandas as pd


class LivePnLCalculator:
    def enrich_positions(self, positions_df: pd.DataFrame | None):
        if positions_df is None or positions_df.empty:
            return pd.DataFrame() if positions_df is None else positions_df

        df = positions_df.copy()
        for col, default in [
            ("shares", 0.0),
            ("avg_entry_price", 0.0),
            ("realized_pnl", 0.0),
            ("current_price", 0.0),
            ("best_bid", None),
        ]:
            if col not in df.columns:
                df[col] = default

        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
        df["avg_entry_price"] = pd.to_numeric(df["avg_entry_price"], errors="coerce").fillna(0.0)
        df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
        df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce").fillna(0.0)

        if "best_bid" in df.columns:
            best_bid = pd.to_numeric(df["best_bid"], errors="coerce")
            df["mark_price"] = best_bid.where(best_bid.notna() & (best_bid > 0), df["current_price"])
        else:
            df["mark_price"] = df["current_price"]

        df["entry_price"] = pd.to_numeric(df.get("entry_price", df["avg_entry_price"]), errors="coerce").fillna(df["avg_entry_price"])
        df["market_value"] = df["shares"] * df["mark_price"]
        df["unrealized_pnl"] = df["shares"] * (df["mark_price"] - df["avg_entry_price"])
        df["total_pnl"] = df["realized_pnl"] + df["unrealized_pnl"]
        df["pnl_source"] = "live_reconciled"
        return df

    def summarize_portfolio(self, positions_df: pd.DataFrame | None):
        enriched = self.enrich_positions(positions_df)
        if enriched is None or enriched.empty:
            return {
                "open_positions": 0,
                "gross_market_value": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_pnl": 0.0,
                "pnl_source": "live_reconciled",
            }
        return {
            "open_positions": int(len(enriched)),
            "gross_market_value": float(enriched["market_value"].sum()),
            "realized_pnl": float(enriched["realized_pnl"].sum()),
            "unrealized_pnl": float(enriched["unrealized_pnl"].sum()),
            "total_pnl": float(enriched["total_pnl"].sum()),
            "pnl_source": "live_reconciled",
        }
