import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TraderAnalytics:
    """
    Builds public-wallet performance analytics from tracked signal/trade logs.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.logs_dir / "trader_analytics.csv"

    def build(self, signals_df: pd.DataFrame, trades_df: pd.DataFrame | None = None):
        if signals_df is None or signals_df.empty:
            return pd.DataFrame()

        signals_df = signals_df.copy()
        if "confidence" in signals_df.columns:
            signals_df["confidence"] = pd.to_numeric(signals_df["confidence"], errors="coerce")

        base = (
            signals_df.groupby("wallet_copied")
            .agg(
                ranked_signals=("wallet_copied", "size"),
                avg_confidence=("confidence", "mean"),
                top_signal_count=("signal_label", lambda x: int((x == "HIGHEST-RANKED PAPER SIGNAL").sum())),
                unique_markets=("market", "nunique"),
            )
            .reset_index()
            .sort_values(by=["avg_confidence", "ranked_signals"], ascending=[False, False])
        )

        if trades_df is not None and not trades_df.empty and "wallet_copied" in trades_df.columns:
            trades_df = trades_df.copy()
            if "fill_price" in trades_df.columns:
                trades_df["fill_price"] = pd.to_numeric(trades_df["fill_price"], errors="coerce")
            trade_counts = (
                trades_df.groupby("wallet_copied")
                .agg(paper_trades=("wallet_copied", "size"), avg_fill_price=("fill_price", "mean"))
                .reset_index()
            )
            base = base.merge(trade_counts, on="wallet_copied", how="left")
        else:
            base["paper_trades"] = 0
            base["avg_fill_price"] = None

        base["paper_trades"] = base["paper_trades"].fillna(0).astype(int)
        return base

    def write(self, signals_df: pd.DataFrame, trades_df: pd.DataFrame | None = None):
        analytics_df = self.build(signals_df, trades_df)
        if analytics_df.empty:
            return analytics_df

        analytics_df.to_csv(self.output_file, index=False)
        logging.info("Saved trader analytics to %s", self.output_file)
        return analytics_df

