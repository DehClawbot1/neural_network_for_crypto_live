import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class WhaleTracker:
    """
    Tracks public wallet activity summaries from scraped signal data.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.logs_dir / "whales.csv"
        self.distribution_file = self.logs_dir / "market_distribution.csv"

    def summarize(self, signals_df: pd.DataFrame):
        if signals_df is None or signals_df.empty:
            return pd.DataFrame()

        grouped = (
            signals_df.groupby("trader_wallet")
            .agg(
                trade_count=("trader_wallet", "size"),
                avg_price=("price", "mean"),
                avg_size=("size", "mean"),
                unique_markets=("market_title", "nunique"),
            )
            .reset_index()
            .sort_values(by=["trade_count", "avg_size"], ascending=[False, False])
        )
        return grouped

    def market_distribution(self, signals_df: pd.DataFrame):
        if signals_df is None or signals_df.empty:
            return pd.DataFrame()

        df = signals_df.copy()
        distribution = (
            df.groupby("market_title")
            .agg(
                unique_wallets=("trader_wallet", "nunique"),
                signal_count=("market_title", "size"),
                avg_price=("price", "mean"),
                avg_size=("size", "mean"),
            )
            .reset_index()
            .sort_values(by=["unique_wallets", "signal_count"], ascending=[False, False])
        )
        return distribution

    def write_summary(self, signals_df: pd.DataFrame):
        summary_df = self.summarize(signals_df)
        distribution_df = self.market_distribution(signals_df)
        if summary_df.empty:
            return summary_df

        summary_df.to_csv(self.output_file, index=False)
        logging.info("Saved whale summary to %s", self.output_file)

        if not distribution_df.empty:
            distribution_df.to_csv(self.distribution_file, index=False)
            logging.info("Saved market distribution to %s", self.distribution_file)

        return summary_df
