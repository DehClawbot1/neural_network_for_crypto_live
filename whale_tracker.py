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

        df = signals_df.copy()
        for col in ["price", "size", "alpha_score", "wallet_alpha_30d", "profit", "net_pnl", "realized_pnl"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        time_col = "timestamp" if "timestamp" in df.columns else "updated_at" if "updated_at" in df.columns else None
        profit_col = "profit" if "profit" in df.columns else "net_pnl" if "net_pnl" in df.columns else "realized_pnl" if "realized_pnl" in df.columns else None
        alpha_col = "alpha_score" if "alpha_score" in df.columns else "wallet_alpha_30d" if "wallet_alpha_30d" in df.columns else None

        grouped = (
            df.groupby("trader_wallet")
            .agg(
                trade_count=("trader_wallet", "size"),
                avg_price=("price", "mean"),
                avg_size=("size", "mean"),
                unique_markets=("market_title", "nunique"),
            )
            .reset_index()
            .sort_values(by=["trade_count", "avg_size"], ascending=[False, False])
        )

        top_market = (
            df.groupby(["trader_wallet", "market_title"]).size().reset_index(name="market_signal_count")
            .sort_values(["trader_wallet", "market_signal_count", "market_title"], ascending=[True, False, True])
            .drop_duplicates(subset=["trader_wallet"], keep="first")
            .rename(columns={"market_title": "market"})[["trader_wallet", "market", "market_signal_count"]]
        )
        grouped = grouped.merge(top_market, on="trader_wallet", how="left")
        grouped["top_market"] = grouped.get("market")

        if alpha_col is not None:
            alpha_df = df.groupby("trader_wallet")[alpha_col].mean().reset_index(name="alpha_score")
            grouped = grouped.merge(alpha_df, on="trader_wallet", how="left")
        else:
            grouped["alpha_score"] = pd.NA

        if profit_col is not None:
            profit_df = df.groupby("trader_wallet")[profit_col].sum().reset_index(name="profit")
            grouped = grouped.merge(profit_df, on="trader_wallet", how="left")
        else:
            grouped["profit"] = pd.NA

        if time_col is not None:
            latest_df = df.groupby("trader_wallet")[time_col].max().reset_index(name="timestamp")
            grouped = grouped.merge(latest_df, on="trader_wallet", how="left")
        else:
            grouped["timestamp"] = pd.NA

        return grouped

    def market_distribution(self, signals_df: pd.DataFrame):
        if signals_df is None or signals_df.empty:
            return pd.DataFrame()

        df = signals_df.copy()
        for col in ["price", "size"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
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
