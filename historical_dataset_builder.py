import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class HistoricalDatasetBuilder:
    """
    Consolidates project logs into a single ML-friendly historical dataset.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.logs_dir / "historical_dataset.csv"

    def _safe_read(self, filename):
        path = self.logs_dir / filename
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def build(self):
        signals_df = self._safe_read("signals.csv")
        trades_df = self._safe_read("daily_summary.txt")
        markets_df = self._safe_read("markets.csv")
        alerts_df = self._safe_read("alerts.csv")
        wallet_alpha_df = self._safe_read("wallet_alpha.csv")
        wallet_alpha_history_df = self._safe_read("wallet_alpha_history.csv")
        btc_targets_df = self._safe_read("btc_targets.csv")

        source_df = raw_candidates_df if not raw_candidates_df.empty else signals_df
        if source_df.empty:
            return pd.DataFrame()

        dataset = source_df.copy()

        rename_map = {
            "market": "market_title",
            "wallet_copied": "trader_wallet",
            "price": "entry_price",
            "side": "outcome_side",
        }
        dataset = dataset.rename(columns={k: v for k, v in rename_map.items() if k in dataset.columns})

        if "timestamp" not in dataset.columns:
            dataset["timestamp"] = pd.NaT

        if not trades_df.empty and all(c in dataset.columns for c in ["market_title", "trader_wallet", "timestamp"]):
            trades_df = trades_df.copy()
            if "market" in trades_df.columns and "market_title" not in trades_df.columns:
                trades_df["market_title"] = trades_df["market"]
            if "wallet_copied" in trades_df.columns and "trader_wallet" not in trades_df.columns:
                trades_df["trader_wallet"] = trades_df["wallet_copied"]
            if "timestamp" in trades_df.columns and all(c in trades_df.columns for c in ["market_title", "trader_wallet"]):
                dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce", format="mixed")
                trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], utc=True, errors="coerce", format="mixed")
                dataset = pd.merge_asof(
                    dataset.sort_values("timestamp"),
                    trades_df.sort_values("timestamp"),
                    on="timestamp",
                    by=["market_title", "trader_wallet"],
                    direction="backward",
                )

        if not markets_df.empty and "market_title" in dataset.columns and "question" in markets_df.columns:
            if "timestamp" in dataset.columns and "timestamp" in markets_df.columns:
                dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce", format="mixed")
                markets_df["timestamp"] = pd.to_datetime(markets_df["timestamp"], utc=True, errors="coerce", format="mixed")
                merged_parts = []
                for market_title, group in dataset.groupby("market_title"):
                    market_history = markets_df[markets_df["question"] == market_title]
                    if market_history.empty:
                        merged_parts.append(group)
                        continue
                    merged_parts.append(
                        pd.merge_asof(
                            group.sort_values("timestamp"),
                            market_history[[c for c in market_history.columns if c in ["timestamp", "question", "liquidity", "volume", "last_trade_price", "url", "best_bid", "best_ask", "slug", "condition_id", "end_date"]]].sort_values("timestamp"),
                            on="timestamp",
                            direction="backward",
                        )
                    )
                dataset = pd.concat(merged_parts, ignore_index=True) if merged_parts else dataset
            else:
                latest_markets = markets_df.drop_duplicates(subset=["question"], keep="last")
                dataset = dataset.merge(
                    latest_markets[[c for c in latest_markets.columns if c in ["question", "liquidity", "volume", "last_trade_price", "url", "best_bid", "best_ask", "slug", "condition_id", "end_date"]]],
                    left_on="market_title",
                    right_on="question",
                    how="left",
                )

        if not alerts_df.empty and "market_title" in dataset.columns and "market" in alerts_df.columns:
            alert_counts = alerts_df.groupby("market").size().reset_index(name="alert_count")
            dataset = dataset.merge(alert_counts, left_on="market_title", right_on="market", how="left")
            dataset["alert_count"] = dataset["alert_count"].fillna(0).astype(int)

        if not wallet_alpha_df.empty and "trader_wallet" in dataset.columns and "wallet_copied" in wallet_alpha_df.columns:
            dataset = dataset.merge(wallet_alpha_df, left_on="trader_wallet", right_on="wallet_copied", how="left")

        if not wallet_alpha_history_df.empty and "trader_wallet" in dataset.columns and "timestamp" in dataset.columns:
            dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce", format="mixed")
            wallet_alpha_history_df["timestamp"] = pd.to_datetime(wallet_alpha_history_df["timestamp"], utc=True, errors="coerce", format="mixed")
            dataset = dataset.sort_values(["trader_wallet", "timestamp"])
            wallet_alpha_history_df = wallet_alpha_history_df.sort_values(["wallet_copied", "timestamp"])
            merged_parts = []
            for wallet, group in dataset.groupby("trader_wallet"):
                history = wallet_alpha_history_df[wallet_alpha_history_df["wallet_copied"] == wallet]
                if history.empty:
                    merged_parts.append(group)
                    continue
                merged_parts.append(pd.merge_asof(group.sort_values("timestamp"), history.sort_values("timestamp"), on="timestamp", direction="backward"))
            dataset = pd.concat(merged_parts, ignore_index=True) if merged_parts else dataset

        if not btc_targets_df.empty and "timestamp" in dataset.columns:
            btc_targets_df["timestamp"] = pd.to_datetime(btc_targets_df["timestamp"], utc=True, errors="coerce", format="mixed")
            dataset = pd.merge_asof(
                dataset.sort_values("timestamp"),
                btc_targets_df[[c for c in btc_targets_df.columns if c in ["timestamp", "btc_price", "btc_spot_return_5m", "btc_spot_return_15m", "btc_realized_vol_15m", "btc_volume_proxy"]]].sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )

        if "best_ask" in dataset.columns and "best_bid" in dataset.columns:
            dataset["spread"] = (dataset["best_ask"].fillna(0) - dataset["best_bid"].fillna(0)).abs()
        if "end_date" in dataset.columns and "timestamp" in dataset.columns:
            dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce", format="mixed")
            dataset["end_date"] = pd.to_datetime(dataset["end_date"], utc=True, errors="coerce", format="mixed")
            dataset["time_to_close_minutes"] = (dataset["end_date"] - dataset["timestamp"]).dt.total_seconds().div(60)

        return dataset

    def write(self):
        dataset = self.build()
        if dataset.empty:
            return dataset

        dataset.to_csv(self.output_file, index=False)
        logging.info("Saved historical dataset to %s", self.output_file)
        return dataset

