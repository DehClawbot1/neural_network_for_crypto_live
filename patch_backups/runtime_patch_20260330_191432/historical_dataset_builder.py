import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class HistoricalDatasetBuilder:
    """
    Consolidate project logs into a single ML-friendly historical dataset.

    Fixes:
    - prefer execution_log.csv over daily_summary.txt
    - use safer merges that do not depend on fragile dynamic key-list lengths
    - normalize timestamps before merge_asof
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

    def _safe_merge_asof(self, left, right, on, by=None):
        if left.empty or right.empty or on not in left.columns or on not in right.columns:
            return left
        work_left = left.copy().sort_values(on)
        work_right = right.copy().sort_values(on)
        return pd.merge_asof(work_left, work_right, on=on, by=by, direction="backward")

    def build(self):
        signals_df = self._safe_read("signals.csv")
        trades_df = self._safe_read("execution_log.csv")
        if trades_df.empty:
            trades_df = self._safe_read("daily_summary.txt")
        markets_df = self._safe_read("markets.csv")
        alerts_df = self._safe_read("alerts.csv")
        wallet_alpha_df = self._safe_read("wallet_alpha.csv")
        wallet_alpha_history_df = self._safe_read("wallet_alpha_history.csv")
        btc_targets_df = self._safe_read("btc_targets.csv")

        if signals_df.empty:
            return pd.DataFrame()

        dataset = signals_df.copy()
        rename_map = {
            "market": "market_title",
            "wallet_copied": "trader_wallet",
            "price": "entry_price",
            "side": "outcome_side",
        }
        dataset = dataset.rename(columns={k: v for k, v in rename_map.items() if k in dataset.columns})
        if "timestamp" not in dataset.columns:
            dataset["timestamp"] = pd.NaT
        dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce")

        if not trades_df.empty:
            trade_cols = [c for c in ["timestamp", "market", "wallet_copied", "fill_price", "size_usdc", "action_type"] if c in trades_df.columns]
            trade_view = trades_df[trade_cols].copy()
            if "timestamp" in trade_view.columns:
                trade_view["timestamp"] = pd.to_datetime(trade_view["timestamp"], utc=True, errors="coerce")
            if "market_title" in dataset.columns and "market" in trade_view.columns:
                dataset = dataset.merge(
                    trade_view,
                    left_on=[c for c in ["market_title", "trader_wallet"] if c in dataset.columns],
                    right_on=[c for c in ["market", "wallet_copied"] if c in trade_view.columns],
                    how="left",
                )

        if not markets_df.empty:
            market_name_col = "question" if "question" in markets_df.columns else "market_title" if "market_title" in markets_df.columns else None
            if market_name_col and "market_title" in dataset.columns:
                if "timestamp" in markets_df.columns:
                    markets_df = markets_df.copy()
                    markets_df["timestamp"] = pd.to_datetime(markets_df["timestamp"], utc=True, errors="coerce")
                    merged_parts = []
                    for market_title, group in dataset.groupby("market_title", dropna=False):
                        market_history = markets_df[markets_df[market_name_col] == market_title]
                        if market_history.empty:
                            merged_parts.append(group)
                            continue
                        cols = [c for c in ["timestamp", market_name_col, "liquidity", "volume", "last_trade_price", "url", "best_bid", "best_ask", "slug", "condition_id", "end_date"] if c in market_history.columns]
                        merged_parts.append(pd.merge_asof(group.sort_values("timestamp"), market_history[cols].sort_values("timestamp"), on="timestamp", direction="backward"))
                    dataset = pd.concat(merged_parts, ignore_index=True) if merged_parts else dataset
                else:
                    latest_markets = markets_df.drop_duplicates(subset=[market_name_col], keep="last")
                    cols = [c for c in [market_name_col, "liquidity", "volume", "last_trade_price", "url", "best_bid", "best_ask", "slug", "condition_id", "end_date"] if c in latest_markets.columns]
                    dataset = dataset.merge(latest_markets[cols], left_on="market_title", right_on=market_name_col, how="left")

        if not alerts_df.empty and "market_title" in dataset.columns:
            alert_market_col = "market" if "market" in alerts_df.columns else "market_title" if "market_title" in alerts_df.columns else None
            if alert_market_col:
                alert_counts = alerts_df.groupby(alert_market_col).size().reset_index(name="alert_count")
                dataset = dataset.merge(alert_counts, left_on="market_title", right_on=alert_market_col, how="left")
                dataset["alert_count"] = dataset["alert_count"].fillna(0).astype(int)

        if not wallet_alpha_df.empty and "trader_wallet" in dataset.columns:
            wallet_key = "wallet_copied" if "wallet_copied" in wallet_alpha_df.columns else "trader_wallet" if "trader_wallet" in wallet_alpha_df.columns else None
            if wallet_key:
                dataset = dataset.merge(wallet_alpha_df, left_on="trader_wallet", right_on=wallet_key, how="left")

        if not wallet_alpha_history_df.empty and "trader_wallet" in dataset.columns and "timestamp" in dataset.columns:
            history_key = "wallet_copied" if "wallet_copied" in wallet_alpha_history_df.columns else "trader_wallet" if "trader_wallet" in wallet_alpha_history_df.columns else None
            if history_key and "timestamp" in wallet_alpha_history_df.columns:
                wallet_alpha_history_df = wallet_alpha_history_df.copy()
                wallet_alpha_history_df["timestamp"] = pd.to_datetime(wallet_alpha_history_df["timestamp"], utc=True, errors="coerce")
                merged_parts = []
                for wallet, group in dataset.groupby("trader_wallet", dropna=False):
                    history = wallet_alpha_history_df[wallet_alpha_history_df[history_key] == wallet]
                    if history.empty:
                        merged_parts.append(group)
                        continue
                    merged_parts.append(pd.merge_asof(group.sort_values("timestamp"), history.sort_values("timestamp"), on="timestamp", direction="backward"))
                dataset = pd.concat(merged_parts, ignore_index=True) if merged_parts else dataset

        if not btc_targets_df.empty and "timestamp" in dataset.columns and "timestamp" in btc_targets_df.columns:
            btc_targets_df = btc_targets_df.copy()
            btc_targets_df["timestamp"] = pd.to_datetime(btc_targets_df["timestamp"], utc=True, errors="coerce")
            cols = [c for c in ["timestamp", "btc_price", "btc_spot_return_5m", "btc_spot_return_15m", "btc_realized_vol_15m", "btc_volume_proxy"] if c in btc_targets_df.columns]
            dataset = pd.merge_asof(dataset.sort_values("timestamp"), btc_targets_df[cols].sort_values("timestamp"), on="timestamp", direction="backward")

        if "best_ask" in dataset.columns and "best_bid" in dataset.columns:
            dataset["spread"] = (pd.to_numeric(dataset["best_ask"], errors="coerce").fillna(0) - pd.to_numeric(dataset["best_bid"], errors="coerce").fillna(0)).abs()
        if "end_date" in dataset.columns and "timestamp" in dataset.columns:
            dataset["end_date"] = pd.to_datetime(dataset["end_date"], utc=True, errors="coerce")
            dataset["time_to_close_minutes"] = (dataset["end_date"] - dataset["timestamp"]).dt.total_seconds().div(60)

        return dataset

    def write(self):
        dataset = self.build()
        if dataset.empty:
            return dataset
        dataset.to_csv(self.output_file, index=False)
        logging.info("Saved historical dataset to %s", self.output_file)
        return dataset
