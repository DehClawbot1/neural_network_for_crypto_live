import logging
import os
from pathlib import Path

import pandas as pd
from entry_snapshot_enrichment import enrich_frame_with_entry_snapshots

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
        # pd.merge_asof raises if the merge key has NaT/null on the left side.
        # Split off null-key rows, merge the valid ones, then concat back.
        mask = left[on].notna()
        if not mask.any():
            return left
        valid = left[mask].copy().sort_values(on)
        work_right = right.copy().sort_values(on)
        merged = pd.merge_asof(valid, work_right, on=on, by=by, direction="backward")
        if mask.all():
            return merged
        return pd.concat([merged, left[~mask]], ignore_index=True)

    def _dedupe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        return df.loc[:, ~df.columns.duplicated()].copy()

    def _parse_logged_timestamp_series(self, series: pd.Series) -> pd.Series:
        local_tz = os.getenv("BOT_LOG_LOCAL_TIMEZONE", "Europe/Lisbon")
        raw = pd.to_datetime(series, errors="coerce", utc=False, format="mixed")
        if getattr(raw.dt, "tz", None) is None:
            try:
                return raw.dt.tz_localize(local_tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
            except Exception:
                return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
        return raw.dt.tz_convert("UTC")

    def _coalesce_market_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = self._dedupe_columns(df)
        if "market_title" in out.columns and "market" in out.columns:
            out["market_title"] = out["market_title"].fillna(out["market"])
            out = out.drop(columns=["market"])
        elif "market" in out.columns and "market_title" not in out.columns:
            out = out.rename(columns={"market": "market_title"})
        return self._dedupe_columns(out)

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
        btc_live_df = self._safe_read("btc_live_snapshot.csv")
        technical_regime_df = self._safe_read("technical_regime_snapshot.csv")
        portfolio_curve_df = self._safe_read("portfolio_equity_curve.csv")

        dataset = enrich_frame_with_entry_snapshots(signals_df, logs_dir=self.logs_dir)
        if dataset.empty:
            return pd.DataFrame()
        rename_map = {
            "market": "market_title",
            "wallet_copied": "trader_wallet",
            "price": "entry_price",
            "side": "outcome_side",
        }
        dataset = dataset.rename(columns={k: v for k, v in rename_map.items() if k in dataset.columns})
        dataset = self._coalesce_market_columns(dataset)
        dataset = self._dedupe_columns(dataset)
        if "timestamp" not in dataset.columns:
            dataset["timestamp"] = pd.NaT
        dataset["timestamp"] = self._parse_logged_timestamp_series(dataset["timestamp"])

        if not trades_df.empty:
            trade_cols = [c for c in ["timestamp", "market", "wallet_copied", "fill_price", "size_usdc", "action_type"] if c in trades_df.columns]
            trade_view = trades_df[trade_cols].copy()
            if "timestamp" in trade_view.columns:
                trade_view["timestamp"] = self._parse_logged_timestamp_series(trade_view["timestamp"])
                trade_view = trade_view[trade_view["timestamp"].notna()].copy()
            if "market" in trade_view.columns and "market_title" not in trade_view.columns:
                trade_view = trade_view.rename(columns={"market": "market_title"})
            if "wallet_copied" in trade_view.columns and "trader_wallet" not in trade_view.columns:
                trade_view = trade_view.rename(columns={"wallet_copied": "trader_wallet"})
            merge_keys = [c for c in ["market_title", "trader_wallet"] if c in dataset.columns and c in trade_view.columns]
            if merge_keys:
                dataset = dataset.merge(trade_view, on=merge_keys, how="left", suffixes=("", "_trade"))
                dataset = self._dedupe_columns(dataset)

        if not markets_df.empty:
            markets_df = self._coalesce_market_columns(markets_df)
            market_name_col = "question" if "question" in markets_df.columns else "market_title" if "market_title" in markets_df.columns else None
            if market_name_col and "market_title" in dataset.columns:
                if "timestamp" in markets_df.columns:
                    markets_df = markets_df.copy()
                    markets_df["timestamp"] = self._parse_logged_timestamp_series(markets_df["timestamp"])
                    markets_df = markets_df[markets_df["timestamp"].notna()].copy()
                    merged_parts = []
                    for market_title, group in dataset.groupby("market_title", dropna=False):
                        market_history = markets_df[markets_df[market_name_col] == market_title].copy()
                        market_history = market_history[market_history["timestamp"].notna()].copy()
                        if market_history.empty:
                            merged_parts.append(group)
                            continue
                        cols = [c for c in ["timestamp", market_name_col, "liquidity", "volume", "last_trade_price", "url", "best_bid", "best_ask", "slug", "condition_id", "end_date"] if c in market_history.columns]
                        merged = self._safe_merge_asof(group, market_history[cols], on="timestamp")
                        merged_parts.append(self._dedupe_columns(merged))
                    dataset = pd.concat(merged_parts, ignore_index=True) if merged_parts else dataset
                    dataset = self._dedupe_columns(dataset)
                else:
                    latest_markets = markets_df.drop_duplicates(subset=[market_name_col], keep="last")
                    cols = [c for c in [market_name_col, "liquidity", "volume", "last_trade_price", "url", "best_bid", "best_ask", "slug", "condition_id", "end_date"] if c in latest_markets.columns]
                    dataset = dataset.merge(latest_markets[cols], left_on="market_title", right_on=market_name_col, how="left")
                    dataset = self._dedupe_columns(dataset)

        if not alerts_df.empty and "market_title" in dataset.columns:
            alert_market_col = "market" if "market" in alerts_df.columns else "market_title" if "market_title" in alerts_df.columns else None
            if alert_market_col:
                alert_counts = alerts_df.groupby(alert_market_col).size().reset_index(name="alert_count")
                dataset = dataset.merge(alert_counts, left_on="market_title", right_on=alert_market_col, how="left")
                dataset["alert_count"] = dataset["alert_count"].fillna(0).astype(int)
                dataset = self._dedupe_columns(dataset)

        if not wallet_alpha_df.empty and "trader_wallet" in dataset.columns:
            wallet_key = "wallet_copied" if "wallet_copied" in wallet_alpha_df.columns else "trader_wallet" if "trader_wallet" in wallet_alpha_df.columns else None
            if wallet_key:
                dataset = dataset.merge(wallet_alpha_df, left_on="trader_wallet", right_on=wallet_key, how="left")
                dataset = self._dedupe_columns(dataset)

        if not wallet_alpha_history_df.empty and "trader_wallet" in dataset.columns and "timestamp" in dataset.columns:
            history_key = "wallet_copied" if "wallet_copied" in wallet_alpha_history_df.columns else "trader_wallet" if "trader_wallet" in wallet_alpha_history_df.columns else None
            if history_key and "timestamp" in wallet_alpha_history_df.columns:
                wallet_alpha_history_df = wallet_alpha_history_df.copy()
                wallet_alpha_history_df["timestamp"] = self._parse_logged_timestamp_series(wallet_alpha_history_df["timestamp"])
                wallet_alpha_history_df = wallet_alpha_history_df[wallet_alpha_history_df["timestamp"].notna()].copy()
                merged_parts = []
                for wallet, group in dataset.groupby("trader_wallet", dropna=False):
                    history = wallet_alpha_history_df[wallet_alpha_history_df[history_key] == wallet]
                    if history.empty:
                        merged_parts.append(group)
                        continue
                    merged_parts.append(self._dedupe_columns(self._safe_merge_asof(group, history, on="timestamp")))
                dataset = pd.concat(merged_parts, ignore_index=True) if merged_parts else dataset
                dataset = self._dedupe_columns(dataset)

        if not btc_targets_df.empty and "timestamp" in dataset.columns and "timestamp" in btc_targets_df.columns:
            btc_targets_df = btc_targets_df.copy()
            btc_targets_df["timestamp"] = self._parse_logged_timestamp_series(btc_targets_df["timestamp"])
            btc_targets_df = btc_targets_df[btc_targets_df["timestamp"].notna()].copy()
            cols = [c for c in ["timestamp", "btc_price", "btc_spot_return_5m", "btc_spot_return_15m", "btc_realized_vol_15m", "btc_volume_proxy"] if c in btc_targets_df.columns]
            dataset = self._safe_merge_asof(dataset, btc_targets_df[cols], on="timestamp")
            dataset = self._dedupe_columns(dataset)

        if not btc_live_df.empty and "timestamp" in dataset.columns:
            btc_live_df = btc_live_df.copy()
            ts_col = "btc_live_timestamp" if "btc_live_timestamp" in btc_live_df.columns else "timestamp" if "timestamp" in btc_live_df.columns else None
            if ts_col:
                btc_live_df[ts_col] = pd.to_datetime(btc_live_df[ts_col], utc=True, errors="coerce")
                btc_live_df = btc_live_df[btc_live_df[ts_col].notna()].copy()
                live_cols = [
                    c
                    for c in [
                        ts_col,
                        "btc_live_price",
                        "btc_live_spot_price",
                        "btc_live_index_price",
                        "btc_live_mark_price",
                        "btc_live_price_kalman",
                        "btc_live_spot_price_kalman",
                        "btc_live_index_price_kalman",
                        "btc_live_mark_price_kalman",
                        "btc_live_funding_rate",
                        "btc_live_source_quality",
                        "btc_live_source_quality_score",
                        "btc_live_source_divergence_bps",
                        "btc_live_spot_index_basis_bps",
                        "btc_live_mark_index_basis_bps",
                        "btc_live_mark_spot_basis_bps",
                        "btc_live_spot_index_basis_bps_kalman",
                        "btc_live_mark_index_basis_bps_kalman",
                        "btc_live_mark_spot_basis_bps_kalman",
                        "btc_live_return_1m",
                        "btc_live_return_5m",
                        "btc_live_return_15m",
                        "btc_live_return_1h",
                        "btc_live_return_1m_kalman",
                        "btc_live_return_5m_kalman",
                        "btc_live_return_15m_kalman",
                        "btc_live_return_1h_kalman",
                        "btc_live_volatility_proxy",
                        "btc_live_bias",
                        "btc_live_confluence",
                        "btc_live_confluence_kalman",
                        "btc_live_index_ready",
                    ]
                    if c in btc_live_df.columns
                ]
                live_view = btc_live_df[live_cols].sort_values(ts_col).rename(columns={ts_col: "timestamp"})
                dataset = self._safe_merge_asof(dataset, live_view, on="timestamp")
                dataset = self._dedupe_columns(dataset)

        if not technical_regime_df.empty and "timestamp" in dataset.columns:
            technical_regime_df = technical_regime_df.copy()
            ts_col = "technical_timestamp" if "technical_timestamp" in technical_regime_df.columns else "timestamp" if "timestamp" in technical_regime_df.columns else None
            if ts_col:
                technical_regime_df[ts_col] = pd.to_datetime(technical_regime_df[ts_col], utc=True, errors="coerce")
                technical_regime_df = technical_regime_df[technical_regime_df[ts_col].notna()].copy()
                regime_cols = [
                    c
                    for c in [
                        ts_col,
                        "market_structure",
                        "trend_score",
                        "btc_atr_pct_15m",
                        "btc_realized_vol_1h",
                        "btc_realized_vol_4h",
                        "btc_volatility_regime",
                        "btc_volatility_regime_score",
                        "btc_trend_persistence",
                        "btc_rsi_14",
                        "btc_rsi_distance_mid",
                        "btc_rsi_divergence_score",
                        "btc_macd",
                        "btc_macd_signal",
                        "btc_macd_hist",
                        "btc_macd_hist_slope",
                        "btc_momentum_regime",
                        "btc_momentum_confluence",
                        "btc_market_regime_label",
                        "btc_market_regime_score",
                        "btc_market_regime_trend_score",
                        "btc_market_regime_volatility_score",
                        "btc_market_regime_chaos_score",
                        "btc_market_regime_stability_score",
                        "btc_market_regime_is_calm",
                        "btc_market_regime_is_trend",
                        "btc_market_regime_is_volatile",
                        "btc_market_regime_is_chaotic",
                        "btc_market_regime_primary_model",
                        "btc_market_regime_confidence_multiplier",
                        "btc_market_regime_weight_legacy",
                        "btc_market_regime_weight_stage1",
                        "btc_market_regime_weight_stage2",
                    ]
                    if c in technical_regime_df.columns
                ]
                regime_view = technical_regime_df[regime_cols].sort_values(ts_col).rename(columns={ts_col: "timestamp"})
                dataset = self._safe_merge_asof(dataset, regime_view, on="timestamp")
                dataset = self._dedupe_columns(dataset)

        if not portfolio_curve_df.empty and "timestamp" in dataset.columns and "timestamp" in portfolio_curve_df.columns:
            portfolio_curve_df = portfolio_curve_df.copy()
            portfolio_curve_df["timestamp"] = pd.to_datetime(portfolio_curve_df["timestamp"], utc=True, errors="coerce")
            portfolio_curve_df = portfolio_curve_df[portfolio_curve_df["timestamp"].notna()].copy()
            if not portfolio_curve_df.empty:
                portfolio_curve_df = portfolio_curve_df.sort_values("timestamp")
                portfolio_curve_df["open_positions_unrealized_pnl_pct_total"] = (
                    pd.to_numeric(portfolio_curve_df.get("unrealized_pnl", 0.0), errors="coerce").fillna(0.0)
                    / pd.to_numeric(portfolio_curve_df.get("entry_notional", 0.0), errors="coerce").replace(0, pd.NA)
                ).fillna(0.0)
                portfolio_view = portfolio_curve_df.rename(
                    columns={
                        "open_positions": "open_positions_count",
                        "entry_notional": "open_positions_negotiated_value_total",
                        "gross_market_value": "open_positions_current_value_total",
                        "unrealized_pnl": "open_positions_unrealized_pnl_total",
                    }
                )
                portfolio_cols = [
                    c
                    for c in [
                        "timestamp",
                        "open_positions_count",
                        "open_positions_negotiated_value_total",
                        "open_positions_current_value_total",
                        "open_positions_unrealized_pnl_total",
                        "open_positions_unrealized_pnl_pct_total",
                    ]
                    if c in portfolio_view.columns
                ]
                merged = self._safe_merge_asof(dataset, portfolio_view[portfolio_cols], on="timestamp")
                for col in portfolio_cols:
                    if col == "timestamp" or col not in merged.columns:
                        continue
                    if col not in dataset.columns:
                        dataset[col] = merged[col]
                    else:
                        dataset[col] = dataset[col].fillna(merged[col])
                dataset = self._dedupe_columns(dataset)

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
