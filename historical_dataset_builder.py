import logging
import os
import re
from pathlib import Path

import pandas as pd
from brain_paths import filter_frame_for_brain, resolve_brain_context
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

    def __init__(self, logs_dir="logs", *, shared_logs_dir=None, brain_context=None, brain_id=None, market_family=None):
        if brain_context is None and (brain_id or market_family):
            brain_context = resolve_brain_context(
                market_family,
                brain_id=brain_id,
                shared_logs_dir=shared_logs_dir or logs_dir,
            )
        self.brain_context = brain_context
        self.logs_dir = Path(brain_context.logs_dir if brain_context is not None else logs_dir)
        self.shared_logs_dir = Path(
            brain_context.shared_logs_dir if brain_context is not None else (shared_logs_dir or logs_dir)
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.logs_dir / "historical_dataset.csv"
        self.btc_live_merge_tolerance = pd.Timedelta(
            os.getenv("BTC_LIVE_MERGE_TOLERANCE", "12h") or "12h"
        )
        self.technical_merge_tolerance = pd.Timedelta(
            os.getenv("TECHNICAL_REGIME_MERGE_TOLERANCE", "12h") or "12h"
        )

    def _safe_read(self, filename):
        path = self.shared_logs_dir / filename
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _safe_read_local(self, filename):
        path = self.logs_dir / filename
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _safe_merge_asof(self, left, right, on, by=None, **kwargs):
        if left.empty or right.empty or on not in left.columns or on not in right.columns:
            return left
        left = left.copy()
        right = right.copy()
        try:
            left_on_series = left[on]
            right_on_series = right[on]
            if str(left_on_series.dtype) != str(right_on_series.dtype):
                left[on] = pd.to_datetime(left_on_series, utc=True, errors="coerce", format="mixed")
                right[on] = pd.to_datetime(right_on_series, utc=True, errors="coerce", format="mixed")
        except Exception:
            pass
        # pd.merge_asof raises if the merge key has NaT/null on the left side.
        # Split off null-key rows, merge the valid ones, then concat back.
        mask = left[on].notna()
        if not mask.any():
            return left
        valid = left[mask].copy().sort_values(on)
        work_right = right.copy().sort_values(on)
        merged = pd.merge_asof(valid, work_right, on=on, by=by, direction="backward", **kwargs)
        if mask.all():
            return merged
        return pd.concat([merged, left[~mask]], ignore_index=True)

    def _dedupe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        return df.loc[:, ~df.columns.duplicated()].copy()

    def _coalesce_suffix_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        suffix_map: dict[str, list[str]] = {}
        for column in list(out.columns):
            match = re.match(r"^(?P<base>.+?)(?P<suffix>_(?:x|y|trade|target))$", str(column))
            if not match:
                continue
            suffix_map.setdefault(match.group("base"), []).append(column)

        for base_name, suffix_columns in suffix_map.items():
            ordered_columns = []
            if base_name in out.columns:
                ordered_columns.append(base_name)
            ordered_columns.extend([column for column in suffix_columns if column in out.columns])
            if not ordered_columns:
                continue
            coalesce_frame = out[ordered_columns].copy()
            out[base_name] = coalesce_frame.apply(
                lambda row: next((value for value in row.tolist() if pd.notna(value)), pd.NA),
                axis=1,
            )
            out = out.drop(columns=[column for column in suffix_columns if column in out.columns])
        return self._dedupe_columns(out)

    def _concat_frame_parts(self, parts: list[pd.DataFrame], fallback: pd.DataFrame) -> pd.DataFrame:
        cleaned_parts: list[pd.DataFrame] = []
        for part in parts or []:
            if part is None or part.empty:
                continue
            cleaned = part.dropna(axis=1, how="all")
            if cleaned.empty:
                continue
            cleaned_parts.append(cleaned)
        if not cleaned_parts:
            return fallback
        if len(cleaned_parts) == 1:
            return cleaned_parts[0].copy()
        return pd.concat(cleaned_parts, ignore_index=True, sort=False)

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

    def _apply_numeric_feature_priors(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        market_family_series = (
            out.get("market_family", pd.Series("", index=out.index))
            .astype(str)
            .str.lower()
        )
        weather_mask = market_family_series.str.startswith("weather_temperature")
        btc_mask = ~weather_mask

        current_price_series = pd.to_numeric(
            out.get("current_price", out.get("entry_price", out.get("btc_price", 0.0))),
            errors="coerce",
        )
        btc_price_series = pd.to_numeric(
            out.get("btc_price", out.get("current_price", out.get("entry_price", 0.0))),
            errors="coerce",
        )

        series_priors = {
            "wallet_trade_count_30d": 0.0,
            "wallet_alpha_30d": 0.0,
            "wallet_avg_forward_return_15m": 0.0,
            "wallet_signal_precision_tp": 0.5,
            "wallet_recent_streak": 0.0,
            "btc_fee_pressure_score": 0.5,
            "btc_mempool_congestion_score": 0.5,
            "btc_network_activity_score": 0.5,
            "btc_network_stress_score": 0.5,
            "trend_score": 0.5,
            "btc_atr_pct_15m": 0.0,
            "btc_realized_vol_1h": 0.0,
            "btc_realized_vol_4h": 0.0,
            "btc_volatility_regime_score": 0.5,
            "btc_trend_persistence": 0.5,
            "btc_rsi_14": 50.0,
            "btc_rsi_distance_mid": 0.0,
            "btc_rsi_divergence_score": 0.0,
            "btc_macd": 0.0,
            "btc_macd_signal": 0.0,
            "btc_macd_hist": 0.0,
            "btc_macd_hist_slope": 0.0,
            "btc_momentum_confluence": 0.0,
            "btc_live_source_quality_score": 0.5,
            "btc_live_source_divergence_bps": 0.0,
            "btc_live_funding_rate": 0.0,
            "btc_live_spot_index_basis_bps": 0.0,
            "btc_live_mark_index_basis_bps": 0.0,
            "btc_live_mark_spot_basis_bps": 0.0,
            "btc_live_spot_index_basis_bps_kalman": 0.0,
            "btc_live_mark_index_basis_bps_kalman": 0.0,
            "btc_live_mark_spot_basis_bps_kalman": 0.0,
            "btc_live_return_1m": 0.0,
            "btc_live_return_5m": 0.0,
            "btc_live_return_15m": 0.0,
            "btc_live_return_1h": 0.0,
            "btc_live_return_1m_kalman": 0.0,
            "btc_live_return_5m_kalman": 0.0,
            "btc_live_return_15m_kalman": 0.0,
            "btc_live_return_1h_kalman": 0.0,
            "btc_live_volatility_proxy": 0.0,
            "btc_live_confluence": 0.0,
            "btc_live_confluence_kalman": 0.0,
            "btc_market_regime_score": 0.5,
            "btc_market_regime_trend_score": 0.5,
            "btc_market_regime_volatility_score": 0.5,
            "btc_market_regime_chaos_score": 0.5,
            "btc_market_regime_stability_score": 0.5,
            "btc_market_regime_confidence_multiplier": 1.0,
            "btc_market_regime_weight_legacy": 1.0 / 3.0,
            "btc_market_regime_weight_stage1": 1.0 / 3.0,
            "btc_market_regime_weight_stage2": 1.0 / 3.0,
            "open_positions_count": 0.0,
            "open_positions_negotiated_value_total": 0.0,
            "open_positions_max_payout_total": 0.0,
            "open_positions_current_value_total": 0.0,
            "open_positions_unrealized_pnl_total": 0.0,
            "open_positions_unrealized_pnl_pct_total": 0.0,
            "open_positions_avg_to_now_price_change_pct_mean": 0.0,
            "open_positions_avg_to_now_price_change_pct_min": 0.0,
            "open_positions_avg_to_now_price_change_pct_max": 0.0,
            "open_positions_winner_count": 0.0,
            "open_positions_loser_count": 0.0,
            "wallet_temp_hit_rate_90d": 0.5,
            "wallet_temp_realized_pnl_90d": 0.0,
            "wallet_region_score": 0.5,
            "wallet_temp_range_skill": 0.5,
            "wallet_temp_threshold_skill": 0.5,
            "wallet_quality_score": 0.5,
            "wallet_state_confidence": 0.5,
            "wallet_state_freshness_score": 0.5,
            "wallet_size_change_score": 0.0,
            "wallet_agreement_score": 0.5,
            "source_wallet_size_delta_ratio": 0.0,
            "liquidity_score": 0.0,
            "volume_score": 0.0,
            "market_structure_score": 0.5,
            "execution_quality_score": 0.0,
            "forecast_p_hit_interval": 0.5,
            "forecast_margin_to_lower_c": 0.0,
            "forecast_margin_to_upper_c": 0.0,
            "forecast_uncertainty_c": 0.0,
            "forecast_drift_c": 0.0,
            "weather_fair_probability_yes": 0.5,
            "weather_fair_probability_side": 0.5,
            "weather_market_probability": 0.5,
            "weather_forecast_edge": 0.0,
            "weather_forecast_margin_score": 0.5,
            "weather_forecast_stability_score": 0.5,
        }

        price_series_priors = {
            "btc_live_price": btc_price_series,
            "btc_live_spot_price": btc_price_series,
            "btc_live_index_price": btc_price_series,
            "btc_live_mark_price": btc_price_series,
            "btc_live_price_kalman": btc_price_series,
            "btc_live_spot_price_kalman": btc_price_series,
            "btc_live_index_price_kalman": btc_price_series,
            "btc_live_mark_price_kalman": btc_price_series,
            "current_price": current_price_series,
            "entry_price": current_price_series,
        }

        boolean_priors = {
            "btc_market_regime_is_calm": 0.0,
            "btc_market_regime_is_trend": 0.0,
            "btc_market_regime_is_volatile": 0.0,
            "btc_market_regime_is_chaotic": 0.0,
            "btc_live_index_ready": 0.0,
            "btc_live_index_feed_available": 0.0,
            "btc_live_mark_feed_available": 0.0,
        }

        updated_columns = {}
        new_columns = {}

        for column, prior in series_priors.items():
            if column not in out.columns:
                new_columns[column] = pd.Series(prior, index=out.index)
            else:
                updated_columns[column] = pd.to_numeric(out[column], errors="coerce").fillna(prior)

        for column, prior_series in price_series_priors.items():
            if column not in out.columns:
                new_columns[column] = pd.Series(prior_series, index=out.index)
            else:
                updated_columns[column] = pd.to_numeric(out[column], errors="coerce").fillna(prior_series)

        derived_fill_pairs = [
            ("btc_live_price_kalman", "btc_live_price"),
            ("btc_live_spot_price_kalman", "btc_live_spot_price"),
            ("btc_live_index_price_kalman", "btc_live_index_price"),
            ("btc_live_mark_price_kalman", "btc_live_mark_price"),
            ("btc_live_return_1m_kalman", "btc_live_return_1m"),
            ("btc_live_return_5m_kalman", "btc_live_return_5m"),
            ("btc_live_return_15m_kalman", "btc_live_return_15m"),
            ("btc_live_return_1h_kalman", "btc_live_return_1h"),
            ("btc_live_confluence_kalman", "btc_live_confluence"),
            ("btc_live_spot_index_basis_bps_kalman", "btc_live_spot_index_basis_bps"),
            ("btc_live_mark_index_basis_bps_kalman", "btc_live_mark_index_basis_bps"),
            ("btc_live_mark_spot_basis_bps_kalman", "btc_live_mark_spot_basis_bps"),
        ]
        for derived_col, raw_col in derived_fill_pairs:
            if derived_col in out.columns and raw_col in out.columns:
                derived_series = pd.to_numeric(out[derived_col], errors="coerce")
                raw_series = pd.to_numeric(out[raw_col], errors="coerce")
                updated_columns[derived_col] = derived_series.where(derived_series.notna(), raw_series)
            elif raw_col in out.columns and derived_col not in out.columns:
                new_columns[derived_col] = pd.to_numeric(out[raw_col], errors="coerce")

        for column, prior in boolean_priors.items():
            if column not in out.columns:
                new_columns[column] = pd.Series(prior, index=out.index)
            else:
                updated_columns[column] = pd.to_numeric(out[column], errors="coerce").fillna(prior)

        if "spread" in out.columns:
            updated_columns["spread"] = pd.to_numeric(out["spread"], errors="coerce").fillna(0.0)
        else:
            new_columns["spread"] = pd.Series(0.0, index=out.index)

        if updated_columns:
            out = out.assign(**updated_columns)
        if new_columns:
            out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)

        # Weather family specific neutral priors should not inherit BTC price proxies.
        weather_price_columns = [
            "current_price",
            "entry_price",
        ]
        weather_boolean_columns = [
            "wallet_watchlist_approved",
            "wallet_state_gate_pass",
            "weather_parseable",
            "forecast_ready",
            "forecast_stale",
            "weather_forecast_confirms_direction",
            "weather_threshold_conflict",
        ]
        for column in weather_price_columns:
            if column in out.columns:
                out.loc[weather_mask, column] = pd.to_numeric(out.loc[weather_mask, column], errors="coerce").fillna(0.5)
        weather_bool_updates = {}
        weather_bool_new = {}
        for column in weather_boolean_columns:
            if column not in out.columns:
                weather_bool_new[column] = pd.Series(0.0, index=out.index)
            else:
                weather_bool_updates[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        if weather_bool_updates:
            out = out.assign(**weather_bool_updates)
        if weather_bool_new:
            out = pd.concat([out, pd.DataFrame(weather_bool_new, index=out.index)], axis=1)

        # BTC-only live/index fields should stay neutral on weather rows.
        btc_only_price_columns = [
            "btc_live_price",
            "btc_live_spot_price",
            "btc_live_index_price",
            "btc_live_mark_price",
            "btc_live_price_kalman",
            "btc_live_spot_price_kalman",
            "btc_live_index_price_kalman",
            "btc_live_mark_price_kalman",
        ]
        for column in btc_only_price_columns:
            if column in out.columns:
                out.loc[weather_mask, column] = pd.to_numeric(out.loc[weather_mask, column], errors="coerce").fillna(0.0)

        return out

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
        weather_wallet_snapshot_df = self._safe_read_local("weather_wallet_state_snapshot.csv")

        dataset = enrich_frame_with_entry_snapshots(signals_df, logs_dir=self.logs_dir)
        if dataset.empty:
            return pd.DataFrame()
        if self.brain_context is not None:
            dataset = filter_frame_for_brain(dataset, self.brain_context)
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
        dataset = self._coalesce_suffix_columns(dataset)
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
                dataset = self._coalesce_suffix_columns(dataset)
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
                        merged_parts.append(self._coalesce_suffix_columns(self._dedupe_columns(merged)))
                    dataset = self._concat_frame_parts(merged_parts, dataset)
                    dataset = self._coalesce_suffix_columns(dataset)
                    dataset = self._dedupe_columns(dataset)
                else:
                    latest_markets = markets_df.drop_duplicates(subset=[market_name_col], keep="last")
                    cols = [c for c in [market_name_col, "liquidity", "volume", "last_trade_price", "url", "best_bid", "best_ask", "slug", "condition_id", "end_date"] if c in latest_markets.columns]
                    dataset = dataset.merge(latest_markets[cols], left_on="market_title", right_on=market_name_col, how="left")
                    dataset = self._coalesce_suffix_columns(dataset)
                    dataset = self._dedupe_columns(dataset)

        if not alerts_df.empty and "market_title" in dataset.columns:
            alert_market_col = "market" if "market" in alerts_df.columns else "market_title" if "market_title" in alerts_df.columns else None
            if alert_market_col:
                alert_counts = alerts_df.groupby(alert_market_col).size().reset_index(name="alert_count")
                dataset = dataset.merge(alert_counts, left_on="market_title", right_on=alert_market_col, how="left")
                dataset["alert_count"] = dataset["alert_count"].fillna(0).astype(int)
                dataset = self._coalesce_suffix_columns(dataset)
                dataset = self._dedupe_columns(dataset)

        if not wallet_alpha_df.empty and "trader_wallet" in dataset.columns:
            wallet_key = "wallet_copied" if "wallet_copied" in wallet_alpha_df.columns else "trader_wallet" if "trader_wallet" in wallet_alpha_df.columns else None
            if wallet_key:
                dataset = dataset.merge(wallet_alpha_df, left_on="trader_wallet", right_on=wallet_key, how="left")
                dataset = self._coalesce_suffix_columns(dataset)
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
                    merged_parts.append(self._coalesce_suffix_columns(self._dedupe_columns(self._safe_merge_asof(group, history, on="timestamp"))))
                dataset = self._concat_frame_parts(merged_parts, dataset)
                dataset = self._coalesce_suffix_columns(dataset)
                dataset = self._dedupe_columns(dataset)

        if not btc_targets_df.empty and "timestamp" in dataset.columns and "timestamp" in btc_targets_df.columns:
            btc_targets_df = btc_targets_df.copy()
            btc_targets_df["timestamp"] = self._parse_logged_timestamp_series(btc_targets_df["timestamp"])
            btc_targets_df = btc_targets_df[btc_targets_df["timestamp"].notna()].copy()
            cols = [c for c in ["timestamp", "btc_price", "btc_spot_return_5m", "btc_spot_return_15m", "btc_realized_vol_15m", "btc_volume_proxy"] if c in btc_targets_df.columns]
            dataset = self._safe_merge_asof(dataset, btc_targets_df[cols], on="timestamp")
            dataset = self._coalesce_suffix_columns(dataset)
            dataset = self._dedupe_columns(dataset)

        # ------------------------------------------------------------------
        # Synthetic backfill: when btc_live_snapshot.csv is empty, derive
        # live-equivalent features from btc_targets columns already merged.
        # ------------------------------------------------------------------
        _live_backfill_applied = False
        if btc_live_df.empty and "btc_price" in dataset.columns:
            logging.info("btc_live_snapshot.csv empty — synthesising live features from btc_targets columns.")
            _bp = pd.to_numeric(dataset.get("btc_price"), errors="coerce")
            for col in ["btc_live_price", "btc_live_spot_price", "btc_live_index_price",
                        "btc_live_mark_price", "btc_live_price_kalman", "btc_live_spot_price_kalman",
                        "btc_live_index_price_kalman", "btc_live_mark_price_kalman"]:
                if col not in dataset.columns:
                    dataset[col] = _bp
            # Returns — map from btc_targets return columns
            _return_map = {
                "btc_live_return_5m": "btc_spot_return_5m",
                "btc_live_return_15m": "btc_spot_return_15m",
            }
            for live_col, src_col in _return_map.items():
                if live_col not in dataset.columns and src_col in dataset.columns:
                    dataset[live_col] = pd.to_numeric(dataset[src_col], errors="coerce")
            # Kalman returns = same as raw (no live filter available)
            for suffix in ["5m", "15m"]:
                raw = f"btc_live_return_{suffix}"
                kal = f"btc_live_return_{suffix}_kalman"
                if kal not in dataset.columns and raw in dataset.columns:
                    dataset[kal] = dataset[raw]
            # Approximate 1m return from 5m / 5
            if "btc_live_return_1m" not in dataset.columns and "btc_live_return_5m" in dataset.columns:
                dataset["btc_live_return_1m"] = dataset["btc_live_return_5m"] / 5.0
                dataset["btc_live_return_1m_kalman"] = dataset["btc_live_return_1m"]
            # Approximate 1h return from 15m * 4
            if "btc_live_return_1h" not in dataset.columns and "btc_live_return_15m" in dataset.columns:
                dataset["btc_live_return_1h"] = dataset["btc_live_return_15m"] * 4.0
                dataset["btc_live_return_1h_kalman"] = dataset["btc_live_return_1h"]
            # Volatility proxy from btc_realized_vol_15m
            if "btc_live_volatility_proxy" not in dataset.columns and "btc_realized_vol_15m" in dataset.columns:
                dataset["btc_live_volatility_proxy"] = pd.to_numeric(dataset["btc_realized_vol_15m"], errors="coerce")
            # Basis features = 0 (single-source historical, no spot/index divergence)
            for basis_col in ["btc_live_source_divergence_bps", "btc_live_spot_index_basis_bps",
                              "btc_live_mark_index_basis_bps", "btc_live_mark_spot_basis_bps",
                              "btc_live_spot_index_basis_bps_kalman", "btc_live_mark_index_basis_bps_kalman",
                              "btc_live_mark_spot_basis_bps_kalman"]:
                if basis_col not in dataset.columns:
                    dataset[basis_col] = 0.0
            # Quality / boolean flags
            if "btc_live_source_quality_score" not in dataset.columns:
                dataset["btc_live_source_quality_score"] = 0.5  # moderate confidence for historical proxy
            if "btc_live_funding_rate" not in dataset.columns:
                dataset["btc_live_funding_rate"] = 0.0
            # Confluence = mean of available return signals
            _ret_cols = [c for c in dataset.columns if c.startswith("btc_live_return_") and "kalman" not in c]
            if "btc_live_confluence" not in dataset.columns and _ret_cols:
                dataset["btc_live_confluence"] = dataset[_ret_cols].mean(axis=1).fillna(0.0)
                dataset["btc_live_confluence_kalman"] = dataset["btc_live_confluence"]
            # Boolean ready flags
            for flag_col in ["btc_live_index_ready", "btc_live_index_feed_available", "btc_live_mark_feed_available"]:
                if flag_col not in dataset.columns:
                    dataset[flag_col] = 1.0  # mark as available (proxied from candle data)
            _live_backfill_applied = True
            logging.info("Synthetic live feature backfill applied (%d rows).", len(dataset))

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
                dataset = self._safe_merge_asof(
                    dataset,
                    live_view,
                    on="timestamp",
                    tolerance=self.btc_live_merge_tolerance,
                )
                dataset = self._coalesce_suffix_columns(dataset)
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
                dataset = self._safe_merge_asof(
                    dataset,
                    regime_view,
                    on="timestamp",
                    tolerance=self.technical_merge_tolerance,
                )
                dataset = self._coalesce_suffix_columns(dataset)
                dataset = self._dedupe_columns(dataset)

        # ------------------------------------------------------------------
        # Sentiment feature enrichment: use REAL historical data where
        # an archive exists, then fall back to empirically-motivated priors
        # for sources that have no free historical API.
        #
        # Disk cache: FGI and Trends data are cached in the shared logs dir
        # for up to SENTIMENT_CACHE_HOURS (default 12h) to avoid re-fetching
        # on every training run.  Delete the cache files to force a refresh.
        #
        # Sources with real history (merged by timestamp):
        #   - Fear & Greed Index (alternative.me) — daily back to Feb 2018
        #   - Google Trends "bitcoin" — weekly back to 5 years
        #
        # Sources with NO history (prior-only):
        #   - Twitter/X sentiment  — only current snapshot available
        #   - Reddit sentiment     — only current snapshot available
        #
        # Priors for Twitter/Reddit are mildly positive (crypto communities
        # historically show a ~55-60% positive post ratio, not perfectly
        # neutral). Using 0.0 everywhere trains the model on a fictional
        # neutral world that never existed.
        # ------------------------------------------------------------------
        _sent_api_fgi_rows = 0
        _sent_api_trends_rows = 0
        _SENTIMENT_CACHE_HOURS = float(os.getenv("SENTIMENT_CACHE_HOURS", "12") or 12)
        _FGI_CACHE_PATH    = self.shared_logs_dir / "_cache_fgi_history.csv"
        _TRENDS_CACHE_PATH = self.shared_logs_dir / "_cache_gtrends_history.csv"

        def _load_or_fetch_csv(cache_path, fetch_fn, label):
            """Load from disk cache if fresh; otherwise fetch and save."""
            import time as _time
            _max_age_sec = _SENTIMENT_CACHE_HOURS * 3600
            if cache_path.exists():
                _age = _time.time() - cache_path.stat().st_mtime
                if _age < _max_age_sec:
                    try:
                        _cached = pd.read_csv(cache_path)
                        logging.info(
                            "Sentiment cache HIT for %s (age=%.1fh, rows=%d)",
                            label, _age / 3600, len(_cached),
                        )
                        return _cached
                    except Exception:
                        pass
            logging.info("Sentiment cache MISS for %s — fetching from API", label)
            _fresh = fetch_fn()
            if not _fresh.empty:
                try:
                    _fresh.to_csv(cache_path, index=False)
                except Exception:
                    pass
            return _fresh

        if "timestamp" in dataset.columns and dataset["timestamp"].notna().any():
            try:
                from btc_sentiment_features import BTCSentimentFeatures as _BSF
                _bsf = _BSF()

                # --- Fear & Greed Index: full history since 2018 ---
                try:
                    _fgi_df = _load_or_fetch_csv(
                        _FGI_CACHE_PATH,
                        lambda: _bsf.fetch_fear_greed_history(limit=0),
                        "FGI",
                    )
                    if not _fgi_df.empty and "timestamp" in _fgi_df.columns:
                        _fgi_df["timestamp"] = pd.to_datetime(_fgi_df["timestamp"], utc=True, errors="coerce")
                        _fgi_df = _fgi_df[_fgi_df["timestamp"].notna()].sort_values("timestamp")
                        # merge_asof requires both sides to be sorted with no NaT on left.
                        # Filter out NaT timestamps before sorting to avoid the
                        # "Merge keys contain null values on left side" error.
                        _ds_ts = dataset["timestamp"].copy()
                        _ds_valid_ts_mask = _ds_ts.notna()
                        _ds_sorted_idx = _ds_ts[_ds_valid_ts_mask].sort_values().index
                        _ds_sorted = dataset.loc[_ds_sorted_idx].copy()
                        _fgi_merge = pd.merge_asof(
                            _ds_sorted[["timestamp"]].reset_index(),
                            _fgi_df[["timestamp", "fgi_value"]],
                            on="timestamp",
                            direction="backward",
                            tolerance=pd.Timedelta("2d"),
                        )
                        _fgi_map = _fgi_merge.set_index("index")["fgi_value"]
                        dataset["fgi_value"] = (
                            pd.to_numeric(dataset.get("fgi_value"), errors="coerce")
                            .fillna(_fgi_map.reindex(dataset.index))
                        )
                        _sent_api_fgi_rows = int(_fgi_map.notna().sum())
                        logging.info(
                            "Sentiment: merged real FGI history into %d/%d dataset rows",
                            _sent_api_fgi_rows, len(dataset),
                        )
                except Exception as _fgi_exc:
                    logging.warning("Sentiment: FGI history fetch failed (%s) — using prior", _fgi_exc)

                # --- Google Trends: 5-year weekly history ---
                try:
                    _gt_df = _load_or_fetch_csv(
                        _TRENDS_CACHE_PATH,
                        lambda: _bsf.fetch_google_trends(keyword="bitcoin", timeframe="today 5-y"),
                        "GoogleTrends",
                    )
                    if not _gt_df.empty and "timestamp" in _gt_df.columns and "gtrends_bitcoin" in _gt_df.columns:
                        _gt_df["timestamp"] = pd.to_datetime(_gt_df["timestamp"], utc=True, errors="coerce")
                        _gt_df = _gt_df[_gt_df["timestamp"].notna()].sort_values("timestamp")
                        _gt_valid_mask = dataset["timestamp"].notna()
                        _gt_sorted_idx = dataset.loc[_gt_valid_mask, "timestamp"].sort_values().index
                        _gt_merge = pd.merge_asof(
                            dataset.loc[_gt_sorted_idx, ["timestamp"]].reset_index(),
                            _gt_df[["timestamp", "gtrends_bitcoin"]],
                            on="timestamp",
                            direction="backward",
                            tolerance=pd.Timedelta("8d"),
                        )
                        _gt_map = _gt_merge.set_index("index")["gtrends_bitcoin"]
                        dataset["gtrends_bitcoin"] = (
                            pd.to_numeric(dataset.get("gtrends_bitcoin"), errors="coerce")
                            .fillna(_gt_map.reindex(dataset.index))
                        )
                        _sent_api_trends_rows = int(_gt_map.notna().sum())
                        logging.info(
                            "Sentiment: merged real Google Trends history into %d/%d dataset rows",
                            _sent_api_trends_rows, len(dataset),
                        )
                except Exception as _gt_exc:
                    logging.warning("Sentiment: Google Trends fetch failed (%s) — using prior", _gt_exc)

                # --- Recompute ALL derived sentiment features from real values ---
                # This populates fgi_normalized, fgi_extreme_fear, fgi_contrarian,
                # fgi_momentum, gtrends_zscore, gtrends_spike, etc. from actual data.
                try:
                    dataset = _bsf._compute_derived_features(dataset)
                    logging.info("Sentiment: recomputed derived features from real FGI/Trends data")
                except Exception as _deriv_exc:
                    logging.warning("Sentiment: derived feature recompute failed: %s", _deriv_exc)

            except ImportError:
                logging.warning("Sentiment: BTCSentimentFeatures not importable — skipping real-data merge")

        # ------------------------------------------------------------------
        # Final fallback priors for any remaining NaN / unarchivable columns.
        #
        # Priors are set from empirical crypto-community baselines, NOT 0.0:
        #   - FGI ~50 (neutral midpoint, long-run average is ~52)
        #   - Twitter/Reddit raw sentiment: mild positive bias (+0.07)
        #     Crypto communities post ~55-60% positive content; VADER returns
        #     ~+0.05 to +0.10 for neutral-to-slightly-positive crypto discussion.
        #   - Engagement and post-count proxies: 0 (unknown = no signal)
        #   - Binary flags (bullish/bearish, spike): 0 (no strong signal)
        #   - z-scores / momentum: 0 (unknown = no deviation from mean)
        # ------------------------------------------------------------------
        _CRYPTO_TWITTER_PRIOR = float(os.getenv("PRIOR_TWITTER_SENTIMENT", "0.07"))
        _CRYPTO_REDDIT_PRIOR  = float(os.getenv("PRIOR_REDDIT_SENTIMENT",  "0.06"))
        _sentiment_defaults = {
            # FGI — only falls back here if API failed
            "fgi_value":            52.0,   # long-run average since 2018 ≈ 52
            "fgi_normalized":        0.52,
            "fgi_extreme_fear":      0.0,
            "fgi_extreme_greed":     0.0,
            "fgi_contrarian":        0.04,  # |52-50|/50
            "fgi_momentum":          0.0,
            "fgi_momentum_3d":       0.0,
            # Twitter/X — no history API; mild positive prior
            "twitter_sentiment":           _CRYPTO_TWITTER_PRIOR,
            "twitter_post_count":          0.0,
            "twitter_sentiment_pos":       max(0.0, _CRYPTO_TWITTER_PRIOR),
            "twitter_sentiment_neg":       0.0,
            "twitter_engagement_proxy":    0.0,
            "twitter_sentiment_zscore":    0.0,   # prior = at the mean
            "twitter_bullish":             0.0,   # prior = no strong signal
            "twitter_bearish":             0.0,
            "twitter_sentiment_momentum":  0.0,
            # Reddit — same reasoning as Twitter
            "reddit_sentiment":            _CRYPTO_REDDIT_PRIOR,
            "reddit_post_count":           0.0,
            "reddit_sentiment_pos":        max(0.0, _CRYPTO_REDDIT_PRIOR),
            "reddit_sentiment_neg":        0.0,
            "reddit_sentiment_zscore":     0.0,
            "reddit_bullish":              0.0,
            "reddit_bearish":              0.0,
            "reddit_sentiment_momentum":   0.0,
            # Google Trends — only falls back here if API failed
            "gtrends_bitcoin":    50.0,   # baseline normalized search interest
            "gtrends_zscore":      0.0,
            "gtrends_spike":       0.0,
            "gtrends_momentum":    0.0,
            # Composite
            "sentiment_score":     _CRYPTO_TWITTER_PRIOR * 0.5 + _CRYPTO_REDDIT_PRIOR * 0.5,
            "is_overheated_long":  0.0,
            "btc_funding_rate":    0.0,
        }
        _sent_filled = 0
        for _sc, _sv in _sentiment_defaults.items():
            if _sc not in dataset.columns:
                dataset[_sc] = _sv
                _sent_filled += 1
            else:
                dataset[_sc] = pd.to_numeric(dataset[_sc], errors="coerce").fillna(_sv)
        logging.info(
            "Sentiment enrichment complete: FGI_api=%d rows, Trends_api=%d rows, "
            "prior_fallback_cols=%d (twitter_prior=%.3f, reddit_prior=%.3f)",
            _sent_api_fgi_rows, _sent_api_trends_rows, _sent_filled,
            _CRYPTO_TWITTER_PRIOR, _CRYPTO_REDDIT_PRIOR,
        )

        # ------------------------------------------------------------------
        # Internal BTC trend features — derived from btc_live_snapshot.csv.
        #
        # These are more reliable than Google Trends (no API rate limits,
        # no external dependency) and directly measure BTC price momentum,
        # volatility, and crowd activity as proxies for "retail interest".
        #
        # Google Trends correlates with: price moves > 5%, volatility spikes,
        # ATH approaches. Our internal momentum features capture the same
        # signal from first-party data.
        #
        # Features added (all prefixed btc_internal_):
        #   btc_internal_ret_1h         — 1h Kalman return (fast trend)
        #   btc_internal_ret_4h         — 4h rolling return (medium trend)
        #   btc_internal_vol_ratio      — recent vol / baseline vol (fear proxy)
        #   btc_internal_trend_score    — composite: momentum × (1 + vol_ratio)
        #   btc_internal_trend_up       — binary: is recent return positive?
        #   btc_internal_price_vs_7d_ma — price z-score vs 7d rolling mean
        # ------------------------------------------------------------------
        _btc_live_for_trends = btc_live_df.copy() if not btc_live_df.empty else pd.DataFrame()
        _btc_internal_rows = 0
        if not _btc_live_for_trends.empty and "timestamp" in dataset.columns:
            try:
                _ts_col = next(
                    (c for c in ["btc_live_timestamp", "timestamp"] if c in _btc_live_for_trends.columns),
                    None
                )
                _price_col = next(
                    (c for c in ["btc_live_price_kalman", "btc_live_price"] if c in _btc_live_for_trends.columns),
                    None
                )
                _ret_1h_col = next(
                    (c for c in ["btc_live_return_1h_kalman", "btc_live_return_1h"] if c in _btc_live_for_trends.columns),
                    None
                )
                _vol_col = "btc_live_volatility_proxy" if "btc_live_volatility_proxy" in _btc_live_for_trends.columns else None

                if _ts_col and _price_col:
                    _bt = _btc_live_for_trends.copy()
                    _bt["_ts"] = pd.to_datetime(_bt[_ts_col], utc=True, errors="coerce")
                    _bt = _bt[_bt["_ts"].notna()].sort_values("_ts")
                    _price = pd.to_numeric(_bt[_price_col], errors="coerce")

                    # 4h return: price change over rolling 16-row window (15m candles × 16 = 4h)
                    _bt["btc_internal_ret_4h"] = _price.pct_change(periods=16).clip(-0.5, 0.5)

                    # 1h return from snapshot column (already computed) or 4-row rolling
                    if _ret_1h_col:
                        _bt["btc_internal_ret_1h"] = pd.to_numeric(_bt[_ret_1h_col], errors="coerce").clip(-0.5, 0.5)
                    else:
                        _bt["btc_internal_ret_1h"] = _price.pct_change(periods=4).clip(-0.5, 0.5)

                    # Volatility ratio: rolling 4h std / rolling 7d std (fear spike proxy)
                    if _vol_col:
                        _vol = pd.to_numeric(_bt[_vol_col], errors="coerce")
                        _vol_4h = _vol.rolling(16, min_periods=4).mean()
                        _vol_7d = _vol.rolling(672, min_periods=16).mean().replace(0, float("nan"))
                        _bt["btc_internal_vol_ratio"] = (_vol_4h / _vol_7d).clip(0, 5).fillna(1.0)
                    else:
                        _vol_4h = _price.rolling(16, min_periods=4).std()
                        _vol_7d = _price.rolling(672, min_periods=16).std().replace(0, float("nan"))
                        _bt["btc_internal_vol_ratio"] = (_vol_4h / _vol_7d).clip(0, 5).fillna(1.0)

                    # Price vs 7-day moving average (z-score normalised)
                    _ma7 = _price.rolling(672, min_periods=16).mean()
                    _std7 = _price.rolling(672, min_periods=16).std().replace(0, float("nan"))
                    _bt["btc_internal_price_vs_7d_ma"] = ((_price - _ma7) / _std7).clip(-4, 4).fillna(0.0)

                    # Composite trend score: 4h return weighted by volatility regime
                    _bt["btc_internal_trend_score"] = (
                        _bt["btc_internal_ret_4h"].fillna(0) * (1.0 + _bt["btc_internal_vol_ratio"].fillna(1.0))
                    ).clip(-2, 2)
                    _bt["btc_internal_trend_up"] = (_bt["btc_internal_ret_4h"].fillna(0) > 0).astype(float)

                    _trend_cols = [
                        "btc_internal_ret_1h",
                        "btc_internal_ret_4h",
                        "btc_internal_vol_ratio",
                        "btc_internal_price_vs_7d_ma",
                        "btc_internal_trend_score",
                        "btc_internal_trend_up",
                    ]
                    _bt_view = _bt[["_ts"] + _trend_cols].rename(columns={"_ts": "timestamp"})

                    # merge_asof into dataset — only rows with valid timestamps
                    _ds_valid_ts = dataset["timestamp"].notna()
                    _ds_idx_sorted = dataset.loc[_ds_valid_ts, "timestamp"].sort_values().index
                    _trend_merge = pd.merge_asof(
                        dataset.loc[_ds_idx_sorted, ["timestamp"]].reset_index(),
                        _bt_view.sort_values("timestamp"),
                        on="timestamp",
                        direction="backward",
                        tolerance=pd.Timedelta("4h"),
                    )
                    for _tc in _trend_cols:
                        _tc_map = _trend_merge.set_index("index")[_tc]
                        # dataset.get(col) returns None (scalar) for missing columns,
                        # which makes pd.to_numeric return float64 NaN — no .fillna().
                        # Always pass a Series to pd.to_numeric.
                        _existing = (
                            dataset[_tc]
                            if _tc in dataset.columns
                            else pd.Series(dtype=float, index=dataset.index)
                        )
                        dataset[_tc] = pd.to_numeric(_existing, errors="coerce").fillna(
                            _tc_map.reindex(dataset.index)
                        ).fillna(0.0)  # neutral prior for rows outside the 4h merge window

                    _btc_internal_rows = int(_trend_merge[_trend_cols[0]].notna().sum())
                    logging.info(
                        "BTC internal trend: computed %d trend features, merged into %d/%d dataset rows",
                        len(_trend_cols), _btc_internal_rows, len(dataset),
                    )
            except Exception as _btrend_exc:
                logging.warning("BTC internal trend features failed: %s", _btrend_exc)

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
                dataset = self._coalesce_suffix_columns(dataset)
                dataset = self._dedupe_columns(dataset)

        if "best_ask" in dataset.columns and "best_bid" in dataset.columns:
            dataset["spread"] = (pd.to_numeric(dataset["best_ask"], errors="coerce").fillna(0) - pd.to_numeric(dataset["best_bid"], errors="coerce").fillna(0)).abs()
        if "end_date" in dataset.columns and "timestamp" in dataset.columns:
            dataset["end_date"] = pd.to_datetime(dataset["end_date"], utc=True, errors="coerce")
            dataset["time_to_close_minutes"] = (dataset["end_date"] - dataset["timestamp"]).dt.total_seconds().div(60)

        if not weather_wallet_snapshot_df.empty and "market_family" in dataset.columns:
            weather_wallet_snapshot_df = weather_wallet_snapshot_df.copy()
            if "market_family" in weather_wallet_snapshot_df.columns:
                weather_wallet_snapshot_df = weather_wallet_snapshot_df[
                    weather_wallet_snapshot_df["market_family"].astype(str).str.startswith("weather_temperature")
                ].copy()
            if not weather_wallet_snapshot_df.empty:
                if "timestamp" in weather_wallet_snapshot_df.columns:
                    weather_wallet_snapshot_df["timestamp"] = self._parse_logged_timestamp_series(weather_wallet_snapshot_df["timestamp"])
                weather_merge_candidates = [
                    "token_id",
                    "condition_id",
                    "outcome_side",
                    "market_title",
                    "trader_wallet",
                ]
                weather_merge_keys = [
                    column
                    for column in weather_merge_candidates
                    if column in dataset.columns and column in weather_wallet_snapshot_df.columns
                ]
                if weather_merge_keys:
                    sort_col = "timestamp" if "timestamp" in weather_wallet_snapshot_df.columns else weather_merge_keys[0]
                    latest_weather = weather_wallet_snapshot_df.sort_values(sort_col).drop_duplicates(
                        subset=weather_merge_keys,
                        keep="last",
                    )
                    dataset = dataset.merge(
                        latest_weather,
                        on=weather_merge_keys,
                        how="left",
                        suffixes=("", "_weather_snapshot"),
                    )
                    dataset = self._coalesce_suffix_columns(dataset)
                    dataset = self._dedupe_columns(dataset)

        dataset = self._coalesce_suffix_columns(dataset)
        dataset = self._apply_numeric_feature_priors(dataset)

        if self.brain_context is not None and not dataset.empty:
            dataset = filter_frame_for_brain(dataset, self.brain_context)
        return dataset

    def write(self):
        from model_feature_safety import clean_dataframe_for_training

        dataset = self.build()
        if dataset.empty:
            return dataset
        dataset = clean_dataframe_for_training(dataset, context="historical_dataset")
        dataset.to_csv(self.output_file, index=False)
        logging.info("Saved historical dataset to %s", self.output_file)
        
        # Phase 5: Version every dataset (store raw source snapshots with timestamps)
        try:
            import datetime
            ts_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            vintage_dir = self.logs_dir / "vintages"
            vintage_dir.mkdir(parents=True, exist_ok=True)
            vintage_path = vintage_dir / f"historical_dataset_{ts_str}.csv"
            dataset.to_csv(vintage_path, index=False)
            logging.info("Saved PiT dataset vintage to %s", vintage_path)
        except Exception as exc:
            logging.warning("Failed to save vintage: %s", exc)
            
        return dataset
