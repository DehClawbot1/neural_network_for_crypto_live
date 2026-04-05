import os
from pathlib import Path

import pandas as pd


def _safe_merge_asof(left, right, on, **kwargs):
    """merge_asof that tolerates NaT/null in the left merge key."""
    if left.empty or right.empty or on not in left.columns or on not in right.columns:
        return left
    mask = left[on].notna()
    if not mask.any():
        return left
    valid = left[mask].copy().sort_values(on)
    work_right = right.copy().sort_values(on)
    merged = pd.merge_asof(valid, work_right, on=on, **kwargs)
    if mask.all():
        return merged
    return pd.concat([merged, left[~mask]], ignore_index=True)


class ContractTargetBuilder:
    """
    Build contract-level labels from token-level CLOB price history.

    Key fixes:
    - support both market and market_title signal schemas
    - compute TP-before-SL / MFE / MAE on the full max-hold window, not the short forward-return window
    - keep forward_return_15m anchored to the last price inside the 15m window
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.signals_file = self.logs_dir / "signals.csv"
        self.raw_candidates_file = self.logs_dir / "raw_candidates.csv"
        self.markets_file = self.logs_dir / "markets.csv"
        self.clob_history_file = self.logs_dir / "clob_price_history.csv"
        self.btc_live_file = self.logs_dir / "btc_live_snapshot.csv"
        self.technical_regime_file = self.logs_dir / "technical_regime_snapshot.csv"
        self.output_file = self.logs_dir / "contract_targets.csv"

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _select_token_id(self, signal_row, market_row):
        side = str(signal_row.get("side", signal_row.get("outcome", signal_row.get("outcome_side", "YES")))).upper()
        if side == "NO":
            return market_row.get("no_token_id")
        return market_row.get("yes_token_id")

    def _parse_logged_timestamp_series(self, series: pd.Series) -> pd.Series:
        local_tz = os.getenv("BOT_LOG_LOCAL_TIMEZONE", "Europe/Lisbon")
        raw = pd.to_datetime(series, errors="coerce", utc=False, format="mixed")
        if getattr(raw.dt, "tz", None) is None:
            try:
                return raw.dt.tz_localize(local_tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
            except Exception:
                return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
        return raw.dt.tz_convert("UTC")

    def _path_stats(self, entry_price, path_prices, tp_move, sl_move):
        if path_prices.empty:
            return None, None, None, None
        moves = [(float(price) - float(entry_price)) / float(entry_price) for price in path_prices] # BUG FIX 5: Normalize to ROI
        mfe = max(moves) if moves else None
        mae = min(moves) if moves else None
        tp_hit_idx = next((i for i, move in enumerate(moves) if move >= tp_move), None)
        sl_hit_idx = next((i for i, move in enumerate(moves) if move <= -sl_move), None)
        tp_before_sl = int(tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx))
        target_up = int((float(path_prices.iloc[-1]) - float(entry_price)) > 0)
        return tp_before_sl, mfe, mae, target_up

    def build(self, forward_minutes=15, max_hold_minutes=60, tp_move=0.04, sl_move=0.03):
        signals_df = self._safe_read(self.signals_file)
        if signals_df.empty:
            signals_df = self._safe_read(self.raw_candidates_file)
        markets_df = self._safe_read(self.markets_file)
        history_df = self._safe_read(self.clob_history_file)
        btc_live_df = self._safe_read(self.btc_live_file)
        technical_regime_df = self._safe_read(self.technical_regime_file)

        if signals_df.empty or markets_df.empty or history_df.empty:
            return pd.DataFrame()

        market_name_col = "market_title" if "market_title" in signals_df.columns else "market" if "market" in signals_df.columns else None
        market_lookup_col = "question" if "question" in markets_df.columns else "market_title" if "market_title" in markets_df.columns else None
        if market_name_col is None or market_lookup_col is None:
            return pd.DataFrame()

        signals_df = signals_df.copy()
        signals_df["timestamp"] = self._parse_logged_timestamp_series(signals_df["timestamp"])
        if not btc_live_df.empty:
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
                        "btc_live_funding_rate",
                        "btc_live_source_quality",
                        "btc_live_source_quality_score",
                        "btc_live_source_divergence_bps",
                        "btc_live_spot_index_basis_bps",
                        "btc_live_mark_index_basis_bps",
                        "btc_live_mark_spot_basis_bps",
                        "btc_live_return_1m",
                        "btc_live_return_5m",
                        "btc_live_return_15m",
                        "btc_live_return_1h",
                        "btc_live_volatility_proxy",
                        "btc_live_bias",
                        "btc_live_confluence",
                        "btc_live_index_ready",
                    ]
                    if c in btc_live_df.columns
                ]
                live_view = btc_live_df[live_cols].sort_values(ts_col).rename(columns={ts_col: "timestamp"})
                signals_df = _safe_merge_asof(signals_df, live_view, on="timestamp", direction="backward")
        if not technical_regime_df.empty:
            technical_regime_df = technical_regime_df.copy()
            ts_col = "technical_timestamp" if "technical_timestamp" in technical_regime_df.columns else "timestamp" if "timestamp" in technical_regime_df.columns else None
            if ts_col:
                technical_regime_df[ts_col] = pd.to_datetime(technical_regime_df[ts_col], utc=True, errors="coerce")
                technical_regime_df = technical_regime_df[technical_regime_df[ts_col].notna()].copy()
                regime_cols = [
                    c
                    for c in [
                        ts_col,
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
                    ]
                    if c in technical_regime_df.columns
                ]
                regime_view = technical_regime_df[regime_cols].sort_values(ts_col).rename(columns={ts_col: "timestamp"})
                signals_df = _safe_merge_asof(signals_df, regime_view, on="timestamp", direction="backward")
        history_df = history_df.copy()
        history_df["token_id"] = history_df["token_id"].astype(str)
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], utc=True, errors="coerce")
        history_df = history_df.dropna(subset=["timestamp", "token_id"]).sort_values(["token_id", "timestamp"]).reset_index(drop=True)
        markets_latest = markets_df.drop_duplicates(subset=[market_lookup_col], keep="last")
        market_lookup = markets_latest.set_index(market_lookup_col).to_dict("index")

        rows = []
        history_groups = dict(tuple(history_df.groupby("token_id"))) # BUG FIX 1: O(1) lookups
        for _, signal_row in signals_df.iterrows():
            signal_ts = signal_row.get("timestamp")
            if pd.isna(signal_ts):
                continue
            market_row = market_lookup.get(signal_row.get(market_name_col))
            if not market_row:
                continue

            token_id = signal_row.get("token_id") or self._select_token_id(signal_row.to_dict(), market_row)
            if not token_id:
                continue
            token_id = str(token_id)

            token_history = history_groups.get(token_id, pd.DataFrame()).copy() # BUG FIX 1: Prevent pipeline freeze
            if token_history.empty:
                continue

            history_before = token_history[token_history["timestamp"] <= signal_ts]
            if history_before.empty:
                continue
            anchor_row = history_before.iloc[-1]
            entry_price = float(anchor_row.get("price", signal_row.get("current_price", 0.5)))
            if entry_price <= 0:
                continue

            full_future_window = token_history[
                (token_history["timestamp"] > signal_ts)
                & (token_history["timestamp"] <= signal_ts + pd.Timedelta(minutes=max_hold_minutes))
            ].copy()
            if full_future_window.empty:
                continue

            forward_window = full_future_window[
                full_future_window["timestamp"] <= signal_ts + pd.Timedelta(minutes=forward_minutes)
            ]
            if forward_window.empty:
                forward_window = full_future_window.head(1)

            forward_return = (float(forward_window["price"].iloc[-1]) - entry_price) / entry_price
            # tp_before_sl/mfe/mae must be computed against the full max-hold path
            path_moves = full_future_window["price"].astype(float)
            moves = [(float(price) - entry_price) / entry_price for price in path_moves] # BUG FIX 5: Normalize to ROI
            mfe = max(moves) if moves else None
            mae = min(moves) if moves else None
            tp_hit_idx = next((i for i, move in enumerate(moves) if move >= tp_move), None)
            sl_hit_idx = next((i for i, move in enumerate(moves) if move <= -sl_move), None)
            tp_before_sl = int(tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx))
            target_up = int((float(path_moves.iloc[-1]) - entry_price) > 0)

            row = signal_row.to_dict()
            row.update(
                {
                    "token_id": token_id,
                    "entry_price": entry_price,
                    "anchor_timestamp": anchor_row.get("timestamp"),
                    "forward_return_15m": forward_return,
                    "tp_before_sl_60m": tp_before_sl,
                    "mfe_60m": mfe,
                    "mae_60m": mae,
                    "target_up": target_up,
                }
            )
            rows.append(row)

        return pd.DataFrame(rows)

    def write(self, forward_minutes=15, max_hold_minutes=60, tp_move=0.04, sl_move=0.03):
        df = self.build(
            forward_minutes=forward_minutes,
            max_hold_minutes=max_hold_minutes,
            tp_move=tp_move,
            sl_move=sl_move,
        )
        if not df.empty:
            df.to_csv(self.output_file, index=False)
        return df
