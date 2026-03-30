from pathlib import Path

import pandas as pd


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
        self.markets_file = self.logs_dir / "markets.csv"
        self.clob_history_file = self.logs_dir / "clob_price_history.csv"
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
        markets_df = self._safe_read(self.markets_file)
        history_df = self._safe_read(self.clob_history_file)

        if signals_df.empty or markets_df.empty or history_df.empty:
            return pd.DataFrame()

        market_name_col = "market_title" if "market_title" in signals_df.columns else "market" if "market" in signals_df.columns else None
        market_lookup_col = "question" if "question" in markets_df.columns else "market_title" if "market_title" in markets_df.columns else None
        if market_name_col is None or market_lookup_col is None:
            return pd.DataFrame()

        signals_df = signals_df.copy()
        signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"], utc=True, errors="coerce")
        history_df = history_df.copy()
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

            token_history = history_groups.get(str(token_id), pd.DataFrame()).copy() # BUG FIX 1: Prevent pipeline freeze
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
