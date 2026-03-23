from pathlib import Path

import pandas as pd


class ContractTargetBuilder:
    """
    Build contract-level labels from real token-level CLOB price history anchored to signal timestamps.
    Research/paper-trading only.
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

    def _compute_path_labels(self, entry_price, future_window, tp_move, sl_move):
        if future_window.empty:
            return {}

        future_prices = future_window["price"].astype(float)
        forward_return = (float(future_prices.iloc[-1]) - entry_price) / entry_price if entry_price else None
        moves = [(float(price) - entry_price) for price in future_prices]
        mfe = max(moves) if moves else None
        mae = min(moves) if moves else None
        max_price = float(future_prices.max()) if len(future_prices) else None
        best_exit_price = max_price
        tp_hit_idx = next((i for i, move in enumerate(moves) if move >= tp_move), None)
        sl_hit_idx = next((i for i, move in enumerate(moves) if move <= -sl_move), None)
        tp_before_sl = int(tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx < sl_hit_idx))
        target_up = int((forward_return or 0) > 0)
        time_to_tp = None
        if tp_hit_idx is not None:
            time_to_tp = float((future_window.iloc[tp_hit_idx]["timestamp"] - future_window.iloc[0]["timestamp"]).total_seconds() / 60.0)
        time_to_max_pnl = None
        if moves:
            max_idx = int(moves.index(mfe))
            time_to_max_pnl = float((future_window.iloc[max_idx]["timestamp"] - future_window.iloc[0]["timestamp"]).total_seconds() / 60.0)
        return {
            "forward_return_15m": forward_return,
            "tp_before_sl_60m": tp_before_sl,
            "tp_hit_before_sl": tp_before_sl,
            "mfe_60m": mfe,
            "mae_60m": mae,
            "max_price_within_horizon": max_price,
            "best_exit_price_within_horizon": best_exit_price,
            "time_to_tp": time_to_tp,
            "time_to_max_pnl": time_to_max_pnl,
            "target_up": target_up,
        }

    def build(self, forward_minutes=15, max_hold_minutes=60, tp_move=0.04, sl_move=0.03):
        signals_df = self._safe_read(self.signals_file)
        markets_df = self._safe_read(self.markets_file)
        history_df = self._safe_read(self.clob_history_file)

        if signals_df.empty or markets_df.empty or history_df.empty:
            return pd.DataFrame()
        if "market" not in signals_df.columns or "question" not in markets_df.columns:
            return pd.DataFrame()

        signals_df = signals_df.copy()
        signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"], utc=True, errors="coerce", format="mixed")
        history_df = history_df.copy()
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], utc=True, errors="coerce", format="mixed")
        history_df = history_df.dropna(subset=["timestamp", "token_id"]).sort_values(["token_id", "timestamp"]).reset_index(drop=True)

        market_lookup = markets_df.drop_duplicates(subset=["question"], keep="last").set_index("question").to_dict("index")

        rows = []
        for _, signal_row in signals_df.iterrows():
            market_title = signal_row.get("market")
            market_row = market_lookup.get(market_title)
            if not market_row:
                continue

            token_id = self._select_token_id(signal_row.to_dict(), market_row)
            if not token_id:
                continue

            token_history = history_df[history_df["token_id"].astype(str) == str(token_id)].copy()
            if token_history.empty:
                continue

            signal_ts = signal_row.get("timestamp")
            if pd.isna(signal_ts):
                continue

            history_before = token_history[token_history["timestamp"] <= signal_ts]
            if history_before.empty:
                continue
            anchor_row = history_before.iloc[-1]
            entry_price = float(anchor_row.get("price", signal_row.get("current_price", 0.5)))

            future_window = token_history[
                (token_history["timestamp"] > signal_ts)
                & (token_history["timestamp"] <= signal_ts + pd.Timedelta(minutes=max_hold_minutes))
            ].copy()
            if future_window.empty:
                continue

            labels = self._compute_path_labels(
                entry_price,
                future_window,
                tp_move,
                sl_move,
            )

            row = signal_row.to_dict()
            row.update(
                {
                    "token_id": token_id,
                    "entry_price": entry_price,
                    "anchor_timestamp": anchor_row.get("timestamp"),
                    **labels,
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

