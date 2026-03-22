from pathlib import Path

import pandas as pd

from pnl_engine import PNLEngine


class PathReplaySimulator:
    """
    Replay future token price bars after each signal and compute TP/SL/time-stop outcomes.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.targets_file = self.logs_dir / "contract_targets.csv"
        self.history_file = self.logs_dir / "clob_price_history.csv"
        self.output_file = self.logs_dir / "path_replay_backtest.csv"

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def simulate(self, capital_usdc=50.0, tp_move=0.04, sl_move=0.03, max_hold_minutes=60):
        targets_df = self._safe_read(self.targets_file)
        history_df = self._safe_read(self.history_file)
        if targets_df.empty or history_df.empty or "token_id" not in targets_df.columns:
            return pd.DataFrame()

        targets_df["timestamp"] = pd.to_datetime(targets_df["timestamp"], utc=True, errors="coerce")
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], utc=True, errors="coerce")
        history_df = history_df.dropna(subset=["timestamp", "token_id"]).sort_values(["token_id", "timestamp"]).reset_index(drop=True)

        trades = []
        for _, row in targets_df.iterrows():
            token_id = row.get("token_id")
            signal_ts = row.get("timestamp")
            if pd.isna(signal_ts) or pd.isna(token_id):
                continue

            token_history = history_df[history_df["token_id"].astype(str) == str(token_id)].copy()
            if token_history.empty:
                continue

            history_before = token_history[token_history["timestamp"] <= signal_ts]
            if history_before.empty:
                continue
            anchor_row = history_before.iloc[-1]
            entry_price = float(anchor_row.get("price", row.get("entry_price", 0.5)))

            future_path = token_history[
                (token_history["timestamp"] > signal_ts)
                & (token_history["timestamp"] <= signal_ts + pd.Timedelta(minutes=max_hold_minutes))
            ].copy()
            if future_path.empty:
                continue

            shares = PNLEngine.shares_from_capital(capital_usdc, entry_price)
            entry_time = signal_ts
            exit_price = float(future_path.iloc[-1].get("price", entry_price))
            exit_time = future_path.iloc[-1].get("timestamp")
            exit_reason = "time_stop"
            mfe = 0.0
            mae = 0.0
            peak_pnl = 0.0
            max_drawdown = 0.0

            for _, future_row in future_path.iterrows():
                future_price = float(future_row.get("price", entry_price))
                move = future_price - entry_price
                pnl = shares * move
                mfe = max(mfe, move)
                mae = min(mae, move)
                peak_pnl = max(peak_pnl, pnl)
                max_drawdown = min(max_drawdown, pnl - peak_pnl)

                if move >= tp_move:
                    exit_price = future_price
                    exit_time = future_row.get("timestamp")
                    exit_reason = "take_profit"
                    break
                if move <= -sl_move:
                    exit_price = future_price
                    exit_time = future_row.get("timestamp")
                    exit_reason = "stop_loss"
                    break
                if float(row.get("confidence", 1.0) or 1.0) < 0.45:
                    exit_price = future_price
                    exit_time = future_row.get("timestamp")
                    exit_reason = "confidence_drop"
                    break

            gross_pnl = PNLEngine.mark_to_market_pnl(capital_usdc, entry_price, exit_price)
            trades.append(
                {
                    "market": row.get("market", row.get("market_title")),
                    "wallet_copied": row.get("wallet_copied", row.get("trader_wallet")),
                    "token_id": token_id,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "holding_minutes": (exit_time - entry_time).total_seconds() / 60.0 if pd.notna(exit_time) and pd.notna(entry_time) else None,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "confidence": row.get("confidence"),
                    "exit_reason": exit_reason,
                    "gross_pnl": gross_pnl,
                    "net_pnl": gross_pnl,
                    "mfe": mfe,
                    "mae": mae,
                    "max_drawdown": max_drawdown,
                }
            )

        return pd.DataFrame(trades)

    def write(self, **kwargs):
        df = self.simulate(**kwargs)
        if not df.empty:
            df.to_csv(self.output_file, index=False)
        return df
