from pathlib import Path

import pandas as pd

from pnl_engine import PNLEngine


class PathReplaySimulator:
    """
    Replay future contract prices after each signal and compute TP/SL-based paper outcomes.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.targets_file = self.logs_dir / "contract_targets.csv"
        self.output_file = self.logs_dir / "path_replay_backtest.csv"

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def simulate(self, capital_usdc=50.0, tp_move=0.04, sl_move=0.03, max_holding_rows=24):
        df = self._safe_read(self.targets_file)
        if df.empty or "market" not in df.columns or "current_price" not in df.columns:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values([c for c in ["market", "timestamp"] if c in df.columns]).reset_index(drop=True)

        trades = []
        for market, group in df.groupby("market"):
            group = group.reset_index(drop=True)
            for i, row in group.iterrows():
                entry_price = float(row.get("current_price", 0.5))
                shares = PNLEngine.shares_from_capital(capital_usdc, entry_price)
                path = group.iloc[i + 1 : i + 1 + max_holding_rows]
                if path.empty:
                    continue

                entry_time = row.get("timestamp")
                exit_price = float(path.iloc[-1].get("current_price", entry_price))
                exit_time = path.iloc[-1].get("timestamp")
                exit_reason = "time_stop"
                mfe = 0.0
                mae = 0.0
                peak_pnl = 0.0
                max_drawdown = 0.0

                for _, future_row in path.iterrows():
                    future_price = float(future_row.get("current_price", entry_price))
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

                gross_pnl = PNLEngine.mark_to_market_pnl(capital_usdc, entry_price, exit_price, side="BUY")
                trades.append(
                    {
                        "market": market,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "holding_rows": len(path),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "shares": shares,
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
