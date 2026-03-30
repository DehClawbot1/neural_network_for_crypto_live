import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class StrategyBacktester:
    """
    Summary reporter for the event-driven replay backtest.
    Uses path_replay_backtest.csv as the source of truth.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.replay_file = self.logs_dir / "path_replay_backtest.csv"
        self.output_file = self.logs_dir / "backtest_summary.csv"
        self.by_market_file = self.logs_dir / "backtest_by_market.csv"
        self.by_wallet_file = self.logs_dir / "backtest_by_wallet.csv"

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def run(self):
        df = self._safe_read(self.replay_file)
        if df.empty or "net_pnl" not in df.columns:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        pnl = df["net_pnl"].astype(float)
        wins = (pnl > 0).sum()
        losses = (pnl <= 0).sum()
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.nan
        sharpe_like = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0
        cumulative = pnl.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

        hold_col = "holding_minutes" if "holding_minutes" in df.columns else "holding_rows" if "holding_rows" in df.columns else "holding_time" if "holding_time" in df.columns else None
        avg_hold = float(df[hold_col].astype(float).mean()) if hold_col else np.nan

        summary = pd.DataFrame(
            [
                {
                    "trades": len(df),
                    "win_rate": float(wins / len(df)) if len(df) else 0.0,
                    "average_pnl": float(pnl.mean()),
                    "gross_profit": float(gross_profit),
                    "gross_loss": float(gross_loss),
                    "profit_factor": profit_factor,
                    "sharpe_like": sharpe_like,
                    "max_drawdown": max_drawdown,
                    "average_hold_time": avg_hold,
                }
            ]
        )

        by_market = pd.DataFrame()
        if "market" in df.columns:
            by_market = (
                df.groupby("market")
                .agg(trades=("market", "size"), average_pnl=("net_pnl", "mean"), total_pnl=("net_pnl", "sum"), win_rate=("net_pnl", lambda s: float((s > 0).mean())))
                .reset_index()
                .sort_values("total_pnl", ascending=False)
            )

        by_wallet = pd.DataFrame()
        if "wallet_copied" in df.columns:
            by_wallet = (
                df.groupby("wallet_copied")
                .agg(trades=("wallet_copied", "size"), average_pnl=("net_pnl", "mean"), total_pnl=("net_pnl", "sum"), win_rate=("net_pnl", lambda s: float((s > 0).mean())))
                .reset_index()
                .sort_values("total_pnl", ascending=False)
            )

        return summary, by_market, by_wallet

    def write(self, *_args, **_kwargs):
        summary, by_market, by_wallet = self.run()
        if summary.empty:
            return summary

        summary.to_csv(self.output_file, index=False)
        if not by_market.empty:
            by_market.to_csv(self.by_market_file, index=False)
        if not by_wallet.empty:
            by_wallet.to_csv(self.by_wallet_file, index=False)
        logging.info("Saved replay backtest summary to %s", self.output_file)
        return summary


if __name__ == "__main__":
    summary = StrategyBacktester(logs_dir=Path("logs")).write()
    print(summary if not summary.empty else "No path_replay_backtest.csv found yet.")

