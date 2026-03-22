import logging
from pathlib import Path

import pandas as pd

from rl_trainer import train_model
from stage1_models import Stage1Models
from stage2_temporal_models import Stage2TemporalModels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Retrainer:
    """
    Outcome-driven retraining trigger for paper/research mode.
    Prefers closed trades and replay episodes over raw dataset size.
    """

    def __init__(self, logs_dir="logs", closed_trade_threshold=100, replay_threshold=200):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.closed_file = self.logs_dir / "closed_positions.csv"
        self.replay_file = self.logs_dir / "path_replay_backtest.csv"
        self.backtest_summary_file = self.logs_dir / "backtest_summary.csv"
        self.status_file = self.logs_dir / "retrainer_status.txt"
        self.status_csv = self.logs_dir / "model_status.csv"
        self.closed_trade_threshold = closed_trade_threshold
        self.replay_threshold = replay_threshold

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _write_status(self, closed_rows: int, replay_rows: int, action: str):
        self.status_file.write_text(action + "\n", encoding="utf-8")
        progress_closed = round(closed_rows / self.closed_trade_threshold, 4) if self.closed_trade_threshold else 0
        progress_replay = round(replay_rows / self.replay_threshold, 4) if self.replay_threshold else 0
        pd.DataFrame(
            [
                {
                    "closed_trade_rows": closed_rows,
                    "closed_trade_threshold": self.closed_trade_threshold,
                    "replay_rows": replay_rows,
                    "replay_threshold": self.replay_threshold,
                    "progress_ratio": max(progress_closed, progress_replay),
                    "last_action": action,
                }
            ]
        ).to_csv(self.status_csv, index=False)

    def maybe_retrain(self):
        closed_df = self._safe_read(self.closed_file)
        replay_df = self._safe_read(self.replay_file)
        backtest_df = self._safe_read(self.backtest_summary_file)

        closed_rows = len(closed_df)
        replay_rows = len(replay_df)
        pnl_degraded = False
        if not backtest_df.empty and "average_pnl" in backtest_df.columns:
            pnl_degraded = float(backtest_df.iloc[-1].get("average_pnl", 0.0) or 0.0) < 0

        should_retrain = (
            closed_rows >= self.closed_trade_threshold
            or replay_rows >= self.replay_threshold
            or pnl_degraded
        )

        if not should_retrain:
            self._write_status(
                closed_rows,
                replay_rows,
                f"Not enough real outcomes for retraining yet: closed={closed_rows}/{self.closed_trade_threshold}, replay={replay_rows}/{self.replay_threshold}",
            )
            return False

        logging.info("Outcome-based retraining triggered. Refreshing supervised models and replay-aware RL model...")
        Stage1Models(logs_dir=self.logs_dir).train()
        Stage2TemporalModels(logs_dir=self.logs_dir).train()
        train_model(timesteps=5000)
        self._write_status(closed_rows, replay_rows, f"Outcome-based retraining triggered with closed={closed_rows}, replay={replay_rows}.")
        return True
