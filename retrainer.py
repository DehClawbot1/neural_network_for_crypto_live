import json
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

    Thresholds are intentionally low (5/10) so early outcomes can retrain quickly.
    """

    def __init__(self, logs_dir="logs", closed_trade_threshold=5, replay_threshold=10, weights_dir="weights"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.closed_file = self.logs_dir / "closed_positions.csv"
        self.replay_file = self.logs_dir / "path_replay_backtest.csv"
        self.backtest_summary_file = self.logs_dir / "backtest_summary.csv"
        self.time_split_file = self.logs_dir / "time_split_eval.csv"
        self.status_file = self.logs_dir / "retrainer_status.txt"
        self.status_csv = self.logs_dir / "model_status.csv"
        self.registry_file = self.weights_dir / "model_registry.csv"
        self.state_file = self.logs_dir / "retrainer_state.json"
        self.closed_trade_threshold = closed_trade_threshold
        self.replay_threshold = replay_threshold
        self._last_retrained_closed_count = self._load_last_retrained_closed_count()

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _safe_float(self, value, default=0.0):
        try:
            coerced = pd.to_numeric(value, errors="coerce")
            if pd.isna(coerced):
                return float(default)
            return float(coerced)
        except Exception:
            return float(default)

    def _load_last_retrained_closed_count(self):
        if not self.state_file.exists():
            return 0
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            return int(payload.get("last_retrained_closed_count", 0) or 0)
        except Exception:
            return 0

    def _persist_last_retrained_closed_count(self):
        payload = {"last_retrained_closed_count": int(self._last_retrained_closed_count)}
        try:
            self.state_file.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

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

    def _current_registry_row(self):
        registry = self._safe_read(self.registry_file)
        if registry.empty:
            return None
        return registry.iloc[-1].to_dict()

    def _promote_if_better(self, closed_rows, replay_rows):
        backtest_df = self._safe_read(self.backtest_summary_file)
        time_split_df = self._safe_read(self.time_split_file)
        if backtest_df.empty:
            return False, "Candidate metrics unavailable for promotion."

        candidate = backtest_df.iloc[-1].to_dict()
        if not time_split_df.empty:
            candidate.update(time_split_df.iloc[-1].to_dict())
        champion = self._current_registry_row()

        if champion is None:
            promoted = True
        else:
            candidate_avg_pnl = self._safe_float(candidate.get("average_pnl"), 0.0)
            candidate_profit_factor = self._safe_float(candidate.get("profit_factor"), 0.0)
            candidate_max_drawdown = self._safe_float(candidate.get("max_drawdown"), -999.0)
            champion_avg_pnl = self._safe_float(champion.get("average_pnl"), -999.0)
            champion_profit_factor = self._safe_float(champion.get("profit_factor"), 0.0)
            champion_max_drawdown = self._safe_float(champion.get("max_drawdown"), -999.0)
            promoted = (
                candidate_avg_pnl >= champion_avg_pnl
                and candidate_profit_factor >= champion_profit_factor
                and candidate_max_drawdown >= champion_max_drawdown
            )

        if not promoted:
            return False, "Candidate did not beat champion on promotion metrics."

        row = {
            "model_version": pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S"),
            "training_data_window": "rolling_outcome_based",
            "replay_rows_used": replay_rows,
            "closed_trades_used": closed_rows,
            "average_pnl": self._safe_float(candidate.get("average_pnl"), 0.0),
            "profit_factor": self._safe_float(candidate.get("profit_factor"), 0.0),
            "max_drawdown": self._safe_float(candidate.get("max_drawdown"), 0.0),
            "test_accuracy": self._safe_float(candidate.get("test_accuracy"), 0.0),
            "promoted_at": pd.Timestamp.utcnow().isoformat(),
        }
        pd.DataFrame([row]).to_csv(self.registry_file, mode="a", header=not self.registry_file.exists(), index=False)
        return True, f"Promoted candidate model {row['model_version']}"

    def maybe_retrain(self):
        closed_df = self._safe_read(self.closed_file)
        replay_df = self._safe_read(self.replay_file)
        backtest_df = self._safe_read(self.backtest_summary_file)

        closed_rows = len(closed_df)
        replay_rows = len(replay_df)
        new_closed = closed_rows - self._last_retrained_closed_count
        pnl_degraded = False
        if not backtest_df.empty and "average_pnl" in backtest_df.columns:
            pnl_degraded = self._safe_float(backtest_df.iloc[-1].get("average_pnl"), 0.0) < 0

        should_retrain = (
            new_closed >= self.closed_trade_threshold
            or replay_rows >= self.replay_threshold
            or pnl_degraded
        )

        if not should_retrain:
            self._write_status(
                closed_rows,
                replay_rows,
                f"Waiting for {self.closed_trade_threshold} new closed trades to retrain: "
                f"new_closed={new_closed}/{self.closed_trade_threshold}, "
                f"total_closed={closed_rows}, replay={replay_rows}/{self.replay_threshold}",
            )
            return False

        logging.info(
            "Retraining triggered: %d new closed trades (threshold=%d). Refreshing models...",
            new_closed,
            self.closed_trade_threshold,
        )
        Stage1Models(logs_dir=self.logs_dir).train()
        Stage2TemporalModels(logs_dir=self.logs_dir).train()
        train_model(timesteps=5000)
        promoted, message = self._promote_if_better(closed_rows, replay_rows)

        self._last_retrained_closed_count = closed_rows
        self._persist_last_retrained_closed_count()
        self._write_status(closed_rows, replay_rows, message)
        return promoted
