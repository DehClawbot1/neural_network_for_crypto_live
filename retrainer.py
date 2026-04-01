import json
import logging
import os
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
        state_payload = self._load_state_payload()
        self._last_retrained_closed_count = int(state_payload.get("last_retrained_closed_count", 0) or 0)
        self._last_retrained_at = str(state_payload.get("last_retrained_at") or "")

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

    def _load_state_payload(self):
        if not self.state_file.exists():
            return {}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _persist_state(self):
        payload = {
            "last_retrained_closed_count": int(self._last_retrained_closed_count),
            "last_retrained_at": str(self._last_retrained_at or ""),
        }
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

    def maybe_retrain(self, force=False, reason="scheduled_cycle_check"):
        closed_df = self._safe_read(self.closed_file)
        replay_df = self._safe_read(self.replay_file)
        backtest_df = self._safe_read(self.backtest_summary_file)

        closed_rows = len(closed_df)
        replay_rows = len(replay_df)
        new_closed = closed_rows - self._last_retrained_closed_count
        pnl_degraded = False
        if not backtest_df.empty and "average_pnl" in backtest_df.columns:
            pnl_degraded = self._safe_float(backtest_df.iloc[-1].get("average_pnl"), 0.0) < 0

        force_every_closed_trade = os.getenv("RETRAIN_ON_EVERY_CLOSED_TRADE", "true").strip().lower() in {"1", "true", "yes", "on"}
        if force and not force_every_closed_trade:
            force = False

        min_interval_seconds = max(0, int(os.getenv("RETRAIN_MIN_INTERVAL_SECONDS", "0") or 0))
        if self._last_retrained_at:
            try:
                last_retrained_ts = pd.to_datetime(self._last_retrained_at, utc=True, errors="coerce")
            except Exception:
                last_retrained_ts = pd.NaT
        else:
            last_retrained_ts = pd.NaT
        if pd.notna(last_retrained_ts) and min_interval_seconds > 0:
            seconds_since_last = (pd.Timestamp.now(tz="UTC") - last_retrained_ts).total_seconds()
            if seconds_since_last < min_interval_seconds:
                self._write_status(
                    closed_rows,
                    replay_rows,
                    f"Retrain deferred ({reason}): cooldown active {int(seconds_since_last)}s/{min_interval_seconds}s",
                )
                return False

        should_retrain = (
            force
            or
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

        rl_timesteps = max(
            256,
            int(
                os.getenv(
                    "FORCED_RETRAIN_RL_TIMESTEPS" if force else "RETRAIN_RL_TIMESTEPS",
                    "1500" if force else "5000",
                )
                or ("1500" if force else "5000")
            ),
        )
        logging.info(
            "Retraining triggered (%s): %d new closed trades (threshold=%d). Refreshing models with rl_timesteps=%d...",
            reason,
            new_closed,
            self.closed_trade_threshold,
            rl_timesteps,
        )
        Stage1Models(logs_dir=self.logs_dir).train()
        Stage2TemporalModels(logs_dir=self.logs_dir).train()
        train_model(timesteps=rl_timesteps)
        promoted, message = self._promote_if_better(closed_rows, replay_rows)

        self._last_retrained_closed_count = closed_rows
        self._last_retrained_at = pd.Timestamp.utcnow().isoformat()
        self._persist_state()
        self._write_status(closed_rows, replay_rows, f"{message} | reason={reason} | rl_timesteps={rl_timesteps}")
        return promoted
