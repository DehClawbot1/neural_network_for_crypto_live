from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from csv_utils import safe_csv_append

from trade_quality import classify_exit_reason_family

logger = logging.getLogger(__name__)


class PerformanceGovernor:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.closed_file = self.logs_dir / "closed_positions.csv"
        self.output_file = self.logs_dir / "performance_governor.csv"

    def _safe_read(self, path: Path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _env_float(self, name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)) or default)
        except Exception:
            return float(default)

    def _rolling_metrics(self, df: pd.DataFrame, window: int) -> dict:
        work = df.tail(window).copy()
        if work.empty:
            return {
                "window": int(window),
                "trades": 0,
                "win_rate": 0.0,
                "average_pnl": 0.0,
                "profit_factor": 0.0,
                "realized_drawdown": 0.0,
                "stop_loss_rate": 0.0,
                "forced_exit_rate": 0.0,
                "rl_exit_rate": 0.0,
                "operational_close_rate": 0.0,
            }
        pnl_col = "net_realized_pnl" if "net_realized_pnl" in work.columns else "realized_pnl"
        pnl = pd.to_numeric(work.get(pnl_col), errors="coerce").fillna(0.0)
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float((-pnl[pnl < 0]).sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        running = pnl.cumsum()
        peak = running.cummax()
        drawdown = (peak - running).max() if not running.empty else 0.0
        reasons = work.get("close_reason", pd.Series("", index=work.index)).fillna("").astype(str).str.strip().str.lower()
        exit_family = work.get("exit_reason_family", pd.Series("", index=work.index)).fillna("").astype(str).str.strip().str.lower()
        if exit_family.eq("").all():
            exit_family = reasons.map(classify_exit_reason_family)
        forced_mask = exit_family.isin({"hard_stop", "technical_invalidation", "time_stop"})
        stop_mask = reasons.eq("stop_loss") | reasons.eq("trajectory_panic_exit")
        rl_mask = exit_family.eq("rl_discretionary")
        operational_mask = exit_family.eq("operational")
        return {
            "window": int(window),
            "trades": int(len(work)),
            "win_rate": float((pnl > 0).mean()),
            "average_pnl": float(pnl.mean()),
            "profit_factor": float(profit_factor),
            "realized_drawdown": float(drawdown),
            "stop_loss_rate": float(stop_mask.mean()),
            "forced_exit_rate": float(forced_mask.mean()),
            "rl_exit_rate": float(rl_mask.mean()),
            "operational_close_rate": float(operational_mask.mean()),
        }

    def evaluate(self) -> dict:
        closed_df = self._safe_read(self.closed_file)
        work = closed_df.copy()
        if not work.empty and "closed_at" in work.columns:
            work["closed_at"] = pd.to_datetime(work["closed_at"], utc=True, errors="coerce")
            work = work.sort_values("closed_at")
        windows = [50, 100, 200]
        metrics = [self._rolling_metrics(work, window) for window in windows]

        lvl1 = {
            "min_win_rate": self._env_float("GOV_LEVEL1_MIN_WIN_RATE", 0.38),
            "min_profit_factor": self._env_float("GOV_LEVEL1_MIN_PROFIT_FACTOR", 0.80),
            "max_avg_pnl_loss": self._env_float("GOV_LEVEL1_MAX_NEGATIVE_AVG_PNL", -0.10),
            "max_drawdown": self._env_float("GOV_LEVEL1_MAX_DRAWDOWN", 30.0),
            "max_rl_exit_rate": self._env_float("GOV_LEVEL1_MAX_RL_EXIT_RATE", 0.55),
            "max_operational_close_rate": self._env_float("GOV_LEVEL1_MAX_OPERATIONAL_CLOSE_RATE", 0.25),
        }
        lvl2 = {
            "min_win_rate": self._env_float("GOV_LEVEL2_MIN_WIN_RATE", 0.30),
            "min_profit_factor": self._env_float("GOV_LEVEL2_MIN_PROFIT_FACTOR", 0.60),
            "max_avg_pnl_loss": self._env_float("GOV_LEVEL2_MAX_NEGATIVE_AVG_PNL", -0.25),
            "max_drawdown": self._env_float("GOV_LEVEL2_MAX_DRAWDOWN", 60.0),
            "max_rl_exit_rate": self._env_float("GOV_LEVEL2_MAX_RL_EXIT_RATE", 0.70),
            "max_operational_close_rate": self._env_float("GOV_LEVEL2_MAX_OPERATIONAL_CLOSE_RATE", 0.40),
        }

        failures_level1 = []
        failures_level2 = []
        for metric in metrics:
            if metric["trades"] < 10:
                continue
            if metric["win_rate"] < lvl1["min_win_rate"]:
                failures_level1.append(f"win_rate_{metric['window']}")
            if metric["profit_factor"] < lvl1["min_profit_factor"]:
                failures_level1.append(f"profit_factor_{metric['window']}")
            if metric["average_pnl"] <= lvl1["max_avg_pnl_loss"]:
                failures_level1.append(f"average_pnl_{metric['window']}")
            if metric["realized_drawdown"] > lvl1["max_drawdown"]:
                failures_level1.append(f"drawdown_{metric['window']}")
            if metric["rl_exit_rate"] > lvl1["max_rl_exit_rate"]:
                failures_level1.append(f"rl_exit_rate_{metric['window']}")
            if metric["operational_close_rate"] > lvl1["max_operational_close_rate"]:
                failures_level1.append(f"operational_close_rate_{metric['window']}")

            if metric["win_rate"] < lvl2["min_win_rate"]:
                failures_level2.append(f"win_rate_{metric['window']}")
            if metric["profit_factor"] < lvl2["min_profit_factor"]:
                failures_level2.append(f"profit_factor_{metric['window']}")
            if metric["average_pnl"] <= lvl2["max_avg_pnl_loss"]:
                failures_level2.append(f"average_pnl_{metric['window']}")
            if metric["realized_drawdown"] > lvl2["max_drawdown"]:
                failures_level2.append(f"drawdown_{metric['window']}")
            if metric["rl_exit_rate"] > lvl2["max_rl_exit_rate"]:
                failures_level2.append(f"rl_exit_rate_{metric['window']}")
            if metric["operational_close_rate"] > lvl2["max_operational_close_rate"]:
                failures_level2.append(f"operational_close_rate_{metric['window']}")

        level = 0
        if failures_level2:
            level = 2
        elif failures_level1:
            level = 1

        policy = {
            0: {"size_multiplier": 1.0, "min_confidence": 0.0, "min_liquidity_score": 0.0, "force_min_size": False, "top_signal_only": False},
            1: {"size_multiplier": self._env_float("GOV_LEVEL1_SIZE_MULTIPLIER", 0.65), "min_confidence": self._env_float("GOV_LEVEL1_MIN_ENTRY_CONFIDENCE", 0.06), "min_liquidity_score": self._env_float("GOV_LEVEL1_MIN_LIQUIDITY_SCORE", 0.10), "force_min_size": False, "top_signal_only": False},
            2: {"size_multiplier": self._env_float("GOV_LEVEL2_SIZE_MULTIPLIER", 0.45), "min_confidence": self._env_float("GOV_LEVEL2_MIN_ENTRY_CONFIDENCE", 0.05), "min_liquidity_score": self._env_float("GOV_LEVEL2_MIN_LIQUIDITY_SCORE", 0.12), "force_min_size": True, "top_signal_only": True},
        }[level]

        latest = metrics[-1] if metrics else {}
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "governor_level": int(level),
            "reason": ",".join(failures_level2 if level == 2 else failures_level1),
            "windows": metrics,
            **policy,
            "live_win_rate": float(latest.get("win_rate", 0.0) or 0.0),
            "live_average_pnl": float(latest.get("average_pnl", 0.0) or 0.0),
            "live_profit_factor": float(latest.get("profit_factor", 0.0) or 0.0),
            "live_realized_drawdown": float(latest.get("realized_drawdown", 0.0) or 0.0),
            "rl_exit_rate": float(latest.get("rl_exit_rate", 0.0) or 0.0),
            "operational_close_rate": float(latest.get("operational_close_rate", 0.0) or 0.0),
        }
        self._append_state(state)
        logger.info("PERFORMANCE_GOVERNOR %s", json.dumps({k: v for k, v in state.items() if k != "windows"}, separators=(",", ":"), sort_keys=True))
        return state

    def _append_state(self, state: dict):
        flat = dict(state)
        flat["windows_json"] = json.dumps(state.get("windows", []), separators=(",", ":"))
        flat.pop("windows", None)
        df = pd.DataFrame([flat])
        safe_csv_append(self.output_file, df)
