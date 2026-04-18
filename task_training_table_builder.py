from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TASK_SPECS = {
    "entry_edge": {
        "eligible_only": True,
        "required_labels": ["entry_quality", "entry_edge_realized", "trade_outcome_label"],
    },
    "fill_probability": {
        "eligible_only": False,
        "required_labels": ["actual_execution_path"],
    },
    "slippage_liquidity": {
        "eligible_only": False,
        "required_labels": ["execution_quality", "slippage_error"],
    },
    "exit_quality": {
        "eligible_only": True,
        "required_labels": ["exit_quality", "exit_regret"],
    },
    "regime_calibration": {
        "eligible_only": False,
        "required_labels": ["technical_regime_bucket"],
    },
}


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _truthy_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "on"})


class TaskTrainingTableBuilder:
    """Split canonical family datasets into task-specific offline training tables."""

    def __init__(self, logs_dir: str | Path = "logs") -> None:
        self.logs_dir = Path(logs_dir)
        self.families = ("btc", "weather_temperature")

    def _family_canonical_path(self, family: str) -> Path:
        return self.logs_dir / family / "canonical_dataset.csv"

    def _task_dir(self, family: str) -> Path:
        return self.logs_dir / family / "task_training"

    def _build_task_frame(self, df: pd.DataFrame, task_name: str) -> pd.DataFrame:
        spec = TASK_SPECS[task_name]
        task_df = df.copy()

        if spec["eligible_only"]:
            task_df = task_df[_truthy_series(task_df, "learning_eligible")].copy()

        required = [c for c in spec["required_labels"] if c in task_df.columns]
        if required:
            task_df = task_df.dropna(subset=required, how="any").copy()

        if task_name == "fill_probability" and "actual_execution_path" in task_df.columns:
            task_df["fill_success"] = (
                task_df["actual_execution_path"].astype(str).str.lower() == "live_exit_filled"
            ).astype(int)

        if task_name == "slippage_liquidity" and "actual_execution_path" in task_df.columns:
            task_df = task_df[
                task_df["actual_execution_path"].astype(str).str.lower().ne("")
            ].copy()

        task_df["task_model_name"] = task_name
        task_df["task_learning_eligible"] = True
        return task_df.reset_index(drop=True)

    def write(self) -> dict[str, dict[str, pd.DataFrame]]:
        results: dict[str, dict[str, pd.DataFrame]] = {}
        for family in self.families:
            canonical = _safe_read(self._family_canonical_path(family))
            if canonical.empty:
                continue
            family_results: dict[str, pd.DataFrame] = {}
            out_dir = self._task_dir(family)
            out_dir.mkdir(parents=True, exist_ok=True)
            for task_name in TASK_SPECS:
                task_df = self._build_task_frame(canonical, task_name)
                out_path = out_dir / f"{task_name}.csv"
                task_df.to_csv(out_path, index=False)
                family_results[task_name] = task_df
                logger.info(
                    "Task table [%s/%s]: %d rows -> %s",
                    family,
                    task_name,
                    len(task_df),
                    out_path,
                )
            results[family] = family_results
        return results
