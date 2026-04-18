from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
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

    def _bootstrap_weather_canonical(self) -> pd.DataFrame:
        contract_path = self.logs_dir / "weather_temperature" / "contract_targets.csv"
        hist_path = self.logs_dir / "weather_temperature" / "historical_dataset.csv"
        decisions_path = self.logs_dir / "weather_temperature" / "candidate_decisions.csv"

        contracts = _safe_read(contract_path)
        if contracts.empty:
            return pd.DataFrame()

        contracts = contracts.copy()
        if "timestamp" in contracts.columns:
            contracts["timestamp"] = pd.to_datetime(contracts["timestamp"], errors="coerce", utc=True)
        else:
            return pd.DataFrame()

        if "target_up" not in contracts.columns:
            return pd.DataFrame()

        contracts = contracts[contracts["target_up"].notna()].copy()
        if contracts.empty:
            return pd.DataFrame()

        contracts["weather_contract_resolved_yes"] = pd.to_numeric(contracts["target_up"], errors="coerce")
        contracts["market_family"] = "weather_temperature"
        if "market" not in contracts.columns and "market_title" in contracts.columns:
            contracts["market"] = contracts["market_title"]
        if "signal_timestamp" not in contracts.columns:
            contracts["signal_timestamp"] = contracts["timestamp"]
        if "fill_timestamp" not in contracts.columns:
            contracts["fill_timestamp"] = contracts["timestamp"]
        if "close_timestamp" not in contracts.columns:
            contracts["close_timestamp"] = contracts["timestamp"]

        decisions = _safe_read(decisions_path)
        if not decisions.empty:
            decisions = decisions.copy()
            if "created_at" in decisions.columns:
                decisions["timestamp"] = pd.to_datetime(decisions["created_at"], errors="coerce", utc=True)
            join_keys = [c for c in ["token_id", "condition_id", "outcome_side"] if c in contracts.columns and c in decisions.columns]
            if "timestamp" in decisions.columns and join_keys:
                decisions = decisions[decisions["timestamp"].notna()].copy()
                contracts = pd.merge_asof(
                    contracts[contracts["timestamp"].notna()].sort_values("timestamp"),
                    decisions.sort_values("timestamp"),
                    on="timestamp",
                    by=join_keys,
                    direction="backward",
                    tolerance=pd.Timedelta("14d"),
                    suffixes=("", "_decision"),
                )
                for left_col, right_col in [
                    ("market", "market_decision"),
                    ("confidence", "confidence_decision"),
                    ("p_tp_before_sl", "p_tp_before_sl_decision"),
                    ("expected_return", "expected_return_decision"),
                    ("final_decision", "final_decision_decision"),
                ]:
                    if right_col in contracts.columns:
                        if left_col not in contracts.columns:
                            contracts[left_col] = contracts[right_col]
                        else:
                            contracts[left_col] = contracts[left_col].fillna(contracts[right_col])

        hist = _safe_read(hist_path)
        if not hist.empty and "timestamp" in hist.columns:
            hist = hist.copy()
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce", utc=True)
            join_keys = [c for c in ["token_id", "condition_id", "outcome_side"] if c in contracts.columns and c in hist.columns]
            if join_keys:
                hist = hist[hist["timestamp"].notna()].copy()
                contracts = pd.merge_asof(
                    contracts[contracts["timestamp"].notna()].sort_values("timestamp"),
                    hist.sort_values("timestamp"),
                    on="timestamp",
                    by=join_keys,
                    direction="backward",
                    tolerance=pd.Timedelta("14d"),
                    suffixes=("", "_hist"),
                )
                for col in ["market_title", "weather_location", "weather_question_type", "liquidity_score", "volume_score", "open_positions_count"]:
                    hist_col = f"{col}_hist"
                    if hist_col in contracts.columns:
                        if col not in contracts.columns:
                            contracts[col] = contracts[hist_col]
                        else:
                            contracts[col] = contracts[col].fillna(contracts[hist_col])

        market_prob = pd.to_numeric(contracts.get("weather_market_probability", 0.5), errors="coerce").fillna(0.5)
        resolved_yes = pd.to_numeric(contracts["weather_contract_resolved_yes"], errors="coerce").fillna(0.0)
        contracts["entry_edge_realized"] = (resolved_yes - market_prob).clip(lower=-1.0, upper=1.0)
        contracts["trade_outcome_label"] = np.where(
            contracts["entry_edge_realized"] > 0,
            "win",
            np.where(contracts["entry_edge_realized"] < 0, "loss", "flat"),
        )
        contracts["entry_quality"] = (contracts["entry_edge_realized"] > 0).astype(int)
        contracts["exit_quality"] = resolved_yes.astype(int)
        contracts["exit_regret"] = (market_prob - resolved_yes).abs()
        contracts["learning_eligible"] = True
        contracts["contaminated_learning_row"] = False
        contracts["learning_exclusion_reason"] = "clean"
        contracts["reconciliation_artifact_flag"] = False
        contracts["operational_close_flag"] = False
        contracts["actual_execution_path"] = ""
        contracts["slippage_error"] = np.nan
        contracts["execution_quality"] = np.nan
        contracts["signal_label"] = contracts.get("signal_label", contracts.get("final_decision", "WEATHER_RESOLVED"))

        regime_source = None
        for candidate in [
            "forecast_uncertainty_c",
            "weather_forecast_stability_score",
            "time_left",
            "expected_return",
            "liquidity_score",
        ]:
            if candidate in contracts.columns:
                series = pd.to_numeric(contracts[candidate], errors="coerce")
                if series.notna().sum() >= 30 and series.nunique(dropna=True) >= 3:
                    regime_source = series
                    break
        if regime_source is not None:
            ranked = regime_source.rank(method="first")
            buckets = pd.qcut(ranked, q=3, labels=["stable", "balanced", "uncertain"])
            contracts["technical_regime_bucket"] = buckets.astype(str)
        else:
            contracts["technical_regime_bucket"] = contracts.get("weather_question_type", "weather_contract").fillna("weather_contract")

        return contracts.reset_index(drop=True)

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
            if canonical.empty and family == "weather_temperature":
                canonical = self._bootstrap_weather_canonical()
                if not canonical.empty:
                    out_path = self._family_canonical_path(family)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    canonical.to_csv(out_path, index=False)
                    logger.info(
                        "Bootstrapped weather canonical dataset: %d rows -> %s",
                        len(canonical),
                        out_path,
                    )
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
