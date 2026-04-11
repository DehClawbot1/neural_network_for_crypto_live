import json
import logging
import os
from pathlib import Path

import pandas as pd

from model_artifact_staging import (
    PROMOTABLE_MODEL_FILENAMES,
    build_candidate_weights_dir,
    promote_candidate_artifacts,
)
from baseline_models import BaselineModels
from model_registry import ModelRegistry
from model_registry_runtime import evaluate_artifact_against_dataset
from rl_trainer import train_model
from supervised_models import SupervisedModels
from stage1_models import Stage1Models
from stage2_temporal_models import Stage2TemporalModels
from trade_quality import enrich_quality_frame
from weather_temperature_trainer import WeatherTemperatureTrainer

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

    def _env_float(self, name, default):
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        return self._safe_float(raw, default)

    def _env_int(self, name, default):
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return int(default)
        return int(self._safe_float(raw, default))

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

    def _append_csv_row(self, path, row, *, sort_by=None):
        existing = self._safe_read(path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        if sort_by and sort_by in updated.columns:
            try:
                sort_key = pd.to_datetime(updated[sort_by], errors="coerce", utc=True)
                # Keep legacy rows with blank timestamps at the top so the newest
                # real attempt remains at the bottom for tail-based inspection.
                updated = (
                    updated.assign(_sort_key=sort_key)
                    .sort_values("_sort_key", na_position="first")
                    .drop(columns=["_sort_key"])
                )
            except Exception:
                pass
        updated.to_csv(path, index=False)
        return updated

    def _emit_retrain_verdict(
        self,
        *,
        verdict,
        reason,
        closed_rows,
        replay_rows,
        rl_timesteps=0,
        candidate_row=None,
        promotion_block_reason="",
        candidate_beats_champion=None,
    ):
        candidate_row = self._with_live_summary(candidate_row)
        promotion_gate_passed = candidate_row.get("promotion_gate_passed")
        payload = {
            "verdict": verdict,
            "reason": reason,
            "closed_rows": int(closed_rows),
            "replay_rows": int(replay_rows),
            "rl_timesteps": int(rl_timesteps or 0),
            "model_version": candidate_row.get("model_version", ""),
            "activation_status": candidate_row.get("activation_status", verdict),
            "promotion_block_reason": promotion_block_reason or candidate_row.get("promotion_block_reason", ""),
            "candidate_beats_champion": candidate_beats_champion,
            "average_pnl": round(self._safe_float(candidate_row.get("average_pnl"), 0.0), 4),
            "profit_factor": round(self._safe_float(candidate_row.get("profit_factor"), 0.0), 4),
            "live_closed_trades": int(candidate_row.get("live_closed_trades", 0) or 0),
            "live_recent_closed_rows": int(candidate_row.get("live_recent_closed_rows", 0) or 0),
            "live_recent_quality_scope_closed_trades": int(candidate_row.get("live_recent_quality_scope_closed_trades", 0) or 0),
            "live_recent_reconciliation_close_count": int(candidate_row.get("live_recent_reconciliation_close_count", 0) or 0),
            "live_average_pnl": round(self._safe_float(candidate_row.get("live_average_pnl"), 0.0), 4),
            "live_profit_factor": round(self._safe_float(candidate_row.get("live_profit_factor"), 0.0), 4),
            "live_recent_reconciliation_close_ratio": round(self._safe_float(candidate_row.get("live_recent_reconciliation_close_ratio"), 0.0), 4),
            "live_recent_quality_scope_ratio": round(self._safe_float(candidate_row.get("live_recent_quality_scope_ratio"), 0.0), 4),
            "live_learning_eligible_ratio": round(self._safe_float(candidate_row.get("live_learning_eligible_ratio"), 0.0), 4),
            "live_entry_context_complete_ratio": round(self._safe_float(candidate_row.get("live_entry_context_complete_ratio"), 0.0), 4),
            "live_unknown_signal_label_ratio": round(self._safe_float(candidate_row.get("live_unknown_signal_label_ratio"), 0.0), 4),
            "live_operational_close_ratio": round(self._safe_float(candidate_row.get("live_operational_close_ratio"), 0.0), 4),
            "promotion_gate_evaluated": promotion_gate_passed is not None,
            "promotion_gate_passed": promotion_gate_passed,
            "promotion_gate_failures": candidate_row.get("promotion_gate_failures", ""),
            "promotion_gate_context": candidate_row.get("promotion_gate_context", ""),
        }
        logging.info("LATEST_RETRAIN_VERDICT %s", json.dumps(payload, separators=(",", ":"), sort_keys=True))

    def _retrain_btc_forecast(self, candidate_weights_dir):
        """Retrain BTC price forecast models (multi-timeframe if data available)."""
        # Load trade feedback weights for sample weighting
        feedback_weights = None
        try:
            from btc_trade_feedback import BTCTradeFeedback
            feedback_weights = BTCTradeFeedback(logs_dir=str(self.logs_dir)).get_sample_weights_for_retraining()
            if feedback_weights:
                logging.info("BTC retrain using trade feedback: accuracy=%.1f%% error_scale=%.3f",
                             feedback_weights.get("overall_accuracy", 0) * 100,
                             feedback_weights.get("error_scale", 1.0))
        except Exception as exc:
            logging.debug("BTC trade feedback not available for retraining: %s", exc)

        try:
            from btc_multitimeframe import BTCMultiTimeframeForecaster

            forecaster = BTCMultiTimeframeForecaster(
                weights_dir=str(candidate_weights_dir),
                logs_dir=str(self.logs_dir),
            )
            results = forecaster.train_all(enrich_derivatives=False, feedback_weights=feedback_weights)
            for tf, metrics in results.items():
                if "error" in metrics:
                    logging.warning("BTC %s retrain error: %s", tf, metrics["error"])
                else:
                    logging.info(
                        "BTC %s retrained: DirAcc=%.2f%% ClsAcc=%.2f%%",
                        tf,
                        metrics.get("direction_accuracy", 0) * 100,
                        metrics.get("classifier_accuracy", 0) * 100,
                    )
        except Exception as exc:
            # Fallback to single model if multi-timeframe fails
            try:
                from btc_forecast_model import BTCForecastModel
                from btc_price_dataset import BTCPriceDatasetBuilder

                builder = BTCPriceDatasetBuilder(logs_dir=str(self.logs_dir))
                dataset = builder.load_dataset()
                if dataset.empty or len(dataset) < 100:
                    logging.info("BTC forecast retrain skipped: dataset has %d rows", len(dataset))
                    return
                model = BTCForecastModel(weights_dir=str(candidate_weights_dir), logs_dir=str(self.logs_dir))
                metrics = model.train(dataset, feedback_weights=feedback_weights)
                logging.info("BTC forecast retrained (single model): %s", metrics)
            except Exception as exc2:
                logging.warning("BTC forecast retrain failed (non-blocking): %s / %s", exc, exc2)

    def _write_status(self, closed_rows: int, replay_rows: int, action: str):
        self.status_file.write_text(action + "\n", encoding="utf-8")
        progress_closed = round(closed_rows / self.closed_trade_threshold, 4) if self.closed_trade_threshold else 0
        progress_replay = round(replay_rows / self.replay_threshold, 4) if self.replay_threshold else 0
        self._append_csv_row(
            self.status_csv,
            {
                "attempted_at": pd.Timestamp.utcnow().isoformat(),
                "closed_trade_rows": closed_rows,
                "closed_trade_threshold": self.closed_trade_threshold,
                "replay_rows": replay_rows,
                "replay_threshold": self.replay_threshold,
                "progress_ratio": max(progress_closed, progress_replay),
                "last_action": action,
                "status_schema": "base_retrainer_v3",
            },
            sort_by="attempted_at",
        )

    def _record_model_status(
        self,
        *,
        attempted_at,
        closed_rows,
        replay_rows,
        action,
        reason,
        rl_timesteps=0,
        candidate_row=None,
        activation_status="blocked",
        promotion_block_reason="",
        candidate_beats_champion=None,
        extra_fields=None,
    ):
        candidate_row = self._with_live_summary(candidate_row)
        promotion_gate_passed = candidate_row.get("promotion_gate_passed")
        progress_closed = round(closed_rows / self.closed_trade_threshold, 4) if self.closed_trade_threshold else 0
        progress_replay = round(replay_rows / self.replay_threshold, 4) if self.replay_threshold else 0
        row = {
            "attempted_at": attempted_at,
            "closed_trade_rows": int(closed_rows),
            "closed_trade_threshold": int(self.closed_trade_threshold),
            "replay_rows": int(replay_rows),
            "replay_threshold": int(self.replay_threshold),
            "progress_ratio": max(progress_closed, progress_replay),
            "last_action": action,
            "trigger_reason": reason,
            "rl_timesteps": int(rl_timesteps or 0),
            "activation_status": activation_status,
            "promotion_block_reason": promotion_block_reason or "",
            "candidate_beats_champion": candidate_beats_champion,
            "model_version": candidate_row.get("model_version", ""),
            "average_pnl": self._safe_float(candidate_row.get("average_pnl"), 0.0),
            "profit_factor": self._safe_float(candidate_row.get("profit_factor"), 0.0),
            "max_drawdown": self._safe_float(candidate_row.get("max_drawdown"), 0.0),
            "test_accuracy": self._safe_float(candidate_row.get("test_accuracy"), 0.0),
            "live_closed_trades": int(candidate_row.get("live_closed_trades", 0) or 0),
            "live_recent_closed_rows": int(candidate_row.get("live_recent_closed_rows", 0) or 0),
            "live_recent_quality_scope_closed_trades": int(candidate_row.get("live_recent_quality_scope_closed_trades", 0) or 0),
            "live_recent_reconciliation_close_count": int(candidate_row.get("live_recent_reconciliation_close_count", 0) or 0),
            "live_win_rate": self._safe_float(candidate_row.get("live_win_rate"), 0.0),
            "live_average_pnl": self._safe_float(candidate_row.get("live_average_pnl"), 0.0),
            "live_profit_factor": self._safe_float(candidate_row.get("live_profit_factor"), 0.0),
            "live_recent_reconciliation_close_ratio": self._safe_float(candidate_row.get("live_recent_reconciliation_close_ratio"), 0.0),
            "live_recent_quality_scope_ratio": self._safe_float(candidate_row.get("live_recent_quality_scope_ratio"), 0.0),
            "live_learning_eligible_ratio": self._safe_float(candidate_row.get("live_learning_eligible_ratio"), 0.0),
            "live_entry_context_complete_ratio": self._safe_float(candidate_row.get("live_entry_context_complete_ratio"), 0.0),
            "live_unknown_signal_label_ratio": self._safe_float(candidate_row.get("live_unknown_signal_label_ratio"), 0.0),
            "live_operational_close_ratio": self._safe_float(candidate_row.get("live_operational_close_ratio"), 0.0),
            "promotion_gate_evaluated": promotion_gate_passed is not None,
            "promotion_gate_passed": promotion_gate_passed,
            "promotion_gate_failures": candidate_row.get("promotion_gate_failures", ""),
            "promotion_gate_context": candidate_row.get("promotion_gate_context", ""),
            "status_schema": "base_retrainer_v3",
        }
        if extra_fields:
            row.update(extra_fields)
        self._append_csv_row(self.status_csv, row, sort_by="attempted_at")
        return row

    def _with_live_summary(self, candidate_row=None):
        merged = dict(candidate_row or {})
        live_summary = self._build_live_validation_summary()
        for key, value in live_summary.items():
            if key not in merged or merged.get(key) in [None, "", 0, 0.0, False]:
                merged[key] = value
        if not merged.get("promotion_gate_context"):
            merged["promotion_gate_context"] = self._format_live_window_context(merged)
        return merged

    def _format_live_window_context(self, candidate_row) -> str:
        recent_closed_rows = int(candidate_row.get("live_recent_closed_rows", 0) or 0)
        if recent_closed_rows <= 0:
            return ""
        reconciliation_count = int(candidate_row.get("live_recent_reconciliation_close_count", 0) or 0)
        quality_scope_count = int(candidate_row.get("live_recent_quality_scope_closed_trades", 0) or 0)
        return (
            f"{reconciliation_count}/{recent_closed_rows} recent closes were reconciliation-driven; "
            f"evaluating quality gates on the remaining {quality_scope_count}."
        )

    def _profit_factor_from_pnl(self, pnl_series):
        pnl = pd.to_numeric(pnl_series, errors="coerce").dropna()
        if pnl.empty:
            return 0.0
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = abs(float(pnl[pnl < 0].sum()))
        if gross_loss <= 0:
            return gross_profit if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _build_live_validation_summary(self):
        closed_df = self._safe_read(self.closed_file)
        summary = {
            "live_closed_trades": 0,
            "live_recent_closed_rows": 0,
            "live_recent_quality_scope_closed_trades": 0,
            "live_recent_reconciliation_close_count": 0,
            "live_win_rate": 0.0,
            "live_average_pnl": 0.0,
            "live_profit_factor": 0.0,
            "live_validation_window": 0,
            "live_recent_reconciliation_close_ratio": 0.0,
            "live_recent_quality_scope_ratio": 0.0,
            "live_learning_eligible_ratio": 0.0,
            "live_entry_context_complete_ratio": 0.0,
            "live_unknown_signal_label_ratio": 0.0,
            "live_operational_close_ratio": 0.0,
        }
        if closed_df.empty:
            return summary

        work = closed_df.copy()
        if "closed_at" in work.columns:
            work["closed_at"] = pd.to_datetime(work["closed_at"], errors="coerce", utc=True)
            work = work.sort_values("closed_at")

        lookback = max(1, self._env_int("PROMOTION_LIVE_LOOKBACK_TRADES", 200))
        work = work.tail(lookback).copy()
        work = enrich_quality_frame(work, logs_dir=self.logs_dir)
        if work.empty:
            return summary
        summary["live_validation_window"] = int(lookback)
        summary["live_recent_closed_rows"] = int(len(work.index))
        reconciliation_mask = work.get("reconciliation_close_flag", pd.Series(False, index=work.index)).astype(bool)
        summary["live_recent_reconciliation_close_count"] = int(reconciliation_mask.sum())
        summary["live_recent_reconciliation_close_ratio"] = float(reconciliation_mask.mean()) if len(work.index) else 0.0
        quality_scope = work[~reconciliation_mask].copy()
        summary["live_recent_quality_scope_closed_trades"] = int(len(quality_scope.index))
        summary["live_recent_quality_scope_ratio"] = (
            float(len(quality_scope.index) / len(work.index)) if len(work.index) else 0.0
        )
        if quality_scope.empty:
            return summary
        learning_eligible_mask = quality_scope.get("learning_eligible", pd.Series(False, index=quality_scope.index)).astype(bool)
        complete_mask = quality_scope.get("entry_context_complete", pd.Series(False, index=quality_scope.index)).astype(bool)
        operational_mask = quality_scope.get("operational_close_flag", pd.Series(False, index=quality_scope.index)).astype(bool)
        quality_scope_signal = (
            quality_scope.get("signal_label", pd.Series("", index=quality_scope.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        summary["live_unknown_signal_label_ratio"] = (
            float(quality_scope_signal.isin({"", "UNKNOWN"}).mean()) if len(quality_scope.index) else 0.0
        )
        summary["live_learning_eligible_ratio"] = float(learning_eligible_mask.mean()) if len(quality_scope.index) else 0.0
        summary["live_entry_context_complete_ratio"] = float(complete_mask.mean()) if len(quality_scope.index) else 0.0
        summary["live_operational_close_ratio"] = float(operational_mask.mean()) if len(quality_scope.index) else 0.0
        work = quality_scope[learning_eligible_mask & complete_mask].copy()

        pnl_column = "net_realized_pnl" if "net_realized_pnl" in work.columns else "realized_pnl"
        pnl = pd.to_numeric(work.get(pnl_column), errors="coerce").dropna()
        if pnl.empty:
            return summary

        summary["live_closed_trades"] = int(len(pnl))
        summary["live_win_rate"] = float((pnl > 0).mean())
        summary["live_average_pnl"] = float(pnl.mean())
        summary["live_profit_factor"] = float(self._profit_factor_from_pnl(pnl))
        return summary

    def _evaluate_promotion_quality_gates(self, candidate_row):
        live_summary = self._build_live_validation_summary()
        candidate_row.update(live_summary)

        min_profit_factor = self._env_float("PROMOTION_MIN_PROFIT_FACTOR", 1.05)
        min_average_pnl = self._env_float("PROMOTION_MIN_AVERAGE_PNL", 0.0)
        min_live_closed_trades = max(1, self._env_int("PROMOTION_MIN_LIVE_CLOSED_TRADES", 25))
        min_live_profit_factor = self._env_float("PROMOTION_MIN_LIVE_PROFIT_FACTOR", 1.00)
        min_live_average_pnl = self._env_float("PROMOTION_MIN_LIVE_AVERAGE_PNL", 0.0)
        min_learning_eligible_ratio = self._env_float("PROMOTION_MIN_LEARNING_ELIGIBLE_RATIO", 0.65)
        min_entry_context_complete_ratio = self._env_float("PROMOTION_MIN_ENTRY_CONTEXT_COMPLETE_RATIO", 0.80)
        max_operational_close_ratio = self._env_float("PROMOTION_MAX_OPERATIONAL_CLOSE_RATIO", 0.30)
        max_unknown_signal_label_ratio = self._env_float("PROMOTION_MAX_UNKNOWN_SIGNAL_LABEL_RATIO", 0.20)

        failures = []
        candidate_profit_factor = self._safe_float(candidate_row.get("profit_factor"), 0.0)
        candidate_average_pnl = self._safe_float(candidate_row.get("average_pnl"), 0.0)
        if candidate_profit_factor < min_profit_factor:
            failures.append(
                f"profit_factor {candidate_profit_factor:.3f} < required {min_profit_factor:.3f}"
            )
        if candidate_average_pnl <= min_average_pnl:
            failures.append(
                f"average_pnl {candidate_average_pnl:.4f} <= required {min_average_pnl:.4f}"
            )

        live_closed_trades = int(candidate_row.get("live_closed_trades", 0) or 0)
        live_profit_factor = self._safe_float(candidate_row.get("live_profit_factor"), 0.0)
        live_average_pnl = self._safe_float(candidate_row.get("live_average_pnl"), 0.0)
        live_learning_eligible_ratio = self._safe_float(candidate_row.get("live_learning_eligible_ratio"), 0.0)
        live_entry_context_complete_ratio = self._safe_float(candidate_row.get("live_entry_context_complete_ratio"), 0.0)
        live_operational_close_ratio = self._safe_float(candidate_row.get("live_operational_close_ratio"), 1.0)
        live_unknown_signal_label_ratio = self._safe_float(candidate_row.get("live_unknown_signal_label_ratio"), 1.0)
        if live_closed_trades < min_live_closed_trades:
            failures.append(
                f"live_closed_trades {live_closed_trades} < required {min_live_closed_trades}"
            )
        if live_profit_factor < min_live_profit_factor:
            failures.append(
                f"live_profit_factor {live_profit_factor:.3f} < required {min_live_profit_factor:.3f}"
            )
        if live_average_pnl <= min_live_average_pnl:
            failures.append(
                f"live_average_pnl {live_average_pnl:.4f} <= required {min_live_average_pnl:.4f}"
            )
        if live_learning_eligible_ratio < min_learning_eligible_ratio:
            failures.append(
                f"live_learning_eligible_ratio {live_learning_eligible_ratio:.3f} < required {min_learning_eligible_ratio:.3f}"
            )
        if live_entry_context_complete_ratio < min_entry_context_complete_ratio:
            failures.append(
                f"live_entry_context_complete_ratio {live_entry_context_complete_ratio:.3f} < required {min_entry_context_complete_ratio:.3f}"
            )
        if live_operational_close_ratio > max_operational_close_ratio:
            failures.append(
                f"live_operational_close_ratio {live_operational_close_ratio:.3f} > allowed {max_operational_close_ratio:.3f}"
            )
        if live_unknown_signal_label_ratio > max_unknown_signal_label_ratio:
            failures.append(
                f"live_unknown_signal_label_ratio {live_unknown_signal_label_ratio:.3f} > allowed {max_unknown_signal_label_ratio:.3f}"
            )

        candidate_row["promotion_gate_context"] = self._format_live_window_context(candidate_row)
        candidate_row["promotion_gate_passed"] = not failures
        candidate_row["promotion_gate_failures"] = " | ".join(failures)
        return not failures, failures

    def _normalize_registry(self, registry: pd.DataFrame):
        if registry is None or registry.empty:
            return pd.DataFrame()
        required = ["model_version"]
        if not all(col in registry.columns for col in required):
            return pd.DataFrame()
        work = registry.copy()
        work["model_version"] = work["model_version"].astype(str).str.strip()
        if "attempted_at" in work.columns:
            work["attempted_at"] = pd.to_datetime(work["attempted_at"], errors="coerce", utc=True)
        else:
            work["attempted_at"] = pd.NaT
        if "promoted_at" in work.columns:
            work["promoted_at"] = pd.to_datetime(work["promoted_at"], errors="coerce", utc=True)
        else:
            work["promoted_at"] = pd.NaT
        work = work[work["model_version"] != ""].copy()
        if work.empty:
            return pd.DataFrame()
        sort_col = "attempted_at" if work["attempted_at"].notna().any() else "promoted_at"
        return work.sort_values(sort_col)

    def _current_registry_row(self):
        registry = self._safe_read(self.registry_file)
        if registry.empty:
            return None
        registry = self._normalize_registry(registry)
        if registry.empty:
            return None
        if "activation_status" in registry.columns:
            promoted = registry[registry["activation_status"].astype(str).str.lower().isin({"promoted", "active", ""})]
            if not promoted.empty:
                return promoted.iloc[-1].to_dict()
            return None
        return registry.iloc[-1].to_dict()

    def _build_candidate_registry_row(self, closed_rows, replay_rows):
        backtest_df = self._safe_read(self.backtest_summary_file)
        time_split_df = self._safe_read(self.time_split_file)
        if backtest_df.empty:
            return None, None, "Candidate metrics unavailable for promotion."

        candidate = backtest_df.iloc[-1].to_dict()
        if not time_split_df.empty:
            candidate.update(time_split_df.iloc[-1].to_dict())
        champion = self._current_registry_row()

        row = {
            "model_version": pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S"),
            "training_data_window": "rolling_outcome_based",
            "replay_rows_used": replay_rows,
            "closed_trades_used": closed_rows,
            "average_pnl": self._safe_float(candidate.get("average_pnl"), 0.0),
            "profit_factor": self._safe_float(candidate.get("profit_factor"), 0.0),
            "max_drawdown": self._safe_float(candidate.get("max_drawdown"), 0.0),
            "test_accuracy": self._safe_float(candidate.get("test_accuracy"), 0.0),
            "attempted_at": pd.Timestamp.utcnow().isoformat(),
            "promoted_at": "",
            "activation_status": "candidate",
            "promotion_block_reason": "",
        }
        gates_passed, failures = self._evaluate_promotion_quality_gates(row)
        row["live_validation_passed"] = gates_passed
        row["live_validation_failures"] = " | ".join(failures)
        return row, champion, None

    def _candidate_beats_champion(self, candidate_row, champion):
        if champion is None:
            return True

        candidate_avg_pnl = self._safe_float(candidate_row.get("average_pnl"), 0.0)
        candidate_profit_factor = self._safe_float(candidate_row.get("profit_factor"), 0.0)
        candidate_max_drawdown = self._safe_float(candidate_row.get("max_drawdown"), -999.0)
        champion_avg_pnl = self._safe_float(champion.get("average_pnl"), -999.0)
        champion_profit_factor = self._safe_float(champion.get("profit_factor"), 0.0)
        champion_max_drawdown = self._safe_float(champion.get("max_drawdown"), -999.0)
        return (
            candidate_avg_pnl >= champion_avg_pnl
            and candidate_profit_factor >= champion_profit_factor
            and candidate_max_drawdown >= champion_max_drawdown
        )

    def _register_attempt(self, row):
        self._append_csv_row(self.registry_file, row, sort_by="attempted_at")
        return f"Recorded candidate model {row['model_version']} ({row.get('activation_status', 'unknown')})"

    def _promote_if_better(self, closed_rows, replay_rows):
        candidate_row, champion, error_message = self._build_candidate_registry_row(closed_rows, replay_rows)
        if candidate_row is None:
            return False, error_message
        if not self._candidate_beats_champion(candidate_row, champion):
            return False, "Candidate did not beat champion on promotion metrics."
        candidate_row["activation_status"] = "promoted"
        candidate_row["promotion_block_reason"] = ""
        return True, self._register_attempt(candidate_row)

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
                action = f"Retrain deferred ({reason}): cooldown active {int(seconds_since_last)}s/{min_interval_seconds}s"
                self._write_status(closed_rows, replay_rows, action)
                self._emit_retrain_verdict(
                    verdict="skipped_cooldown",
                    reason=reason,
                    closed_rows=closed_rows,
                    replay_rows=replay_rows,
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
            action = (
                f"Waiting for {self.closed_trade_threshold} new closed trades to retrain: "
                f"new_closed={new_closed}/{self.closed_trade_threshold}, "
                f"total_closed={closed_rows}, replay={replay_rows}/{self.replay_threshold}"
            )
            self._write_status(closed_rows, replay_rows, action)
            self._emit_retrain_verdict(
                verdict="waiting_threshold",
                reason=reason,
                closed_rows=closed_rows,
                replay_rows=replay_rows,
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
        candidate_weights_dir = build_candidate_weights_dir(self.weights_dir, prefix="base_retrain")
        SupervisedModels(logs_dir=self.logs_dir, weights_dir=candidate_weights_dir).train()
        Stage1Models(logs_dir=self.logs_dir, weights_dir=candidate_weights_dir).train()
        Stage2TemporalModels(logs_dir=self.logs_dir, weights_dir=candidate_weights_dir).train()
        BaselineModels(logs_dir=self.logs_dir, weights_dir=candidate_weights_dir).train()
        WeatherTemperatureTrainer(logs_dir=self.logs_dir, weights_dir=candidate_weights_dir).train()
        train_model(timesteps=rl_timesteps, save_path=candidate_weights_dir / "ppo_polytrader")
        self._retrain_btc_forecast(candidate_weights_dir)

        candidate_row, champion, error_message = self._build_candidate_registry_row(closed_rows, replay_rows)
        attempted_at = pd.Timestamp.utcnow().isoformat()
        candidate_beats_champion = None
        if candidate_row is None:
            promoted = False
            message = error_message
            promotion_block_reason = error_message or "candidate_metrics_unavailable"
        elif not bool(candidate_row.get("promotion_gate_passed", False)):
            promoted = False
            candidate_row["attempted_at"] = attempted_at
            candidate_row["activation_status"] = "blocked_quality_gate"
            gate_failure_reason = candidate_row.get("promotion_gate_failures") or "quality_gate_failed"
            gate_context = candidate_row.get("promotion_gate_context", "")
            candidate_row["promotion_block_reason"] = (
                f"{gate_failure_reason} | {gate_context}" if gate_context else gate_failure_reason
            )
            message = (
                "Candidate failed promotion quality gates: "
                f"{gate_failure_reason}"
            )
            if gate_context:
                message = f"{message} | {gate_context}"
            promotion_block_reason = candidate_row["promotion_block_reason"]
        elif not self._candidate_beats_champion(candidate_row, champion):
            promoted = False
            candidate_beats_champion = False
            candidate_row["attempted_at"] = attempted_at
            candidate_row["activation_status"] = "blocked_champion"
            candidate_row["promotion_block_reason"] = "candidate_did_not_beat_champion"
            message = "Candidate did not beat champion on promotion metrics."
            promotion_block_reason = candidate_row["promotion_block_reason"]
        else:
            candidate_beats_champion = True
            promotion_result = promote_candidate_artifacts(
                candidate_weights_dir,
                self.weights_dir,
                filenames=PROMOTABLE_MODEL_FILENAMES,
                backup_label="base_retrain",
            )
            promoted_files = promotion_result.get("promoted_files", [])
            if not promoted_files:
                promoted = False
                candidate_row["attempted_at"] = attempted_at
                candidate_row["activation_status"] = "blocked_missing_artifacts"
                candidate_row["promotion_block_reason"] = "no_staged_artifacts_produced"
                message = "Candidate promotion skipped: no staged artifacts were produced."
                promotion_block_reason = candidate_row["promotion_block_reason"]
            else:
                promoted = True
                candidate_row["attempted_at"] = attempted_at
                candidate_row["promoted_at"] = attempted_at
                candidate_row["activation_status"] = "promoted"
                candidate_row["promotion_block_reason"] = ""
                registry_message = self._register_attempt(candidate_row)
                message = (
                    f"{registry_message} | activated={len(promoted_files)} "
                    f"| rollback_dir={promotion_result.get('rollback_dir')}"
                )
                promotion_block_reason = ""

        if candidate_row is not None and not promoted:
            self._register_attempt(candidate_row)

        self._last_retrained_closed_count = closed_rows
        self._last_retrained_at = pd.Timestamp.utcnow().isoformat()
        self._persist_state()
        self.status_file.write_text(f"{message} | reason={reason} | rl_timesteps={rl_timesteps}\n", encoding="utf-8")
        self._record_model_status(
            attempted_at=attempted_at,
            closed_rows=closed_rows,
            replay_rows=replay_rows,
            action=message,
            reason=reason,
            rl_timesteps=rl_timesteps,
            candidate_row=candidate_row,
            activation_status=(
                candidate_row.get("activation_status")
                if candidate_row is not None
                else ("promoted" if promoted else "blocked")
            ),
            promotion_block_reason=promotion_block_reason,
            candidate_beats_champion=candidate_beats_champion,
        )
        self._emit_retrain_verdict(
            verdict=("promoted" if promoted else "blocked"),
            reason=reason,
            closed_rows=closed_rows,
            replay_rows=replay_rows,
            rl_timesteps=rl_timesteps,
            candidate_row=candidate_row,
            promotion_block_reason=promotion_block_reason,
            candidate_beats_champion=candidate_beats_champion,
        )

        registry = ModelRegistry(logs_dir=str(self.logs_dir))
        registry_run_id = str(candidate_row.get("model_version") if candidate_row else pd.Timestamp.utcnow().strftime("retrain_%Y%m%d%H%M%S"))
        registry_rows = []
        registry_rows.extend(
            evaluate_artifact_against_dataset(
                run_id=registry_run_id,
                dataset_file=self.logs_dir / "contract_targets.csv",
                artifact_path=candidate_weights_dir / "tp_classifier.joblib",
                artifact_group="btc_tabular_classifier",
                market_family="btc",
                target_col="tp_before_sl_60m",
            ).to_dict("records")
        )
        registry_rows.extend(
            evaluate_artifact_against_dataset(
                run_id=registry_run_id,
                dataset_file=self.logs_dir / "contract_targets.csv",
                artifact_path=candidate_weights_dir / "return_regressor.joblib",
                artifact_group="btc_tabular_regressor",
                market_family="btc",
                target_col="forward_return_15m",
            ).to_dict("records")
        )
        registry_rows.extend(
            evaluate_artifact_against_dataset(
                run_id=registry_run_id,
                dataset_file=self.logs_dir / "contract_targets.csv",
                artifact_path=candidate_weights_dir / "stage1_tp_classifier.joblib",
                artifact_group="stage1_classifier",
                market_family="btc",
                target_col="tp_before_sl_60m",
            ).to_dict("records")
        )
        registry_rows.extend(
            evaluate_artifact_against_dataset(
                run_id=registry_run_id,
                dataset_file=self.logs_dir / "contract_targets.csv",
                artifact_path=candidate_weights_dir / "stage1_return_regressor.joblib",
                artifact_group="stage1_regressor",
                market_family="btc",
                target_col="forward_return_15m",
            ).to_dict("records")
        )
        registry_rows.extend(
            evaluate_artifact_against_dataset(
                run_id=registry_run_id,
                dataset_file=self.logs_dir / "sequence_dataset.csv",
                artifact_path=candidate_weights_dir / "stage2_temporal_classifier.joblib",
                artifact_group="stage2_temporal_classifier",
                market_family="btc",
                target_col="tp_before_sl_60m",
            ).to_dict("records")
        )
        registry_rows.extend(
            evaluate_artifact_against_dataset(
                run_id=registry_run_id,
                dataset_file=self.logs_dir / "sequence_dataset.csv",
                artifact_path=candidate_weights_dir / "stage2_temporal_regressor.joblib",
                artifact_group="stage2_temporal_regressor",
                market_family="btc",
                target_col="forward_return_15m",
            ).to_dict("records")
        )
        registry_rows.extend(
            evaluate_artifact_against_dataset(
                run_id=registry_run_id,
                dataset_file=self.logs_dir / "contract_targets.csv",
                artifact_path=candidate_weights_dir / "weather_temperature_model.joblib",
                artifact_group="weather_temperature_classifier",
                market_family="weather_temperature",
                target_col="target_up",
                market_family_prefix="weather_temperature",
            ).to_dict("records")
        )
        baseline_df = self._safe_read(self.logs_dir / "baseline_eval.csv")
        if not baseline_df.empty:
            baseline_df = baseline_df.copy()
            baseline_df["run_id"] = registry_run_id
            baseline_df["promotion_status"] = "evaluation_only"
            baseline_df["promotion_reason"] = "baseline_report_only"
            baseline_df["beats_champion"] = None
            baseline_df["is_champion"] = False
            baseline_df["promotion_gate_passed"] = None
            baseline_df["notes"] = baseline_df.get("notes", pd.Series("", index=baseline_df.index)).fillna("")
            registry_rows.extend(baseline_df.to_dict("records"))
        if registry_rows:
            for row in registry_rows:
                if str(row.get("promotion_status") or "") == "evaluation_only":
                    continue
                row["promotion_status"] = "promoted" if promoted else "blocked"
                row["promotion_reason"] = "" if promoted else promotion_block_reason
                row["beats_champion"] = candidate_beats_champion
                row["is_champion"] = bool(promoted)
                row["promotion_gate_passed"] = bool(candidate_row.get("promotion_gate_passed")) if candidate_row else None
                row["notes"] = reason
            registry.register_rows(registry_rows)
            registry.write_regime_model_comparison()
            registry.write_decision_profit_audit()
        return promoted
