"""
BTC Forecast Walk-Forward Live Evaluator

Logs every prediction vs actual outcome to track whether the model
is improving or degrading in production.

Each cycle:
  1. Record the current prediction (direction, confidence, predicted return)
  2. When enough time has passed (15 candles = 3.75h for 15m), look back
     and compare the prediction to what actually happened
  3. Append the result to logs/btc_forecast_eval.csv

Metrics tracked per prediction:
  - predicted_direction, predicted_return, confidence
  - actual_return, actual_direction
  - correct (bool), confident_correct (bool)
  - Rolling accuracy over last 50/200 predictions

Usage in supervisor.py:
    from btc_forecast_eval import BTCForecastEvaluator
    evaluator = BTCForecastEvaluator()

    # After prediction:
    evaluator.record_prediction(btc_fc, current_price)

    # Each cycle (evaluates any matured predictions):
    evaluator.evaluate_matured(current_price)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BTCForecastEvaluator:
    """
    Walk-forward live evaluator for BTC price predictions.

    Records predictions in a pending queue, then evaluates them
    once the forecast horizon has elapsed (default: 15 candles = 3.75h).
    """

    def __init__(
        self,
        logs_dir: str = "logs",
        horizon_seconds: int = 15 * 15 * 60,  # 15 candles * 15 min = 3.75 hours
        confidence_threshold: float = 0.52,
    ):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.eval_path = self.logs_dir / "btc_forecast_eval.csv"
        self.horizon_seconds = horizon_seconds
        self.confidence_threshold = confidence_threshold

        # Pending predictions waiting to be evaluated
        self._pending: deque[dict] = deque(maxlen=500)
        self._lock = threading.Lock()

        # Rolling accuracy tracking (in-memory)
        self._recent_results: deque[dict] = deque(maxlen=200)

        # Load existing eval data to seed rolling stats
        self._load_recent_results()

    # ------------------------------------------------------------------
    # Record a new prediction
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        prediction: dict,
        current_price: float,
        source: str = "multi_timeframe",
    ) -> None:
        """
        Record a new prediction for future evaluation.

        Args:
            prediction: dict from BTCMultiTimeframeForecaster.predict() or BTCForecastModel.predict()
            current_price: BTC price at prediction time
            source: label for where the prediction came from
        """
        if not prediction.get("btc_forecast_ready"):
            return

        entry = {
            "predict_ts": datetime.now(timezone.utc).isoformat(),
            "predict_epoch": time.time(),
            "entry_price": current_price,
            "predicted_direction": prediction.get("btc_predicted_direction", 0),
            "predicted_return": prediction.get("btc_predicted_return_15", 0.0),
            "confidence": prediction.get("btc_forecast_confidence", 0.0),
            "source": source,
            # Multi-timeframe metadata
            "mtf_agreement": prediction.get("btc_mtf_agreement", 0.0),
            "mtf_n_agree": prediction.get("btc_mtf_n_agree", 0),
            "mtf_n_total": prediction.get("btc_mtf_n_total", 0),
            "mtf_source": prediction.get("btc_mtf_source", "unknown"),
        }

        with self._lock:
            self._pending.append(entry)

        logger.debug(
            "BTCForecastEval: recorded prediction dir=%d conf=%.3f price=%.2f",
            entry["predicted_direction"], entry["confidence"], current_price,
        )

    # ------------------------------------------------------------------
    # Evaluate matured predictions
    # ------------------------------------------------------------------

    def evaluate_matured(self, current_price: float) -> list[dict]:
        """
        Check pending predictions and evaluate any that have matured
        (enough time has passed since prediction).

        Args:
            current_price: current BTC price

        Returns:
            List of evaluated prediction dicts (empty if none matured).
        """
        now = time.time()
        matured = []

        with self._lock:
            remaining = deque(maxlen=500)
            for entry in self._pending:
                age = now - entry["predict_epoch"]
                if age >= self.horizon_seconds:
                    # This prediction has matured — evaluate it
                    result = self._evaluate_single(entry, current_price, now)
                    matured.append(result)
                else:
                    remaining.append(entry)
            self._pending = remaining

        # Append results to CSV and rolling tracker
        for result in matured:
            self._append_result(result)
            self._recent_results.append(result)

        if matured:
            stats = self.rolling_stats()
            logger.info(
                "BTCForecastEval: evaluated %d predictions | "
                "Rolling accuracy: %.1f%% (last %d) | "
                "Confident accuracy: %.1f%% | Avg return: %.4f%%",
                len(matured),
                stats.get("accuracy_pct", 0),
                stats.get("n_evaluated", 0),
                stats.get("confident_accuracy_pct", 0),
                stats.get("avg_actual_return_pct", 0),
            )

        return matured

    def _evaluate_single(self, entry: dict, exit_price: float, now: float) -> dict:
        """Evaluate a single matured prediction."""
        entry_price = entry["entry_price"]
        predicted_dir = entry["predicted_direction"]
        confidence = entry["confidence"]

        # Actual outcome
        if entry_price > 0:
            actual_return = (exit_price - entry_price) / entry_price
        else:
            actual_return = 0.0

        actual_direction = 1 if actual_return > 0 else (-1 if actual_return < 0 else 0)

        # Correctness
        correct = (predicted_dir == actual_direction) if predicted_dir != 0 else False
        is_confident = confidence >= self.confidence_threshold
        confident_correct = correct if is_confident else None  # None = not confident enough to count

        # Profit if we traded the prediction
        if predicted_dir != 0:
            pnl_pct = actual_return * predicted_dir  # positive if correct direction
        else:
            pnl_pct = 0.0

        result = {
            **entry,
            "eval_ts": datetime.now(timezone.utc).isoformat(),
            "eval_epoch": now,
            "exit_price": exit_price,
            "actual_return": round(actual_return, 6),
            "actual_direction": actual_direction,
            "correct": correct,
            "is_confident": is_confident,
            "confident_correct": confident_correct,
            "pnl_pct": round(pnl_pct, 6),
            "horizon_seconds": round(now - entry["predict_epoch"], 1),
        }

        return result

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def rolling_stats(self, window: int | None = None) -> dict:
        """
        Compute rolling evaluation statistics.

        Args:
            window: number of recent predictions to consider (None = all in buffer)

        Returns:
            Dict with accuracy metrics.
        """
        results = list(self._recent_results)
        if window:
            results = results[-window:]

        if not results:
            return {
                "n_evaluated": 0,
                "accuracy_pct": 0.0,
                "confident_accuracy_pct": 0.0,
                "avg_confidence": 0.0,
                "avg_actual_return_pct": 0.0,
                "avg_pnl_pct": 0.0,
                "signal_rate": 0.0,
            }

        # Filter to only predictions that had a directional signal
        directional = [r for r in results if r.get("predicted_direction", 0) != 0]
        confident = [r for r in directional if r.get("is_confident")]

        n_total = len(results)
        n_directional = len(directional)
        n_confident = len(confident)

        accuracy = sum(1 for r in directional if r.get("correct")) / n_directional if n_directional > 0 else 0
        conf_accuracy = sum(1 for r in confident if r.get("confident_correct")) / n_confident if n_confident > 0 else 0
        avg_conf = float(np.mean([r.get("confidence", 0) for r in directional])) if directional else 0
        avg_return = float(np.mean([r.get("actual_return", 0) for r in results])) * 100
        avg_pnl = float(np.mean([r.get("pnl_pct", 0) for r in directional])) * 100 if directional else 0

        return {
            "n_evaluated": n_total,
            "n_directional": n_directional,
            "n_confident": n_confident,
            "accuracy_pct": round(accuracy * 100, 2),
            "confident_accuracy_pct": round(conf_accuracy * 100, 2),
            "avg_confidence": round(avg_conf, 4),
            "avg_actual_return_pct": round(avg_return, 4),
            "avg_pnl_pct": round(avg_pnl, 4),
            "signal_rate": round(n_directional / n_total, 4) if n_total > 0 else 0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _append_result(self, result: dict) -> None:
        """Append a single evaluation result to the CSV log."""
        try:
            from csv_utils import safe_csv_append

            # Select columns for CSV (skip large nested fields)
            row = {
                "predict_ts": result.get("predict_ts"),
                "eval_ts": result.get("eval_ts"),
                "entry_price": result.get("entry_price"),
                "exit_price": result.get("exit_price"),
                "predicted_direction": result.get("predicted_direction"),
                "predicted_return": result.get("predicted_return"),
                "confidence": result.get("confidence"),
                "actual_return": result.get("actual_return"),
                "actual_direction": result.get("actual_direction"),
                "correct": result.get("correct"),
                "is_confident": result.get("is_confident"),
                "confident_correct": result.get("confident_correct"),
                "pnl_pct": result.get("pnl_pct"),
                "horizon_seconds": result.get("horizon_seconds"),
                "source": result.get("source"),
                "mtf_agreement": result.get("mtf_agreement"),
                "mtf_n_agree": result.get("mtf_n_agree"),
                "mtf_n_total": result.get("mtf_n_total"),
                "mtf_source": result.get("mtf_source"),
            }
            df = pd.DataFrame([row])
            safe_csv_append(self.eval_path, df)
        except Exception as exc:
            logger.warning("BTCForecastEval: failed to append result: %s", exc)

    def _load_recent_results(self) -> None:
        """Load recent evaluation results from disk to seed rolling stats."""
        try:
            if self.eval_path.exists() and self.eval_path.stat().st_size > 0:
                df = pd.read_csv(self.eval_path, engine="python", on_bad_lines="skip")
                if not df.empty:
                    # Load last 200 rows
                    for _, row in df.tail(200).iterrows():
                        self._recent_results.append(row.to_dict())
                    logger.info(
                        "BTCForecastEval: loaded %d historical results from %s",
                        len(self._recent_results), self.eval_path,
                    )
        except Exception as exc:
            logger.debug("BTCForecastEval: could not load history: %s", exc)

    # ------------------------------------------------------------------
    # Status / summary
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def summary(self) -> dict:
        """Return a summary dict suitable for logging or telemetry."""
        stats = self.rolling_stats()
        stats["pending_predictions"] = self.pending_count
        stats["eval_log_path"] = str(self.eval_path)
        return stats
