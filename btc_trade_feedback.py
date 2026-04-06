"""
btc_trade_feedback.py
=====================
Compares BTC price predictions at trade entry vs actual BTC price at exit,
computes prediction accuracy per trade, and feeds that back into the
forecast model via sample weighting.

For every closed trade that has entry/exit BTC data:
  - Was the BTC direction prediction correct?
  - How far off was the predicted return vs actual BTC move?
  - Did trades with correct BTC predictions win more often?

The feedback weights are saved to logs/btc_forecast_feedback.csv and
consumed by btc_forecast_model.py during retraining to up-weight
market regimes where the model was accurate and down-weight where it
was not.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from csv_utils import safe_csv_append

logger = logging.getLogger(__name__)


class BTCTradeFeedback:
    """Analyze BTC prediction accuracy per closed trade and produce feedback."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.closed_file = self.logs_dir / "closed_positions.csv"
        self.feedback_file = self.logs_dir / "btc_forecast_feedback.csv"
        self.summary_file = self.logs_dir / "btc_forecast_feedback_summary.csv"

    def _safe_read(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Core: analyze closed trades
    # ------------------------------------------------------------------

    def analyze(self) -> pd.DataFrame:
        """
        Read closed_positions.csv, compute BTC prediction accuracy for each
        trade, and return the enriched DataFrame.

        Returns only trades that have both entry and exit BTC data.
        """
        df = self._safe_read(self.closed_file)
        if df.empty:
            return pd.DataFrame()

        # Filter to trades with BTC forecast data
        required = ["entry_btc_predicted_direction", "entry_btc_price", "exit_btc_price"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        df["entry_btc_predicted_direction"] = pd.to_numeric(
            df["entry_btc_predicted_direction"], errors="coerce"
        ).fillna(0).astype(int)
        df["entry_btc_price"] = pd.to_numeric(df["entry_btc_price"], errors="coerce")
        df["exit_btc_price"] = pd.to_numeric(df["exit_btc_price"], errors="coerce")
        df["entry_btc_predicted_return"] = pd.to_numeric(
            df.get("entry_btc_predicted_return", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0.0)
        df["entry_btc_forecast_confidence"] = pd.to_numeric(
            df.get("entry_btc_forecast_confidence", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0.0)

        # Only trades with valid BTC entry/exit prices
        mask = (df["entry_btc_price"] > 0) & (df["exit_btc_price"] > 0)
        work = df[mask].copy()
        if work.empty:
            return pd.DataFrame()

        # -- BTC actual move during the trade --
        work["btc_actual_return"] = (
            (work["exit_btc_price"] - work["entry_btc_price"]) / work["entry_btc_price"]
        )
        work["btc_actual_direction"] = np.sign(work["btc_actual_return"]).astype(int)

        # -- Prediction accuracy --
        has_prediction = work["entry_btc_predicted_direction"] != 0
        work["btc_direction_correct"] = np.where(
            has_prediction,
            (work["entry_btc_predicted_direction"] == work["btc_actual_direction"]).astype(int),
            np.nan,
        )
        work["btc_return_error"] = np.where(
            has_prediction,
            work["entry_btc_predicted_return"] - work["btc_actual_return"],
            np.nan,
        )
        work["btc_return_abs_error"] = np.abs(work["btc_return_error"])

        # -- Trade outcome --
        pnl_col = "net_realized_pnl" if "net_realized_pnl" in work.columns else "realized_pnl"
        work["trade_pnl"] = pd.to_numeric(work.get(pnl_col, 0), errors="coerce").fillna(0.0)
        work["trade_won"] = (work["trade_pnl"] > 0).astype(int)

        # -- BTC prediction profit (if we traded BTC direction) --
        work["btc_prediction_pnl"] = np.where(
            has_prediction,
            work["btc_actual_return"] * work["entry_btc_predicted_direction"],
            0.0,
        )

        return work

    # ------------------------------------------------------------------
    # Feedback weights for retraining
    # ------------------------------------------------------------------

    def compute_feedback_weights(self, lookback: int = 200) -> dict:
        """
        Compute feedback weights from the most recent closed trades.

        Returns a dict with:
          - btc_direction_accuracy: fraction of correct directional calls
          - btc_avg_return_error: mean absolute return error
          - btc_prediction_profitable: fraction of trades where BTC pred was profitable
          - regime_weights: dict of {regime: weight} for sample weighting
          - win_rate_when_btc_correct: win rate when BTC direction was right
          - win_rate_when_btc_wrong: win rate when BTC direction was wrong
        """
        work = self.analyze()
        if work.empty:
            return {}

        recent = work.tail(lookback)
        has_pred = recent["entry_btc_predicted_direction"] != 0
        directional = recent[has_pred]

        if directional.empty:
            return {}

        correct = directional["btc_direction_correct"].sum()
        total = len(directional)
        accuracy = correct / total if total > 0 else 0.0

        avg_abs_error = float(directional["btc_return_abs_error"].mean())
        prediction_profitable = float((directional["btc_prediction_pnl"] > 0).mean())

        # Win rates segmented by BTC prediction correctness
        btc_correct_mask = directional["btc_direction_correct"] == 1
        btc_wrong_mask = directional["btc_direction_correct"] == 0
        win_when_correct = float(directional.loc[btc_correct_mask, "trade_won"].mean()) if btc_correct_mask.any() else 0.0
        win_when_wrong = float(directional.loc[btc_wrong_mask, "trade_won"].mean()) if btc_wrong_mask.any() else 0.0

        # Regime-level weights: accuracy per volatility/technical regime
        regime_weights = {}
        for regime_col in ["volatility_bucket", "technical_regime_bucket"]:
            if regime_col in directional.columns:
                for regime, group in directional.groupby(regime_col, dropna=False):
                    if len(group) >= 3:
                        regime_acc = float(group["btc_direction_correct"].mean())
                        regime_weights[f"{regime_col}:{regime}"] = round(regime_acc, 4)

        feedback = {
            "btc_direction_accuracy": round(accuracy, 4),
            "btc_avg_return_error": round(avg_abs_error, 6),
            "btc_prediction_profitable": round(prediction_profitable, 4),
            "win_rate_when_btc_correct": round(win_when_correct, 4),
            "win_rate_when_btc_wrong": round(win_when_wrong, 4),
            "n_trades_with_btc_data": total,
            "regime_weights": regime_weights,
        }

        logger.info(
            "BTC trade feedback: accuracy=%.1f%% avg_error=%.4f%% "
            "win_when_correct=%.1f%% win_when_wrong=%.1f%% (n=%d)",
            accuracy * 100, avg_abs_error * 100,
            win_when_correct * 100, win_when_wrong * 100, total,
        )

        return feedback

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write_feedback(self) -> pd.DataFrame:
        """Analyze, persist per-trade feedback, and summary stats."""
        work = self.analyze()
        if work.empty:
            return pd.DataFrame()

        # Per-trade feedback rows
        feedback_cols = [
            "opened_at", "closed_at", "market", "token_id",
            "entry_btc_predicted_direction", "entry_btc_predicted_return",
            "entry_btc_forecast_confidence", "entry_btc_price", "exit_btc_price",
            "entry_btc_mtf_agreement", "entry_btc_mtf_source",
            "btc_actual_return", "btc_actual_direction",
            "btc_direction_correct", "btc_return_error", "btc_return_abs_error",
            "btc_prediction_pnl", "trade_pnl", "trade_won",
            "close_reason", "signal_label",
        ]
        available = [c for c in feedback_cols if c in work.columns]
        feedback_df = work[available].copy()
        feedback_df.to_csv(self.feedback_file, index=False)
        logger.info("Wrote %d BTC trade feedback rows to %s", len(feedback_df), self.feedback_file)

        # Summary stats
        stats = self.compute_feedback_weights()
        if stats:
            summary_row = {k: v for k, v in stats.items() if k != "regime_weights"}
            summary_row["regime_weights"] = str(stats.get("regime_weights", {}))
            safe_csv_append(self.summary_file, pd.DataFrame([summary_row]))

        return feedback_df

    # ------------------------------------------------------------------
    # Sample weights for btc_forecast_model retraining
    # ------------------------------------------------------------------

    def get_sample_weights_for_retraining(self, lookback: int = 200) -> dict:
        """
        Return a dict that btc_forecast_model can use to adjust sample weights.

        Keys:
          - "regime_accuracy": {regime_label: accuracy} for regime-aware weighting
          - "overall_accuracy": float, overall directional accuracy
          - "error_scale": float, 1.0 / (1.0 + avg_abs_error) for return calibration
        """
        feedback = self.compute_feedback_weights(lookback=lookback)
        if not feedback:
            return {}

        accuracy = feedback.get("btc_direction_accuracy", 0.5)
        avg_error = feedback.get("btc_avg_return_error", 0.01)
        regime_weights = feedback.get("regime_weights", {})

        return {
            "overall_accuracy": accuracy,
            "error_scale": round(1.0 / (1.0 + avg_error), 4),
            "regime_accuracy": regime_weights,
        }
