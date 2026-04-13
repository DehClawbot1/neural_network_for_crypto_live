"""
alpha_engine_btc.py
Phase 6 Stage A — BTC Alpha Engine

Wraps the existing BTC model stack and adds isotonic calibration.

Outputs (per candidate):
  fair_prob      — calibrated probability of YES (up)
  edge           — raw edge score
  ev             — expected value after cost
  trade_quality  — composite quality score
  regime         — BTC regime label
  calibrated_prob — isotonic-calibrated fair_prob
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        num = float(value)
        return float(default) if not np.isfinite(num) else num
    except Exception:
        return float(default)


def _safe_import(module_name: str):
    try:
        import importlib
        return importlib.import_module(module_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# BTCCalibrationModel — isotonic regression wrapper
# ---------------------------------------------------------------------------
class BTCCalibrationModel:
    """
    Fits an isotonic regression calibrator on (raw_prob, actual_outcome)
    pairs from alpha_feedback_clean.csv and applies it at runtime.

    Falls back to identity (no calibration) when there isn't enough data.
    """

    MIN_ROWS = 30

    def __init__(self, logs_dir: str = "logs", weights_dir: str = "weights") -> None:
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self._calibrator = None
        self._loaded = False

    def _alpha_feedback_df(self) -> pd.DataFrame:
        path = self.logs_dir / "alpha_feedback_clean.csv"
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def fit(self) -> bool:
        """Fit calibrator from alpha feedback. Returns True if fitted."""
        df = self._alpha_feedback_df()
        if df.empty or len(df) < self.MIN_ROWS:
            return False
        if "entry_p_tp" not in df.columns or "direction_correct" not in df.columns:
            return False

        try:
            from sklearn.isotonic import IsotonicRegression
            X = pd.to_numeric(df["entry_p_tp"], errors="coerce").fillna(0.5).values
            y = pd.to_numeric(df["direction_correct"], errors="coerce").fillna(0.5).values
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(X, y)
            logger.info("BTCCalibrationModel: fitted on %d rows", len(df))
            return True
        except Exception as exc:
            logger.warning("BTCCalibrationModel: fit failed: %s", exc)
            return False

    def calibrate(self, raw_prob: float) -> float:
        """Apply calibration to a raw probability. Falls back to identity."""
        if self._calibrator is None:
            return raw_prob
        try:
            return float(np.clip(self._calibrator.predict([raw_prob])[0], 0.0, 1.0))
        except Exception:
            return raw_prob


# ---------------------------------------------------------------------------
# BTCAlphaEngine
# ---------------------------------------------------------------------------
class BTCAlphaEngine:
    """
    Stage A — BTC Alpha Engine.

    Orchestrates:
      A1. BTC tabular alpha model (LightGBM via existing BTCMultiTimeframeForecaster)
      A2. BTC sequence model  (Stage2TemporalModels)
      A3. BTC regime model    (BTCRegimeRouter)
      A4. BTC calibration     (BTCCalibrationModel)

    Usage:
        engine = BTCAlphaEngine(logs_dir="logs/btc", weights_dir="weights/btc")
        output = engine.score(candidate_row)
        # output: {"fair_prob": 0.62, "edge": 0.08, "ev": 0.05, ...}
    """

    def __init__(
        self,
        logs_dir: str = "logs",
        weights_dir: str = "weights",
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self._calibrator = BTCCalibrationModel(logs_dir=str(self.logs_dir), weights_dir=str(self.weights_dir))
        self._calibrator_fitted = False
        self._forecaster = None
        self._regime_router = None

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------
    def _ensure_forecaster(self) -> None:
        if self._forecaster is not None:
            return
        try:
            from btc_multitimeframe import BTCMultiTimeframeForecaster
            self._forecaster = BTCMultiTimeframeForecaster(
                weights_dir=str(self.weights_dir),
                logs_dir=str(self.logs_dir),
            )
            logger.info("BTCAlphaEngine: loaded BTCMultiTimeframeForecaster")
        except Exception as exc:
            logger.warning("BTCAlphaEngine: BTCMultiTimeframeForecaster unavailable: %s", exc)

    def _ensure_regime_router(self) -> None:
        if self._regime_router is not None:
            return
        try:
            from btc_regime_router import BTCRegimeRouter
            self._regime_router = BTCRegimeRouter(
                logs_dir=str(self.logs_dir),
                weights_dir=str(self.weights_dir),
            )
            logger.info("BTCAlphaEngine: loaded BTCRegimeRouter")
        except Exception as exc:
            logger.warning("BTCAlphaEngine: BTCRegimeRouter unavailable: %s", exc)

    def _ensure_calibrator(self) -> None:
        if not self._calibrator_fitted:
            self._calibrator_fitted = self._calibrator.fit()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def score(self, candidate_row: dict[str, Any]) -> dict[str, Any]:
        """
        Score a single BTC candidate row.
        Returns btc_alpha_output dict.
        Falls back gracefully when models are unavailable.
        """
        self._ensure_forecaster()
        self._ensure_regime_router()
        self._ensure_calibrator()

        # --- A1: tabular alpha ----------------------------------------
        fair_prob = _safe_float(candidate_row.get("p_tp_before_sl", candidate_row.get("fair_prob")), 0.5)
        ev = _safe_float(candidate_row.get("expected_return", candidate_row.get("ev")), 0.0)
        edge = _safe_float(candidate_row.get("edge_score", candidate_row.get("edge")), 0.0)

        if self._forecaster is not None:
            try:
                forecast = self._forecaster.predict_row(candidate_row)
                fair_prob = _safe_float(forecast.get("prob_up_15m", fair_prob), fair_prob)
                ev = _safe_float(forecast.get("expected_return", ev), ev)
            except Exception as exc:
                logger.debug("BTCAlphaEngine: forecaster.predict_row failed: %s", exc)

        # --- A3: regime -----------------------------------------------
        regime = str(candidate_row.get("btc_market_regime_label", "unknown"))
        if self._regime_router is not None:
            try:
                regime = str(self._regime_router.classify(candidate_row) or regime)
            except Exception:
                pass

        # --- A4: calibration ------------------------------------------
        calibrated_prob = self._calibrator.calibrate(fair_prob)

        # --- trade quality composite ----------------------------------
        trade_quality = float(np.clip(
            (calibrated_prob - 0.5) * 0.50     # directional edge
            + np.clip(ev * 5.0, -1.0, 1.0) * 0.30  # EV contribution
            + np.clip(edge * 8.0, -1.0, 1.0) * 0.20,  # edge score
            -1.0, 1.0,
        ))

        return {
            "fair_prob": round(fair_prob, 6),
            "edge": round(edge, 6),
            "ev": round(ev, 6),
            "trade_quality": round(trade_quality, 6),
            "regime": regime,
            "calibrated_prob": round(calibrated_prob, 6),
        }

    def refresh_calibration(self) -> bool:
        """Re-fit calibration from latest alpha feedback. Call daily."""
        self._calibrator_fitted = self._calibrator.fit()
        return self._calibrator_fitted
