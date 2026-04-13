"""
alpha_engine_weather.py
Phase 6 Stage B — Weather Alpha Engine

Wraps the existing weather model stack and adds revision + uncertainty
models.

Outputs (per candidate):
  fair_prob          — calibrated resolution probability
  revision_dir       — predicted next forecast revision direction (+1/-1/0)
  uncertainty_penalty — how uncertain the current forecast state is
  ev                 — expected value after cost
  trade_quality      — composite quality score
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


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# WeatherRevisionModel
# ---------------------------------------------------------------------------
class WeatherRevisionModel:
    """
    Predicts how the forecast will move before resolution:
      +1 (forecast will rise), 0 (stable), -1 (forecast will fall).

    Trained from weather alpha_feedback where we track revision_direction.
    Falls back to 0 (no prediction) until enough data accumulates.
    """

    MIN_ROWS = 50

    def __init__(self, weights_dir: str = "weights", logs_dir: str = "logs") -> None:
        self.weights_dir = Path(weights_dir)
        self.logs_dir = Path(logs_dir)
        self._model = None
        self._features: list[str] = []

    def fit(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < self.MIN_ROWS:
            return False
        if "weather_forecast_revision_direction" not in df.columns:
            return False
        try:
            import lightgbm as lgb
            feature_cols = [c for c in df.columns if c not in {
                "weather_forecast_revision_direction", "attributed_at", "report_id",
                "token_id", "condition_id", "alpha_verdict", "outcome_class",
            } and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
            if not feature_cols:
                return False
            X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            y = pd.to_numeric(df["weather_forecast_revision_direction"], errors="coerce").fillna(0)
            self._model = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, random_state=42, verbose=-1)
            self._model.fit(X, y)
            self._features = feature_cols
            logger.info("WeatherRevisionModel: fitted on %d rows", len(df))
            return True
        except Exception as exc:
            logger.warning("WeatherRevisionModel: fit failed: %s", exc)
            return False

    def predict(self, row: dict[str, Any]) -> int:
        if self._model is None or not self._features:
            return 0
        try:
            X = pd.DataFrame([{f: _safe_float(row.get(f)) for f in self._features}])
            return int(self._model.predict(X)[0])
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# WeatherUncertaintyModel
# ---------------------------------------------------------------------------
class WeatherUncertaintyModel:
    """
    Classifies whether the current forecast state is too unstable to trade.
    Returns an uncertainty_penalty in [0, 1].

    Trained from weather contract_targets where ensemble_spread or
    source_disagreement columns are available.
    """

    def __init__(self, weights_dir: str = "weights", logs_dir: str = "logs") -> None:
        self.weights_dir = Path(weights_dir)
        self.logs_dir = Path(logs_dir)

    def predict(self, row: dict[str, Any]) -> float:
        """
        Phase 5: Manual uncertainty heuristic stripped. 
        Will return 0.0 until a unified uncertainty ML model is provided.
        """
        return 0.0


# ---------------------------------------------------------------------------
# WeatherAlphaEngine
# ---------------------------------------------------------------------------
class WeatherAlphaEngine:
    """
    Stage B — Weather Alpha Engine.

    Orchestrates:
      B1. Weather fair-value model (via existing WeatherTemperatureTrainer)
      B2. Weather revision model   (WeatherRevisionModel)
      B3. Weather uncertainty model (WeatherUncertaintyModel)
      B4. Weather calibration model (isotonic, shared calibrator)

    Usage:
        engine = WeatherAlphaEngine(logs_dir="logs/weather_temperature", weights_dir="weights/weather_temperature")
        output = engine.score(candidate_row)
    """

    def __init__(
        self,
        logs_dir: str = "logs",
        weights_dir: str = "weights",
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self._revision_model = WeatherRevisionModel(weights_dir=str(self.weights_dir), logs_dir=str(self.logs_dir))
        self._uncertainty_model = WeatherUncertaintyModel(weights_dir=str(self.weights_dir), logs_dir=str(self.logs_dir))
        self._calibrator = None
        self._calibrator_fitted = False
        self._temperature_model = None

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------
    def _ensure_temperature_model(self) -> None:
        if self._temperature_model is not None:
            return
        try:
            from weather_temperature_trainer import WeatherTemperatureTrainer
            from brain_paths import resolve_brain_context, WEATHER_FAMILY
            ctx = resolve_brain_context(WEATHER_FAMILY, shared_logs_dir=str(self.logs_dir.parent), shared_weights_dir=str(self.weights_dir.parent))
            self._temperature_model = WeatherTemperatureTrainer(brain_context=ctx, weights_dir=str(self.weights_dir))
            logger.info("WeatherAlphaEngine: loaded WeatherTemperatureTrainer")
        except Exception as exc:
            logger.warning("WeatherAlphaEngine: WeatherTemperatureTrainer unavailable: %s", exc)

    def _ensure_calibrator(self) -> None:
        if self._calibrator_fitted:
            return
        try:
            from sklearn.isotonic import IsotonicRegression
            alpha_path = self.logs_dir / "alpha_feedback_clean.csv"
            df = _safe_read(alpha_path)
            if df.empty or len(df) < 30:
                return
            if "entry_p_tp" not in df.columns or "direction_correct" not in df.columns:
                return
            X = pd.to_numeric(df["entry_p_tp"], errors="coerce").fillna(0.5).values
            y = pd.to_numeric(df["direction_correct"], errors="coerce").fillna(0.5).values
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(X, y)
            self._calibrator_fitted = True
            logger.info("WeatherAlphaEngine: calibrator fitted on %d rows", len(df))
        except Exception as exc:
            logger.warning("WeatherAlphaEngine: calibrator fit failed: %s", exc)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def score(self, candidate_row: dict[str, Any]) -> dict[str, Any]:
        """Score a single weather candidate. Returns weather_alpha_output dict."""
        self._ensure_temperature_model()
        self._ensure_calibrator()

        # --- B1: fair-value probability --------------------------------
        fair_prob = _safe_float(candidate_row.get("p_tp_before_sl", candidate_row.get("fair_prob")), 0.5)
        ev = _safe_float(candidate_row.get("expected_return", candidate_row.get("ev")), 0.0)

        if self._temperature_model is not None:
            try:
                pred = self._temperature_model.predict_row(candidate_row)
                if pred:
                    fair_prob = _safe_float(pred.get("resolution_prob", fair_prob), fair_prob)
                    ev = _safe_float(pred.get("ev", ev), ev)
            except Exception:
                pass

        # --- B2: revision direction ------------------------------------
        revision_dir = self._revision_model.predict(candidate_row)

        # --- B3: uncertainty penalty -----------------------------------
        uncertainty_penalty = self._uncertainty_model.predict(candidate_row)

        # --- B4: calibration ------------------------------------------
        calibrated_prob = fair_prob
        if self._calibrator is not None:
            try:
                calibrated_prob = float(np.clip(
                    self._calibrator.predict([fair_prob])[0], 0.0, 1.0
                ))
            except Exception:
                pass

        # Penalize EV by uncertainty
        ev_penalized = ev * max(0.0, 1.0 - uncertainty_penalty)

        # Trade quality
        trade_quality = float(np.clip(
            (calibrated_prob - 0.5) * 0.50
            + np.clip(ev_penalized * 5.0, -1.0, 1.0) * 0.30
            + (revision_dir * 0.1) * 0.20,
            -1.0, 1.0,
        ))

        return {
            "fair_prob": round(fair_prob, 6),
            "revision_dir": int(revision_dir),
            "uncertainty_penalty": round(uncertainty_penalty, 6),
            "ev": round(ev_penalized, 6),
            "trade_quality": round(trade_quality, 6),
            "calibrated_prob": round(calibrated_prob, 6),
        }

    def refresh_calibration(self) -> bool:
        """Re-fit calibration from latest alpha feedback."""
        self._calibrator_fitted = False
        self._ensure_calibrator()
        return self._calibrator_fitted
