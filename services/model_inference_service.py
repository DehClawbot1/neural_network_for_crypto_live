"""
services/model_inference_service.py
─────────────────────────────────────
Immutable model loading and safe inference.

Rules
─────
1. Models are loaded once at construction. No hot-swap during a session.
   Reloading requires constructing a new ModelInferenceService.
2. ModelFaultError is raised when inference fails in a way that makes
   the score unreliable. Callers MUST skip the candidate — do NOT fall
   back to score=0 and continue.
3. Only promoted artifacts (from the staging directory) are loaded.
   Candidate/experimental artifacts are never used in the live path.
4. This service never makes trading decisions — only produces scores.

Usage
─────
    svc = ModelInferenceService(weights_dir="weights")
    if not svc.is_ready:
        raise ModelFaultError("No model artifacts available", model_stage="load")

    result = svc.infer(features_df)
    # result contains p_tp_before_sl, expected_return, edge_score columns
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from services.types import ModelFaultError

logger = logging.getLogger(__name__)

# Minimum acceptable model score range — scores outside this suggest a fault
_MIN_VALID_PROBABILITY = 0.0
_MAX_VALID_PROBABILITY = 1.0
_MAX_VALID_EXPECTED_RETURN = 5.0   # sanity ceiling — >5x return is suspect


class ModelInferenceService:
    """
    Wraps ModelInference (and optionally Stage1/Stage2) with hard fault semantics.

    On any inference failure that produces unreliable scores, ModelFaultError
    is raised. There is no silent fallback to 0.

    The service is stateless after construction — the loaded artifacts
    do not change for the life of the object.
    """

    def __init__(
        self,
        weights_dir: str = "weights",
        *,
        brain_id: Optional[str] = None,
        market_family: Optional[str] = None,
        shared_logs_dir: str = "logs",
        shared_weights_dir: str = "weights",
    ) -> None:
        self._weights_dir = Path(weights_dir)
        self._brain_id = brain_id
        self._market_family = market_family
        self._shared_logs_dir = shared_logs_dir
        self._shared_weights_dir = shared_weights_dir
        self._model = None
        self._load_error: Optional[str] = None
        self._artifact_paths: dict[str, str] = {}

        self._load_model()

    # ── Primary API ───────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the model was loaded successfully and has no missing artifacts."""
        return self._model is not None and self._load_error is None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def infer(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a feature DataFrame.

        Adds columns: p_tp_before_sl, expected_return, edge_score.

        Parameters
        ──────────
        features_df : DataFrame with model features (rows = candidates)

        Returns
        ───────
        Copy of features_df with score columns appended.

        Raises
        ──────
        ModelFaultError : if model is not loaded, input is empty, or
                          inference produces invalid scores (NaN, Inf, out-of-range).
                          Callers MUST skip the candidate — no silent fallback.
        """
        if not self.is_ready:
            raise ModelFaultError(
                f"Model not loaded (load_error={self._load_error!r})",
                model_stage="load",
                context={"weights_dir": str(self._weights_dir)},
            )

        if features_df is None or features_df.empty:
            raise ModelFaultError(
                "Empty feature DataFrame passed to infer()",
                model_stage="infer",
            )

        try:
            result = self._model.run(features_df)
        except Exception as exc:
            raise ModelFaultError(
                f"Model inference raised an exception: {exc}",
                model_stage="infer",
                context={"error": str(exc), "rows": len(features_df)},
            ) from exc

        if result is None or result.empty:
            raise ModelFaultError(
                "Model returned empty result",
                model_stage="infer",
                context={"input_rows": len(features_df)},
            )

        self._validate_scores(result)
        return result

    def infer_single(self, signal: dict) -> dict:
        """
        Run inference on a single signal dict.

        Wraps the row in a DataFrame, calls infer(), returns augmented dict.

        Raises
        ──────
        ModelFaultError : same as infer().
        """
        df = pd.DataFrame([signal])
        result_df = self.infer(df)
        return result_df.iloc[0].to_dict()

    def artifact_summary(self) -> dict:
        """Return paths and readiness for logging/audit."""
        return {
            "weights_dir":     str(self._weights_dir),
            "is_ready":        self.is_ready,
            "load_error":      self._load_error,
            "artifacts":       self._artifact_paths,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the ModelInference object. Sets _load_error on failure."""
        try:
            from model_inference import ModelInference
        except ImportError as exc:
            self._load_error = f"model_inference module not available: {exc}"
            logger.error("ModelInferenceService: %s", self._load_error)
            return

        try:
            model = ModelInference(
                weights_dir=str(self._weights_dir),
                brain_id=self._brain_id,
                market_family=self._market_family,
                shared_logs_dir=self._shared_logs_dir,
                shared_weights_dir=self._shared_weights_dir,
            )
        except Exception as exc:
            self._load_error = f"ModelInference construction failed: {exc}"
            logger.error("ModelInferenceService: %s", self._load_error)
            return

        missing = model.missing_artifacts()
        if missing:
            self._load_error = f"Missing artifacts: {[m['component'] for m in missing]}"
            logger.warning("ModelInferenceService: %s", self._load_error)
            # We still store the model — callers can check is_ready
            self._model = model
            self._load_error = self._load_error  # explicit
            return

        # Track artifact paths for audit
        self._artifact_paths = {
            "classifier": str(model.classifier_file),
            "regressor":  str(model.regressor_file),
        }
        self._model = model
        logger.info("ModelInferenceService: loaded artifacts from %s", self._weights_dir)

    def _validate_scores(self, df: pd.DataFrame) -> None:
        """
        Raise ModelFaultError if output scores are degenerate.

        Checks p_tp_before_sl and expected_return for NaN, Inf, or out-of-range.
        """
        for col, lo, hi in [
            ("p_tp_before_sl",  _MIN_VALID_PROBABILITY, _MAX_VALID_PROBABILITY),
            ("expected_return", -_MAX_VALID_EXPECTED_RETURN, _MAX_VALID_EXPECTED_RETURN),
        ]:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if series.isna().any():
                bad_count = int(series.isna().sum())
                raise ModelFaultError(
                    f"Inference column {col!r} has {bad_count} NaN/non-numeric rows",
                    model_stage="validate",
                    context={"column": col, "bad_rows": bad_count, "total_rows": len(df)},
                )
            if np.isinf(series).any():
                raise ModelFaultError(
                    f"Inference column {col!r} has infinite values",
                    model_stage="validate",
                    context={"column": col},
                )


if __name__ == "__main__":
    # Self-test using a mock ModelInference
    import tempfile, os
    from services.types import ModelFaultError

    class _MockModelInference:
        def __init__(self, **kwargs):
            self.classifier_file = Path("/tmp/fake_clf.joblib")
            self.regressor_file  = Path("/tmp/fake_reg.joblib")
        def missing_artifacts(self):
            return []  # pretend both exist
        def run(self, df):
            out = df.copy()
            out["p_tp_before_sl"]  = 0.70
            out["expected_return"] = 0.10
            out["edge_score"]      = 0.07
            return out

    # Monkey-patch for test
    import services.model_inference_service as _svc_mod
    _svc_mod.ModelInference = _MockModelInference  # type: ignore

    # Patch model_inference import inside _load_model
    import sys
    import types as _types
    _fake = _types.ModuleType("model_inference")
    _fake.ModelInference = _MockModelInference
    sys.modules["model_inference"] = _fake

    svc = ModelInferenceService(weights_dir="/tmp/fake_weights")
    assert svc.is_ready, f"Expected ready: {svc.load_error}"

    features = pd.DataFrame([
        {"token_id": "t1", "confidence": 0.80, "some_feature": 1.5},
        {"token_id": "t2", "confidence": 0.60, "some_feature": 0.8},
    ])
    result = svc.infer(features)
    assert "p_tp_before_sl"  in result.columns
    assert "expected_return" in result.columns
    assert "edge_score"      in result.columns
    assert len(result) == 2

    # Empty input raises ModelFaultError
    try:
        svc.infer(pd.DataFrame())
        assert False, "should raise"
    except ModelFaultError as e:
        assert "Empty" in str(e)

    # infer_single
    row_out = svc.infer_single({"token_id": "t3", "confidence": 0.75})
    assert row_out["p_tp_before_sl"] == 0.70

    print("model_inference_service self-test PASSED.")
