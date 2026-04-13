"""
execution_engine.py
Phase 6 Stage C — Execution Engine

Predicts fill probability, slippage, and liquidity failure risk
from orderbook / execution features.

Models separated by Entry / Exit:
  C1. FillProbabilityModel  — predicts fill_prob_5s / fill_prob_30s
  C2. SlippageModel         — predicts expected_slippage
  C3. LiquidityFailureModel — predicts liquidity_failure_risk
  C4. ExecutionEngine.score()  — composite execution quality
  C5. compute_pnl_after_cost()   — real cost model

Uses Bayesian shrinkage + Heuristic fallback until 50 rows accumulate.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_TRAINING_ROWS = 5
_FULL_CONFIDENCE_ROWS = 50.0

_FILL_PROB_FEATURES = [
    "spread_at_entry", "requested_size", "fill_latency_ms",
    "depth_best_bid", "depth_best_ask", "queue_imbalance",
    "quote_instability", "time_pressure",
]
_SLIPPAGE_FEATURES = [
    "spread_at_entry", "requested_size", "fill_latency_ms",
    "depth_best_bid", "depth_best_ask", "queue_imbalance",
    "order_aggressiveness",
]
_LIQUIDITY_FEATURES = [
    "spread_at_entry", "requested_size",
    "depth_best_bid", "depth_best_ask",
    "fill_latency_ms", "order_aggressiveness",
]


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


def _feature_row(row: dict[str, Any], features: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{f: _safe_float(row.get(f)) for f in features}])


# ---------------------------------------------------------------------------
# Heuristic fallbacks
# ---------------------------------------------------------------------------
def _heuristic_fill_prob(row: dict[str, Any]) -> tuple[float, float]:
    """Heuristic fill probability when model is unavailable."""
    spread = _safe_float(row.get("spread_at_entry"), 0.05)
    size = _safe_float(row.get("requested_size"), 10.0)
    base = max(0.40, 0.95 - spread * 3.0 - size / 500.0)
    return float(np.clip(base, 0.30, 0.95)), float(np.clip(base + 0.07, 0.35, 0.99))


def _heuristic_slippage(row: dict[str, Any]) -> float:
    """Heuristic slippage when model is unavailable."""
    spread = _safe_float(row.get("spread_at_entry"), 0.02)
    size = _safe_float(row.get("requested_size"), 10.0)
    return float(np.clip(spread * 0.30 + size / 2000.0, 0.0, 0.10))


def _heuristic_liquidity_failure(row: dict[str, Any]) -> float:
    """Heuristic liquidity failure risk when model is unavailable."""
    spread = _safe_float(row.get("spread_at_entry"), 0.02)
    size = _safe_float(row.get("requested_size"), 10.0)
    return float(np.clip(spread * 2.0 + size / 1000.0, 0.0, 0.50))


# ---------------------------------------------------------------------------
# FillProbabilityModel
# ---------------------------------------------------------------------------
class FillProbabilityModel:
    def __init__(self, side: str = "entry", weights_dir: str = "weights") -> None:
        self.side = side
        self.weights_dir = Path(weights_dir)
        self._model_5s = None
        self._model_30s = None
        self._features = _FILL_PROB_FEATURES
        self._sample_count = 0

    def load(self) -> bool:
        path_5s = self.weights_dir / f"exec_fill_prob_5s_{self.side}.joblib"
        path_30s = self.weights_dir / f"exec_fill_prob_30s_{self.side}.joblib"
        try:
            if path_5s.exists():
                payload = joblib.load(path_5s)
                self._model_5s = payload.get("model") if isinstance(payload, dict) else payload
                self._features = payload.get("features", self._features) if isinstance(payload, dict) else self._features
                self._sample_count = payload.get("sample_count", self._sample_count) if isinstance(payload, dict) else self._sample_count
            if path_30s.exists():
                payload = joblib.load(path_30s)
                self._model_30s = payload.get("model") if isinstance(payload, dict) else payload
            return self._model_5s is not None
        except Exception as exc:
            logger.debug("FillProbabilityModel[%s]: load failed: %s", self.side, exc)
            return False

    def train(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < _MIN_TRAINING_ROWS:
            return False
        try:
            import lightgbm as lgb
            feature_cols = [f for f in self._features if f in df.columns]
            if not feature_cols:
                return False
            X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            sw = df.get("sample_weight", pd.Series(1.0, index=df.index))

            for target_col, attr in [("actual_filled", "_model_5s"), ("actual_filled", "_model_30s")]:
                if target_col not in df.columns:
                    continue
                y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
                mdl = lgb.LGBMClassifier(n_estimators=100, num_leaves=10, min_data_in_leaf=3, random_state=42, verbose=-1)
                mdl.fit(X, y, sample_weight=sw)
                setattr(self, attr, mdl)
                # Persist
                suffix = "5s" if attr == "_model_5s" else "30s"
                path = self.weights_dir / f"exec_fill_prob_{suffix}_{self.side}.joblib"
                self._sample_count = len(df)
                joblib.dump({"model": mdl, "features": feature_cols, "sample_count": self._sample_count}, path)
                logger.info("FillProbabilityModel[%s]: trained %s on %d rows", self.side, suffix, len(df))
            return self._model_5s is not None
        except Exception as exc:
            logger.warning("FillProbabilityModel[%s]: train failed: %s", self.side, exc)
            return False

    def predict(self, row: dict[str, Any]) -> tuple[float, float]:
        p5_heur, p30_heur = _heuristic_fill_prob(row)
        if self._model_5s is None and self._model_30s is None:
            return p5_heur, p30_heur
        
        ml_weight = float(np.clip(self._sample_count / _FULL_CONFIDENCE_ROWS, 0.0, 1.0))
        X = _feature_row(row, self._features)
        
        p5, p30 = p5_heur, p30_heur
        try:
            if self._model_5s is not None:
                p5_ml = float(self._model_5s.predict_proba(X)[0][1])
                p5 = ml_weight * p5_ml + (1.0 - ml_weight) * p5_heur
        except Exception:
            pass
        try:
            if self._model_30s is not None:
                p30_ml = float(self._model_30s.predict_proba(X)[0][1])
                p30 = ml_weight * p30_ml + (1.0 - ml_weight) * p30_heur
        except Exception:
            pass
        return float(np.clip(p5, 0.0, 1.0)), float(np.clip(p30, 0.0, 1.0))


# ---------------------------------------------------------------------------
# SlippageModel
# ---------------------------------------------------------------------------
class SlippageModel:
    def __init__(self, side: str = "entry", weights_dir: str = "weights") -> None:
        self.side = side
        self.weights_dir = Path(weights_dir)
        self._model = None
        self._features = _SLIPPAGE_FEATURES
        self._sample_count = 0

    def load(self) -> bool:
        path = self.weights_dir / f"exec_slippage_model_{self.side}.joblib"
        try:
            if path.exists():
                payload = joblib.load(path)
                self._model = payload.get("model") if isinstance(payload, dict) else payload
                self._features = payload.get("features", self._features) if isinstance(payload, dict) else self._features
                self._sample_count = payload.get("sample_count", self._sample_count) if isinstance(payload, dict) else self._sample_count
            return self._model is not None
        except Exception:
            return False

    def train(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < _MIN_TRAINING_ROWS:
            return False
        # Train on actual_slippage or sum of slippage components if available
        # But for robustness, just require the column
        target_col = "actual_slippage"
        if target_col not in df.columns:
            return False
            
        try:
            import lightgbm as lgb
            feature_cols = [f for f in self._features if f in df.columns]
            if not feature_cols:
                return False
            X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
            sw = df.get("sample_weight", pd.Series(1.0, index=df.index))
            self._model = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, min_data_in_leaf=3, random_state=42, verbose=-1)
            self._model.fit(X, y, sample_weight=sw)
            self._sample_count = len(df)
            path = self.weights_dir / f"exec_slippage_model_{self.side}.joblib"
            joblib.dump({"model": self._model, "features": feature_cols, "sample_count": self._sample_count}, path)
            logger.info("SlippageModel[%s]: trained on %d rows", self.side, self._sample_count)
            return True
        except Exception as exc:
            logger.warning("SlippageModel[%s]: train failed: %s", self.side, exc)
            return False

    def predict(self, row: dict[str, Any]) -> float:
        heur = _heuristic_slippage(row)
        if self._model is None:
            return heur
        ml_weight = float(np.clip(self._sample_count / _FULL_CONFIDENCE_ROWS, 0.0, 1.0))
        try:
            X = _feature_row(row, self._features)
            ml_pred = float(np.clip(self._model.predict(X)[0], -0.1, 0.3))
            return ml_weight * ml_pred + (1.0 - ml_weight) * heur
        except Exception:
            return heur


# ---------------------------------------------------------------------------
# LiquidityFailureModel
# ---------------------------------------------------------------------------
class LiquidityFailureModel:
    def __init__(self, side: str = "entry", weights_dir: str = "weights") -> None:
        self.side = side
        self.weights_dir = Path(weights_dir)
        self._model = None
        self._features = _LIQUIDITY_FEATURES
        self._sample_count = 0

    def load(self) -> bool:
        path = self.weights_dir / f"exec_liquidity_failure_model_{self.side}.joblib"
        try:
            if path.exists():
                payload = joblib.load(path)
                self._model = payload.get("model") if isinstance(payload, dict) else payload
                self._features = payload.get("features", self._features) if isinstance(payload, dict) else self._features
                self._sample_count = payload.get("sample_count", self._sample_count) if isinstance(payload, dict) else self._sample_count
            return self._model is not None
        except Exception:
            return False

    def train(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < _MIN_TRAINING_ROWS:
            return False
        try:
            import lightgbm as lgb
            feature_cols = [f for f in self._features if f in df.columns]
            if not feature_cols:
                return False
            # Build target
            filled = pd.to_numeric(df.get("actual_filled", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
            latency = pd.to_numeric(df.get("fill_latency_ms", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
            y = ((filled == 0) | (latency > 30000)).astype(int)
            X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            sw = df.get("sample_weight", pd.Series(1.0, index=df.index))
            self._model = lgb.LGBMClassifier(n_estimators=100, num_leaves=10, min_data_in_leaf=3, random_state=42, verbose=-1)
            self._model.fit(X, y, sample_weight=sw)
            self._sample_count = len(df)
            path = self.weights_dir / f"exec_liquidity_failure_model_{self.side}.joblib"
            joblib.dump({"model": self._model, "features": feature_cols, "sample_count": self._sample_count}, path)
            logger.info("LiquidityFailureModel[%s]: trained on %d rows", self.side, self._sample_count)
            return True
        except Exception as exc:
            logger.warning("LiquidityFailureModel[%s]: train failed: %s", self.side, exc)
            return False

    def predict(self, row: dict[str, Any]) -> float:
        heur = _heuristic_liquidity_failure(row)
        if self._model is None:
            return heur
        ml_weight = float(np.clip(self._sample_count / _FULL_CONFIDENCE_ROWS, 0.0, 1.0))
        try:
            X = _feature_row(row, self._features)
            ml_pred = float(np.clip(self._model.predict_proba(X)[0][1], 0.0, 1.0))
            return ml_weight * ml_pred + (1.0 - ml_weight) * heur
        except Exception:
            return heur


# ---------------------------------------------------------------------------
# ExecutionEngine  (Stage C orchestrator)
# ---------------------------------------------------------------------------
class ExecutionEngine:
    """
    Stage C — Execution Engine.

    Combines models across entry/exit sides into a direct edge-adjustment.
    """

    def __init__(self, weights_dir: str = "weights", logs_dir: str = "logs") -> None:
        self.weights_dir = Path(weights_dir)
        self.logs_dir = Path(logs_dir)
        
        self._entry_fill_model = FillProbabilityModel(side="entry", weights_dir=str(self.weights_dir))
        self._entry_slippage_model = SlippageModel(side="entry", weights_dir=str(self.weights_dir))
        self._entry_liquidity_model = LiquidityFailureModel(side="entry", weights_dir=str(self.weights_dir))
        
        self._exit_fill_model = FillProbabilityModel(side="exit", weights_dir=str(self.weights_dir))
        self._exit_slippage_model = SlippageModel(side="exit", weights_dir=str(self.weights_dir))
        self._exit_liquidity_model = LiquidityFailureModel(side="exit", weights_dir=str(self.weights_dir))
        
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._entry_fill_model.load()
        self._entry_slippage_model.load()
        self._entry_liquidity_model.load()
        
        self._exit_fill_model.load()
        self._exit_slippage_model.load()
        self._exit_liquidity_model.load()
        
        self._loaded = True

    def train(self, df: pd.DataFrame | None = None) -> dict[str, bool]:
        """Train sub-models. Here we split identical rows for both right now
           unless entry/exit flags exist in df."""
        if df is None:
            exec_path = self.logs_dir / "execution_feedback.csv"
            df = _safe_read(exec_path)
        
        # If we had row.side == 'entry', we would filter here.
        # For compatibility with mixed schema, train both on the same dataset
        # downstream pipelines should supply isolated entry/exit row files eventually.
        results = {
            "entry_fill": self._entry_fill_model.train(df),
            "entry_slippage": self._entry_slippage_model.train(df),
            "entry_liquidity": self._entry_liquidity_model.train(df),
        }
        self._loaded = True
        return results

    def score(self, candidate_row: dict[str, Any], side: str = "entry") -> dict[str, Any]:
        """Score a single candidate row for execution quality."""
        self.load()

        if side == "exit":
            fill_5s, fill_30s = self._exit_fill_model.predict(candidate_row)
            slippage = self._exit_slippage_model.predict(candidate_row)
            liq_fail = self._exit_liquidity_model.predict(candidate_row)
        else:
            fill_5s, fill_30s = self._entry_fill_model.predict(candidate_row)
            slippage = self._entry_slippage_model.predict(candidate_row)
            liq_fail = self._entry_liquidity_model.predict(candidate_row)

        alpha_edge = _safe_float(candidate_row.get("risk_adjusted_ev", candidate_row.get("edge_score", 0.0)), 0.0)

        # Meta-engine Execution Quality Modifier
        # execution adjusted edge = (alpha * fill_prob) - expected_slippage - (liq_fail * penalty)
        penalty_scalar = alpha_edge if alpha_edge > 0 else 0.1
        adjusted_edge = (alpha_edge * fill_30s) - abs(slippage) - (liq_fail * penalty_scalar)

        # Legacy bounded quality score
        quality = float(np.clip(
            fill_30s * 0.45 
            + (1.0 - np.clip(slippage / 0.05, 0.0, 1.0)) * 0.35 
            + (1.0 - liq_fail) * 0.20, 
            0.0, 1.0,
        ))

        return {
            "fill_prob_5s": round(fill_5s, 6),
            "fill_prob_30s": round(fill_30s, 6),
            "expected_slippage": round(slippage, 6),
            "liquidity_failure_risk": round(liq_fail, 6),
            "quality_score": round(quality, 6),
            "execution_adjusted_edge": round(adjusted_edge, 6),
        }

    def score_batch(self, df: pd.DataFrame, side: str = "entry") -> pd.DataFrame:
        """Batch scoring for stage3_hybrid pipeline."""
        if df is None or df.empty:
            return df
        
        self.load()
        out = pd.DataFrame(index=df.index)
        out["exec_fill_prob_30s"] = 0.0
        out["exec_expected_slippage"] = 0.0
        out["exec_liquidity_fail_risk"] = 0.0
        
        # Simple loop since models are tree-based predict (fast enough for small batch)
        # We can optimize via pure LGBM predict over the whole DF later.
        for idx, row in df.iterrows():
            scores = self.score(row.to_dict(), side=side)
            out.loc[idx, "exec_fill_prob_30s"] = scores["fill_prob_30s"]
            out.loc[idx, "exec_expected_slippage"] = scores["expected_slippage"]
            out.loc[idx, "exec_liquidity_fail_risk"] = scores["liquidity_failure_risk"]
            out.loc[idx, "execution_adjusted_edge"] = scores["execution_adjusted_edge"]
            
        return out

    def compute_pnl_after_cost(self, path_return: float, candidate_row: dict[str, Any]) -> float:
        self.load()
        spread_cost = _safe_float(candidate_row.get("spread_at_entry"), 0.02) * 0.5
        slippage_cost = abs(self._entry_slippage_model.predict(candidate_row))
        fee = _safe_float(os.getenv("PLATFORM_FEE_RATE", "0.001"), 0.001)
        total_cost = spread_cost + slippage_cost + fee
        return float(path_return - total_cost)
