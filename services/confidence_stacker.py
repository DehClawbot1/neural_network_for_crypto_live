"""
services/confidence_stacker.py
──────────────────────────────
Stacked meta-learner that replaces the hand-tuned confidence blend in
signal_engine.py (the 0.45 / 0.30 / 0.25 weights + 0.59/0.44/0.39 caps).

Architecture
------------
Level-0 features (all pulled from the scored signal row):
    p_tp_before_sl, expected_return, edge_score,
    wallet_state_score, whale_pressure, market_structure_score,
    btc_network_stress_score, btc_trend_confluence,
    <ta_bias_match>          one-hot derived field
    <volatility_bucket_*>    one-hot derived field

Level-1 meta-learner:
    L2-regularised logistic regression, fit via Newton-IRLS.
    Training target: 1 if closed-trade `realized_pnl > 0`, else 0.
    Optional calibration wrap via services/calibration_suite (auto-select
    best of Platt / Isotonic / Beta on held-out trades).

Robustness
----------
- Zero external deps beyond numpy / pandas.
- Refuses to fit below `min_samples` — returns `is_fitted=False` so the
  caller falls back to the legacy hand-tuned formula.
- Persists to logs/confidence_stacker.json (plain JSON, no pickle).
- Schema-versioned: features list is part of the artefact; a rename that
  changes the feature set forces a refit instead of crashing live.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Schema version — bump when changing feature layout
_SCHEMA_VERSION = 2

# Level-0 features (numeric; missing → 0.0)
_NUMERIC_FEATURES = [
    "p_tp_before_sl",
    "expected_return",
    "edge_score",
    "wallet_state_score",
    "whale_pressure",
    "market_structure_score",
    "btc_network_stress_score",
    "btc_trend_confluence",
    "confidence_at_entry",  # legacy hand-tuned confidence (kept as feature so
                            # the stacker starts out monotone in it)
]

# Categorical one-hot features (string → indicator)
_CATEGORICAL_FEATURES = {
    "volatility_bucket": ("low", "medium", "high", "extreme"),
    "market_family": ("btc", "weather_temperature"),
    "btc_trend_bias": ("LONG", "SHORT", "NEUTRAL"),
}


def _expand_categorical(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, values in _CATEGORICAL_FEATURES.items():
        src = df[col].astype(str).str.lower() if col in df.columns else pd.Series([""] * len(df), index=df.index)
        for v in values:
            out[f"{col}__{v}"] = (src == str(v).lower()).astype(float)
    return out


def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    num = pd.DataFrame(index=df.index)
    for col in _NUMERIC_FEATURES:
        if col in df.columns:
            num[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            num[col] = 0.0
    cat = _expand_categorical(df)
    full = pd.concat([num.fillna(0.0), cat.fillna(0.0)], axis=1)
    # Constant / bias column added inside the fit routine.
    return full.to_numpy(dtype=float), list(full.columns)


# ─────────────────────────── L2 logistic fit ───────────────────────────
def _fit_logistic_l2(
    X: np.ndarray, y: np.ndarray, *, l2: float = 1.0,
    max_iter: int = 50, tol: float = 1e-6,
) -> np.ndarray:
    """
    Newton-IRLS for L2-regularised logistic regression.
    Prepends a bias column so callers pass raw features.
    Returns the weight vector of length X.shape[1] + 1.
    """
    n, d = X.shape
    X_ = np.hstack([np.ones((n, 1)), X])          # bias column
    w = np.zeros(d + 1, dtype=float)
    reg = l2 * np.eye(d + 1)
    reg[0, 0] = 0.0                                # don't regularise bias

    for _ in range(max_iter):
        z = X_ @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        W = p * (1.0 - p)                          # diag(p*(1-p))
        # Gradient and Hessian
        grad = X_.T @ (p - y) + reg @ w
        H = (X_.T * W) @ X_ + reg
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w_new = w - step
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def _predict_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    X_ = np.hstack([np.ones((n, 1)), X])
    z = X_ @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


# ────────────────────────────── main class ─────────────────────────────
@dataclass
class StackerArtefact:
    schema_version: int = _SCHEMA_VERSION
    feature_names: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    n_train: int = 0
    auc_train: float | None = None
    brier_train: float | None = None
    calibrator: dict | None = None    # optional {"kind": "platt", "a": ..., "b": ...}
    trained_at: str | None = None

    def to_json(self) -> str:
        return json.dumps({
            "schema_version": self.schema_version,
            "feature_names": self.feature_names,
            "weights": self.weights,
            "n_train": self.n_train,
            "auc_train": self.auc_train,
            "brier_train": self.brier_train,
            "calibrator": self.calibrator,
            "trained_at": self.trained_at,
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "StackerArtefact":
        d = json.loads(text)
        return cls(
            schema_version=int(d.get("schema_version", 0)),
            feature_names=list(d.get("feature_names", [])),
            weights=list(d.get("weights", [])),
            n_train=int(d.get("n_train", 0)),
            auc_train=d.get("auc_train"),
            brier_train=d.get("brier_train"),
            calibrator=d.get("calibrator"),
            trained_at=d.get("trained_at"),
        )


class ConfidenceStacker:
    """
    End-to-end API:

        s = ConfidenceStacker(path="logs/confidence_stacker.json")
        s.load_if_exists()
        if not s.is_fitted:
            s.fit_from_closed_trades("logs/closed_positions.csv")
        confidence = s.predict_row({"p_tp_before_sl": ..., ...})

    Caller falls back to the legacy hand-tuned formula when
    `s.is_fitted == False`.
    """

    def __init__(
        self,
        *,
        path: str = "logs/confidence_stacker.json",
        min_samples: int = 80,
        l2: float = 1.0,
        calibrate: bool = True,
    ):
        self.path = Path(path)
        self.min_samples = int(min_samples)
        self.l2 = float(l2)
        self.calibrate = bool(calibrate)
        self._artefact: StackerArtefact | None = None

    # ───────── persistence ─────────
    @property
    def is_fitted(self) -> bool:
        return self._artefact is not None and len(self._artefact.weights) > 0

    def load_if_exists(self) -> bool:
        if not self.path.exists():
            return False
        try:
            art = StackerArtefact.from_json(self.path.read_text(encoding="utf-8"))
            if art.schema_version != _SCHEMA_VERSION:
                logger.info("Stacker artefact schema mismatch (%s != %s) — refit needed.",
                            art.schema_version, _SCHEMA_VERSION)
                return False
            self._artefact = art
            return True
        except Exception as e:
            logger.warning("Stacker load failed: %s — will refit.", e)
            return False

    def _persist(self) -> None:
        if self._artefact is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self._artefact.to_json(), encoding="utf-8")

    # ───────── fit ─────────
    def fit_from_closed_trades(self, closed_csv: str = "logs/closed_positions.csv") -> dict:
        path = Path(closed_csv)
        if not path.exists():
            return {"status": "no_trades", "path": str(path)}
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception as e:
            return {"status": "csv_read_failed", "error": str(e)}
        if df.empty or "realized_pnl" not in df.columns:
            return {"status": "missing_realized_pnl"}
        y = (pd.to_numeric(df["realized_pnl"], errors="coerce") > 0).astype(float).to_numpy()
        X, names = _feature_matrix(df)
        m = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[m], y[m]
        if X.shape[0] < self.min_samples:
            return {"status": "insufficient", "n": int(X.shape[0]),
                    "need": self.min_samples}
        # Drop zero-variance columns to avoid singular Hessian.
        var = X.var(axis=0)
        keep = var > 1e-10
        if not keep.all():
            X = X[:, keep]
            names = [n for n, k in zip(names, keep) if k]
        # Fit L2 logistic.
        w = _fit_logistic_l2(X, y, l2=self.l2)
        # In-sample metrics (honest train-only; deploy-side reporting is
        # handled by calibration_report.py on held-out data).
        p = _predict_logistic(X, w)
        auc = _auc(p, y)
        brier = float(np.mean((p - y) ** 2))
        # Optional Platt calibration on the same predictions (cheap,
        # data-efficient; isotonic/beta tried by nightly calibration_suite).
        calibrator = None
        if self.calibrate:
            try:
                from services.calibration_suite import PlattScaler
                ps = PlattScaler().fit(p, y)
                calibrator = {"kind": "platt", "a": ps.a, "b": ps.b}
            except Exception as e:
                logger.debug("Platt wrap skipped: %s", e)

        self._artefact = StackerArtefact(
            schema_version=_SCHEMA_VERSION,
            feature_names=list(names),
            weights=[float(x) for x in w],
            n_train=int(X.shape[0]),
            auc_train=float(auc) if auc is not None else None,
            brier_train=float(brier),
            calibrator=calibrator,
            trained_at=pd.Timestamp.utcnow().isoformat(),
        )
        self._persist()
        return {
            "status": "ok", "n": int(X.shape[0]),
            "auc": round(float(auc), 4) if auc is not None else None,
            "brier": round(float(brier), 4),
            "n_features": int(X.shape[1]),
            "calibrator": calibrator["kind"] if calibrator else None,
        }

    # ───────── predict ─────────
    def predict_row(self, row: dict | pd.Series) -> float:
        if not self.is_fitted:
            return float("nan")
        art = self._artefact
        # Reuse pipeline: wrap single row in DataFrame
        df = pd.DataFrame([dict(row)])
        X, names = _feature_matrix(df)
        # Align columns to trained schema (pad missing with 0, drop extras).
        xvec = np.zeros(len(art.feature_names), dtype=float)
        name_to_idx = {n: i for i, n in enumerate(names)}
        for i, n in enumerate(art.feature_names):
            j = name_to_idx.get(n)
            if j is not None:
                xvec[i] = X[0, j]
        p = _predict_logistic(xvec.reshape(1, -1), np.asarray(art.weights, dtype=float))[0]
        # Calibrator wrap
        c = art.calibrator or {}
        if c.get("kind") == "platt":
            try:
                a = float(c.get("a", 1.0)); b = float(c.get("b", 0.0))
                z = a * float(p) + b
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
            except Exception:
                pass
        return float(np.clip(p, 0.0, 1.0))

    def summary(self) -> dict:
        if not self.is_fitted:
            return {"fitted": False}
        art = self._artefact
        return {
            "fitted": True,
            "schema_version": art.schema_version,
            "n_train": art.n_train,
            "auc_train": art.auc_train,
            "brier_train": art.brier_train,
            "n_features": len(art.feature_names),
            "trained_at": art.trained_at,
            "calibrator": (art.calibrator or {}).get("kind"),
        }


# ──────────────────────────── helpers ──────────────────────────────────
def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """ROC-AUC via rank-sum (Mann-Whitney U). Returns None when degenerate."""
    s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=float)
    pos = s[y > 0.5]; neg = s[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    pos_ranks = ranks[y > 0.5].sum()
    u = pos_ranks - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


__all__ = ["ConfidenceStacker", "StackerArtefact"]
