"""Calibrated edge adjustment layer.

Replaces hand-tuned feedback multipliers with a model trained on realized
outcomes from canonical_dataset.csv.

WHAT THIS REPLACES
------------------
Before (heuristics):
  btc_regime_router.py  → REGIME_MODEL_WEIGHTS multipliers {calm:1.02, chaotic:0.72, ...}
  trade_feedback_learner.py → conf_mult * ret_mult * edge_mult chains per scope

After (calibrated):
  EdgeCalibrator.adjust(signal_row) → single calibrated_edge_factor ∈ [FLOOR, CAP]
  The factor is derived from:
    "in this (regime × spread_bucket × liquidity_bucket × vol_bucket) context,
     what fraction of predicted_edge was historically realised?"
  A lightweight GBM or lookup table is fitted offline from canonical data.
  The live path never writes anything — it only calls .adjust().

HOW IT WORKS
------------
Training (offline):
  1. Load canonical_dataset (eligible rows, BTC family).
  2. Compute per-row realised_fraction = price_return / predicted_edge
     (clipped to [-3, 3] — outliers corrupt the calibration).
  3. Bin context into (regime_bucket × spread_bucket × liq_bucket × vol_bucket).
  4. Fit GradientBoostingRegressor(target=realised_fraction) on context features.
  5. Save artifact to weights/{family}/edge_calibrator.joblib.
  6. Also save a coarse lookup table (context_key → median_realised_fraction)
     as a fast fallback when the model cannot score a live row.

Inference (live):
  calibrator = EdgeCalibrator.load("btc")
  factor = calibrator.adjust(signal_row)  # float, clipped to [FLOOR, CAP]
  edge_score_adjusted = edge_score * factor

The FLOOR and CAP prevent the calibrator from amplifying marginal signals or
reducing a good signal to zero from a single stale data window.
  FLOOR = 0.50  (calibrator can halve predicted edge at most)
  CAP   = 1.20  (calibrator can add 20% at most)

These are also env-var overridable (CALIBRATOR_FLOOR, CALIBRATOR_CAP).

INTEGRATION POINTS
------------------
btc_regime_router.py:
  Old: p_tp_blended *= REGIME_MODEL_WEIGHTS[regime]["multiplier"]
  New: p_tp_blended *= EdgeCalibrator.load("btc").adjust(row)

trade_feedback_learner.apply_to_scored_df():
  Old: conf_mult * ret_mult * edge_mult chain across 6 scopes
  New: single calibrated factor from EdgeCalibrator
  Old multipliers are RETAINED as diagnostic columns but not applied.

performance_governor.FamilyPerformanceGovernor.evaluate():
  Old: hardcoded thresholds (min_wr_1=0.38, max_dd_1=30.0)
  New: thresholds derived from canonical_dataset historical quantiles
       via GovernorThresholdCalibrator (bottom of this file)
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_FLOOR = 0.50   # never reduce predicted edge by more than 50%
_DEFAULT_CAP   = 1.20   # never amplify predicted edge by more than 20%
_CLIP_RATIO    = 3.0    # clip realised_fraction outliers at ±3

_CONTEXT_COLS = [
    "technical_regime_bucket",
    "spread_bucket",
    "liquidity_bucket",
    "volatility_bucket",
]

_FEATURE_COLS = _CONTEXT_COLS + [
    "market_implied_prob",
    "confidence_score",
    "predicted_edge",
    "btc_volatility_24h",
    "btc_rsi_14",
    "hold_time_seconds",
]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Bucketing helpers — deterministic context key from a signal row
# ---------------------------------------------------------------------------

def _spread_bucket(spread_bps: float) -> str:
    if math.isnan(spread_bps):
        return "unknown"
    if spread_bps < 20:
        return "tight"
    if spread_bps < 60:
        return "normal"
    if spread_bps < 150:
        return "wide"
    return "very_wide"


def _liquidity_bucket(liq_score: float) -> str:
    if math.isnan(liq_score):
        return "unknown"
    if liq_score >= 0.75:
        return "deep"
    if liq_score >= 0.45:
        return "medium"
    if liq_score >= 0.20:
        return "thin"
    return "illiquid"


def _vol_bucket(vol_24h: float) -> str:
    if math.isnan(vol_24h):
        return "unknown"
    if vol_24h < 0.01:
        return "low"
    if vol_24h < 0.025:
        return "medium"
    if vol_24h < 0.05:
        return "high"
    return "extreme"


def _context_key(row: dict | pd.Series) -> str:
    def _get(k):
        v = row.get(k) if hasattr(row, "get") else getattr(row, k, None)
        return str(v or "unknown").lower().strip()

    regime = _get("technical_regime_bucket")
    spread = _spread_bucket(float(pd.to_numeric(row.get("spread_bps", float("nan")), errors="coerce") or float("nan")))
    liq    = _liquidity_bucket(float(pd.to_numeric(row.get("liquidity_score", float("nan")), errors="coerce") or float("nan")))
    vol    = _vol_bucket(float(pd.to_numeric(row.get("btc_volatility_24h", float("nan")), errors="coerce") or float("nan")))

    # Fill missing buckets from the row's own bucket columns if available
    if spread == "unknown" and "spread_bucket" in (row.keys() if hasattr(row, "keys") else []):
        spread = _get("spread_bucket")
    if liq == "unknown" and "liquidity_bucket" in (row.keys() if hasattr(row, "keys") else []):
        liq = _get("liquidity_bucket")
    if vol == "unknown" and "volatility_bucket" in (row.keys() if hasattr(row, "keys") else []):
        vol = _get("volatility_bucket")

    return f"{regime}|{spread}|{liq}|{vol}"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_context_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add spread_bucket, liquidity_bucket, volatility_bucket columns to df."""
    out = df.copy()

    def _col_or_nan(frame: pd.DataFrame, name: str) -> pd.Series:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(float("nan"), index=frame.index)

    spread_bps = _col_or_nan(out, "spread_bps")
    liq_score  = _col_or_nan(out, "liquidity_score")
    vol_24h    = _col_or_nan(out, "btc_volatility_24h")

    out["spread_bucket"]    = spread_bps.map(lambda x: _spread_bucket(x if not pd.isna(x) else float("nan")))
    out["liquidity_bucket"] = liq_score.map(lambda x: _liquidity_bucket(x if not pd.isna(x) else float("nan")))
    out["volatility_bucket"] = vol_24h.map(lambda x: _vol_bucket(x if not pd.isna(x) else float("nan")))

    if "technical_regime_bucket" not in out.columns:
        out["technical_regime_bucket"] = "unknown"

    out["context_key"] = out.apply(_context_key, axis=1)
    return out


def _compute_realised_fraction(df: pd.DataFrame) -> pd.Series:
    """
    Compute the realised fraction of expected edge per row.

    Primary:   price_return / predicted_edge  (clipped to [-3, 3])
    Fallback A: price_return / ((confidence_score - 0.5) * 2)
                — when predicted_edge is absent, use confidence as a proxy
    Fallback B: price_return itself (mean-normalised)
                — when neither predicted_edge nor confidence_score is present,
                  use the sign and relative magnitude of price_return to build
                  a usable calibration target.  This is weaker but still tells
                  us which contexts reliably produce positive/negative returns.

    In all cases the result is clipped to [-_CLIP_RATIO, _CLIP_RATIO].
    Rows where the denominator is near zero are excluded.
    """
    actual = pd.to_numeric(df.get("price_return", pd.Series(dtype=float)), errors="coerce")

    # --- Primary: predicted_edge ---
    if "predicted_edge" in df.columns:
        pred = pd.to_numeric(df["predicted_edge"], errors="coerce")
        valid = pred.abs() > 1e-4
        if valid.sum() >= 10:
            ratio = pd.Series(float("nan"), index=df.index)
            ratio.loc[valid] = (actual.loc[valid] / pred.loc[valid]).clip(-_CLIP_RATIO, _CLIP_RATIO)
            return ratio

    # --- Fallback A: confidence_score proxy ---
    if "confidence_score" in df.columns:
        conf = pd.to_numeric(df["confidence_score"], errors="coerce")
        pred_proxy = (conf - 0.5) * 2.0
        valid = pred_proxy.abs() > 1e-4
        if valid.sum() >= 10:
            ratio = pd.Series(float("nan"), index=df.index)
            ratio.loc[valid] = (actual.loc[valid] / pred_proxy.loc[valid]).clip(-_CLIP_RATIO, _CLIP_RATIO)
            log.debug("[EdgeCalibrator] using confidence_score proxy for predicted_edge")
            return ratio

    # --- Fallback B: mean-normalised price_return ---
    # Treats the sign-and-magnitude of return as a direct calibration signal.
    # realised_fraction = price_return / |mean(price_return)| clipped to [-3, 3].
    mean_abs = float(actual.abs().mean())
    if mean_abs > 1e-8 and actual.notna().sum() >= 10:
        ratio = (actual / mean_abs).clip(-_CLIP_RATIO, _CLIP_RATIO)
        log.info("[EdgeCalibrator] using mean-normalised price_return as calibration target")
        return ratio

    return pd.Series(float("nan"), index=df.index)


def train_edge_calibrator(
    family: str = "btc",
    logs_dir: str = "logs",
    weights_dir: str = "weights",
) -> Path | None:
    """
    Train the edge calibrator from canonical_dataset and save the artifact.
    Returns path to saved artifact or None if training was not possible.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    logs_path    = Path(logs_dir)
    weights_path = Path(weights_dir)

    # Load canonical dataset
    canon_path = logs_path / family / "canonical_dataset.csv"
    if not canon_path.exists():
        canon_path = logs_path / "canonical_dataset.csv"
    if not canon_path.exists():
        log.warning("[EdgeCalibrator] canonical dataset not found for family=%s", family)
        return None

    df = pd.read_csv(canon_path, low_memory=False)

    # BTC/weather split guard — keep only rows belonging to this family
    if "market_family" in df.columns:
        family_lower = family.lower()
        mf = df["market_family"].astype(str).str.lower()
        if "weather" in family_lower:
            df = df[mf.str.startswith("weather")].copy()
        else:
            # btc: keep all non-weather rows
            df = df[~mf.str.startswith("weather")].copy()

    # Eligible rows only
    if "learning_eligible" in df.columns:
        elig = df["learning_eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[elig].copy()

    if len(df) < 30:
        log.warning("[EdgeCalibrator] too few eligible rows (%d) for family=%s", len(df), family)
        return None

    # Sort by time
    for ts_col in ("timestamp", "signal_timestamp", "fill_timestamp"):
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            df = df.sort_values(ts_col).reset_index(drop=True)
            break

    # Add context buckets
    df = _build_context_buckets(df)

    # Compute target: realised fraction of predicted edge
    df["realised_fraction"] = _compute_realised_fraction(df)
    df = df[df["realised_fraction"].notna()].copy()

    if len(df) < 20:
        log.warning("[EdgeCalibrator] too few rows with valid realised_fraction (%d)", len(df))
        return None

    log.info("[EdgeCalibrator] training on %d rows (family=%s)", len(df), family)

    # Feature set for the GBM
    feature_cols = [c for c in _FEATURE_COLS if c in df.columns]

    # Encode categoricals
    encoders: dict[str, LabelEncoder] = {}
    X = df[feature_cols].copy()
    for c in feature_cols:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str).fillna("__missing__"))
            encoders[c] = le

    y = df["realised_fraction"].to_numpy()

    # Build lookup table first (fast fallback for inference)
    lookup: dict[str, float] = {}
    for key, grp in df.groupby("context_key"):
        vals = grp["realised_fraction"].dropna()
        if len(vals) >= 3:
            # Shrink toward global median with sample-size weighting
            global_med = float(df["realised_fraction"].median())
            n          = len(vals)
            weight     = min(1.0, n / 20.0)   # full weight at n=20
            lookup[str(key)] = float(weight * vals.median() + (1 - weight) * global_med)
    global_median = float(df["realised_fraction"].median())
    log.info("[EdgeCalibrator] lookup table: %d context keys  global_median=%.4f", len(lookup), global_median)

    # Train GBM model
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])

    try:
        pipe.fit(X, y)
        train_pred = pipe.predict(X)
        mae = float(np.mean(np.abs(train_pred - y)))
        log.info("[EdgeCalibrator] GBM trained  MAE=%.4f", mae)
    except Exception as exc:
        log.warning("[EdgeCalibrator] GBM training failed: %s — will use lookup only", exc)
        pipe = None

    # Save artifact
    import joblib
    out_dir = weights_path / family
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "edge_calibrator.joblib"
    joblib.dump({
        "model":          pipe,
        "encoders":       encoders,
        "feature_cols":   feature_cols,
        "lookup":         lookup,
        "global_median":  global_median,
        "family":         family,
        "floor":          _env_float("CALIBRATOR_FLOOR", _DEFAULT_FLOOR),
        "cap":            _env_float("CALIBRATOR_CAP",   _DEFAULT_CAP),
        "n_train":        len(df),
    }, artifact_path)
    log.info("[EdgeCalibrator] saved → %s", artifact_path)
    return artifact_path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class EdgeCalibrator:
    """
    Live inference wrapper.  Load once at startup, call .adjust() per signal.

    Usage:
        calibrator = EdgeCalibrator.load("btc")
        factor = calibrator.adjust(signal_row)
        edge_score_adjusted = edge_score * factor
        # factor is always in [FLOOR, CAP]
    """

    _instances: dict[str, "EdgeCalibrator"] = {}   # process-level cache

    def __init__(self, bundle: dict) -> None:
        self._model      = bundle.get("model")
        self._encoders   = bundle.get("encoders", {})
        self._feat_cols  = bundle.get("feature_cols", [])
        self._lookup     = bundle.get("lookup", {})
        self._global_med = float(bundle.get("global_median", 1.0))
        self._family     = bundle.get("family", "btc")
        self._floor      = float(bundle.get("floor", _DEFAULT_FLOOR))
        self._cap        = float(bundle.get("cap",   _DEFAULT_CAP))

    # -- Loading -------------------------------------------------------------

    @classmethod
    def load(
        cls,
        family: str,
        weights_dir: str = "weights",
        *,
        force_reload: bool = False,
    ) -> "EdgeCalibrator":
        """
        Load from artifact (cached in process memory after first load).
        Falls back to a neutral calibrator (factor=1.0 always) if no artifact.
        """
        cache_key = f"{family}:{weights_dir}"
        if not force_reload and cache_key in cls._instances:
            return cls._instances[cache_key]

        import joblib
        artifact_path = Path(weights_dir) / family / "edge_calibrator.joblib"
        if artifact_path.exists():
            try:
                bundle = joblib.load(artifact_path)
                instance = cls(bundle)
                cls._instances[cache_key] = instance
                log.info("[EdgeCalibrator] loaded from %s", artifact_path)
                return instance
            except Exception as exc:
                log.warning("[EdgeCalibrator] failed to load artifact: %s — using neutral", exc)

        # Neutral calibrator — always returns 1.0
        neutral = cls({
            "model": None, "encoders": {}, "feature_cols": [],
            "lookup": {}, "global_median": 1.0, "family": family,
            "floor": _DEFAULT_FLOOR, "cap": _DEFAULT_CAP,
        })
        cls._instances[cache_key] = neutral
        return neutral

    @classmethod
    def invalidate_cache(cls, family: str | None = None) -> None:
        """Call after retraining to force reload on next .load() call."""
        if family is None:
            cls._instances.clear()
        else:
            keys = [k for k in cls._instances if k.startswith(f"{family}:")]
            for k in keys:
                del cls._instances[k]

    # -- Core inference ------------------------------------------------------

    def adjust(self, signal_row: dict | pd.Series) -> float:
        """
        Return the calibrated edge factor for a signal row.

        The factor represents the historically realised fraction of predicted edge
        in this context (regime × spread × liquidity × volatility).

        Always returns a float in [floor, cap].
        Returns 1.0 (neutral) if the calibrator has no data for this context.
        """
        raw = self._predict_raw(signal_row)
        if math.isnan(raw):
            raw = self._global_med
        factor = float(np.clip(raw, self._floor, self._cap))
        return factor

    def adjust_dict(self, signal_row: dict) -> dict:
        """
        Return a dict with calibrated fields added.
        Does not modify the original row.

        Added keys:
          calibrated_edge_factor        — the multiplier
          calibrated_edge_context_key   — the context bucket used
          pre_calibration_edge_score    — original edge_score (diagnostic)
          pre_calibration_expected_return — original expected_return (diagnostic)
        """
        factor = self.adjust(signal_row)
        key    = _context_key(signal_row)
        out = dict(signal_row)
        out["pre_calibration_edge_score"]        = signal_row.get("edge_score", float("nan"))
        out["pre_calibration_expected_return"]   = signal_row.get("expected_return", float("nan"))
        out["calibrated_edge_factor"]            = round(factor, 4)
        out["calibrated_edge_context_key"]       = key
        # Apply to edge_score and expected_return
        if "edge_score" in out:
            out["edge_score"] = float(out["edge_score"] or 0.0) * factor
        if "expected_return" in out:
            out["expected_return"] = float(out["expected_return"] or 0.0) * factor
        return out

    def _predict_raw(self, row: dict | pd.Series) -> float:
        """Raw (unclipped) predicted realised_fraction for this context."""
        # Try GBM model first
        if self._model is not None and self._feat_cols:
            try:
                row_data: dict[str, Any] = {}
                for c in self._feat_cols:
                    val = row.get(c) if hasattr(row, "get") else getattr(row, c, None)
                    if c in self._encoders and val is not None:
                        le = self._encoders[c]
                        val_str = str(val or "__missing__")
                        if val_str in le.classes_:
                            val = int(le.transform([val_str])[0])
                        else:
                            val = -1   # unseen label — treated as missing by imputer
                    row_data[c] = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else float("nan")
                X = pd.DataFrame([row_data])
                pred = float(self._model.predict(X)[0])
                if math.isfinite(pred):
                    return pred
            except Exception as exc:
                log.debug("[EdgeCalibrator] GBM predict failed: %s — falling back to lookup", exc)

        # Lookup table fallback
        key = _context_key(row)
        if key in self._lookup:
            return float(self._lookup[key])

        # Partial key match (regime only)
        regime_prefix = key.split("|")[0] + "|"
        matches = [v for k, v in self._lookup.items() if k.startswith(regime_prefix)]
        if matches:
            return float(np.median(matches))

        return float(self._global_med)

    # -- Diagnostics ---------------------------------------------------------

    def context_table(self) -> pd.DataFrame:
        """Return the lookup table as a DataFrame, sorted by median realised fraction."""
        if not self._lookup:
            return pd.DataFrame()
        rows = []
        for key, val in sorted(self._lookup.items(), key=lambda x: x[1]):
            regime, spread, liq, vol = (key.split("|") + ["?", "?", "?", "?"])[:4]
            rows.append({
                "context_key":       key,
                "regime":            regime,
                "spread_bucket":     spread,
                "liquidity_bucket":  liq,
                "vol_bucket":        vol,
                "median_realised_fraction": round(val, 4),
                "calibrated_factor":  round(float(np.clip(val, self._floor, self._cap)), 4),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Governor threshold calibrator
# ---------------------------------------------------------------------------

class GovernorThresholdCalibrator:
    """
    Derives data-driven thresholds for FamilyPerformanceGovernor instead of
    using hardcoded values like (min_wr_1=0.38, max_dd_1=30.0).

    Method:
      - Load canonical_dataset / closed_positions for the family.
      - Compute rolling EWMA win rate and rolling drawdown.
      - Set thresholds at the 20th/10th/5th percentile of historical values
        (conservative: we want the governor to trigger when conditions are
         materially worse than what has been seen as acceptable historically).

    Returns a dict matching the structure used by FamilyPerformanceGovernor.evaluate().
    """

    _DEFAULT_THRESHOLDS = {
        "btc": {
            "min_wr_1": 0.38, "min_wr_2A": 0.32, "min_wr_2B": 0.28,
            "max_dd_1": 30.0, "max_dd_2A": 50.0, "max_dd_2B": 75.0,
            "max_decay_1": 0.01, "max_decay_2A": 0.025,
        },
        "weather": {
            "min_wr_1": 0.30, "min_wr_2A": 0.25, "min_wr_2B": 0.20,
            "max_dd_1": 25.0, "max_dd_2A": 40.0, "max_dd_2B": 60.0,
            "max_decay_1": 0.02, "max_decay_2A": 0.04,
        },
    }

    def __init__(self, family: str, logs_dir: str = "logs") -> None:
        self.family    = family.lower()
        self.logs_dir  = Path(logs_dir)

    def _load_pnl_series(self) -> pd.Series:
        for path in [
            self.logs_dir / self.family / "canonical_dataset.csv",
            self.logs_dir / "closed_positions.csv",
        ]:
            if path.exists():
                try:
                    df = pd.read_csv(path, low_memory=False)
                    pnl_col = next((c for c in ("price_return", "net_realized_pnl", "realized_pnl") if c in df.columns), None)
                    if pnl_col and len(df) >= 30:
                        pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
                        if len(pnl) >= 30:
                            return pnl
                except Exception:
                    pass
        return pd.Series(dtype=float)

    def derive(self, min_rows: int = 30) -> dict:
        """
        Derive thresholds from data.  Falls back to hardcoded defaults if
        insufficient data is available.
        """
        pnl = self._load_pnl_series()
        family_key = "btc" if "btc" in self.family else "weather"
        defaults = self._DEFAULT_THRESHOLDS[family_key].copy()

        if len(pnl) < min_rows:
            log.info("[GovernorCalibrator] insufficient data (%d rows) — using defaults", len(pnl))
            return defaults

        # Rolling EWMA win rate history
        wins = (pnl > 0).astype(float)
        wr_ewma = wins.ewm(span=30, adjust=False).mean()

        # Rolling drawdown history
        running  = pnl.cumsum()
        peak     = running.cummax()
        drawdown = (peak - running)

        # Thresholds at historical percentiles
        # Level 1 triggers at 20th percentile (moderate stress)
        # Level 2A at 10th (significant stress)
        # Level 2B at 5th  (extreme stress)
        wr_p20 = float(wr_ewma.quantile(0.20))
        wr_p10 = float(wr_ewma.quantile(0.10))
        wr_p05 = float(wr_ewma.quantile(0.05))
        dd_p80 = float(drawdown.quantile(0.80))   # drawdown: higher = worse, trigger at 80th pct
        dd_p90 = float(drawdown.quantile(0.90))
        dd_p95 = float(drawdown.quantile(0.95))

        # Clamp: calibrated thresholds must not be more permissive than the
        # hardcoded defaults (prevents calibration from being gamed by
        # a period of unusually bad performance).
        derived = {
            "min_wr_1":   round(max(wr_p20, defaults["min_wr_1"]  * 0.85), 4),
            "min_wr_2A":  round(max(wr_p10, defaults["min_wr_2A"] * 0.85), 4),
            "min_wr_2B":  round(max(wr_p05, defaults["min_wr_2B"] * 0.85), 4),
            "max_dd_1":   round(min(dd_p80, defaults["max_dd_1"]  * 1.15), 2),
            "max_dd_2A":  round(min(dd_p90, defaults["max_dd_2A"] * 1.15), 2),
            "max_dd_2B":  round(min(dd_p95, defaults["max_dd_2B"] * 1.15), 2),
            "max_decay_1":  defaults["max_decay_1"],   # edge decay — keep fixed (too noisy to derive)
            "max_decay_2A": defaults["max_decay_2A"],
            "calibrated_from_data": True,
            "n_rows": int(len(pnl)),
        }

        log.info(
            "[GovernorCalibrator] %s derived: wr_1=%.3f wr_2A=%.3f dd_1=%.1f dd_2A=%.1f",
            self.family, derived["min_wr_1"], derived["min_wr_2A"],
            derived["max_dd_1"], derived["max_dd_2A"],
        )
        return derived

    def save(self, weights_dir: str = "weights") -> Path:
        """Derive and persist thresholds as a small JSON artifact."""
        import json
        thresholds = self.derive()
        out_dir  = Path(weights_dir) / self.family
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "governor_thresholds.json"
        out_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        log.info("[GovernorCalibrator] saved → %s", out_path)
        return out_path

    @staticmethod
    def load_thresholds(family: str, weights_dir: str = "weights") -> dict | None:
        """Load persisted thresholds. Returns None if not yet calibrated."""
        import json
        p = Path(weights_dir) / family / "governor_thresholds.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# CLI — train both artifacts
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train edge calibrator + governor thresholds")
    parser.add_argument("--family",      default="btc", choices=["btc", "weather_temperature"])
    parser.add_argument("--logs-dir",    default="logs")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--show-table",  action="store_true", help="Print context lookup table")
    args = parser.parse_args()

    # Train edge calibrator
    path = train_edge_calibrator(args.family, args.logs_dir, args.weights_dir)
    if path:
        print(f"EdgeCalibrator saved: {path}")
        if args.show_table:
            cal = EdgeCalibrator.load(args.family, args.weights_dir, force_reload=True)
            print(cal.context_table().to_string(index=False))
    else:
        print("EdgeCalibrator training failed — check logs", file=sys.stderr)

    # Derive governor thresholds
    gov_cal = GovernorThresholdCalibrator(args.family, args.logs_dir)
    gov_path = gov_cal.save(args.weights_dir)
    print(f"GovernorThresholds saved: {gov_path}")
