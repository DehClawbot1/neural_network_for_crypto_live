"""Task-specific model suite.

Five separate learners, each reading from logs/{family}/task_training/{task}.csv.
No single learner tries to predict everything — each is scoped to one decision layer.

Models:
  EntryEdgeModel          — predicts entry_quality (binary) from signal/microstructure features
  FillProbabilityModel    — predicts fill success from order mechanics (uses ALL rows)
  SlippageLiquidityModel  — predicts execution_quality (binary) from fill mechanics
  ExitQualityModel        — predicts exit_quality (binary) from close context (BTC only)
  RegimeCalibrationModel  — maps live features → technical_regime_bucket (multiclass)

Each model:
  - Reads logs/{family}/task_training/{task}.csv
  - Applies its own row filter
  - Saves weights/{family}/task_{name}.joblib
  - Requires market_family in {"btc", "weather_temperature"}

Entry point:
  TaskModelSuite(family).train_all()
  TaskModelSuite(family).train("entry_edge")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

_SUPPORTED_FAMILIES = {"btc", "weather_temperature"}

# ---------------------------------------------------------------------------
# Feature sets per canonical schema section
# ---------------------------------------------------------------------------

# Section D — microstructure at signal time
_FEAT_MICROSTRUCTURE = [
    "spread_bps",
    "best_bid",
    "best_ask",
    "orderbook_depth_bid",
    "orderbook_depth_ask",
    "market_implied_prob",
    "liquidity_score",
]

# Section E — BTC regime (live at signal time)
_FEAT_BTC_REGIME = [
    "btc_price_usd",
    "btc_rsi_14",
    "btc_volatility_24h",
    "btc_trend_bias",
    "btc_volatility_regime",
    "btc_market_regime_label",
    "btc_funding_rate",
    "btc_volume_24h",
]

# Section G — wallet / portfolio context
_FEAT_WALLET = [
    "wallet_balance_usdc",
    "open_positions_count",
    "portfolio_heat",
    "recent_wallet_activity_5",
    "recent_yes_ratio_5",
]

# Section H — order decision
_FEAT_ORDER_DECISION = [
    "signal_label",
    "confidence_score",
    "predicted_edge",
    "size_usdc",
    "tp_threshold_pct",
    "sl_threshold_pct",
    "brain_id",
    "technical_regime_bucket",
    "entry_gate_passed",
    "supervisor_gate_passed",
]

# Section I — fill result
_FEAT_FILL = [
    "entry_price_filled",
    "slippage_bps",
    "fill_latency_ms",
    "partial_fill_ratio",
    "orderbook_depth_at_fill",
    "spread_at_fill_bps",
]

# Section J — close result
_FEAT_CLOSE = [
    "exit_price",
    "price_return",
    "hold_time_seconds",
    "close_reason",
]

_FEAT_WEATHER_FORECAST = [
    "weather_market_probability",
    "weather_forecast_edge",
    "forecast_margin_to_lower_c",
    "forecast_margin_to_upper_c",
    "forecast_uncertainty_c",
    "liquidity_score",
    "volume_score",
    "open_positions_count",
    "confidence",
    "p_tp_before_sl",
    "expected_return",
    "final_decision",
    "weather_location",
    "weather_question_type",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_task_training_table(family: str, task_name: str, logs_dir: Path) -> pd.DataFrame:
    path = logs_dir / family / "task_training" / f"{task_name}.csv"
    if not path.exists():
        log.warning("Task training table not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp")
    elif "signal_timestamp" in df.columns:
        df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"], errors="coerce", utc=True)
        df = df.sort_values("signal_timestamp")
    return df


def _eligible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with no artifact flags set."""
    if "learning_eligible" in df.columns:
        mask = df["learning_eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
        return df[mask].copy()
    return df.copy()


def _encode_categoricals(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Label-encode object columns in feature_cols in-place, return updated df and cols."""
    result = df.copy()
    for c in feature_cols:
        if c not in result.columns:
            continue
        if result[c].dtype == object:
            le = LabelEncoder()
            result[c] = le.fit_transform(result[c].astype(str).fillna("__missing__"))
    return result, feature_cols


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return only columns that exist in df and have at least one non-null value."""
    seen: set[str] = set()
    present: list[str] = []
    for c in cols:
        if c in seen:
            continue
        if c in df.columns and df[c].notna().any():
            present.append(c)
            seen.add(c)
    return present


def _build_pipeline(use_gbm: bool = False) -> Pipeline:
    clf = (
        GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        if use_gbm
        else RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clf),
    ])


def _evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict[str, Any]:
    preds = model.predict(X_test)
    proba_matrix = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    probs = proba_matrix[:, 1] if (proba_matrix is not None and proba_matrix.shape[1] > 1) else None
    auc = roc_auc_score(y_test, probs) if probs is not None and y_test.nunique() > 1 else float("nan")
    prec = precision_score(y_test, preds, zero_division=0)
    log.info("[%s] AUC=%.4f  Precision=%.4f  n_test=%d", name, auc, prec, len(y_test))
    log.info("[%s] %s", name, classification_report(y_test, preds, zero_division=0))
    return {"auc": auc, "precision": prec, "n_test": len(y_test)}


def _save_artifact(
    weights_dir: Path,
    family: str,
    name: str,
    model: Pipeline,
    features: list[str],
    target: str,
    metadata: dict[str, Any],
    calibrator_model=None,
) -> Path:
    out_dir = weights_dir / family
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"task_{name}.joblib"
    joblib.dump({
        "model": model,
        "features": features,
        "target": target,
        "model_kind": name,
        "market_family": family,
        "metadata": metadata,
        "calibrator_model": calibrator_model,
    }, path)
    log.info("Saved %s", path)
    return path


def _split_temporal(X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
    n = len(X)
    split = int(n * (1 - test_frac))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def _find_temporal_binary_split(
    y: pd.Series,
    *,
    min_train_rows: int = 100,
    min_test_rows: int = 20,
) -> int | None:
    n = len(y)
    if n < (min_train_rows + min_test_rows):
        return None
    for split in range(max(min_train_rows, 1), n - min_test_rows + 1):
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]
        if y_train.nunique() >= 2 and y_test.nunique() >= 2:
            return split
    return None


def _split_temporal_three_way(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.6,
    calib_frac: float = 0.2,
):
    n = len(X)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    return (
        X.iloc[:train_end],
        X.iloc[train_end:calib_end],
        X.iloc[calib_end:],
        y.iloc[:train_end],
        y.iloc[train_end:calib_end],
        y.iloc[calib_end:],
    )


def _apply_binary_calibrator(calibrator, probs: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(probs, dtype=float)
    raw = np.asarray(probs, dtype=float).reshape(-1, 1)
    if hasattr(calibrator, "predict_proba"):
        calibrated = calibrator.predict_proba(raw)
        return calibrated[:, 1] if calibrated.shape[1] > 1 else calibrated[:, 0]
    if hasattr(calibrator, "predict"):
        return np.asarray(calibrator.predict(raw), dtype=float)
    return np.asarray(probs, dtype=float)


# ---------------------------------------------------------------------------
# Individual learners
# ---------------------------------------------------------------------------

@dataclass
class EntryEdgeModel:
    """Predicts entry_quality (binary) from supervisor signal + microstructure.

    Label: entry_quality = 1 when trade result was profitable (entry_edge_realized > 0).
    Filter: learning_eligible rows only.
    """
    family: str
    logs_dir: Path
    weights_dir: Path
    name: str = "entry_edge"

    _TARGET_BY_FAMILY: dict = field(default_factory=lambda: {
        "btc": "entry_quality",
        "weather_temperature": "entry_quality",
    })

    def train(self) -> dict[str, Any] | None:
        df = _eligible_rows(_load_task_training_table(self.family, self.name, self.logs_dir))
        if df.empty:
            log.warning("[entry_edge] No eligible rows for family=%s", self.family)
            return None

        target = self._TARGET_BY_FAMILY.get(self.family, "entry_quality")
        if target not in df.columns or df[target].isna().all():
            log.warning("[entry_edge] Target %s missing or all-NaN", target)
            return None

        candidate_features = (
            _FEAT_WEATHER_FORECAST
            if self.family == "weather_temperature"
            else _FEAT_ORDER_DECISION + _FEAT_MICROSTRUCTURE + _FEAT_WALLET
        )
        feature_cols = _present(df, candidate_features)
        df, feature_cols = _encode_categoricals(df, feature_cols)

        y = df[target].fillna(0).astype(int)
        X = df[feature_cols]

        if y.nunique() < 2:
            log.warning("[entry_edge] Target has only one class — skipping")
            return None

        X_train, X_test, y_train, y_test = _split_temporal(X, y)
        model = _build_pipeline(use_gbm=True)
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test, self.name)
        _save_artifact(self.weights_dir, self.family, self.name, model, feature_cols, target, metrics)
        return metrics


@dataclass
class FillProbabilityModel:
    """Predicts whether a trade fills successfully (live_exit_filled).

    Uses ALL rows (failed fills are the signal — do not filter to eligible only).
    Label: fill_success = 1 when actual_execution_path == 'live_exit_filled'
    """
    family: str
    logs_dir: Path
    weights_dir: Path
    name: str = "fill_probability"

    def train(self) -> dict[str, Any] | None:
        df = _load_task_training_table(self.family, self.name, self.logs_dir)
        if df.empty:
            log.warning("[fill_probability] No data for family=%s", self.family)
            return None

        if "actual_execution_path" not in df.columns:
            log.warning("[fill_probability] actual_execution_path missing")
            return None

        df = df.copy()
        df["fill_success"] = (
            df["actual_execution_path"].astype(str).str.lower() == "live_exit_filled"
        ).astype(int)

        candidate_features = (
            _FEAT_WEATHER_FORECAST
            if self.family == "weather_temperature"
            else _FEAT_ORDER_DECISION + _FEAT_MICROSTRUCTURE + _FEAT_WALLET
        )
        feature_cols = _present(df, candidate_features)
        df, feature_cols = _encode_categoricals(df, feature_cols)

        y = df["fill_success"]
        X = df[feature_cols]

        if y.nunique() < 2:
            log.warning("[fill_probability] Target has only one class — skipping")
            return None

        X_train, X_cal, X_test, y_train, y_cal, y_test = _split_temporal_three_way(X, y)
        if len(X_train) < 20 or len(X_test) < 20:
            log.warning("[fill_probability] Insufficient temporal slices — skipping")
            return None
        model = _build_pipeline(use_gbm=False)
        model.fit(X_train, y_train)
        calibrator = None
        calibration_method = "none"
        if len(X_cal) >= 20 and y_cal.nunique() >= 2:
            try:
                cal_matrix = model.predict_proba(X_cal)
                cal_probs = cal_matrix[:, 1] if cal_matrix.shape[1] > 1 else cal_matrix[:, 0]
                calibrator = LogisticRegression(random_state=42)
                calibrator.fit(cal_probs.reshape(-1, 1), y_cal.astype(int))
                calibration_method = "platt"
            except Exception as exc:
                log.warning("[fill_probability] Calibration skipped: %s", exc)
                calibrator = None

        proba_matrix = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        raw_probs = proba_matrix[:, 1] if (proba_matrix is not None and proba_matrix.shape[1] > 1) else None
        if raw_probs is not None:
            calibrated_probs = _apply_binary_calibrator(calibrator, raw_probs)
            preds = (calibrated_probs >= 0.5).astype(int)
            auc = roc_auc_score(y_test, calibrated_probs) if y_test.nunique() > 1 else float("nan")
            prec = precision_score(y_test, preds, zero_division=0)
            metrics = {
                "auc": auc,
                "precision": prec,
                "n_test": len(y_test),
                "calibration_method": calibration_method,
            }
            log.info("[%s] AUC=%.4f  Precision=%.4f  n_test=%d  calibrator=%s", self.name, auc, prec, len(y_test), calibration_method)
            log.info("[%s] %s", self.name, classification_report(y_test, preds, zero_division=0))
        else:
            metrics = _evaluate(model, X_test, y_test, self.name)
            metrics["calibration_method"] = calibration_method
        _save_artifact(
            self.weights_dir,
            self.family,
            self.name,
            model,
            feature_cols,
            "fill_success",
            metrics,
            calibrator_model=calibrator,
        )
        return metrics


@dataclass
class SlippageLiquidityModel:
    """Predicts execution_quality (binary) from fill mechanics.

    Label: execution_quality = 1 when slippage was within expected bounds.
    Filter: rows where actual_execution_path == 'live_exit_filled' (actual fills only).
    """
    family: str
    logs_dir: Path
    weights_dir: Path
    name: str = "slippage_liquidity"

    def train(self) -> dict[str, Any] | None:
        df = _load_task_training_table(self.family, self.name, self.logs_dir)
        if df.empty:
            log.warning("[slippage_liquidity] No data for family=%s", self.family)
            return None

        if "actual_execution_path" in df.columns:
            df = df[df["actual_execution_path"].astype(str).str.lower() == "live_exit_filled"].copy()

        target = "execution_quality"
        if target not in df.columns or df[target].isna().all():
            # Derive from slippage_error if available
            if "slippage_error" in df.columns:
                median_slip = df["slippage_error"].median()
                df[target] = (df["slippage_error"].fillna(median_slip) <= median_slip).astype(int)
                log.info("[slippage_liquidity] Derived execution_quality from slippage_error median split")
            else:
                log.warning("[slippage_liquidity] Target %s missing — skipping", target)
                return None

        candidate_features = (
            _FEAT_WEATHER_FORECAST
            if self.family == "weather_temperature"
            else _FEAT_FILL + _FEAT_MICROSTRUCTURE + ["size_usdc", "btc_volatility_24h", "btc_volatility_regime"]
        )
        feature_cols = _present(df, candidate_features)
        df, feature_cols = _encode_categoricals(df, feature_cols)

        y = df[target].fillna(0).astype(int)
        X = df[feature_cols]

        if len(df) < 30 or y.nunique() < 2:
            log.warning("[slippage_liquidity] Insufficient data (n=%d) or one class — skipping", len(df))
            return None

        X_train, X_test, y_train, y_test = _split_temporal(X, y)
        model = _build_pipeline(use_gbm=False)
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test, self.name)
        _save_artifact(self.weights_dir, self.family, self.name, model, feature_cols, target, metrics)
        return metrics


@dataclass
class ExitQualityModel:
    """Predicts exit_quality (binary) from close context.

    BTC only: label is exit_quality = 1 when exit_regret <= 0.01 (captured >99% of MFE).
    Weather: uses contract resolution outcome as surrogate.
    Filter: learning_eligible rows only.
    """
    family: str
    logs_dir: Path
    weights_dir: Path
    name: str = "exit_quality"

    _TARGET_BY_FAMILY: dict = field(default_factory=lambda: {
        "btc": "exit_quality",
        "weather_temperature": "weather_contract_resolved_yes",
    })

    def train(self) -> dict[str, Any] | None:
        df = _eligible_rows(_load_task_training_table(self.family, self.name, self.logs_dir))
        if df.empty:
            log.warning("[exit_quality] No eligible rows for family=%s", self.family)
            return None

        target = self._TARGET_BY_FAMILY.get(self.family, "exit_quality")
        if target not in df.columns or df[target].isna().all():
            log.warning("[exit_quality] Target %s missing or all-NaN for family=%s", target, self.family)
            return None

        # Drop NaN target rows
        df = df[df[target].notna()].copy()

        candidate_features = (
            _FEAT_WEATHER_FORECAST
            if self.family == "weather_temperature"
            else _FEAT_CLOSE + _FEAT_ORDER_DECISION + _FEAT_WALLET + _FEAT_BTC_REGIME
        )
        feature_cols = _present(df, candidate_features)
        df, feature_cols = _encode_categoricals(df, feature_cols)

        y = df[target].fillna(0).astype(int)
        X = df[feature_cols]

        if len(df) < 20 or y.nunique() < 2:
            log.warning("[exit_quality] Insufficient data (n=%d) or one class — skipping", len(df))
            return None

        split = _find_temporal_binary_split(y, min_train_rows=100, min_test_rows=20)
        if split is None:
            log.warning("[exit_quality] Could not find temporal split with both classes in train and test — skipping")
            return None
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model = _build_pipeline(use_gbm=True)
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test, self.name)
        _save_artifact(self.weights_dir, self.family, self.name, model, feature_cols, target, metrics)
        return metrics


@dataclass
class RegimeCalibrationModel:
    """Maps live features → technical_regime_bucket (multiclass classifier).

    Reads historical_dataset.csv when BTC regime features are absent from canonical
    (0% coverage is expected until historical_dataset timestamps align better).
    Falls back gracefully to canonical if historical is missing.
    """
    family: str
    logs_dir: Path
    weights_dir: Path
    name: str = "regime_calibration"

    _TARGET = "technical_regime_bucket"

    def _load_data(self) -> pd.DataFrame:
        # Try historical_dataset first (richest regime coverage)
        hist_path = self.logs_dir / self.family / "historical_dataset.csv"
        if not hist_path.exists():
            hist_path = self.logs_dir / "historical_dataset.csv"

        if hist_path.exists():
            df = pd.read_csv(hist_path, low_memory=False)
            if self._TARGET in df.columns and df[self._TARGET].notna().any():
                log.info("[regime_calibration] Using historical_dataset (%d rows)", len(df))
                return df

        # Fall back to canonical
        log.info("[regime_calibration] Falling back to task training table")
        return _load_task_training_table(self.family, self.name, self.logs_dir)

    def train(self) -> dict[str, Any] | None:
        df = self._load_data()
        if df.empty:
            log.warning("[regime_calibration] No data for family=%s", self.family)
            return None

        if self._TARGET not in df.columns or df[self._TARGET].isna().all():
            log.warning("[regime_calibration] Target %s missing — skipping", self._TARGET)
            return None

        df = df[df[self._TARGET].notna()].copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.sort_values("timestamp")

        candidate_features = (
            _FEAT_WEATHER_FORECAST
            if self.family == "weather_temperature"
            else _FEAT_BTC_REGIME + _FEAT_MICROSTRUCTURE + _FEAT_WALLET
        )
        feature_cols = _present(df, candidate_features)
        df, feature_cols = _encode_categoricals(df, feature_cols)

        if not feature_cols:
            log.warning("[regime_calibration] No features present — skipping")
            return None

        y_raw = df[self._TARGET].astype(str).str.lower()
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw), index=df.index)
        X = df[feature_cols]

        if len(df) < 30 or y.nunique() < 2:
            log.warning("[regime_calibration] Insufficient data (n=%d, classes=%d) — skipping", len(df), y.nunique())
            return None

        X_train, X_test, y_train, y_test = _split_temporal(X, y)
        model = _build_pipeline(use_gbm=False)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, preds)
        log.info("[regime_calibration] Accuracy=%.4f  n_test=%d  classes=%s", acc, len(y_test), list(le.classes_))
        metrics = {"accuracy": acc, "n_test": len(y_test), "classes": list(le.classes_)}

        _save_artifact(
            self.weights_dir, self.family, self.name, model, feature_cols,
            self._TARGET, {**metrics, "label_encoder_classes": list(le.classes_)},
        )
        return metrics


# ---------------------------------------------------------------------------
# Suite entry point
# ---------------------------------------------------------------------------

_ALL_MODEL_NAMES = {"entry_edge", "fill_probability", "slippage_liquidity", "exit_quality", "regime_calibration"}


class TaskModelSuite:
    """Train one or all task-specific models for a given market family.

    Usage:
        suite = TaskModelSuite("btc")
        suite.train_all()
        suite.train("entry_edge")
    """

    def __init__(
        self,
        family: str,
        logs_dir: str = "logs",
        weights_dir: str = "weights",
    ) -> None:
        family = family.lower().strip()
        if family not in _SUPPORTED_FAMILIES:
            raise ValueError(f"Unsupported family '{family}'. Must be one of {_SUPPORTED_FAMILIES}")
        self.family = family
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)

    def _make_learner(self, name: str):
        kw = dict(family=self.family, logs_dir=self.logs_dir, weights_dir=self.weights_dir)
        if name == "entry_edge":
            return EntryEdgeModel(**kw)
        if name == "fill_probability":
            return FillProbabilityModel(**kw)
        if name == "slippage_liquidity":
            return SlippageLiquidityModel(**kw)
        if name == "exit_quality":
            return ExitQualityModel(**kw)
        if name == "regime_calibration":
            return RegimeCalibrationModel(**kw)
        raise ValueError(f"Unknown model name '{name}'. Choose from {_ALL_MODEL_NAMES}")

    def train(self, name: str) -> dict[str, Any] | None:
        """Train a single model by name."""
        learner = self._make_learner(name)
        log.info("=== Training %s [family=%s] ===", name, self.family)
        return learner.train()

    def train_all(self) -> dict[str, dict[str, Any] | None]:
        """Train all 5 task models. Returns results dict keyed by model name."""
        results = {}
        for name in ["entry_edge", "fill_probability", "slippage_liquidity", "exit_quality", "regime_calibration"]:
            results[name] = self.train(name)
        log.info("=== Suite complete for family=%s ===", self.family)
        log.info("Results: %s", {k: v for k, v in results.items() if v is not None})
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train task-specific model suite")
    parser.add_argument("--family", default="btc", choices=sorted(_SUPPORTED_FAMILIES))
    parser.add_argument("--model", default="all", help="Model name or 'all'")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--weights-dir", default="weights")
    args = parser.parse_args()

    suite = TaskModelSuite(args.family, logs_dir=args.logs_dir, weights_dir=args.weights_dir)
    if args.model == "all":
        suite.train_all()
    else:
        result = suite.train(args.model)
        if result:
            print(result)
