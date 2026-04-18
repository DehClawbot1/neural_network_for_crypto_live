from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
from model_feature_safety import drop_all_nan_features
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)

_MIN_TRAIN_ROWS = 50
_MIN_TEST_ROWS = 20


class WalkForwardEvaluator:
    """
    Walk-forward evaluation with PURGING and EMBARGO to eliminate time leakage
    from overlapping feature windows and autocorrelated residuals.

    Concepts (López de Prado, Advances in Financial ML, ch. 7):
    - PURGE: drop training samples whose *label window* overlaps any test
      sample's *feature window*. Without this, the model trains on data that
      encodes post-test-sample information.
    - EMBARGO: further drop training samples within `embargo_bars` of each
      test fold boundary, to suppress short-term autocorrelation leakage at
      the seam.

    Research/paper-trading only.
    """

    FEATURE_COLUMNS = DEFAULT_TABULAR_FEATURE_COLUMNS

    def __init__(
        self,
        logs_dir="logs",
        *,
        embargo_bars: int = 24,
        label_window_bars: int = 1,
        n_splits: int = 1,
    ):
        self.logs_dir = Path(logs_dir)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.output_file = self.logs_dir / "walk_forward_eval.csv"
        if embargo_bars < 0:
            raise ValueError("embargo_bars must be >= 0")
        if label_window_bars < 1:
            raise ValueError("label_window_bars must be >= 1")
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.embargo_bars = int(embargo_bars)
        self.label_window_bars = int(label_window_bars)
        self.n_splits = int(n_splits)

    @staticmethod
    def purge_train_indices(
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        *,
        embargo_bars: int,
        label_window_bars: int,
    ) -> np.ndarray:
        """
        Remove training rows whose label window [i, i+label_window_bars) overlaps
        any test row, AND rows within `embargo_bars` of the test-fold boundary.

        Assumes indices are positional (0..N-1) after sorting by time.
        """
        if test_idx.size == 0:
            return train_idx
        test_min = int(test_idx.min())
        test_max = int(test_idx.max())
        # Purge: any train row i whose [i, i+label_window_bars-1] intersects [test_min, test_max]
        purge_lo = test_min - label_window_bars + 1
        purge_hi = test_max
        # Embargo: forbid train rows in a band around the test fold
        embargo_lo = test_min - embargo_bars
        embargo_hi = test_max + embargo_bars
        lo = min(purge_lo, embargo_lo)
        hi = max(purge_hi, embargo_hi)
        mask = (train_idx < lo) | (train_idx > hi)
        return train_idx[mask]

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _numeric_feature_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        normalized = {}
        truthy = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "on": 1.0, "off": 0.0}
        for column in columns:
            series = df[column]
            if pd.api.types.is_bool_dtype(series):
                normalized[column] = series.astype(float)
                continue
            if pd.api.types.is_numeric_dtype(series):
                normalized[column] = pd.to_numeric(series, errors="coerce")
                continue
            text = series.astype(str).str.strip().str.lower()
            mapped = text.map(truthy)
            numeric = pd.to_numeric(series, errors="coerce")
            normalized[column] = numeric.where(numeric.notna(), mapped)
        return pd.DataFrame(normalized, index=df.index)

    def evaluate(self, train_ratio=0.7):
        df = self._safe_read(self.dataset_file)
        if df.empty or "target_up" not in df.columns:
            return pd.DataFrame()

        candidates = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        if not candidates:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        df = df.reset_index(drop=True)
        split_idx = int(len(df) * train_ratio)
        all_idx = np.arange(len(df))
        test_idx = all_idx[split_idx:]
        train_idx_raw = all_idx[:split_idx]
        train_idx = self.purge_train_indices(
            train_idx_raw,
            test_idx,
            embargo_bars=self.embargo_bars,
            label_window_bars=self.label_window_bars,
        )
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        if train_df.empty or test_df.empty:
            return pd.DataFrame()
        usable, _ = drop_all_nan_features(train_df, candidates, context="walk_forward_evaluator")
        if not usable:
            return pd.DataFrame()

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]
        )
        train_x = self._numeric_feature_frame(train_df, usable)
        test_x = self._numeric_feature_frame(test_df, usable)
        model.fit(train_x, train_df["target_up"].astype(int))
        preds = model.predict(test_x)
        acc = accuracy_score(test_df["target_up"].astype(int), preds)

        result = pd.DataFrame([
            {
                "train_rows": len(train_df),
                "train_rows_raw": int(train_idx_raw.size),
                "purged_rows": int(train_idx_raw.size - train_idx.size),
                "embargo_bars": self.embargo_bars,
                "label_window_bars": self.label_window_bars,
                "test_rows": len(test_df),
                "accuracy": acc,
                "positive_rate": float(np.mean(preds)),
            }
        ])
        result.to_csv(self.output_file, index=False)
        return result


# ---------------------------------------------------------------------------
# Task-model walk-forward evaluation (offline learning loop)
# ---------------------------------------------------------------------------

# Feature sets matching task_model_suite.py
_FEAT_MICROSTRUCTURE = [
    "spread_bps", "best_bid", "best_ask",
    "orderbook_depth_bid", "orderbook_depth_ask",
    "market_implied_prob", "liquidity_score",
]
_FEAT_BTC_REGIME = [
    "btc_price_usd", "btc_rsi_14", "btc_volatility_24h",
    "btc_trend_bias", "btc_volatility_regime", "btc_market_regime_label",
    "btc_funding_rate", "btc_volume_24h",
]
_FEAT_WALLET = [
    "wallet_balance_usdc", "open_positions_count", "portfolio_heat",
    "recent_wallet_activity_5", "recent_yes_ratio_5",
]
_FEAT_ORDER_DECISION = [
    "signal_label", "confidence_score", "predicted_edge", "size_usdc",
    "tp_threshold_pct", "sl_threshold_pct", "brain_id",
    "technical_regime_bucket", "entry_gate_passed", "supervisor_gate_passed",
]
_FEAT_FILL = [
    "entry_price_filled", "slippage_bps", "fill_latency_ms",
    "partial_fill_ratio", "orderbook_depth_at_fill", "spread_at_fill_bps",
]
_FEAT_CLOSE = [
    "exit_price", "price_return", "hold_time_seconds", "close_reason",
]

_TASK_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "entry_edge": {
        "target_btc": "entry_quality",
        "target_weather": "entry_quality",
        "features": _FEAT_ORDER_DECISION + _FEAT_MICROSTRUCTURE + _FEAT_WALLET,
        "eligible_only": True,
    },
    "fill_probability": {
        "target_btc": "fill_success",
        "target_weather": "fill_success",
        "features": _FEAT_ORDER_DECISION + _FEAT_MICROSTRUCTURE + _FEAT_WALLET,
        "eligible_only": False,
        "derive_target": "fill_success",
    },
    "slippage_liquidity": {
        "target_btc": "execution_quality",
        "target_weather": "execution_quality",
        "features": _FEAT_FILL + _FEAT_MICROSTRUCTURE + ["size_usdc", "btc_volatility_24h", "btc_volatility_regime"],
        "eligible_only": False,
    },
    "exit_quality": {
        "target_btc": "exit_quality",
        "target_weather": "weather_contract_resolved_yes",
        "features": _FEAT_CLOSE + _FEAT_ORDER_DECISION + _FEAT_WALLET + _FEAT_BTC_REGIME,
        "eligible_only": True,
    },
    "regime_calibration": {
        "target_btc": "technical_regime_bucket",
        "target_weather": "technical_regime_bucket",
        "features": _FEAT_BTC_REGIME + _FEAT_MICROSTRUCTURE + _FEAT_WALLET,
        "eligible_only": False,
        "multiclass": True,
    },
}


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    auc: float
    precision: float
    accuracy: float
    positive_rate: float


@dataclass
class TaskWFReport:
    model_name: str
    market_family: str
    target: str
    n_folds_requested: int
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_auc(self) -> float:
        vals = [f.auc for f in self.folds if not np.isnan(f.auc)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_precision(self) -> float:
        vals = [f.precision for f in self.folds if not np.isnan(f.precision)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def std_auc(self) -> float:
        vals = [f.auc for f in self.folds if not np.isnan(f.auc)]
        return float(np.std(vals)) if len(vals) > 1 else float("nan")

    def summary(self) -> str:
        lines = [
            f"TaskWF [{self.model_name} / {self.market_family}]  target={self.target}",
            f"  folds={len(self.folds)}/{self.n_folds_requested}  "
            f"AUC={self.mean_auc:.4f}±{self.std_auc:.4f}  prec={self.mean_precision:.4f}",
        ]
        for fr in self.folds:
            lines.append(
                f"    fold {fr.fold:02d}  train={fr.n_train:4d}  test={fr.n_test:3d}  "
                f"AUC={fr.auc:.4f}  prec={fr.precision:.4f}  acc={fr.accuracy:.4f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "market_family": self.market_family,
            "target": self.target,
            "n_folds": len(self.folds),
            "mean_auc": self.mean_auc,
            "std_auc": self.std_auc,
            "mean_precision": self.mean_precision,
        }


@dataclass
class PromotionVerdict:
    model_name: str
    market_family: str
    promote: bool
    reason: str
    candidate_auc: float
    incumbent_auc: float

    def summary(self) -> str:
        return (
            f"[{self.model_name}/{self.market_family}] promote={self.promote}  "
            f"cand={self.candidate_auc:.4f}  incumb={self.incumbent_auc:.4f}  "
            f"reason={self.reason}"
        )


class TaskWalkForwardEvaluator:
    """Walk-forward evaluator for task_model_suite models.

    Reads from logs/{family}/task_training/{task}.csv (or historical_dataset for
    regime_calibration).  Runs TimeSeriesSplit CV.  Supports comparing a
    candidate artifact against an incumbent artifact for promotion gating.

    Usage:
        ev = TaskWalkForwardEvaluator("btc")
        report = ev.evaluate("entry_edge")
        verdict = ev.compare("weights/btc/task_entry_edge.joblib",
                             "weights/btc/task_entry_edge_incumbent.joblib",
                             "entry_edge")
    """

    def __init__(self, family: str, logs_dir: str = "logs", n_splits: int = 5) -> None:
        self.family = family.lower().strip()
        self.logs_dir = Path(logs_dir)
        self.n_splits = n_splits

    # -- helpers -------------------------------------------------------------

    def _load_dataset(self, model_name: str) -> pd.DataFrame:
        if model_name == "regime_calibration":
            for p in [
                self.logs_dir / self.family / "historical_dataset.csv",
                self.logs_dir / "historical_dataset.csv",
            ]:
                if p.exists():
                    df = pd.read_csv(p, low_memory=False)
                    if "technical_regime_bucket" in df.columns and df["technical_regime_bucket"].notna().any():
                        return self._sort_time(df)

        for p in [
            self.logs_dir / self.family / "task_training" / f"{model_name}.csv",
            self.logs_dir / self.family / "canonical_dataset.csv",
            self.logs_dir / "canonical_dataset.csv",
        ]:
            if p.exists():
                return self._sort_time(pd.read_csv(p, low_memory=False))
        return pd.DataFrame()

    @staticmethod
    def _sort_time(df: pd.DataFrame) -> pd.DataFrame:
        for col in ("timestamp", "signal_timestamp", "fill_timestamp"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
                return df.sort_values(col).reset_index(drop=True)
        return df

    @staticmethod
    def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
        return [c for c in cols if c in df.columns and df[c].notna().any()]

    @staticmethod
    def _encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c in out.columns and out[c].dtype == object:
                le = LabelEncoder()
                out[c] = le.fit_transform(out[c].astype(str).fillna("__missing__"))
        return out

    def _prepare(self, df: pd.DataFrame, model_name: str):
        spec = _TASK_MODEL_SPECS.get(model_name)
        if spec is None:
            return None
        target = spec["target_btc"] if self.family == "btc" else spec["target_weather"]

        if spec.get("derive_target") == "fill_success" and "fill_success" not in df.columns:
            if "actual_execution_path" not in df.columns:
                return None
            df = df.copy()
            df["fill_success"] = (
                df["actual_execution_path"].astype(str).str.lower() == "live_exit_filled"
            ).astype(int)

        if target == "execution_quality" and "execution_quality" not in df.columns:
            if "slippage_error" in df.columns:
                med = df["slippage_error"].median()
                df = df.copy()
                df["execution_quality"] = (df["slippage_error"].fillna(med) <= med).astype(int)
            else:
                return None

        if target not in df.columns or df[target].isna().all():
            return None

        if spec.get("eligible_only") and "learning_eligible" in df.columns:
            mask = df["learning_eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
            df = df[mask].copy()

        df = df[df[target].notna()].copy()
        feature_cols = self._present(df, spec["features"])
        if not feature_cols:
            return None

        df = self._encode(df, feature_cols)
        X = df[feature_cols]
        multiclass = spec.get("multiclass", False)
        if multiclass:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(df[target].astype(str).str.lower()), index=df.index)
        else:
            y = df[target].fillna(0).astype(int)
        return X, y, target, multiclass

    @staticmethod
    def _eval_fold(model, X_test, y_test, multiclass: bool):
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        if multiclass:
            prec = precision_score(y_test, preds, average="macro", zero_division=0)
            return float("nan"), prec, acc
        prec = precision_score(y_test, preds, zero_division=0)
        auc = float("nan")
        if y_test.nunique() > 1:
            try:
                pm = model.predict_proba(X_test)
                prob = pm[:, 1] if pm.shape[1] > 1 else pm[:, 0]
                auc = roc_auc_score(y_test, prob)
            except Exception:
                pass
        return auc, prec, acc

    # -- public --------------------------------------------------------------

    def evaluate(self, model_name: str) -> TaskWFReport:
        """Train from scratch with TimeSeriesSplit and return fold metrics."""
        spec = _TASK_MODEL_SPECS.get(model_name, {})
        target = spec.get("target_btc" if self.family == "btc" else "target_weather", "?")
        report = TaskWFReport(model_name, self.family, target, self.n_splits)

        df = self._load_dataset(model_name)
        if df.empty:
            return report

        prepared = self._prepare(df, model_name)
        if prepared is None:
            return report

        X, y, target, multiclass = prepared
        report.target = target

        if len(X) < (_MIN_TRAIN_ROWS + _MIN_TEST_ROWS):
            log.warning("[TaskWFE] %s/%s: too few rows (%d)", self.family, model_name, len(X))
            return report

        use_gbm = model_name in ("entry_edge", "exit_quality")
        n_splits = min(self.n_splits, max(2, len(X) // (_MIN_TRAIN_ROWS + _MIN_TEST_ROWS)))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            if len(X_tr) < _MIN_TRAIN_ROWS or len(X_te) < _MIN_TEST_ROWS:
                continue
            if y_tr.nunique() < 2:
                continue

            clf = (
                GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
                if use_gbm
                else RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1)
            )
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", clf),
            ])
            try:
                pipe.fit(X_tr, y_tr)
                auc, prec, acc = self._eval_fold(pipe, X_te, y_te, multiclass)
            except Exception as exc:
                log.warning("[TaskWFE] fold %d failed: %s", fold_idx, exc)
                continue

            pos_rate = float(y_te.mean()) if not multiclass else float("nan")
            report.folds.append(FoldResult(fold_idx, len(X_tr), len(X_te), auc, prec, acc, pos_rate))

        log.info("[TaskWFE] %s", report.summary().splitlines()[0])
        return report

    def score_artifact(self, artifact_path: str | Path, model_name: str) -> TaskWFReport:
        """Score an already-trained artifact on walk-forward test windows."""
        import joblib

        artifact_path = Path(artifact_path)
        spec = _TASK_MODEL_SPECS.get(model_name, {})
        target = spec.get("target_btc" if self.family == "btc" else "target_weather", "?")
        report = TaskWFReport(model_name, self.family, target, self.n_splits)

        if not artifact_path.exists():
            return report
        bundle = joblib.load(artifact_path)
        model = bundle["model"]
        features: list[str] = bundle["features"]

        df = self._load_dataset(model_name)
        if df.empty:
            return report
        prepared = self._prepare(df, model_name)
        if prepared is None:
            return report

        X_all, y_all, target, multiclass = prepared
        report.target = target
        present = [f for f in features if f in X_all.columns]
        if not present:
            return report
        X_all = X_all[present]

        n_splits = min(self.n_splits, max(2, len(X_all) // (_MIN_TRAIN_ROWS + _MIN_TEST_ROWS)))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold_idx, (_, test_idx) in enumerate(tscv.split(X_all)):
            X_te = X_all.iloc[test_idx]
            y_te = y_all.iloc[test_idx]
            if len(X_te) < _MIN_TEST_ROWS:
                continue
            try:
                auc, prec, acc = self._eval_fold(model, X_te, y_te, multiclass)
            except Exception as exc:
                log.warning("[TaskWFE] score_artifact fold %d: %s", fold_idx, exc)
                continue
            pos_rate = float(y_te.mean()) if not multiclass else float("nan")
            report.folds.append(FoldResult(fold_idx, 0, len(X_te), auc, prec, acc, pos_rate))
        return report

    def compare(
        self,
        candidate_path: str | Path,
        incumbent_path: str | Path | None,
        model_name: str,
    ) -> PromotionVerdict:
        """Return a PromotionVerdict comparing candidate vs incumbent artifacts."""
        cand_report = self.score_artifact(candidate_path, model_name)
        c_auc = cand_report.mean_auc
        c_prec = cand_report.mean_precision

        if incumbent_path is None or not Path(str(incumbent_path)).exists():
            return PromotionVerdict(
                model_name, self.family, promote=True,
                reason="no_incumbent",
                candidate_auc=c_auc, incumbent_auc=float("nan"),
            )

        inc_report = self.score_artifact(incumbent_path, model_name)
        i_auc = inc_report.mean_auc
        i_prec = inc_report.mean_precision

        if not np.isnan(c_auc) and not np.isnan(i_auc):
            promote = c_auc >= i_auc
            reason = f"auc: {c_auc:.4f} {'≥' if promote else '<'} {i_auc:.4f}"
        else:
            promote = c_prec >= i_prec
            reason = f"precision fallback: {c_prec:.4f} {'≥' if promote else '<'} {i_prec:.4f}"

        return PromotionVerdict(model_name, self.family, promote, reason, c_auc, i_auc)
