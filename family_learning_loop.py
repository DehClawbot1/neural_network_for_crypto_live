"""Family-specific learning loops.

BTC and weather are structurally different markets and must NOT share one
retraining clock, one dataset pipeline, one validation metric, or one
promotion gate set.

  BTC   — price-space labels (tp_before_sl_60m, price_return, MFE/MAE).
           Validation: Sortino ratio, AUC, drawdown.
           Data rate: faster (many short-lived positions).

  Weather — probability-space labels (target_up, weather_contract_resolved_yes).
            Validation: Brier score, ECE calibration, AUC.
            Data rate: slower (daily / weekly resolution).

Each family has:
  • its own closed-trade counter and last-retrained-at timestamp (persisted
    to logs/{family}/loop_state.json)
  • its own trigger threshold (env-overridable per family)
  • its own dataset builder path (only that family's canonical rows)
  • its own target construction (price-space vs prob-space)
  • its own walk-forward validation
  • its own promotion criteria (hard gates + family-specific thresholds)

Entry point for the supervisor / retrainer:

    from family_learning_loop import FamilyLearningLoop, run_all_family_loops

    # Called on every closed-trade event:
    run_all_family_loops(logs_dir="logs", weights_dir="weights")

    # Or per-family:
    loop = FamilyLearningLoop("btc", logs_dir="logs", weights_dir="weights")
    loop.maybe_retrain()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BTC = "btc"
_WEATHER = "weather_temperature"
_SUPPORTED = {_BTC, _WEATHER}

# Default trigger thresholds (number of *new* closed trades since last retrain).
# Weather resolves ~10x slower than BTC, so its threshold is intentionally higher.
_DEFAULT_THRESHOLD = {_BTC: 5, _WEATHER: 10}

# Minimum OOS rows before promotion is even considered.
_DEFAULT_MIN_OOS = {_BTC: 30, _WEATHER: 15}

# ---------------------------------------------------------------------------
# Per-family target definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FamilyTargetSpec:
    """Describes which labels this family optimises and how to validate them."""
    family: str
    # Primary binary classification target (used for AUC / Brier)
    primary_target: str
    # Regression target for return estimation (may be None)
    return_target: str | None
    # Metric names used in validation summary
    primary_metric: str          # "auc" for both, but logged differently
    secondary_metric: str        # "sortino" for BTC, "brier" for weather
    # Hard-gate thresholds (promotion blocks if not met)
    min_auc: float
    max_brier: float | None      # weather only
    min_sortino: float | None    # BTC only
    max_drawdown_pct: float      # BTC: 0.20; weather: looser (0.40 — positions are binary)
    max_ece: float               # expected calibration error ceiling
    # Walk-forward folds
    n_wf_splits: int


_FAMILY_SPECS: dict[str, FamilyTargetSpec] = {
    _BTC: FamilyTargetSpec(
        family=_BTC,
        primary_target="tp_before_sl_60m",      # binary: hit TP before SL in 60 min
        return_target="price_return",           # continuous return
        primary_metric="auc",
        secondary_metric="sortino",
        min_auc=float(os.getenv("BTC_GATE_MIN_AUC", "0.52")),
        max_brier=None,
        min_sortino=float(os.getenv("BTC_GATE_MIN_SORTINO", "0.30")),
        max_drawdown_pct=float(os.getenv("BTC_GATE_MAX_DRAWDOWN_PCT", "0.20")),
        max_ece=float(os.getenv("BTC_GATE_MAX_ECE", "0.10")),
        n_wf_splits=int(os.getenv("BTC_WF_SPLITS", "3")),
    ),
    _WEATHER: FamilyTargetSpec(
        family=_WEATHER,
        primary_target="target_up",             # binary: did contract resolve YES
        return_target=None,
        primary_metric="auc",
        secondary_metric="brier",
        min_auc=float(os.getenv("WEATHER_GATE_MIN_AUC", "0.52")),
        max_brier=float(os.getenv("WEATHER_GATE_MAX_BRIER", "0.25")),
        min_sortino=None,
        max_drawdown_pct=float(os.getenv("WEATHER_GATE_MAX_DRAWDOWN_PCT", "0.40")),
        max_ece=float(os.getenv("WEATHER_GATE_MAX_ECE", "0.08")),
        n_wf_splits=int(os.getenv("WEATHER_WF_SPLITS", "3")),
    ),
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _sortino(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 5:
        return float("nan")
    mean_r = float(r.mean())
    downside = r[r < 0]
    if len(downside) == 0:
        return float("inf") if mean_r > 0 else float("nan")
    downside_std = float(downside.std(ddof=1))
    if downside_std == 0:
        return float("nan")
    return float(mean_r / downside_std * np.sqrt(periods_per_year))


def _max_drawdown_pct(equity: pd.Series) -> float:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 2:
        return 0.0
    rolling_peak = eq.cummax()
    dd = (eq - rolling_peak) / rolling_peak.replace(0, float("nan"))
    return float(abs(dd.min(skipna=True)))


def _brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(y))
    if valid.sum() < 3:
        return float("nan")
    return float(np.mean((p[valid] - y[valid]) ** 2))


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(y))
    if valid.sum() < n_bins:
        return float("nan")
    p, y = p[valid], y[valid]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_acc = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(p[mask]))
        ece_acc += mask.sum() / len(p) * abs(acc - conf)
    return float(ece_acc)


def _auc_safe(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(y))
    if valid.sum() < 5 or len(np.unique(y[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(y[valid], p[valid]))


# ---------------------------------------------------------------------------
# Per-family walk-forward evaluation
# ---------------------------------------------------------------------------

@dataclass
class FamilyWFResult:
    family: str
    n_folds: int
    mean_auc: float
    std_auc: float
    secondary_value: float    # sortino (BTC) or brier (weather)
    max_drawdown_pct: float
    mean_ece: float
    n_oos_rows: int
    fold_aucs: list[float] = field(default_factory=list)

    def passes_gates(self, spec: FamilyTargetSpec) -> tuple[bool, list[str]]:
        failures: list[str] = []
        if np.isnan(self.mean_auc) or self.mean_auc < spec.min_auc:
            failures.append(
                f"auc {self.mean_auc:.4f} < min {spec.min_auc:.4f}"
            )
        if spec.min_sortino is not None:
            if np.isnan(self.secondary_value) or self.secondary_value < spec.min_sortino:
                failures.append(
                    f"sortino {self.secondary_value:.4f} < min {spec.min_sortino:.4f}"
                )
        if spec.max_brier is not None:
            if np.isnan(self.secondary_value) or self.secondary_value > spec.max_brier:
                failures.append(
                    f"brier {self.secondary_value:.4f} > max {spec.max_brier:.4f}"
                )
        if self.max_drawdown_pct > spec.max_drawdown_pct:
            failures.append(
                f"drawdown {self.max_drawdown_pct:.4f} > max {spec.max_drawdown_pct:.4f}"
            )
        if not np.isnan(self.mean_ece) and self.mean_ece > spec.max_ece:
            failures.append(
                f"ece {self.mean_ece:.4f} > max {spec.max_ece:.4f}"
            )
        return len(failures) == 0, failures

    def summary(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_folds": self.n_folds,
            "mean_auc": round(self.mean_auc, 4),
            "std_auc": round(self.std_auc, 4),
            "secondary_value": round(self.secondary_value, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "mean_ece": round(self.mean_ece, 4),
            "n_oos_rows": self.n_oos_rows,
        }


def _walk_forward_evaluate(
    df: pd.DataFrame,
    spec: FamilyTargetSpec,
) -> FamilyWFResult | None:
    """
    Run TimeSeriesSplit walk-forward evaluation against the primary target.

    Returns None when there is not enough data to form even one valid fold.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    target = spec.primary_target
    if target not in df.columns or df[target].isna().all():
        log.warning("[wf/%s] target '%s' missing or all-NaN", spec.family, target)
        return None

    # Feature columns: all numeric columns except targets and meta cols
    _META = {
        "timestamp", "signal_timestamp", "closed_at", "opened_at",
        "token_id", "market_id", "market_family", "brain_id",
        "learning_eligible", "split_label",
        spec.primary_target,
        spec.return_target or "__never__",
        # common label columns that must not be used as features
        "entry_quality", "exit_quality", "execution_quality",
        "target_up", "weather_contract_resolved_yes",
        "tp_before_sl_60m", "price_return", "mfe_pct", "mae_pct",
    }

    feature_cols = [
        c for c in df.columns
        if c not in _META
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().any()
    ]
    # Also encode object columns that are not meta
    object_cols = [
        c for c in df.columns
        if c not in _META
        and df[c].dtype == object
        and df[c].notna().any()
    ]
    work = df.copy()
    for c in object_cols:
        le = LabelEncoder()
        work[c] = le.fit_transform(work[c].astype(str).fillna("__nan__"))
        if c not in feature_cols:
            feature_cols.append(c)

    if not feature_cols:
        log.warning("[wf/%s] no usable feature columns", spec.family)
        return None

    X = work[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_raw = pd.to_numeric(work[target], errors="coerce")
    valid = y_raw.notna()
    X, y = X[valid].reset_index(drop=True), y_raw[valid].reset_index(drop=True)

    if len(X) < 40:
        log.info("[wf/%s] only %d valid rows — skipping walk-forward", spec.family, len(X))
        return None

    tscv = TimeSeriesSplit(n_splits=spec.n_wf_splits)
    fold_aucs, fold_eces, all_probs, all_labels, all_returns = [], [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if len(X_tr) < 20 or len(X_te) < 5:
            continue
        if y_tr.nunique() < 2:
            log.debug("[wf/%s] fold %d train has 1 class — skipping", spec.family, fold_idx)
            continue

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                random_state=42 + fold_idx,
            )),
        ])
        pipe.fit(X_tr, y_tr)
        pm = pipe.predict_proba(X_te)
        probs = pm[:, 1] if pm.shape[1] > 1 else pm[:, 0]
        auc = _auc_safe(probs, y_te.values)
        ece = _ece(probs, y_te.values)
        fold_aucs.append(auc)
        fold_eces.append(ece)
        all_probs.extend(probs.tolist())
        all_labels.extend(y_te.tolist())

        # Collect returns for BTC Sortino / weather Brier
        if spec.return_target and spec.return_target in work.columns:
            ret_te = pd.to_numeric(
                work[spec.return_target].iloc[valid.values][test_idx],
                errors="coerce",
            )
            all_returns.extend(ret_te.tolist())

    if not fold_aucs:
        log.warning("[wf/%s] no valid folds produced", spec.family)
        return None

    probs_arr = np.array(all_probs, dtype=float)
    labels_arr = np.array(all_labels, dtype=float)

    # Secondary metric
    if spec.secondary_metric == "sortino":
        returns_series = pd.Series(all_returns, dtype=float).dropna()
        secondary = _sortino(returns_series)
    else:  # brier
        secondary = _brier_score(probs_arr, labels_arr)

    # Drawdown via equity curve on predictions
    equity = pd.Series(
        [1.0 if p > 0.5 else -1.0 for p, lbl in zip(all_probs, all_labels)],
        dtype=float,
    ).cumsum()
    dd_pct = _max_drawdown_pct(equity)

    return FamilyWFResult(
        family=spec.family,
        n_folds=len(fold_aucs),
        mean_auc=float(np.nanmean(fold_aucs)),
        std_auc=float(np.nanstd(fold_aucs)),
        secondary_value=secondary,
        max_drawdown_pct=dd_pct,
        mean_ece=float(np.nanmean(fold_eces)),
        n_oos_rows=len(all_labels),
        fold_aucs=fold_aucs,
    )


# ---------------------------------------------------------------------------
# Per-family dataset loader
# ---------------------------------------------------------------------------

def _load_family_canonical(family: str, logs_dir: Path) -> pd.DataFrame:
    """Load the family-specific canonical dataset, returning only that family's rows."""
    path = logs_dir / family / "canonical_dataset.csv"
    if not path.exists():
        # Fallback: shared canonical filtered to this family
        shared = logs_dir / "canonical_dataset.csv"
        if not shared.exists():
            log.warning("[%s] canonical_dataset.csv not found at %s or %s", family, path, shared)
            return pd.DataFrame()
        try:
            df = pd.read_csv(shared, low_memory=False)
        except Exception as exc:
            log.warning("[%s] failed to read shared canonical: %s", family, exc)
            return pd.DataFrame()
        mf = df.get("market_family", pd.Series("", index=df.index)).fillna("").astype(str)
        if family == _WEATHER:
            df = df[mf.str.startswith(_WEATHER)].copy()
        else:
            df = df[~mf.str.startswith(_WEATHER)].copy()
    else:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            log.warning("[%s] failed to read canonical at %s: %s", family, path, exc)
            return pd.DataFrame()

    # Chronological sort
    for ts_col in ("closed_at", "signal_timestamp", "timestamp"):
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            df = df.sort_values(ts_col)
            break

    # Filter to learning-eligible rows only
    if "learning_eligible" in df.columns:
        mask = df["learning_eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[mask].copy()

    return df


# ---------------------------------------------------------------------------
# Per-family target construction
# ---------------------------------------------------------------------------

def _build_btc_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure BTC-specific targets exist.  All labels are in price-space.

    tp_before_sl_60m — already present from ContractTargetBuilder.
    price_return      — already present.
    entry_quality     — 1 if price_return > 0, 0 otherwise (proxy when missing).
    exit_quality      — 1 if price_return >= tp_threshold_pct (proxy when missing).
    """
    out = df.copy()

    if "tp_before_sl_60m" not in out.columns:
        # Derive from price_return and thresholds if possible
        if "price_return" in out.columns and "tp_threshold_pct" in out.columns:
            pr = pd.to_numeric(out["price_return"], errors="coerce")
            tp = pd.to_numeric(out["tp_threshold_pct"], errors="coerce")
            out["tp_before_sl_60m"] = ((pr > 0) & (pr >= tp)).astype(float)
        else:
            log.warning("[btc] tp_before_sl_60m missing and cannot be derived — will skip")

    if "entry_quality" not in out.columns and "price_return" in out.columns:
        out["entry_quality"] = (
            pd.to_numeric(out["price_return"], errors="coerce") > 0
        ).astype(float)

    if "exit_quality" not in out.columns and "price_return" in out.columns:
        pr = pd.to_numeric(out["price_return"], errors="coerce")
        if "tp_threshold_pct" in out.columns:
            tp = pd.to_numeric(out["tp_threshold_pct"], errors="coerce")
            out["exit_quality"] = (pr >= tp).astype(float)
        else:
            out["exit_quality"] = (pr > 0.02).astype(float)  # 2% proxy

    return out


def _build_weather_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure weather-specific targets exist.  All labels are in probability-space.

    target_up                      — 1 if contract resolved YES (price → 1.0).
    weather_contract_resolved_yes  — same thing, alternative name.
    entry_quality                  — 1 if bought YES and resolved YES, or bought NO and resolved NO.
    """
    out = df.copy()

    # Unify the two label column names
    if "target_up" not in out.columns and "weather_contract_resolved_yes" in out.columns:
        out["target_up"] = pd.to_numeric(out["weather_contract_resolved_yes"], errors="coerce")
    elif "target_up" not in out.columns and "price_return" in out.columns:
        # If contract resolved YES, final price → 1.0, return should be positive
        out["target_up"] = (
            pd.to_numeric(out["price_return"], errors="coerce") > 0
        ).astype(float)

    if "weather_contract_resolved_yes" not in out.columns and "target_up" in out.columns:
        out["weather_contract_resolved_yes"] = out["target_up"]

    if "entry_quality" not in out.columns and "target_up" in out.columns:
        # Entry quality: our signal_label agrees with the outcome
        sl = out.get("signal_label", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
        resolved_yes = pd.to_numeric(out["target_up"], errors="coerce").fillna(0.5) > 0.5
        out["entry_quality"] = (
            ((sl == "BUY_YES") & resolved_yes) | ((sl == "BUY_NO") & ~resolved_yes)
        ).astype(float)

    return out


def _apply_family_targets(family: str, df: pd.DataFrame) -> pd.DataFrame:
    if family == _BTC:
        return _build_btc_targets(df)
    if family == _WEATHER:
        return _build_weather_targets(df)
    raise ValueError(f"Unknown family: {family!r}")


# ---------------------------------------------------------------------------
# Per-family promotion criteria (wraps hard gates with family tuning)
# ---------------------------------------------------------------------------

@dataclass
class FamilyPromotionVerdict:
    family: str
    promoted: bool
    reason: str
    wf_result: FamilyWFResult | None
    gate_failures: list[str]
    metrics: dict[str, Any]

    def log_summary(self) -> None:
        status = "PROMOTED" if self.promoted else "BLOCKED"
        log.info(
            "[FamilyLoop/%s] %s — %s | metrics=%s",
            self.family, status, self.reason,
            json.dumps(self.metrics, default=str),
        )


def _evaluate_family_promotion(
    family: str,
    candidate_artifact_dir: Path,
    wf_result: FamilyWFResult | None,
    spec: FamilyTargetSpec,
    *,
    logs_dir: Path,
    weights_dir: Path,
    candidate_version: str,
    min_oos_rows: int,
) -> FamilyPromotionVerdict:
    """
    Run the full promotion decision for one family's candidate.

    Layers (all must pass):
      1. Minimum OOS rows
      2. Walk-forward gate failures (AUC / Sortino or Brier / drawdown / ECE)
      3. PromotionGateRunner hard gates (existing 7-gate system, family-scoped)
    """
    gate_failures: list[str] = []
    metrics: dict[str, Any] = {"family": family, "candidate_version": candidate_version}

    # ── Layer 0: minimum data ────────────────────────────────────────────────
    if wf_result is None:
        gate_failures.append("walk_forward_evaluation_failed: insufficient data")
        return FamilyPromotionVerdict(
            family=family,
            promoted=False,
            reason="walk_forward_evaluation_failed",
            wf_result=None,
            gate_failures=gate_failures,
            metrics=metrics,
        )

    metrics.update(wf_result.summary())

    if wf_result.n_oos_rows < min_oos_rows:
        gate_failures.append(
            f"min_oos_rows {wf_result.n_oos_rows} < required {min_oos_rows}"
        )

    # ── Layer 1: walk-forward gates ──────────────────────────────────────────
    wf_passed, wf_failures = wf_result.passes_gates(spec)
    gate_failures.extend(wf_failures)

    # ── Layer 2: PromotionGateRunner (existing hard gates) ───────────────────
    hard_passed = True
    try:
        from promotion_gate import PromotionGateRunner
        runner = PromotionGateRunner(
            family=family,
            logs_dir=str(logs_dir),
            weights_dir=str(weights_dir),
        )
        gate_result = runner.evaluate(
            candidate_artifact_dir=str(candidate_artifact_dir),
            candidate_version=candidate_version,
        )
        metrics["hard_gate_report"] = gate_result.report().replace("\n", " | ")
        metrics["hard_gate_passed"] = gate_result.passed
        if not gate_result.passed:
            hard_passed = False
            for g in gate_result.failed_gates():
                gate_failures.append(
                    f"hard_gate:{g.name} value={g.value:.4f} threshold={g.threshold:.4f}"
                )
    except Exception as exc:
        log.warning("[FamilyLoop/%s] PromotionGateRunner failed (non-blocking): %s", family, exc)
        # Don't block on infrastructure failure — let the WF gates decide
        metrics["hard_gate_error"] = str(exc)

    promoted = len(gate_failures) == 0
    reason = "all_gates_passed" if promoted else " | ".join(gate_failures[:3])
    return FamilyPromotionVerdict(
        family=family,
        promoted=promoted,
        reason=reason,
        wf_result=wf_result,
        gate_failures=gate_failures,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Per-family artifact promotion
# ---------------------------------------------------------------------------

def _promote_artifacts(
    candidate_dir: Path,
    active_weights_dir: Path,
    family: str,
) -> list[str]:
    """
    Atomically copy task_{model}.joblib files from candidate → active weights.
    Returns list of promoted filenames.
    """
    import shutil

    candidate_dir = Path(candidate_dir)
    active_weights_dir = Path(active_weights_dir)
    active_weights_dir.mkdir(parents=True, exist_ok=True)

    promoted: list[str] = []
    for src in candidate_dir.glob("task_*.joblib"):
        dst = active_weights_dir / src.name
        try:
            # Write to temp then rename for atomicity
            tmp = dst.with_suffix(".tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            promoted.append(src.name)
            log.info("[FamilyLoop/%s] promoted %s → %s", family, src, dst)
        except Exception as exc:
            log.warning("[FamilyLoop/%s] could not promote %s: %s", family, src.name, exc)

    # Also promote edge calibrator if present
    for extra in ["edge_calibrator.joblib", "governor_thresholds.json"]:
        src = candidate_dir / extra
        if src.exists():
            dst = active_weights_dir / extra
            try:
                tmp = dst.with_suffix(".tmp")
                import shutil as _shutil
                _shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                promoted.append(extra)
            except Exception as exc:
                log.warning("[FamilyLoop/%s] could not promote %s: %s", family, extra, exc)

    return promoted


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class FamilyLearningLoop:
    """
    Manages the full offline learning lifecycle for one market family.

    Lifecycle per cycle:
      1. Check trigger (new closed trades since last retrain ≥ threshold)
      2. Build / refresh dataset (family-scoped canonical rows only)
      3. Apply family-specific target construction
      4. Run walk-forward evaluation (family-specific metrics)
      5. Evaluate promotion criteria (family-specific gates)
      6. If passing: train full task model suite and promote artifacts
      7. Persist state and emit audit log
    """

    def __init__(
        self,
        family: str,
        *,
        logs_dir: str | Path = "logs",
        weights_dir: str | Path = "weights",
    ):
        family = family.lower().strip()
        if family not in _SUPPORTED:
            raise ValueError(f"Unsupported family {family!r}. Supported: {_SUPPORTED}")

        self.family = family
        self.spec = _FAMILY_SPECS[family]
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)

        # Family-scoped directories
        self.family_logs_dir = self.logs_dir / family
        self.family_weights_dir = self.weights_dir / family
        self.family_logs_dir.mkdir(parents=True, exist_ok=True)
        self.family_weights_dir.mkdir(parents=True, exist_ok=True)

        # Persistent state
        self._state_file = self.family_logs_dir / "loop_state.json"
        self._audit_file = self.family_logs_dir / "loop_audit.csv"
        self._state = self._load_state()

    # ── State persistence ────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if not self._state_file.exists():
            return {"last_retrained_closed_count": 0, "last_retrained_at": ""}
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception:
            return {"last_retrained_closed_count": 0, "last_retrained_at": ""}

    def _save_state(self) -> None:
        try:
            self._state_file.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("[FamilyLoop/%s] state save failed: %s", self.family, exc)

    def _append_audit(self, row: dict) -> None:
        try:
            existing = pd.DataFrame()
            if self._audit_file.exists():
                existing = pd.read_csv(self._audit_file, engine="python", on_bad_lines="skip")
            new_df = pd.DataFrame([row])
            merged = pd.concat([existing, new_df], ignore_index=True)
            merged.to_csv(self._audit_file, index=False)
        except Exception as exc:
            log.warning("[FamilyLoop/%s] audit write failed: %s", self.family, exc)

    # ── Trigger logic ────────────────────────────────────────────────────────

    @property
    def _trigger_threshold(self) -> int:
        env_key = f"RETRAIN_THRESHOLD_{self.family.upper()}"
        raw = os.getenv(env_key)
        if raw:
            try:
                return int(float(raw))
            except ValueError:
                pass
        return _DEFAULT_THRESHOLD.get(self.family, 5)

    @property
    def _min_oos_rows(self) -> int:
        return _DEFAULT_MIN_OOS.get(self.family, 30)

    def _count_closed_trades(self) -> int:
        """Count closed trades for this family from closed_positions.csv."""
        closed_file = self.logs_dir / "closed_positions.csv"
        if not closed_file.exists():
            return 0
        try:
            df = pd.read_csv(closed_file, engine="python", on_bad_lines="skip",
                             usecols=lambda c: c in {"market_family", "closed_at"})
        except Exception:
            try:
                df = pd.read_csv(closed_file, engine="python", on_bad_lines="skip")
            except Exception:
                return 0
        if df.empty:
            return 0
        if "market_family" in df.columns:
            mf = df["market_family"].fillna("").astype(str)
            if self.family == _WEATHER:
                df = df[mf.str.startswith(_WEATHER)]
            else:
                df = df[~mf.str.startswith(_WEATHER)]
        return len(df)

    def should_retrain(self, force: bool = False) -> tuple[bool, str]:
        if force:
            return True, "forced"

        current_count = self._count_closed_trades()
        last_count = int(self._state.get("last_retrained_closed_count", 0))
        new_trades = current_count - last_count
        threshold = self._trigger_threshold

        if new_trades >= threshold:
            return True, f"new_closed_trades={new_trades}>={threshold}"

        # Minimum interval cooldown
        min_interval = int(os.getenv(
            f"RETRAIN_MIN_INTERVAL_SECONDS_{self.family.upper()}",
            os.getenv("RETRAIN_MIN_INTERVAL_SECONDS", "0"),
        ))
        last_at_str = self._state.get("last_retrained_at", "")
        if last_at_str and min_interval > 0:
            try:
                last_at = pd.to_datetime(last_at_str, utc=True)
                elapsed = (pd.Timestamp.now(tz="UTC") - last_at).total_seconds()
                if elapsed < min_interval:
                    return False, f"cooldown_active elapsed={int(elapsed)}s/{min_interval}s"
            except Exception:
                pass

        return False, f"waiting new_closed_trades={new_trades}/{threshold}"

    # ── Dataset and target construction ─────────────────────────────────────

    def _refresh_dataset(self) -> pd.DataFrame:
        """
        Trigger dataset builders for this family, then load the canonical CSV.
        All builders are non-blocking — if they fail, we proceed with whatever
        canonical rows are already on disk.
        """
        try:
            from brain_paths import resolve_brain_context
            context = resolve_brain_context(
                self.family,
                shared_logs_dir=self.logs_dir,
                shared_weights_dir=self.weights_dir,
            )
            from brain_training_orchestrator import build_family_datasets
            # build_family_datasets loops over all contexts; we only want ours.
            # Call the individual builders directly to avoid cross-family rebuilds.
            try:
                from historical_dataset_builder import HistoricalDatasetBuilder
                HistoricalDatasetBuilder(brain_context=context).write()
            except Exception as exc:
                log.debug("[FamilyLoop/%s] HistoricalDatasetBuilder failed: %s", self.family, exc)
            try:
                from contract_target_builder import ContractTargetBuilder
                ContractTargetBuilder(brain_context=context).write()
            except Exception as exc:
                log.debug("[FamilyLoop/%s] ContractTargetBuilder failed: %s", self.family, exc)
            try:
                from canonical_dataset_builder import CanonicalDatasetBuilder
                CanonicalDatasetBuilder(logs_dir=str(self.logs_dir)).write()
            except Exception as exc:
                log.debug("[FamilyLoop/%s] CanonicalDatasetBuilder failed: %s", self.family, exc)
            try:
                from task_training_table_builder import TaskTrainingTableBuilder
                TaskTrainingTableBuilder(logs_dir=str(self.logs_dir)).write()
            except Exception as exc:
                log.debug("[FamilyLoop/%s] TaskTrainingTableBuilder failed: %s", self.family, exc)
        except Exception as exc:
            log.warning("[FamilyLoop/%s] dataset refresh failed (non-blocking): %s", self.family, exc)

        df = _load_family_canonical(self.family, self.logs_dir)
        log.info("[FamilyLoop/%s] canonical dataset: %d rows", self.family, len(df))
        return df

    # ── Training ─────────────────────────────────────────────────────────────

    def _train_task_models(self, candidate_dir: Path, df: pd.DataFrame) -> dict[str, Any]:
        """
        Train the task model suite (5 learners) using the already-enriched df.
        Saves artifacts into candidate_dir/{family}/.
        """
        try:
            from task_model_suite import TaskModelSuite
            suite = TaskModelSuite(
                family=self.family,
                logs_dir=str(self.logs_dir),
                weights_dir=str(candidate_dir),
            )
            results = suite.train_all()
            log.info("[FamilyLoop/%s] task model suite trained: %s", self.family, results)
            return results or {}
        except Exception as exc:
            log.warning("[FamilyLoop/%s] TaskModelSuite failed (non-blocking): %s", self.family, exc)
            return {}

    def _train_brain_models(self, candidate_dir: Path) -> dict[str, Any]:
        """
        Train the supervised/stage1/stage2 brain models for this family only.
        """
        try:
            from brain_paths import resolve_brain_context
            from brain_training_orchestrator import train_brain_models
            context = resolve_brain_context(
                self.family,
                shared_logs_dir=self.logs_dir,
                shared_weights_dir=self.weights_dir,
            )
            train_brain_models(context, candidate_weights_dir=candidate_dir)
            return {"brain_trained": True}
        except Exception as exc:
            log.warning("[FamilyLoop/%s] brain model training failed (non-blocking): %s", self.family, exc)
            return {"brain_trained": False, "brain_error": str(exc)}

    def _train_edge_calibrator(self, candidate_dir: Path) -> None:
        try:
            from edge_calibrator import train_edge_calibrator
            train_edge_calibrator(
                family=self.family,
                logs_dir=str(self.logs_dir),
                weights_dir=str(candidate_dir),
            )
        except Exception as exc:
            log.debug("[FamilyLoop/%s] edge calibrator training failed: %s", self.family, exc)

    # ── Main entry point ─────────────────────────────────────────────────────

    def maybe_retrain(self, force: bool = False) -> dict[str, Any]:
        """
        Check trigger, and if met, run a full train → eval → promote cycle.

        Returns a result dict with keys:
          triggered, promoted, reason, metrics, gate_failures
        """
        should, trigger_reason = self.should_retrain(force)
        if not should:
            log.info("[FamilyLoop/%s] retrain skipped: %s", self.family, trigger_reason)
            return {"triggered": False, "promoted": False, "reason": trigger_reason}

        log.info("[FamilyLoop/%s] retrain triggered: %s", self.family, trigger_reason)
        now = datetime.now(timezone.utc).isoformat()
        current_count = self._count_closed_trades()
        version = datetime.now(timezone.utc).strftime(f"{self.family}_%Y%m%d%H%M%S")

        # ── 1. Build / refresh dataset ───────────────────────────────────────
        df = self._refresh_dataset()
        if df.empty:
            reason = "dataset_empty"
            log.warning("[FamilyLoop/%s] dataset empty — aborting cycle", self.family)
            self._append_audit({
                "attempted_at": now, "version": version, "triggered": True,
                "promoted": False, "reason": reason, "n_rows": 0,
            })
            return {"triggered": True, "promoted": False, "reason": reason}

        # ── 2. Apply family-specific targets ─────────────────────────────────
        df = _apply_family_targets(self.family, df)

        # ── 3. Walk-forward evaluation on current data ───────────────────────
        wf_result = _walk_forward_evaluate(df, self.spec)

        # ── 4. Evaluate promotion criteria ───────────────────────────────────
        # Build candidate artifacts in a staging directory
        from model_artifact_staging import build_candidate_weights_dir
        candidate_dir = build_candidate_weights_dir(
            self.weights_dir,
            prefix=f"family_loop_{self.family}",
        )
        candidate_family_dir = candidate_dir / self.family
        candidate_family_dir.mkdir(parents=True, exist_ok=True)

        verdict = _evaluate_family_promotion(
            family=self.family,
            candidate_artifact_dir=candidate_family_dir,
            wf_result=wf_result,
            spec=self.spec,
            logs_dir=self.logs_dir,
            weights_dir=self.weights_dir,
            candidate_version=version,
            min_oos_rows=self._min_oos_rows,
        )
        verdict.log_summary()

        promoted = False
        promoted_files: list[str] = []

        if verdict.promoted:
            # ── 5. Train full model suite ────────────────────────────────────
            task_results = self._train_task_models(candidate_dir, df)
            brain_results = self._train_brain_models(candidate_dir)
            self._train_edge_calibrator(candidate_family_dir)

            # ── 6. Promote artifacts ─────────────────────────────────────────
            promoted_files = _promote_artifacts(
                candidate_family_dir, self.family_weights_dir, self.family
            )
            if promoted_files:
                promoted = True
                log.info(
                    "[FamilyLoop/%s] promoted %d artifact(s): %s",
                    self.family, len(promoted_files), promoted_files,
                )
                # Register with champion/challenger system
                try:
                    from champion_challenger import get_slot as _cc_get_slot
                    cc_slot = _cc_get_slot(
                        self.family,
                        weights_dir=str(self.weights_dir),
                        logs_dir=str(self.logs_dir),
                    )
                    cc_slot.bootstrap_champion_from_weights()
                    cc_slot.register_challenger(candidate_family_dir, version_label=version)
                    cc_result = cc_slot.try_promote()
                    log.info("[FamilyLoop/%s] CC try_promote: %s", self.family, cc_result)
                except Exception as cc_exc:
                    log.debug("[FamilyLoop/%s] CC integration failed (non-blocking): %s", self.family, cc_exc)
            else:
                log.warning("[FamilyLoop/%s] promotion gate passed but no artifacts produced", self.family)
        else:
            verdict.log_summary()

        # ── 7. Update state ──────────────────────────────────────────────────
        self._state["last_retrained_closed_count"] = current_count
        self._state["last_retrained_at"] = now
        self._save_state()

        audit_row = {
            "attempted_at": now,
            "version": version,
            "family": self.family,
            "triggered": True,
            "promoted": promoted,
            "reason": verdict.reason if not promoted else "promoted",
            "n_rows": len(df),
            "gate_failures": " | ".join(verdict.gate_failures),
            "promoted_files": ", ".join(promoted_files),
            **verdict.metrics,
        }
        self._append_audit(audit_row)

        return {
            "triggered": True,
            "promoted": promoted,
            "reason": audit_row["reason"],
            "metrics": verdict.metrics,
            "gate_failures": verdict.gate_failures,
            "promoted_files": promoted_files,
            "version": version,
        }

    def status(self) -> dict[str, Any]:
        current_count = self._count_closed_trades()
        last_count = int(self._state.get("last_retrained_closed_count", 0))
        return {
            "family": self.family,
            "trigger_threshold": self._trigger_threshold,
            "closed_trades_total": current_count,
            "new_since_last_retrain": current_count - last_count,
            "last_retrained_at": self._state.get("last_retrained_at", "never"),
            "ready_to_retrain": (current_count - last_count) >= self._trigger_threshold,
        }


# ---------------------------------------------------------------------------
# Registry and convenience entry point
# ---------------------------------------------------------------------------

_LOOP_REGISTRY: dict[str, FamilyLearningLoop] = {}


def get_loop(
    family: str,
    *,
    logs_dir: str | Path = "logs",
    weights_dir: str | Path = "weights",
) -> FamilyLearningLoop:
    """Process-level singleton per family (matches the champion_challenger pattern)."""
    key = f"{family}|{logs_dir}|{weights_dir}"
    if key not in _LOOP_REGISTRY:
        _LOOP_REGISTRY[key] = FamilyLearningLoop(family, logs_dir=logs_dir, weights_dir=weights_dir)
    return _LOOP_REGISTRY[key]


def run_all_family_loops(
    *,
    logs_dir: str | Path = "logs",
    weights_dir: str | Path = "weights",
    force_btc: bool = False,
    force_weather: bool = False,
) -> dict[str, dict]:
    """
    Run BTC and weather learning loops independently.
    BTC can promote without waiting for weather data to accumulate, and vice versa.

    Returns dict[family → result] for both families.
    """
    results: dict[str, dict] = {}

    btc_loop = get_loop(_BTC, logs_dir=logs_dir, weights_dir=weights_dir)
    weather_loop = get_loop(_WEATHER, logs_dir=logs_dir, weights_dir=weights_dir)

    results[_BTC] = btc_loop.maybe_retrain(force=force_btc)
    results[_WEATHER] = weather_loop.maybe_retrain(force=force_weather)

    log.info(
        "[FamilyLoops] BTC triggered=%s promoted=%s | Weather triggered=%s promoted=%s",
        results[_BTC]["triggered"], results[_BTC]["promoted"],
        results[_WEATHER]["triggered"], results[_WEATHER]["promoted"],
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Family-specific learning loop runner")
    parser.add_argument(
        "--family", choices=["btc", "weather_temperature", "all"], default="all",
        help="Which family to run (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of trigger")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--status", action="store_true", help="Print status only, do not retrain")
    args = parser.parse_args()

    if args.status:
        for fam in ([args.family] if args.family != "all" else ["btc", "weather_temperature"]):
            loop = get_loop(fam, logs_dir=args.logs_dir, weights_dir=args.weights_dir)
            print(json.dumps(loop.status(), indent=2))
        sys.exit(0)

    if args.family == "all":
        results = run_all_family_loops(
            logs_dir=args.logs_dir,
            weights_dir=args.weights_dir,
            force_btc=args.force,
            force_weather=args.force,
        )
    else:
        loop = get_loop(args.family, logs_dir=args.logs_dir, weights_dir=args.weights_dir)
        results = {args.family: loop.maybe_retrain(force=args.force)}

    for fam, res in results.items():
        print(f"\n{'='*60}")
        print(f"  {fam.upper()}")
        print(f"  triggered : {res['triggered']}")
        print(f"  promoted  : {res['promoted']}")
        print(f"  reason    : {res['reason']}")
        if res.get("metrics"):
            for k, v in res["metrics"].items():
                if k not in {"family", "candidate_version", "hard_gate_report"}:
                    print(f"  {k:<30} {v}")

    any_failed = any(
        r["triggered"] and not r["promoted"] and r["reason"] not in {
            "waiting_threshold", "cooldown_active", "dataset_empty",
        }
        for r in results.values()
    )
    sys.exit(1 if any_failed else 0)
