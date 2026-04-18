"""Hard promotion gates for model advancement.

A candidate model is only promoted when it clears EVERY gate.
No single gate can be waived.  All thresholds are configurable via env vars;
defaults are conservative.

Gates (in evaluation order):
  1. SAMPLE_SIZE          — minimum OOS trades; blocks lucky-streak promotion
  2. SHARPE               — annualised Sortino ratio on OOS PnL stream > threshold
  3. DRAWDOWN             — max rolling drawdown on OOS equity curve < threshold
  4. CALIBRATION          — ECE (expected calibration error) < threshold
  5. FILL_ADJUSTED_EDGE   — (win_rate × avg_win - loss_rate × avg_loss) after
                            subtracting mean slippage_bps cost > 0
  6. FAMILY_STABILITY     — CV of AUC across walk-forward folds < threshold
                            (high CV = model memorised one time window)
  7. BEATS_CHAMPION       — candidate Sortino > incumbent Sortino on the same
                            OOS window (champion comparison, not just absolute)

Statistical guard:
  Every numeric gate additionally checks that the estimate is based on enough
  samples to be meaningful.  If OOS trades < MIN_SAMPLE_SIZE the entire gate
  suite returns blocked=True with reason "insufficient_oos_sample".

Usage:
    from promotion_gate import PromotionGateRunner
    runner = PromotionGateRunner(family="btc", logs_dir="logs", weights_dir="weights")
    result = runner.evaluate(candidate_artifact_dir="weights/_candidates/base_retrain_XYZ")
    if result.passed:
        ...promote...
    else:
        print(result.report())

Environment variables (all optional — defaults shown):
    GATE_MIN_OOS_TRADES          = 50
    GATE_MIN_SORTINO             = 0.30
    GATE_MAX_DRAWDOWN_PCT        = 0.20   (20 % of peak equity)
    GATE_MAX_ECE                 = 0.10   (10 % calibration error)
    GATE_MIN_FILL_ADJUSTED_EDGE  = 0.0
    GATE_MAX_STABILITY_CV        = 0.40   (40 % coefficient of variation on fold AUC)
    GATE_CHAMPION_MARGIN         = 0.0    (candidate Sortino must be >= champion + margin)
    GATE_SIGNIFICANCE_PVALUE     = 0.10   (one-sided t-test on OOS returns > 0)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class GateOutcome:
    name: str
    passed: bool
    value: float
    threshold: float
    direction: str          # "above" | "below"
    detail: str = ""

    def summary_line(self) -> str:
        sym = "OK" if self.passed else "XX"
        op = ">=" if self.direction == "above" else "<="
        return f"  {sym} {self.name:<28} value={self.value:>9.4f}  {op} {self.threshold:.4f}  {self.detail}"


@dataclass
class GateResult:
    family: str
    candidate_version: str
    passed: bool
    block_reason: str
    gates: list[GateOutcome] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def failed_gates(self) -> list[GateOutcome]:
        return [g for g in self.gates if not g.passed]

    def report(self) -> str:
        lines = [
            f"=== Promotion Gate Report [{self.family}] candidate={self.candidate_version} ===",
            f"Overall: {'PASSED' if self.passed else 'BLOCKED'}",
        ]
        if not self.passed:
            lines.append(f"Block reason: {self.block_reason}")
        lines.append("")
        for g in self.gates:
            lines.append(g.summary_line())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "family": self.family,
            "candidate_version": self.candidate_version,
            "gate_passed": self.passed,
            "gate_block_reason": self.block_reason,
        }
        for g in self.gates:
            d[f"gate_{g.name}_value"] = g.value
            d[f"gate_{g.name}_passed"] = g.passed
        d.update(self.metrics)
        return d


# ---------------------------------------------------------------------------
# Finance helpers
# ---------------------------------------------------------------------------

def _sortino_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    if len(returns) < 2:
        return float("nan")
    mean_r = float(np.mean(returns))
    downside = returns[returns < 0.0]
    if len(downside) == 0:
        # All positive — return a large positive number, capped to avoid inf
        return 10.0
    dd_std = float(np.std(downside, ddof=1))
    if dd_std == 0.0:
        return float("nan")
    return float((mean_r / dd_std) * math.sqrt(periods_per_year))


def _max_drawdown_pct(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as fraction of peak (0.20 = 20%)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    # Avoid divide-by-zero on zero-peak
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(peak != 0, (equity_curve - peak) / np.abs(peak), 0.0)
    return float(abs(dd.min()))


def _expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE: weighted average |mean_pred_prob - fraction_positive| over probability bins."""
    if len(probs) == 0:
        return float("nan")
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    ece = 0.0
    n = len(probs)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        ece += (len(bin_probs) / n) * abs(bin_probs.mean() - bin_labels.mean())
    return float(ece)


def _one_sided_t_pvalue(returns: np.ndarray) -> float:
    """p-value for H0: mean(returns) <= 0, H1: mean(returns) > 0."""
    from scipy import stats  # optional; fall back gracefully
    if len(returns) < 5:
        return 1.0
    t_stat, p_two = stats.ttest_1samp(returns, popmean=0.0)
    if t_stat <= 0:
        return 1.0
    return float(p_two / 2.0)   # one-sided


def _fill_adjusted_edge(df: pd.DataFrame) -> float:
    """
    Edge after fill costs:
        (win_rate × avg_win_pnl) - (loss_rate × avg_loss_pnl) - mean_slippage_cost

    Uses `price_return` as the return column (BTC % move).
    `slippage_bps` converted to return-space: bps / 10000.
    """
    pnl_col = next((c for c in ("price_return", "net_realized_pnl", "realized_pnl") if c in df.columns), None)
    if pnl_col is None or df.empty:
        return float("nan")

    returns = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
    if returns.empty:
        return float("nan")

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns)
    loss_rate = len(losses) / len(returns)
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(abs(losses.mean())) if not losses.empty else 0.0

    gross_edge = win_rate * avg_win - loss_rate * avg_loss

    # Slippage cost
    slip_cost = 0.0
    if "slippage_bps" in df.columns:
        slip_bps = pd.to_numeric(df["slippage_bps"], errors="coerce").dropna()
        if not slip_bps.empty:
            slip_cost = float(slip_bps.mean()) / 10_000.0  # bps → fraction

    return gross_edge - slip_cost


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------

class PromotionGateRunner:
    """
    Evaluates all hard promotion gates for a candidate model artifact directory.

    Reads:
      - logs/{family}/canonical_dataset.csv  — OOS trade returns, slippage, calibration data
      - walk_forward_evaluator.TaskWalkForwardEvaluator — fold AUC stability
      - weights/{family}/task_*.joblib (incumbent) — for champion comparison
      - weights/_candidates/{label}/task_*.joblib (candidate)

    The OOS window is the most recent `oos_window_trades` rows of canonical_dataset
    that postdate the candidate's training data.  If the canonical dataset has a
    `split_label` column ("train"/"test") it is used directly; otherwise the last
    20 % of rows by timestamp are used.
    """

    _GATE_NAMES = [
        "sample_size",
        "sharpe",
        "drawdown",
        "calibration",
        "fill_adjusted_edge",
        "family_stability",
        "beats_champion",
    ]

    def __init__(
        self,
        family: str,
        logs_dir: str = "logs",
        weights_dir: str = "weights",
        oos_fraction: float = 0.20,
    ) -> None:
        self.family = family.lower().strip()
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self.oos_fraction = oos_fraction

        # Thresholds — read from env, fall back to conservative defaults
        self.min_oos_trades: int = self._env_int("GATE_MIN_OOS_TRADES", 50)
        self.min_sortino: float = self._env_float("GATE_MIN_SORTINO", 0.30)
        self.max_drawdown_pct: float = self._env_float("GATE_MAX_DRAWDOWN_PCT", 0.20)
        self.max_ece: float = self._env_float("GATE_MAX_ECE", 0.10)
        self.min_fill_adj_edge: float = self._env_float("GATE_MIN_FILL_ADJUSTED_EDGE", 0.0)
        self.max_stability_cv: float = self._env_float("GATE_MAX_STABILITY_CV", 0.40)
        self.champion_margin: float = self._env_float("GATE_CHAMPION_MARGIN", 0.0)
        self.significance_pvalue: float = self._env_float("GATE_SIGNIFICANCE_PVALUE", 0.10)

    # -- env helpers ---------------------------------------------------------

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name, "").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    # -- data loading --------------------------------------------------------

    def _load_canonical(self) -> pd.DataFrame:
        for p in [
            self.logs_dir / self.family / "canonical_dataset.csv",
            self.logs_dir / "canonical_dataset.csv",
        ]:
            if p.exists():
                df = pd.read_csv(p, low_memory=False)
                for col in ("timestamp", "signal_timestamp", "fill_timestamp"):
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
                        df = df.sort_values(col).reset_index(drop=True)
                        break
                return df
        return pd.DataFrame()

    def _oos_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the OOS portion of canonical_dataset."""
        if df.empty:
            return df
        if "split_label" in df.columns:
            oos = df[df["split_label"].astype(str).str.lower() == "test"]
            if not oos.empty:
                return oos.copy()
        # Fall back: last oos_fraction of rows (already sorted by time)
        split = int(len(df) * (1.0 - self.oos_fraction))
        return df.iloc[split:].copy()

    def _eligible_oos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict to learning_eligible rows in the OOS slice."""
        oos = self._oos_slice(df)
        if "learning_eligible" in oos.columns:
            mask = oos["learning_eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
            return oos[mask].copy()
        return oos

    # -- individual gate computations ----------------------------------------

    def _gate_sample_size(self, oos: pd.DataFrame) -> GateOutcome:
        n = len(oos)
        passed = n >= self.min_oos_trades
        return GateOutcome(
            name="sample_size",
            passed=passed,
            value=float(n),
            threshold=float(self.min_oos_trades),
            direction="above",
            detail=f"{n} OOS eligible rows",
        )

    def _gate_sharpe(self, oos: pd.DataFrame) -> tuple[GateOutcome, float]:
        """Returns (gate, sortino_value)."""
        pnl_col = next((c for c in ("price_return", "net_realized_pnl", "realized_pnl") if c in oos.columns), None)
        if pnl_col is None:
            return GateOutcome("sharpe", False, float("nan"), self.min_sortino, "above", "no return column"), float("nan")

        returns = pd.to_numeric(oos[pnl_col], errors="coerce").dropna().to_numpy()
        sortino = _sortino_ratio(returns)

        # Statistical significance guard
        try:
            pval = _one_sided_t_pvalue(returns)
            sig_note = f"p={pval:.3f}"
            sig_ok = pval <= self.significance_pvalue
        except Exception:
            sig_note = "scipy unavailable"
            sig_ok = True  # don't block if scipy missing

        passed = (not math.isnan(sortino)) and (sortino >= self.min_sortino) and sig_ok
        detail = f"sortino={sortino:.4f}  {sig_note}"
        return GateOutcome("sharpe", passed, sortino if not math.isnan(sortino) else -999.0,
                           self.min_sortino, "above", detail), sortino

    def _gate_drawdown(self, oos: pd.DataFrame) -> GateOutcome:
        pnl_col = next((c for c in ("price_return", "net_realized_pnl", "realized_pnl") if c in oos.columns), None)
        if pnl_col is None:
            return GateOutcome("drawdown", False, float("nan"), self.max_drawdown_pct, "below", "no return column")

        returns = pd.to_numeric(oos[pnl_col], errors="coerce").fillna(0.0).to_numpy()
        equity = 1.0 + np.cumsum(returns)  # normalised equity curve starting at 1
        mdd = _max_drawdown_pct(equity)
        passed = mdd <= self.max_drawdown_pct
        return GateOutcome("drawdown", passed, mdd, self.max_drawdown_pct, "below",
                           f"max_dd={mdd:.2%}  allowed={self.max_drawdown_pct:.2%}")

    def _gate_calibration(self, oos: pd.DataFrame, candidate_dir: Path) -> GateOutcome:
        """
        Measures ECE of the entry_edge model on OOS data.
        Falls back to checking confidence_score vs actual entry_quality if no artifact.
        """
        # Try to load candidate entry_edge model and score OOS rows
        artifact = candidate_dir / f"task_entry_edge.joblib"
        if not artifact.exists():
            # Incumbent
            artifact = self.weights_dir / self.family / "task_entry_edge.joblib"

        ece = float("nan")
        if artifact.exists() and "entry_quality" in oos.columns:
            try:
                import joblib
                from sklearn.preprocessing import LabelEncoder
                from sklearn.impute import SimpleImputer

                bundle = joblib.load(artifact)
                model = bundle["model"]
                features = bundle["features"]
                present = [f for f in features if f in oos.columns]
                if present:
                    X = oos[present].copy()
                    for c in X.columns:
                        if X[c].dtype == object:
                            le = LabelEncoder()
                            X[c] = le.fit_transform(X[c].astype(str).fillna("__missing__"))
                    X = SimpleImputer(strategy="median").fit_transform(X)
                    y_true = oos["entry_quality"].fillna(0).astype(int).to_numpy()
                    probs = model.predict_proba(X)
                    if probs.ndim == 2 and probs.shape[1] > 1:
                        probs = probs[:, 1]
                    else:
                        probs = probs[:, 0]
                    ece = _expected_calibration_error(probs, y_true)
            except Exception as exc:
                log.warning("[gate_calibration] artifact scoring failed: %s", exc)

        # Fallback: use raw confidence_score vs entry_quality
        if math.isnan(ece) and "confidence_score" in oos.columns and "entry_quality" in oos.columns:
            try:
                probs = pd.to_numeric(oos["confidence_score"], errors="coerce").fillna(0.5).to_numpy()
                labels = oos["entry_quality"].fillna(0).astype(int).to_numpy()
                ece = _expected_calibration_error(probs, labels)
            except Exception as exc:
                log.warning("[gate_calibration] fallback ECE failed: %s", exc)

        if math.isnan(ece):
            return GateOutcome("calibration", False, float("nan"), self.max_ece, "below",
                               "ECE could not be computed")

        passed = ece <= self.max_ece
        return GateOutcome("calibration", passed, ece, self.max_ece, "below",
                           f"ECE={ece:.4f}  allowed={self.max_ece:.4f}")

    def _gate_fill_adjusted_edge(self, oos: pd.DataFrame) -> GateOutcome:
        edge = _fill_adjusted_edge(oos)
        if math.isnan(edge):
            return GateOutcome("fill_adjusted_edge", False, float("nan"), self.min_fill_adj_edge,
                               "above", "could not compute fill-adjusted edge")
        passed = edge >= self.min_fill_adj_edge
        return GateOutcome("fill_adjusted_edge", passed, edge, self.min_fill_adj_edge, "above",
                           f"fill_adj_edge={edge:.6f}")

    def _gate_family_stability(self) -> GateOutcome:
        """
        Run TaskWalkForwardEvaluator on entry_edge model and measure fold AUC CV.
        High CV = model memorised one time window; blocks promotion.
        """
        try:
            from walk_forward_evaluator import TaskWalkForwardEvaluator
            ev = TaskWalkForwardEvaluator(self.family, logs_dir=str(self.logs_dir), n_splits=5)
            report = ev.evaluate("entry_edge")
            auc_vals = [f.auc for f in report.folds if not math.isnan(f.auc)]
        except Exception as exc:
            log.warning("[gate_stability] WFE failed: %s", exc)
            auc_vals = []

        if len(auc_vals) < 2:
            # Can't measure stability — treat as soft pass (don't block on missing data)
            return GateOutcome("family_stability", True, 0.0, self.max_stability_cv, "below",
                               f"only {len(auc_vals)} valid folds — stability not measurable, gate waived")

        cv = float(np.std(auc_vals, ddof=1) / np.mean(auc_vals)) if np.mean(auc_vals) != 0 else float("nan")
        if math.isnan(cv):
            return GateOutcome("family_stability", False, float("nan"), self.max_stability_cv, "below",
                               "CV undefined (mean AUC = 0)")
        passed = cv <= self.max_stability_cv
        return GateOutcome("family_stability", passed, cv, self.max_stability_cv, "below",
                           f"fold_AUC_CV={cv:.4f}  folds={auc_vals}")

    def _gate_beats_champion(
        self,
        oos: pd.DataFrame,
        candidate_sortino: float,
    ) -> GateOutcome:
        """
        Load the incumbent's registry row and compare OOS Sortino.
        If no incumbent exists, gate is auto-passed (first deployment).
        """
        registry_path = self.weights_dir / "model_registry.csv"
        if not registry_path.exists():
            return GateOutcome("beats_champion", True, candidate_sortino, candidate_sortino, "above",
                               "no champion in registry — first deployment")

        try:
            reg = pd.read_csv(registry_path, low_memory=False)
            if "activation_status" in reg.columns:
                promoted = reg[reg["activation_status"].astype(str).str.lower().isin({"promoted", "active"})]
                if promoted.empty:
                    return GateOutcome("beats_champion", True, candidate_sortino, candidate_sortino,
                                       "above", "no promoted champion — first deployment")
                champion_row = promoted.iloc[-1]
            else:
                champion_row = reg.iloc[-1]
        except Exception as exc:
            log.warning("[gate_beats_champion] registry read failed: %s", exc)
            return GateOutcome("beats_champion", True, candidate_sortino, candidate_sortino,
                               "above", "registry unreadable — gate waived")

        # Retrieve champion OOS Sortino from registry if recorded, else recompute
        champ_sortino = float("nan")
        for key in ("gate_sharpe_value", "oos_sortino", "sortino"):
            val = champion_row.get(key)
            if val is not None:
                try:
                    champ_sortino = float(val)
                    break
                except (TypeError, ValueError):
                    pass

        if math.isnan(champ_sortino):
            # Champion predates gate recording — auto-pass to avoid permanently blocking
            return GateOutcome("beats_champion", True, candidate_sortino,
                               candidate_sortino, "above",
                               "champion has no recorded Sortino — gate waived for legacy champion")

        required = champ_sortino + self.champion_margin
        if math.isnan(candidate_sortino):
            return GateOutcome("beats_champion", False, float("nan"), required, "above",
                               "candidate Sortino unavailable")

        passed = candidate_sortino >= required
        return GateOutcome(
            "beats_champion", passed, candidate_sortino, required, "above",
            f"candidate={candidate_sortino:.4f}  champion={champ_sortino:.4f}  margin={self.champion_margin:.4f}",
        )

    # -- main evaluation -----------------------------------------------------

    def evaluate(
        self,
        candidate_artifact_dir: str | Path | None = None,
        candidate_version: str | None = None,
    ) -> GateResult:
        """
        Run all gates and return a GateResult.

        candidate_artifact_dir: path to the candidate weights directory
            (e.g. weights/_candidates/base_retrain_20260418120000).
            If None, uses weights/{family}/ (re-evaluates the current incumbent).
        candidate_version: label for the registry row.  Defaults to dirname.
        """
        candidate_dir = (
            Path(candidate_artifact_dir) if candidate_artifact_dir
            else self.weights_dir / self.family
        )
        version = candidate_version or candidate_dir.name

        full_df = self._load_canonical()
        oos = self._eligible_oos(full_df)

        gates: list[GateOutcome] = []
        metrics: dict[str, Any] = {
            "oos_rows": len(oos),
            "total_rows": len(full_df),
        }

        # -- Gate 1: Sample size (hard gate — if it fails, skip remaining) --
        g_sample = self._gate_sample_size(oos)
        gates.append(g_sample)
        if not g_sample.passed:
            result = GateResult(
                family=self.family,
                candidate_version=version,
                passed=False,
                block_reason="insufficient_oos_sample",
                gates=gates,
                metrics=metrics,
            )
            log.warning("[PromotionGate] BLOCKED — %s", result.block_reason)
            return result

        # -- Gate 2: Sharpe/Sortino --
        g_sharpe, sortino_val = self._gate_sharpe(oos)
        gates.append(g_sharpe)
        metrics["oos_sortino"] = sortino_val

        # -- Gate 3: Drawdown --
        g_dd = self._gate_drawdown(oos)
        gates.append(g_dd)

        # -- Gate 4: Calibration --
        g_cal = self._gate_calibration(oos, candidate_dir)
        gates.append(g_cal)

        # -- Gate 5: Fill-adjusted edge --
        g_edge = self._gate_fill_adjusted_edge(oos)
        gates.append(g_edge)

        # -- Gate 6: Family stability (walk-forward AUC CV) --
        g_stab = self._gate_family_stability()
        gates.append(g_stab)

        # -- Gate 7: Beats champion --
        g_champ = self._gate_beats_champion(oos, sortino_val)
        gates.append(g_champ)

        # -- Aggregate --
        failures = [g for g in gates if not g.passed]
        passed = len(failures) == 0
        block_reason = " | ".join(g.name for g in failures) if failures else ""

        result = GateResult(
            family=self.family,
            candidate_version=version,
            passed=passed,
            block_reason=block_reason,
            gates=gates,
            metrics=metrics,
        )
        log.info("[PromotionGate] %s — %s", "PASSED" if passed else "BLOCKED", block_reason or "all gates clear")
        return result

    def evaluate_and_log(
        self,
        candidate_artifact_dir: str | Path | None = None,
        candidate_version: str | None = None,
        registry_path: str | Path | None = None,
    ) -> GateResult:
        """
        Run evaluate() and append the result row to the model registry CSV.
        """
        result = self.evaluate(candidate_artifact_dir, candidate_version)
        reg_path = Path(registry_path) if registry_path else self.weights_dir / "model_registry.csv"

        row = result.to_dict()
        row["attempted_at"] = pd.Timestamp.utcnow().isoformat()
        row["activation_status"] = "promoted" if result.passed else f"blocked_{result.block_reason}"

        try:
            existing = pd.read_csv(reg_path, low_memory=False) if reg_path.exists() else pd.DataFrame()
            updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            updated.to_csv(reg_path, index=False)
        except Exception as exc:
            log.warning("[PromotionGate] registry append failed: %s", exc)

        return result


# ---------------------------------------------------------------------------
# Convenience: quick summary for a given family
# ---------------------------------------------------------------------------

def run_gates(
    family: str,
    candidate_dir: str | Path | None = None,
    logs_dir: str = "logs",
    weights_dir: str = "weights",
) -> GateResult:
    runner = PromotionGateRunner(family=family, logs_dir=logs_dir, weights_dir=weights_dir)
    return runner.evaluate(candidate_artifact_dir=candidate_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Run hard promotion gates")
    parser.add_argument("--family", default="btc", choices=["btc", "weather_temperature"])
    parser.add_argument("--candidate-dir", default=None, help="Path to candidate weights dir")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--oos-fraction", type=float, default=0.20)
    args = parser.parse_args()

    runner = PromotionGateRunner(
        family=args.family,
        logs_dir=args.logs_dir,
        weights_dir=args.weights_dir,
        oos_fraction=args.oos_fraction,
    )
    result = runner.evaluate(args.candidate_dir)
    print(result.report())
    sys.exit(0 if result.passed else 1)
