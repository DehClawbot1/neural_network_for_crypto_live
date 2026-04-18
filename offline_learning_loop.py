from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from brain_paths import list_brain_contexts
from brain_training_orchestrator import build_family_datasets

logger = logging.getLogger(__name__)


@dataclass
class OfflinePromotionResult:
    family: str
    model_name: str
    promoted: bool
    reason: str
    candidate_path: str
    incumbent_path: str
    candidate_auc: float
    incumbent_auc: float
    candidate_metrics: dict
    incumbent_metrics: dict
    hard_gate_passed: bool


@dataclass
class PromotionGateMetrics:
    sample_size: int
    sharpe_like: float
    max_drawdown: float
    calibration_error: float
    fill_adjusted_edge: float
    family_stability: float


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def _apply_saved_calibrator(calibrator, probs: pd.Series) -> pd.Series:
    if calibrator is None:
        return pd.Series(probs, index=probs.index, dtype=float)
    raw = np.asarray(pd.to_numeric(probs, errors="coerce").fillna(0.0), dtype=float).reshape(-1, 1)
    if hasattr(calibrator, "predict_proba"):
        calibrated = calibrator.predict_proba(raw)
        vals = calibrated[:, 1] if calibrated.shape[1] > 1 else calibrated[:, 0]
        return pd.Series(vals, index=probs.index, dtype=float)
    if hasattr(calibrator, "predict"):
        return pd.Series(np.asarray(calibrator.predict(raw), dtype=float), index=probs.index)
    return pd.Series(probs, index=probs.index, dtype=float)


def _predict_with_saved_feature_order(model, X_eval: pd.DataFrame, *, proba: bool):
    try:
        return model.predict_proba(X_eval) if proba else model.predict(X_eval)
    except ValueError as exc:
        if "feature names" not in str(exc).lower():
            raise
        raw = X_eval.to_numpy()
        return model.predict_proba(raw) if proba else model.predict(raw)


class OfflineLearningLoop:
    """Collect -> retrain offline -> walk-forward compare -> promote if better."""

    def __init__(self, logs_dir: str | Path = "logs", weights_dir: str | Path = "weights") -> None:
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self.signal_file = self.logs_dir / "offline_learning_signal.json"
        self.report_file = self.logs_dir / "offline_learning_report.json"
        self.min_sample_size = int(float(__import__("os").getenv("OFFLINE_PROMOTION_MIN_SAMPLE_SIZE", "40")))
        self.min_sharpe_delta = float(__import__("os").getenv("OFFLINE_PROMOTION_MIN_SHARPE_DELTA", "0.0"))
        self.max_calibration_error = float(__import__("os").getenv("OFFLINE_PROMOTION_MAX_CALIBRATION_ERROR", "0.25"))
        self.min_fill_adjusted_edge = float(__import__("os").getenv("OFFLINE_PROMOTION_MIN_FILL_ADJUSTED_EDGE", "0.0"))
        self.min_family_stability = float(__import__("os").getenv("OFFLINE_PROMOTION_MIN_FAMILY_STABILITY", "0.45"))
        self.min_stability_group_size = int(float(__import__("os").getenv("OFFLINE_PROMOTION_MIN_STABILITY_GROUP_SIZE", "5")))
        self.metric_epsilon = float(__import__("os").getenv("OFFLINE_PROMOTION_METRIC_EPSILON", "1e-6"))

    def _candidate_dir(self, family: str) -> Path:
        path = self.weights_dir / "_offline_candidates" / family
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _active_family_dir(self, family: str) -> Path:
        path = self.weights_dir / family
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _promote(self, family: str, model_name: str, candidate_path: Path) -> None:
        active_dir = self._active_family_dir(family)
        dest = active_dir / candidate_path.name
        shutil.copy2(candidate_path, dest)

    def _score_binary_artifact(self, artifact_path: Path, evaluator: TaskWalkForwardEvaluator, model_name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
        if not artifact_path.exists():
            return None
        dataset = evaluator._load_dataset(model_name)
        prepared = evaluator._prepare(dataset, model_name)
        if prepared is None:
            return None
        X_all, y_all, _, multiclass = prepared
        if multiclass:
            return None
        bundle = joblib.load(artifact_path)
        model = bundle["model"]
        calibrator = bundle.get("calibrator_model")
        features = [f for f in bundle.get("features", []) if f in X_all.columns]
        if not features:
            return None
        X_eval = X_all[features]
        try:
            probs = _predict_with_saved_feature_order(model, X_eval, proba=True)
            pred_prob = pd.Series(probs[:, 1] if probs.shape[1] > 1 else probs[:, 0], index=X_eval.index)
        except Exception:
            preds = _predict_with_saved_feature_order(model, X_eval, proba=False)
            pred_prob = pd.Series(pd.to_numeric(preds, errors="coerce").fillna(0.0), index=X_eval.index)
        return dataset.loc[X_eval.index].copy(), y_all.loc[X_eval.index].copy(), pred_prob

    def _score_artifact(self, artifact_path: Path, evaluator, model_name: str):
        if not artifact_path.exists():
            return None
        dataset = evaluator._load_dataset(model_name)
        prepared = evaluator._prepare(dataset, model_name)
        if prepared is None:
            return None
        X_all, y_all, _, multiclass = prepared
        bundle = joblib.load(artifact_path)
        model = bundle["model"]
        calibrator = bundle.get("calibrator_model")
        features = [f for f in bundle.get("features", []) if f in X_all.columns]
        if not features:
            return None
        X_eval = X_all[features]
        if multiclass:
            preds = pd.Series(model.predict(X_eval), index=X_eval.index)
            return dataset.loc[X_eval.index].copy(), y_all.loc[X_eval.index].copy(), preds, True
        try:
            probs = _predict_with_saved_feature_order(model, X_eval, proba=True)
            pred_prob = pd.Series(probs[:, 1] if probs.shape[1] > 1 else probs[:, 0], index=X_eval.index)
            pred_prob = _apply_saved_calibrator(calibrator, pred_prob)
        except Exception:
            preds = _predict_with_saved_feature_order(model, X_eval, proba=False)
            pred_prob = pd.Series(pd.to_numeric(pd.Series(preds), errors="coerce").fillna(0.0).to_numpy(), index=X_eval.index)
        return dataset.loc[X_eval.index].copy(), y_all.loc[X_eval.index].copy(), pred_prob, False

    @staticmethod
    def _equity_curve_stats(returns: pd.Series) -> tuple[float, float]:
        values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
        if values.empty:
            return float("nan"), float("nan")
        std = float(values.std(ddof=0))
        sharpe_like = float(values.mean() / std) if std > 0 else float(values.mean())
        equity = values.cumsum()
        running_peak = equity.cummax()
        drawdown = equity - running_peak
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        return sharpe_like, max_drawdown

    @staticmethod
    def _calibration_error(probs: pd.Series, actual: pd.Series) -> float:
        df = pd.DataFrame({"p": pd.to_numeric(probs, errors="coerce"), "y": pd.to_numeric(actual, errors="coerce")}).dropna()
        if len(df) < 10:
            return float("nan")
        df["bin"] = pd.cut(df["p"], bins=10, labels=False, include_lowest=True)
        grouped = df.groupby("bin").agg(p_mean=("p", "mean"), y_mean=("y", "mean"), count=("y", "size"))
        total = float(grouped["count"].sum())
        if total <= 0:
            return float("nan")
        return float(((grouped["count"] / total) * (grouped["p_mean"] - grouped["y_mean"]).abs()).sum())

    def _family_stability(self, frame: pd.DataFrame, edge_col: str, model_name: str) -> float:
        if edge_col not in frame.columns:
            return float("nan")
        if model_name == "entry_edge":
            grouping_candidates = [
                ("technical_regime_bucket",),
                ("market_family", "technical_regime_bucket"),
                ("technical_regime_bucket", "signal_label"),
                ("signal_label",),
                ("market_family",),
            ]
        else:
            grouping_candidates = [
                ("technical_regime_bucket", "signal_label"),
                ("technical_regime_bucket",),
                ("market_family", "technical_regime_bucket"),
                ("signal_label",),
                ("market_family",),
            ]
        group_cols = None
        for candidates in grouping_candidates:
            present = [c for c in candidates if c in frame.columns and frame[c].notna().any()]
            if present:
                group_cols = present
                break
        if group_cols is None:
            return float("nan")
        grouped = frame.groupby(group_cols)[edge_col].agg(["mean", "size"])
        if grouped.empty:
            return float("nan")
        grouped = grouped[grouped["size"] >= self.min_stability_group_size].copy()
        if grouped.empty:
            return float("nan")
        if model_name in {"fill_probability", "slippage_liquidity", "exit_quality"}:
            positive = grouped["mean"] > 0.5
            return float(grouped.loc[positive, "size"].sum() / grouped["size"].sum())
        return float((grouped["mean"] > 0).mean())

    @staticmethod
    def _bounded_return_series(frame: pd.DataFrame, model_name: str, actual: pd.Series) -> pd.Series:
        price_return = pd.to_numeric(frame.get("price_return", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
        price_return = price_return.clip(lower=-1.0, upper=1.0)
        if model_name == "entry_edge":
            return price_return
        if model_name == "fill_probability":
            fill_success = pd.to_numeric(frame.get("fill_success", actual), errors="coerce").fillna(0.0)
            return (price_return * fill_success).clip(lower=-1.0, upper=1.0)
        if model_name == "slippage_liquidity":
            slip = pd.to_numeric(frame.get("slippage_error", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
            return (-slip).clip(lower=-0.05, upper=0.05)
        if model_name == "exit_quality":
            regret = pd.to_numeric(frame.get("exit_regret", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
            return (-regret).clip(lower=-1.0, upper=1.0)
        return price_return

    def _promotion_metrics(self, artifact_path: Path, evaluator: TaskWalkForwardEvaluator, model_name: str) -> PromotionGateMetrics:
        scored = self._score_artifact(artifact_path, evaluator, model_name)
        if scored is None:
            return PromotionGateMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        frame, actual, model_output, multiclass = scored
        family_series = pd.to_numeric(pd.Series(dtype=float), errors="coerce")
        is_weather = False
        if "market_family" in frame.columns:
            family_series = frame["market_family"].astype(str).str.lower()
            is_weather = family_series.str.startswith("weather").any()
        if not is_weather and "weather_contract_resolved_yes" in frame.columns:
            is_weather = pd.to_numeric(frame["weather_contract_resolved_yes"], errors="coerce").notna().any()

        if multiclass:
            correct = (pd.to_numeric(model_output, errors="coerce") == pd.to_numeric(actual, errors="coerce")).astype(float)
            sharpe_like, max_drawdown = self._equity_curve_stats(correct - correct.mean())
            calibration_error = float("nan")
            fill_adjusted_edge = float(correct.mean()) if len(correct) else float("nan")
            family_stability = self._family_stability(frame.assign(_edge=correct), "_edge", model_name)
            return PromotionGateMetrics(
                sample_size=int(len(frame)),
                sharpe_like=sharpe_like,
                max_drawdown=max_drawdown,
                calibration_error=calibration_error,
                fill_adjusted_edge=fill_adjusted_edge,
                family_stability=family_stability,
            )

        if is_weather and model_name == "entry_edge":
            edge_series = pd.to_numeric(frame.get("entry_edge_realized", actual), errors="coerce").fillna(0.0).clip(lower=-1.0, upper=1.0)
        elif is_weather and model_name == "exit_quality":
            resolved = pd.to_numeric(frame.get("weather_contract_resolved_yes", actual), errors="coerce").fillna(0.0)
            market_prob = pd.to_numeric(frame.get("weather_market_probability", 0.5), errors="coerce").fillna(0.5)
            edge_series = (resolved - market_prob).clip(lower=-1.0, upper=1.0)
        else:
            edge_series = self._bounded_return_series(frame, model_name, actual)
        sharpe_like, max_drawdown = self._equity_curve_stats(edge_series)
        calibration_error = self._calibration_error(model_output, actual)
        fill_success = pd.to_numeric(frame.get("fill_success", actual), errors="coerce").fillna(0.0)
        if is_weather and model_name == "entry_edge":
            fill_adjusted_edge = float(edge_series.mean()) if len(edge_series) else float("nan")
            stability_series = pd.to_numeric(frame.get("entry_quality", actual), errors="coerce").fillna(0.0)
        elif model_name == "fill_probability":
            fill_adjusted_edge = float(edge_series.mean()) if len(edge_series) else float("nan")
            stability_series = fill_success
        elif model_name == "slippage_liquidity":
            fill_adjusted_edge = float(edge_series.mean()) if len(edge_series) else float("nan")
            stability_series = pd.to_numeric(frame.get("execution_quality", actual), errors="coerce").fillna(0.0)
        elif model_name == "exit_quality":
            fill_adjusted_edge = float((edge_series * fill_success.where(fill_success.notna(), 1.0)).mean()) if len(edge_series) else float("nan")
            stability_series = pd.to_numeric(frame.get("exit_quality", actual), errors="coerce").fillna(0.0)
        else:
            fill_adjusted_edge = float((edge_series * fill_success.where(fill_success.notna(), 1.0)).mean()) if len(edge_series) else float("nan")
            stability_series = edge_series
        family_stability = self._family_stability(frame.assign(_edge=stability_series), "_edge", model_name)

        return PromotionGateMetrics(
            sample_size=int(len(frame)),
            sharpe_like=sharpe_like,
            max_drawdown=max_drawdown,
            calibration_error=calibration_error,
            fill_adjusted_edge=fill_adjusted_edge,
            family_stability=family_stability,
        )

    def _hard_promotion_gate(
        self,
        model_name: str,
        candidate: PromotionGateMetrics,
        incumbent: PromotionGateMetrics | None,
    ) -> tuple[bool, str]:
        if candidate.sample_size < self.min_sample_size:
            return False, f"sample_size {candidate.sample_size} < {self.min_sample_size}"
        if not np.isnan(candidate.calibration_error) and candidate.calibration_error > self.max_calibration_error:
            return False, f"calibration_error {candidate.calibration_error:.4f} > {self.max_calibration_error:.4f}"
        min_fill_edge = self.min_fill_adjusted_edge
        if model_name == "slippage_liquidity":
            min_fill_edge = min(min_fill_edge, 0.0) - self.metric_epsilon
        if not np.isnan(candidate.fill_adjusted_edge) and candidate.fill_adjusted_edge < min_fill_edge:
            return False, f"fill_adjusted_edge {candidate.fill_adjusted_edge:.4f} < {self.min_fill_adjusted_edge:.4f}"
        if not np.isnan(candidate.family_stability) and candidate.family_stability < self.min_family_stability:
            return False, f"family_stability {candidate.family_stability:.4f} < {self.min_family_stability:.4f}"
        if incumbent is None:
            return True, "no_incumbent_hard_gate_pass"
        if not np.isnan(candidate.sharpe_like) and not np.isnan(incumbent.sharpe_like):
            if candidate.sharpe_like + self.metric_epsilon < incumbent.sharpe_like + self.min_sharpe_delta:
                return False, f"sharpe_like {candidate.sharpe_like:.4f} < incumbent {incumbent.sharpe_like:.4f}"
        if not np.isnan(candidate.max_drawdown) and not np.isnan(incumbent.max_drawdown):
            if candidate.max_drawdown + self.metric_epsilon < incumbent.max_drawdown:
                return False, f"drawdown {candidate.max_drawdown:.4f} worse than incumbent {incumbent.max_drawdown:.4f}"
        if not np.isnan(candidate.calibration_error) and not np.isnan(incumbent.calibration_error):
            if candidate.calibration_error > incumbent.calibration_error + self.metric_epsilon:
                return False, f"calibration_error {candidate.calibration_error:.4f} > incumbent {incumbent.calibration_error:.4f}"
        if not np.isnan(candidate.fill_adjusted_edge) and not np.isnan(incumbent.fill_adjusted_edge):
            if candidate.fill_adjusted_edge + self.metric_epsilon < incumbent.fill_adjusted_edge:
                return False, f"fill_adjusted_edge {candidate.fill_adjusted_edge:.4f} < incumbent {incumbent.fill_adjusted_edge:.4f}"
        if not np.isnan(candidate.family_stability) and not np.isnan(incumbent.family_stability):
            if candidate.family_stability + self.metric_epsilon < incumbent.family_stability:
                return False, f"family_stability {candidate.family_stability:.4f} < incumbent {incumbent.family_stability:.4f}"
        return True, "hard_gate_pass"

    def _task_specific_compare(
        self,
        model_name: str,
        verdict,
        candidate: PromotionGateMetrics,
        incumbent: PromotionGateMetrics | None,
    ) -> tuple[bool, str]:
        if incumbent is None:
            return bool(verdict.promote), verdict.reason
        if model_name != "fill_probability":
            return bool(verdict.promote), verdict.reason

        c_cal = candidate.calibration_error
        i_cal = incumbent.calibration_error
        c_edge = candidate.fill_adjusted_edge
        i_edge = incumbent.fill_adjusted_edge
        c_stability = candidate.family_stability
        i_stability = incumbent.family_stability

        better_calibration = (not np.isnan(c_cal) and not np.isnan(i_cal) and c_cal <= i_cal + self.metric_epsilon)
        better_edge = (not np.isnan(c_edge) and not np.isnan(i_edge) and c_edge + self.metric_epsilon >= i_edge)
        better_stability = (not np.isnan(c_stability) and not np.isnan(i_stability) and c_stability + self.metric_epsilon >= i_stability)

        if better_calibration and better_edge and better_stability:
            return True, (
                "fill_probability value compare: "
                f"calibration_error {c_cal:.4f} <= {i_cal:.4f}, "
                f"fill_adjusted_edge {c_edge:.4f} >= {i_edge:.4f}, "
                f"family_stability {c_stability:.4f} >= {i_stability:.4f}"
            )
        return bool(verdict.promote), verdict.reason

    def run(self) -> list[OfflinePromotionResult]:
        from task_model_suite import _ALL_MODEL_NAMES, TaskModelSuite
        from walk_forward_evaluator import TaskWalkForwardEvaluator

        build_family_datasets(shared_logs_dir=self.logs_dir, shared_weights_dir=self.weights_dir)
        results: list[OfflinePromotionResult] = []

        for context in list_brain_contexts(shared_logs_dir=self.logs_dir, shared_weights_dir=self.weights_dir):
            family = context.market_family
            suite = TaskModelSuite(family, logs_dir=str(self.logs_dir), weights_dir=str(self._candidate_dir(family)))
            evaluator = TaskWalkForwardEvaluator(family, logs_dir=str(self.logs_dir))
            suite.train_all()

            for model_name in sorted(_ALL_MODEL_NAMES):
                candidate_path = self._candidate_dir(family) / family / f"task_{model_name}.joblib"
                incumbent_path = self._active_family_dir(family) / f"task_{model_name}.joblib"
                verdict = evaluator.compare(candidate_path, incumbent_path, model_name)
                candidate_metrics = self._promotion_metrics(candidate_path, evaluator, model_name)
                incumbent_metrics = self._promotion_metrics(incumbent_path, evaluator, model_name) if incumbent_path.exists() else None
                hard_gate_passed, hard_gate_reason = self._hard_promotion_gate(model_name, candidate_metrics, incumbent_metrics)
                compare_passed, compare_reason = self._task_specific_compare(model_name, verdict, candidate_metrics, incumbent_metrics)
                promote = bool(compare_passed and hard_gate_passed and candidate_path.exists())
                if promote:
                    self._promote(family, model_name, candidate_path)
                results.append(
                    OfflinePromotionResult(
                        family=family,
                        model_name=model_name,
                        promoted=promote,
                        reason=compare_reason if promote else f"{compare_reason}; {hard_gate_reason}",
                        candidate_path=str(candidate_path),
                        incumbent_path=str(incumbent_path),
                        candidate_auc=verdict.candidate_auc,
                        incumbent_auc=verdict.incumbent_auc,
                        candidate_metrics=asdict(candidate_metrics),
                        incumbent_metrics=asdict(incumbent_metrics) if incumbent_metrics is not None else {},
                        hard_gate_passed=hard_gate_passed,
                    )
                )

        payload = {
            "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "results": [r.__dict__ for r in results],
        }
        self.report_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.signal_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Offline learning loop wrote %s and %s", self.report_file, self.signal_file)
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    OfflineLearningLoop().run()
