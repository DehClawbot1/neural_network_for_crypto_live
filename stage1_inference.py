from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from brain_paths import resolve_brain_context
from return_calibration import calibrate_return_predictions

try:
    from inference_runtime_guard import report_error as _report_inference_error
except Exception:  # pragma: no cover
    def _report_inference_error(*args, **kwargs):
        return None


class Stage1Inference:
    def __init__(self, weights_dir="weights", *, brain_context=None, brain_id=None, market_family=None, shared_logs_dir="logs", shared_weights_dir="weights"):
        if brain_context is None and (brain_id or market_family):
            brain_context = resolve_brain_context(
                market_family,
                brain_id=brain_id,
                shared_logs_dir=shared_logs_dir,
                shared_weights_dir=shared_weights_dir,
            )
        self.weights_dir = Path(brain_context.weights_dir if brain_context is not None else weights_dir)
        self.classifier_file = self.weights_dir / "stage1_tp_classifier.joblib"
        self.regressor_file = self.weights_dir / "stage1_return_regressor.joblib"

    def missing_artifacts(self):
        missing = []
        if not self.classifier_file.exists():
            missing.append({"component": "classifier", "path": str(self.classifier_file)})
        if not self.regressor_file.exists():
            missing.append({"component": "regressor", "path": str(self.regressor_file)})
        return missing

    def _load(self, path):
        if not path.exists():
            return None
        return joblib.load(path)

    def _prepare_matrix(self, saved, frame: pd.DataFrame):
        if saved is None:
            return None
        feature_names = list(saved.get("features", []))
        if not feature_names:
            return None

        work = frame.copy()
        missing = {col: 0.0 for col in feature_names if col not in work.columns}
        if missing:
            work = work.assign(**missing)

        x = work[feature_names].copy()
        for col in x.columns:
            x[col] = (
                pd.to_numeric(x[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

        preprocessor = saved.get("preprocessor") or saved.get("transformer") or saved.get("scaler")
        if preprocessor is not None:
            x = preprocessor.transform(x)
        return x

    def run(self, features_df: pd.DataFrame):
        if features_df is None or features_df.empty:
            return pd.DataFrame()

        out = features_df.copy()
        out["p_tp_before_sl"] = 0.0
        out["expected_return"] = 0.0

        clf_saved = self._load(self.classifier_file)
        reg_saved = self._load(self.regressor_file)

        if clf_saved is not None:
            try:
                clf = clf_saved["model"] if isinstance(clf_saved, dict) and "model" in clf_saved else clf_saved
                x_clf = self._prepare_matrix(
                    clf_saved if isinstance(clf_saved, dict) else {"features": getattr(clf, "feature_names_in_", [])},
                    out,
                )
                if x_clf is not None:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                            category=UserWarning,
                        )
                        if hasattr(clf, "predict_proba"):
                            probs = clf.predict_proba(x_clf)[:, 1]
                        else:
                            raw = clf.decision_function(x_clf)
                            probs = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
                    out["p_tp_before_sl"] = np.clip(pd.Series(probs, index=out.index).astype(float), 0.0, 1.0)
            except Exception as exc:
                _report_inference_error("stage1_inference.classifier", exc, context="p_tp_before_sl_zero_fallback")
                out["p_tp_before_sl"] = 0.0

        if reg_saved is not None:
            try:
                reg = reg_saved["model"] if isinstance(reg_saved, dict) and "model" in reg_saved else reg_saved
                x_reg = self._prepare_matrix(
                    reg_saved if isinstance(reg_saved, dict) else {"features": getattr(reg, "feature_names_in_", [])},
                    out,
                )
                if x_reg is not None:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                            category=UserWarning,
                        )
                        preds = reg.predict(x_reg)
                    calibration = reg_saved.get("return_calibration") if isinstance(reg_saved, dict) else None
                    out["expected_return"] = calibrate_return_predictions(preds, calibration, index=out.index)
            except Exception as exc:
                _report_inference_error("stage1_inference.regressor", exc, context="expected_return_zero_fallback")
                out["expected_return"] = 0.0

        calibration = reg_saved.get("return_calibration") if isinstance(reg_saved, dict) else {}
        uncertainty_floor = float(calibration.get("uncertainty_floor", 0.02) or 0.02)
        out["return_std"] = np.maximum(abs(out["expected_return"].astype(float)) * 0.35, uncertainty_floor)
        out["lower_confidence_bound"] = out["expected_return"].astype(float) - out["return_std"].astype(float)
        out["edge_score"] = out["p_tp_before_sl"].astype(float) * out["lower_confidence_bound"].astype(float)
        return out
