from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ModelInference:
    """
    Load trained supervised models and emit inference-oriented outputs.

    Key safety fixes:
    - preserve the exact training feature order
    - create missing training columns at inference time instead of silently dropping them
    - coerce inference data to numeric and fill NaNs
    - support optional preprocessor/scaler pipelines stored alongside the model
    - never emit positive-looking outputs if model inference itself failed
    """

    def __init__(self, weights_dir="weights"):
        self.weights_dir = Path(weights_dir)
        self.classifier_file = self.weights_dir / "tp_classifier.joblib"
        self.regressor_file = self.weights_dir / "return_regressor.joblib"

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
        for col in feature_names:
            if col not in work.columns:
                work[col] = 0.0

        x = work[feature_names].copy()
        for col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
        out["edge_score"] = 0.0

        clf_saved = self._load(self.classifier_file)
        reg_saved = self._load(self.regressor_file)

        if clf_saved is not None:
            try:
                clf = clf_saved["model"] if isinstance(clf_saved, dict) and "model" in clf_saved else clf_saved
                x_clf = self._prepare_matrix(clf_saved if isinstance(clf_saved, dict) else {"features": getattr(clf, "feature_names_in_", [])}, out)
                if x_clf is not None:
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(x_clf)[:, 1]
                    else:
                        raw = clf.decision_function(x_clf)
                        probs = 1.0 / (1.0 + np.exp(-raw))
                    out["p_tp_before_sl"] = np.clip(pd.Series(probs, index=out.index).astype(float), 0.0, 1.0)
            except Exception:
                out["p_tp_before_sl"] = 0.0

        if reg_saved is not None:
            try:
                reg = reg_saved["model"] if isinstance(reg_saved, dict) and "model" in reg_saved else reg_saved
                x_reg = self._prepare_matrix(reg_saved if isinstance(reg_saved, dict) else {"features": getattr(reg, "feature_names_in_", [])}, out)
                if x_reg is not None:
                    preds = reg.predict(x_reg)
                    out["expected_return"] = pd.Series(preds, index=out.index).astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            except Exception:
                out["expected_return"] = 0.0

        out["edge_score"] = out["p_tp_before_sl"].astype(float) * out["expected_return"].astype(float)
        return out
