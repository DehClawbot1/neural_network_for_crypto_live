from pathlib import Path

import joblib
import pandas as pd


class Stage1Inference:
    def __init__(self, weights_dir="weights"):
        self.weights_dir = Path(weights_dir)
        self.classifier_file = self.weights_dir / "stage1_tp_classifier.joblib"
        self.regressor_file = self.weights_dir / "stage1_return_regressor.joblib"

    def _load(self, path):
        if not path.exists():
            return None
        return joblib.load(path)

    def run(self, features_df: pd.DataFrame):
        if features_df is None or features_df.empty:
            return pd.DataFrame()

        out = features_df.copy()
        clf_saved = self._load(self.classifier_file)
        reg_saved = self._load(self.regressor_file)

        if clf_saved is not None:
            feat_cols = list(clf_saved["features"])
            X = out.reindex(columns=feat_cols, fill_value=0.0)
            if not X.empty:
                probs = clf_saved["model"].predict_proba(X)[:, 1]
                out["p_tp_before_sl"] = probs
        if reg_saved is not None:
            feat_cols = list(reg_saved["features"])
            X = out.reindex(columns=feat_cols, fill_value=0.0)
            if not X.empty:
                preds = reg_saved["model"].predict(X)
                out["expected_return"] = preds

        if "p_tp_before_sl" not in out.columns:
            out["p_tp_before_sl"] = 0.0
        if "expected_return" not in out.columns:
            out["expected_return"] = 0.0
        out["return_std"] = abs(out["expected_return"].astype(float)) * 0.35
        out["lower_confidence_bound"] = out["expected_return"].astype(float) - out["return_std"].astype(float)
        out["edge_score"] = out["p_tp_before_sl"].astype(float) * out["lower_confidence_bound"].astype(float)
        return out

