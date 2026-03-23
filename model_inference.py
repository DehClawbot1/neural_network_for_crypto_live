from pathlib import Path

import joblib
import pandas as pd


class ModelInference:
    """
    Load trained supervised models and emit inference-oriented outputs.
    """

    def __init__(self, weights_dir="weights"):
        self.weights_dir = Path(weights_dir)
        self.classifier_file = self.weights_dir / "tp_classifier.joblib"
        self.regressor_file = self.weights_dir / "return_regressor.joblib"

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
            clf = clf_saved["model"]
            feat_cols = [c for c in clf_saved["features"] if c in out.columns]
            if feat_cols:
                probs = clf.predict_proba(out[feat_cols])[:, 1]
                out["p_tp_before_sl"] = probs

        if reg_saved is not None:
            reg = reg_saved["model"]
            feat_cols = [c for c in reg_saved["features"] if c in out.columns]
            if feat_cols:
                preds = reg.predict(out[feat_cols])
                out["expected_return"] = preds

        out["p_tp_before_sl"] = out.get("p_tp_before_sl", 0.0)
        out["expected_return"] = out.get("expected_return", 0.0)
        out["edge_score"] = out["p_tp_before_sl"].astype(float) * out["expected_return"].astype(float)
        return out

