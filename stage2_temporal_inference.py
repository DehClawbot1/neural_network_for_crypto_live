from pathlib import Path

import joblib
import pandas as pd


class Stage2TemporalInference:
    def __init__(self, weights_dir="weights"):
        self.weights_dir = Path(weights_dir)
        self.classifier_file = self.weights_dir / "stage2_temporal_classifier.joblib"
        self.regressor_file = self.weights_dir / "stage2_temporal_regressor.joblib"

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
            feat_cols = [c for c in clf_saved["features"] if c in out.columns]
            if feat_cols:
                out["temporal_p_tp_before_sl"] = clf_saved["model"].predict_proba(out[feat_cols])[:, 1]
        if reg_saved is not None:
            feat_cols = [c for c in reg_saved["features"] if c in out.columns]
            if feat_cols:
                out["temporal_expected_return"] = reg_saved["model"].predict(out[feat_cols])

        return out
