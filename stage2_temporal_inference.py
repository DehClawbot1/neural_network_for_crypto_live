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
            feat_cols = list(clf_saved["features"])
            X = out.reindex(columns=feat_cols, fill_value=0.0)
            if not X.empty:
                out["temporal_p_tp_before_sl"] = clf_saved["model"].predict_proba(X)[:, 1]
        if reg_saved is not None:
            feat_cols = list(reg_saved["features"])
            X = out.reindex(columns=feat_cols, fill_value=0.0)
            if not X.empty:
                out["temporal_expected_return"] = reg_saved["model"].predict(X)

        return out

