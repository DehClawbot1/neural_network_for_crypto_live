import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from stage2_temporal_models import Stage2TemporalModels


class TestStage2TemporalModels:
    def setup_method(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.logs_dir = Path(self.test_dir.name)
        self.weights_dir = self.logs_dir / "weights"
        self.model_manager = Stage2TemporalModels(
            logs_dir=str(self.logs_dir),
            weights_dir=str(self.weights_dir),
        )

    def teardown_method(self):
        self.test_dir.cleanup()

    def test_stationarize_features(self):
        df = pd.DataFrame(
            {
                "btc_live_price": [50000, 51000, 52000],
                "spread": [0.01, 0.05, 0.20],
                "trend_score": [0.3, 0.5, 0.7],
            }
        )
        feature_cols = ["btc_live_price", "spread", "trend_score"]

        stationarized = self.model_manager._stationarize_features(df, feature_cols)

        # btc_live_price is log_scale → log1p applied
        assert stationarized["btc_live_price"].iloc[0] == np.log1p(50000)
        # spread is robust_scale → median/IQR normalised (median centred)
        assert abs(float(stationarized["spread"].median())) < 1e-6
        # trend_score is clip01 → untouched by stationarize
        assert stationarized["trend_score"].iloc[0] == 0.3

    def test_balance_binary_frame(self):
        df = pd.DataFrame(
            {
                "feature": np.random.rand(10),
                "tp_before_sl_60m": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        )

        balanced = self.model_manager._balance_binary_frame(df, "tp_before_sl_60m")

        counts = balanced["tp_before_sl_60m"].value_counts()
        assert counts[0] == counts[1]
        assert len(balanced) == 18
