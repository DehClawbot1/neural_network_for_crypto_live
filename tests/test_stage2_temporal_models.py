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
                "entry_price": [0.5, 0.6, 0.7],
                "volume_score": [1000, 2000, 3000],
                "liquidity_score": [500, 1500, 2500],
            }
        )
        feature_cols = ["entry_price", "volume_score", "liquidity_score"]

        stationarized = self.model_manager._stationarize_features(df, feature_cols)

        assert stationarized["entry_price"].iloc[0] == np.log1p(0.5)
        assert round(float(stationarized["volume_score"].mean()), 5) == 0.0
        assert round(float(stationarized["liquidity_score"].mean()), 5) == 0.0

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
