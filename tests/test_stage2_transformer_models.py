import pandas as pd

from stage2_transformer_models import Stage2TransformerModels


class TestStage2TransformerModels:
    def setup_method(self):
        self.transformer_manager = Stage2TransformerModels()

    def test_reshape_sequence_logic(self):
        df = pd.DataFrame(
            {
                "price_lag_0": [0.10, 0.20],
                "price_lag_1": [0.11, 0.21],
                "price_lag_2": [0.12, 0.22],
                "vol_lag_0": [10, 20],
                "vol_lag_1": [11, 21],
                "vol_lag_2": [12, 22],
            }
        )
        feature_cols = df.columns.tolist()

        reshaped = self.transformer_manager._reshape_sequence(df, feature_cols)

        assert reshaped.shape == (2, 3, 2)
        assert reshaped[0, 1, 0] == 0.11

    def test_reshape_sequence_empty(self):
        df = pd.DataFrame({"random_col": [1, 2, 3]})
        reshaped = self.transformer_manager._reshape_sequence(df, ["random_col"])
        assert reshaped is None
