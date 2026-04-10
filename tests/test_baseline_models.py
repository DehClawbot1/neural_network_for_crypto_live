import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline_models import BaselineModels, _curated_standardized_features, _kde_feature_subset, _load_sklearn_baseline_components

_sklearn_available = _load_sklearn_baseline_components() is not None


class TestBaselineModels:
    def setup_method(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.logs_dir = Path(self.test_dir.name)

    def teardown_method(self):
        self.test_dir.cleanup()

    def _make_dataset(self, n=30):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "trader_win_rate": rng.rand(n),
            "normalized_trade_size": rng.rand(n),
            "trend_score": rng.rand(n),
            "btc_volatility_regime_score": rng.rand(n),
            "btc_momentum_confluence": rng.randn(n),
            "btc_market_regime_score": rng.rand(n),
            "sentiment_score": rng.randn(n),
            "liquidity_score": rng.rand(n),
            "tp_before_sl_60m": rng.randint(0, 2, n),
        })
        return df

    @pytest.mark.skipif(not _sklearn_available, reason="sklearn import fails in this environment")
    def test_train_returns_metrics(self):
        df = self._make_dataset(30)
        df.to_csv(self.logs_dir / "contract_targets.csv", index=False)
        model = BaselineModels(logs_dir=str(self.logs_dir))
        result = model.train()
        assert not result.empty
        assert "model_kind" in result.columns
        assert set(result["model_kind"]) == {"lda", "gaussian_nb"}

    def test_empty_dataset(self):
        model = BaselineModels(logs_dir=str(self.logs_dir))
        result = model.train()
        assert result.empty

    def test_curated_features(self):
        df = self._make_dataset(10)
        features = _curated_standardized_features(df)
        assert len(features) > 0
        assert "trader_win_rate" in features

    def test_kde_subset_limited(self):
        df = self._make_dataset(10)
        features = _kde_feature_subset(df)
        assert len(features) <= 8
