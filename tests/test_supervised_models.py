import joblib
import pandas as pd
import pytest

from supervised_models import SupervisedModels, _load_sklearn_supervised

try:
    _SKLEARN_AVAILABLE = _load_sklearn_supervised() is not None
except Exception:
    _SKLEARN_AVAILABLE = False


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn import fails in this environment")
def test_supervised_models_store_sparse_regularization_metadata(tmp_path):
    logs_dir = tmp_path / "logs"
    weights_dir = tmp_path / "weights"
    logs_dir.mkdir()
    weights_dir.mkdir()

    df = pd.DataFrame(
        [
            {
                "timestamp": f"2026-04-10T00:{idx:02d}:00Z",
                "current_price": 0.40 + (idx * 0.01),
                "spread": 0.02 + (0.001 * idx),
                "liquidity_score": 0.20 + (0.03 * idx),
                "volume_score": 0.30 + (0.02 * idx),
                "btc_live_index_price": 68000 + idx,
                "reddit_sentiment": (-0.2 if idx % 2 == 0 else 0.2),
                "twitter_sentiment": (0.1 if idx % 2 == 0 else -0.1),
                "open_positions_unrealized_pnl_pct_total": -0.03 + (0.01 * idx),
                "tp_before_sl_60m": int(idx % 2 == 0),
                "forward_return_15m": -0.04 + (0.02 * idx),
            }
            for idx in range(8)
        ]
    )
    df.to_csv(logs_dir / "contract_targets.csv", index=False)

    usable = SupervisedModels(logs_dir=logs_dir, weights_dir=weights_dir).train()

    assert usable is not None
    clf_saved = joblib.load(weights_dir / "tp_classifier.joblib")
    reg_saved = joblib.load(weights_dir / "return_regressor.joblib")
    assert clf_saved["regularization"] in {"l1", "none"}
    assert clf_saved["model_kind"] in {"logistic_l1", "random_forest_fallback"}
    assert reg_saved["regularization"] in {"l1", "none"}
    assert reg_saved["model_kind"] in {"lasso", "random_forest_fallback"}


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn import fails in this environment")
def test_supervised_models_handles_duplicate_feature_names_in_catalog(tmp_path):
    logs_dir = tmp_path / "logs"
    weights_dir = tmp_path / "weights"
    logs_dir.mkdir()
    weights_dir.mkdir()

    df = pd.DataFrame(
        [
            {
                "timestamp": f"2026-04-10T00:{idx:02d}:00Z",
                "current_price": 0.41 + (idx * 0.01),
                "spread": 0.02 + (0.001 * idx),
                "time_left": 0.6 + (0.01 * idx),
                "liquidity_score": 0.20 + (0.02 * idx),
                "volume_score": 0.30 + (0.03 * idx),
                "market_structure_score": 0.25 + (0.01 * idx),
                "tp_before_sl_60m": int(idx % 2 == 0),
                "forward_return_15m": -0.03 + (0.01 * idx),
            }
            for idx in range(10)
        ]
    )
    df.to_csv(logs_dir / "contract_targets.csv", index=False)

    usable = SupervisedModels(logs_dir=logs_dir, weights_dir=weights_dir).train()

    assert usable is not None
    assert len(usable) == len(set(usable))
