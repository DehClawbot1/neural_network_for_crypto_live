import pandas as pd

from btc_regime_router import apply_regime_model_blend, classify_btc_regime_row


def test_classify_btc_regime_row_identifies_trend_regime():
    regime = classify_btc_regime_row(
        {
            "market_structure": "BULLISH",
            "trend_score": 0.87,
            "btc_trend_bias": "LONG",
            "btc_live_bias": "LONG",
            "btc_trend_confluence": 0.82,
            "btc_momentum_confluence": 0.74,
            "btc_live_confluence": 0.71,
            "btc_live_volatility_proxy": 0.28,
            "btc_volatility_regime_score": 0.30,
            "btc_live_mark_index_basis_bps": 6.0,
            "btc_live_source_divergence_bps": 4.0,
        }
    )

    assert regime["btc_market_regime_label"] == "trend"
    assert regime["btc_market_regime_is_trend"] == 1
    assert regime["btc_market_regime_weight_stage1"] > regime["btc_market_regime_weight_legacy"]


def test_apply_regime_model_blend_uses_regime_weights_and_preserves_outputs():
    df = pd.DataFrame(
        [
            {
                "market_structure": "MIXED",
                "trend_score": 0.52,
                "btc_trend_bias": "LONG",
                "btc_live_bias": "SHORT",
                "btc_trend_confluence": 0.24,
                "btc_momentum_confluence": 0.31,
                "btc_live_confluence": 0.28,
                "btc_live_volatility_proxy": 0.82,
                "btc_volatility_regime_score": 0.76,
                "btc_live_mark_index_basis_bps": 31.0,
                "btc_live_source_divergence_bps": 29.0,
                "legacy_p_tp_before_sl": 0.44,
                "legacy_expected_return": 0.012,
                "legacy_edge_score": 0.00528,
                "stage1_p_tp_before_sl": 0.63,
                "stage1_expected_return": 0.031,
                "stage1_edge_score": 0.01953,
                "stage1_lower_confidence_bound": 0.020,
                "stage1_return_std": 0.011,
                "temporal_p_tp_before_sl": 0.70,
                "temporal_expected_return": 0.042,
            }
        ]
    )

    blended = apply_regime_model_blend(df)
    row = blended.iloc[0]

    assert row["btc_market_regime_label"] == "chaotic"
    assert row["btc_market_regime_weight_stage2"] > row["btc_market_regime_weight_stage1"]
    assert 0.44 <= float(row["p_tp_before_sl"]) <= 0.70
    assert float(row["expected_return"]) > 0.0
    assert float(row["regime_blended_edge_score"]) > 0.0
