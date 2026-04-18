from offline_learning_loop import OfflineLearningLoop, PromotionGateMetrics
import pandas as pd


def test_hard_promotion_gate_blocks_small_lucky_candidate():
    loop = OfflineLearningLoop()
    candidate = PromotionGateMetrics(
        sample_size=12,
        sharpe_like=2.0,
        max_drawdown=-0.1,
        calibration_error=0.05,
        fill_adjusted_edge=0.03,
        family_stability=0.8,
    )
    incumbent = PromotionGateMetrics(
        sample_size=100,
        sharpe_like=0.5,
        max_drawdown=-0.2,
        calibration_error=0.10,
        fill_adjusted_edge=0.01,
        family_stability=0.7,
    )

    passed, reason = loop._hard_promotion_gate("entry_edge", candidate, incumbent)

    assert passed is False
    assert "sample_size" in reason


def test_hard_promotion_gate_blocks_worse_drawdown_or_calibration():
    loop = OfflineLearningLoop()
    candidate = PromotionGateMetrics(
        sample_size=80,
        sharpe_like=0.9,
        max_drawdown=-0.5,
        calibration_error=0.30,
        fill_adjusted_edge=0.03,
        family_stability=0.8,
    )
    incumbent = PromotionGateMetrics(
        sample_size=120,
        sharpe_like=0.7,
        max_drawdown=-0.2,
        calibration_error=0.10,
        fill_adjusted_edge=0.02,
        family_stability=0.7,
    )

    passed, reason = loop._hard_promotion_gate("entry_edge", candidate, incumbent)

    assert passed is False
    assert "calibration_error" in reason or "drawdown" in reason


def test_hard_promotion_gate_passes_stronger_stable_candidate():
    loop = OfflineLearningLoop()
    candidate = PromotionGateMetrics(
        sample_size=120,
        sharpe_like=1.1,
        max_drawdown=-0.1,
        calibration_error=0.06,
        fill_adjusted_edge=0.04,
        family_stability=0.9,
    )
    incumbent = PromotionGateMetrics(
        sample_size=150,
        sharpe_like=0.7,
        max_drawdown=-0.2,
        calibration_error=0.10,
        fill_adjusted_edge=0.01,
        family_stability=0.6,
    )

    passed, reason = loop._hard_promotion_gate("entry_edge", candidate, incumbent)

    assert passed is True
    assert reason == "hard_gate_pass"


def test_fill_probability_compare_prefers_calibrated_execution_value():
    from types import SimpleNamespace

    loop = OfflineLearningLoop()
    verdict = SimpleNamespace(promote=False, reason="auc: 0.46 < 0.47")
    candidate = PromotionGateMetrics(
        sample_size=100,
        sharpe_like=0.1,
        max_drawdown=-0.1,
        calibration_error=0.04,
        fill_adjusted_edge=0.02,
        family_stability=0.9,
    )
    incumbent = PromotionGateMetrics(
        sample_size=100,
        sharpe_like=0.1,
        max_drawdown=-0.1,
        calibration_error=0.05,
        fill_adjusted_edge=0.01,
        family_stability=0.8,
    )

    promote, reason = loop._task_specific_compare("fill_probability", verdict, candidate, incumbent)

    assert promote is True
    assert "value compare" in reason


def test_entry_edge_stability_prefers_regime_first_grouping():
    loop = OfflineLearningLoop()
    frame = pd.DataFrame(
        [
            {"market_family": "btc_other", "technical_regime_bucket": "neutral", "signal_label": "LEGACY", "_edge": 0.06},
            {"market_family": "btc_other", "technical_regime_bucket": "neutral", "signal_label": "LEGACY", "_edge": 0.05},
            {"market_family": "btc_other", "technical_regime_bucket": "neutral", "signal_label": "IGNORE", "_edge": -0.05},
            {"market_family": "btc_other", "technical_regime_bucket": "neutral", "signal_label": "IGNORE", "_edge": -0.04},
            {"market_family": "btc_other", "technical_regime_bucket": "neutral", "signal_label": "LEGACY", "_edge": 0.05},
            {"market_family": "btc_other", "technical_regime_bucket": "neutral", "signal_label": "IGNORE", "_edge": -0.03},
        ]
    )

    stability = loop._family_stability(frame, "_edge", "entry_edge")

    assert abs(stability - 1.0) < 1e-9


def test_slippage_liquidity_stability_uses_execution_quality_support():
    loop = OfflineLearningLoop()
    frame = pd.DataFrame(
        [
            {"technical_regime_bucket": "neutral", "signal_label": "A", "_edge": 0.0, "execution_quality": 1},
            {"technical_regime_bucket": "neutral", "signal_label": "A", "_edge": 0.0, "execution_quality": 1},
            {"technical_regime_bucket": "neutral", "signal_label": "A", "_edge": 0.0, "execution_quality": 1},
            {"technical_regime_bucket": "neutral", "signal_label": "A", "_edge": 0.0, "execution_quality": 1},
            {"technical_regime_bucket": "neutral", "signal_label": "A", "_edge": 0.0, "execution_quality": 1},
            {"technical_regime_bucket": "impulse", "signal_label": "B", "_edge": -0.01, "execution_quality": 0},
        ]
    )

    stability = loop._family_stability(frame.assign(_edge=frame["execution_quality"]), "_edge", "slippage_liquidity")

    assert abs(stability - 1.0) < 1e-9


def test_exit_quality_stability_uses_exit_quality_support():
    loop = OfflineLearningLoop()
    frame = pd.DataFrame(
        [
            {"technical_regime_bucket": "trend", "signal_label": "A", "_edge": 1, "exit_quality": 1},
            {"technical_regime_bucket": "trend", "signal_label": "A", "_edge": 1, "exit_quality": 1},
            {"technical_regime_bucket": "trend", "signal_label": "A", "_edge": 1, "exit_quality": 1},
            {"technical_regime_bucket": "trend", "signal_label": "A", "_edge": 1, "exit_quality": 1},
            {"technical_regime_bucket": "trend", "signal_label": "A", "_edge": 1, "exit_quality": 1},
            {"technical_regime_bucket": "neutral", "signal_label": "B", "_edge": 0, "exit_quality": 0},
        ]
    )

    stability = loop._family_stability(frame.assign(_edge=frame["exit_quality"]), "_edge", "exit_quality")

    assert abs(stability - 1.0) < 1e-9


def test_family_stability_prefers_regime_level_grouping():
    loop = OfflineLearningLoop()
    frame = pd.DataFrame(
        (
            [{"market_family": "btc_other", "technical_regime_bucket": "trend", "signal_label": "A", "_edge": 0.10} for _ in range(5)]
            + [{"market_family": "btc_other", "technical_regime_bucket": "neutral", "signal_label": "B", "_edge": -0.10} for _ in range(5)]
            + [{"market_family": "btc_other", "technical_regime_bucket": "impulse", "signal_label": "A", "_edge": 0.05} for _ in range(5)]
        )
    )

    stability = loop._family_stability(frame, "_edge", "entry_edge")
    assert abs(stability - (2 / 3)) < 1e-9


def test_fill_probability_stability_is_support_weighted():
    loop = OfflineLearningLoop()
    frame = pd.DataFrame(
        [
            {"technical_regime_bucket": "trend", "signal_label": "A", "_edge": 0.9},
            {"technical_regime_bucket": "trend", "signal_label": "A", "_edge": 0.8},
            {"technical_regime_bucket": "trend", "signal_label": "A", "_edge": 0.7},
            {"technical_regime_bucket": "trend", "signal_label": "A", "_edge": 0.9},
            {"technical_regime_bucket": "trend", "signal_label": "A", "_edge": 0.8},
            {"technical_regime_bucket": "bad", "signal_label": "B", "_edge": 0.0},
        ]
    )

    stability = loop._family_stability(frame, "_edge", "fill_probability")

    assert abs(stability - 1.0) < 1e-9
