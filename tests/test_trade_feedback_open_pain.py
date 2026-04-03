from pathlib import Path

import pandas as pd

from signal_engine import SignalEngine
from trade_feedback_learner import TradeFeedbackLearner


def test_open_trade_pain_penalizes_scored_candidates(tmp_path):
    logs_dir = Path(tmp_path)
    positions = pd.DataFrame(
        [
            {
                "status": "OPEN",
                "signal_label": "STRONG PAPER OPPORTUNITY",
                "entry_price": 0.50,
                "current_price": 0.44,
                "max_adverse_excursion_pct": -0.12,
                "max_drawdown_from_peak_pct": 0.08,
                "fast_adverse_move_count": 2,
            }
        ]
    )
    positions.to_csv(logs_dir / "positions.csv", index=False)

    learner = TradeFeedbackLearner(logs_dir=str(logs_dir))
    signal_engine = SignalEngine()
    scored_df = pd.DataFrame(
        [
            {
                "signal_label": "STRONG PAPER OPPORTUNITY",
                "confidence": 0.74,
                "p_tp_before_sl": 0.63,
                "expected_return": 0.03,
                "edge_score": 0.0189,
                "whale_pressure": 0.70,
                "market_structure_score": 0.68,
                "volatility_risk": 0.20,
                "time_decay_score": 0.15,
                "btc_network_activity_score": 0.60,
                "btc_network_stress_score": 0.50,
                "btc_trend_bias": "LONG",
                "outcome_side": "YES",
                "long_fractal_breakout": True,
                "short_fractal_breakout": False,
                "btc_trend_confluence": 1.0,
            }
        ]
    )

    adjusted = learner.apply_to_scored_df(scored_df, signal_engine)

    assert not adjusted.empty
    assert adjusted.iloc[0]["feedback_open_pain_score"] > 0.0
    assert adjusted.iloc[0]["confidence"] < 0.74
    assert "open_pain=" in adjusted.iloc[0]["reason"]


def test_open_trade_pain_tuning_can_be_made_less_aggressive(tmp_path, monkeypatch):
    logs_dir = Path(tmp_path)
    positions = pd.DataFrame(
        [
            {
                "status": "OPEN",
                "signal_label": "STRONG PAPER OPPORTUNITY",
                "entry_price": 0.50,
                "current_price": 0.44,
                "max_adverse_excursion_pct": -0.12,
                "max_drawdown_from_peak_pct": 0.08,
                "fast_adverse_move_count": 2,
            }
        ]
    )
    positions.to_csv(logs_dir / "positions.csv", index=False)

    signal_engine = SignalEngine()
    scored_df = pd.DataFrame(
        [
            {
                "signal_label": "STRONG PAPER OPPORTUNITY",
                "confidence": 0.74,
                "p_tp_before_sl": 0.63,
                "expected_return": 0.03,
                "edge_score": 0.0189,
                "whale_pressure": 0.70,
                "market_structure_score": 0.68,
                "volatility_risk": 0.20,
                "time_decay_score": 0.15,
                "btc_network_activity_score": 0.60,
                "btc_network_stress_score": 0.50,
                "btc_trend_bias": "LONG",
                "outcome_side": "YES",
                "long_fractal_breakout": True,
                "short_fractal_breakout": False,
                "btc_trend_confluence": 1.0,
            }
        ]
    )

    baseline_learner = TradeFeedbackLearner(logs_dir=str(logs_dir))
    baseline = baseline_learner.apply_to_scored_df(scored_df, signal_engine)

    monkeypatch.setenv("OPEN_PAIN_SENSITIVITY", "0.4")
    monkeypatch.setenv("OPEN_PAIN_CONF_PENALTY_MAX", "0.08")
    monkeypatch.setenv("OPEN_PAIN_RET_PENALTY_MAX", "0.10")
    tuned_learner = TradeFeedbackLearner(logs_dir=str(logs_dir))
    tuned = tuned_learner.apply_to_scored_df(scored_df, signal_engine)

    assert tuned.iloc[0]["feedback_open_pain_score"] < baseline.iloc[0]["feedback_open_pain_score"]
    assert tuned.iloc[0]["confidence"] > baseline.iloc[0]["confidence"]


def test_open_trade_pain_handles_missing_optional_columns(tmp_path):
    logs_dir = Path(tmp_path)
    positions = pd.DataFrame(
        [
            {
                "status": "OPEN",
                "signal_label": "STRONG PAPER OPPORTUNITY",
                "entry_price": 0.50,
                "current_price": 0.47,
            }
        ]
    )
    positions.to_csv(logs_dir / "positions.csv", index=False)

    learner = TradeFeedbackLearner(logs_dir=str(logs_dir))
    signal_engine = SignalEngine()
    scored_df = pd.DataFrame(
        [
            {
                "signal_label": "STRONG PAPER OPPORTUNITY",
                "confidence": 0.70,
                "p_tp_before_sl": 0.60,
                "expected_return": 0.02,
                "edge_score": 0.012,
                "whale_pressure": 0.66,
                "market_structure_score": 0.62,
                "volatility_risk": 0.18,
                "time_decay_score": 0.12,
                "btc_network_activity_score": 0.55,
                "btc_network_stress_score": 0.45,
                "btc_trend_bias": "LONG",
                "outcome_side": "YES",
                "long_fractal_breakout": True,
                "short_fractal_breakout": False,
                "btc_trend_confluence": 1.0,
            }
        ]
    )

    adjusted = learner.apply_to_scored_df(scored_df, signal_engine)

    assert not adjusted.empty
    assert "feedback_open_pain_score" in adjusted.columns
    assert adjusted.iloc[0]["feedback_open_pain_score"] >= 0.0
