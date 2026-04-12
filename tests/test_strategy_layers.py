from strategy_layers import EntryRuleLayer, PredictionLayer


def test_prediction_layer_boosts_score_when_profitability_is_strong():
    score = PredictionLayer.select_signal_score(
        {
            "confidence": 0.35,
            "p_tp_before_sl": 0.35,
            "edge_score": 0.08,
            "entry_ev": 0.09,
            "risk_adjusted_ev": 0.07,
        }
    )

    assert score > 0.35
    assert score <= 1.0


def test_prediction_layer_uses_probability_when_profitability_is_absent():
    score = PredictionLayer.select_signal_score(
        {
            "confidence": 0.35,
            "p_tp_before_sl": 0.58,
            "edge_score": 0.0,
            "entry_ev": 0.0,
        }
    )

    # With no profitability backing, the probability-only score is discounted
    # (profitability-first: probability * 0.70 when no edge/return present).
    assert 0.35 < score < 0.58
    assert score > 0.0


def test_prediction_layer_weather_uses_forecast_edge_and_fair_value_gap():
    components = PredictionLayer.select_signal_components(
        {
            "market_family": "weather_temperature_threshold",
            "confidence": 0.34,
            "forecast_p_hit_interval": 0.66,
            "weather_fair_probability_side": 0.71,
            "weather_market_probability": 0.52,
            "weather_forecast_edge": 0.19,
            "weather_forecast_margin_score": 0.73,
            "weather_forecast_stability_score": 0.81,
            "entry_ev": 0.09,
        }
    )

    assert components["probability_signal"] >= 0.66
    assert components["profitability_signal"] > 0.0
    assert components["market_inefficiency_signal"] > 0.0
    assert components["score"] > 0.60


def test_entry_rule_allows_trade_when_trend_regime_confirms_direction():
    layer = EntryRuleLayer(min_score=0.25, max_spread=0.20, min_liquidity=5, min_liquidity_score=0.05)
    result = layer.evaluate(
        {
            "confidence": 0.34,
            "p_tp_before_sl": 0.55,
            "edge_score": 0.05,
            "expected_return": 0.03,
            "spread": 0.02,
            "liquidity": 100.0,
            "outcome_side": "YES",
            "btc_trend_bias": "LONG",
            "alligator_alignment": "BULLISH",
            "adx_value": 24.0,
            "adx_threshold": 18.0,
            "adx_trending": True,
            "price_above_anchored_vwap": True,
            "price_below_anchored_vwap": False,
            "btc_trend_confluence": 1.0,
            "latest_bullish_fractal": 66550.0,
            "long_fractal_breakout": True,
        }
    )

    assert result["allow"] is True
    assert result["ta_entry_ready"] is True
    assert result["ta_bias_conflicts"] is False


def test_entry_rule_penalizes_trade_until_fractal_breakout_confirms():
    layer = EntryRuleLayer(min_score=0.25, max_spread=0.20, min_liquidity=5, min_liquidity_score=0.05)
    result = layer.evaluate(
        {
            "confidence": 0.71,
            "p_tp_before_sl": 0.65,
            "edge_score": 0.06,
            "expected_return": 0.04,
            "spread": 0.02,
            "liquidity": 100.0,
            "outcome_side": "YES",
            "btc_trend_bias": "LONG",
            "alligator_alignment": "BULLISH",
            "adx_value": 24.0,
            "adx_threshold": 18.0,
            "adx_trending": True,
            "price_above_anchored_vwap": True,
            "price_below_anchored_vwap": False,
            "btc_trend_confluence": 1.0,
            "latest_bullish_fractal": 66550.0,
            "long_fractal_breakout": False,
        }
    )

    assert result["allow"] is True
    assert result["ta_trigger_blocked"] is False
    assert result["ta_entry_ready"] is False
    assert result["score_threshold"] > 0.25


def test_entry_rule_penalizes_trade_when_trend_bias_conflicts():
    layer = EntryRuleLayer(min_score=0.25, max_spread=0.20, min_liquidity=5, min_liquidity_score=0.05)
    result = layer.evaluate(
        {
            "confidence": 0.75,
            "p_tp_before_sl": 0.70,
            "edge_score": 0.08,
            "expected_return": 0.05,
            "spread": 0.02,
            "liquidity": 100.0,
            "outcome_side": "NO",
            "btc_trend_bias": "LONG",
            "alligator_alignment": "BULLISH",
            "adx_value": 31.0,
            "adx_threshold": 18.0,
            "adx_trending": True,
            "price_above_anchored_vwap": True,
            "price_below_anchored_vwap": False,
            "btc_trend_confluence": 1.0,
        }
    )

    assert result["allow"] is True
    assert result["macro_veto"] is False
    assert result["ta_bias_conflicts"] is True
    assert result["score_threshold"] > 0.25


def test_entry_rule_uses_spread_and_liquidity_as_penalties_not_hard_blocks():
    layer = EntryRuleLayer(min_score=0.25, max_spread=0.20, min_liquidity=5, min_liquidity_score=0.05)
    result = layer.evaluate(
        {
            "confidence": 0.95,
            "p_tp_before_sl": 0.80,
            "edge_score": 0.10,
            "expected_return": 0.06,
            "spread": 0.32,
            "liquidity": 1.5,
            "outcome_side": "YES",
        }
    )

    assert result["spread_ok"] is False
    assert result["liquidity_ok"] is False
    assert result["spread_penalty"] > 0.0
    assert result["liquidity_penalty"] > 0.0
    assert result["allow"] is True


def test_entry_rule_tightens_when_open_portfolio_is_underwater():
    layer = EntryRuleLayer(min_score=0.25, max_spread=0.20, min_liquidity=5, min_liquidity_score=0.05)

    baseline = layer.evaluate(
        {
            "confidence": 0.27,
            "p_tp_before_sl": 0.52,
            "edge_score": 0.03,
            "expected_return": 0.01,
            "spread": 0.02,
            "liquidity": 100.0,
            "outcome_side": "YES",
        }
    )
    stressed = layer.evaluate(
        {
            "confidence": 0.27,
            "p_tp_before_sl": 0.52,
            "edge_score": 0.03,
            "expected_return": 0.01,
            "spread": 0.02,
            "liquidity": 100.0,
            "outcome_side": "YES",
            "open_positions_count": 3,
            "open_positions_unrealized_pnl_pct_total": -0.07,
            "open_positions_avg_to_now_price_change_pct_mean": -0.06,
            "open_positions_winner_count": 0,
            "open_positions_loser_count": 3,
        }
    )

    assert baseline["allow"] is True
    assert stressed["allow"] is False
    assert stressed["portfolio_pressure_penalty"] > 0.0
    assert stressed["score_threshold"] > baseline["score_threshold"]


def test_weather_entry_rule_exposes_profitability_first_components():
    layer = EntryRuleLayer(min_score=0.25, max_spread=0.20, min_liquidity=5, min_liquidity_score=0.05)
    result = layer.evaluate(
        {
            "market_family": "weather_temperature_threshold",
            "confidence": 0.33,
            "forecast_p_hit_interval": 0.69,
            "weather_fair_probability_side": 0.74,
            "weather_market_probability": 0.51,
            "weather_forecast_edge": 0.23,
            "weather_forecast_margin_score": 0.65,
            "weather_forecast_stability_score": 0.80,
            "weather_parseable": True,
            "forecast_ready": True,
            "forecast_stale": False,
            "weather_forecast_confirms_direction": True,
            "wallet_state_gate_pass": True,
            "liquidity_score": 0.60,
            "spread": 0.03,
        }
    )

    assert result["allow"] is True
    assert result["decision_probability_signal"] >= 0.69
    assert result["decision_profitability_signal"] > 0.0
    assert result["decision_market_inefficiency_signal"] > 0.0
