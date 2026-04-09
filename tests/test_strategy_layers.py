from strategy_layers import EntryRuleLayer


def test_entry_rule_allows_trade_when_trend_regime_confirms_direction():
    layer = EntryRuleLayer(min_score=0.25, max_spread=0.20, min_liquidity=5, min_liquidity_score=0.05)
    result = layer.evaluate(
        {
            "confidence": 0.34,
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
            "spread": 0.02,
            "liquidity": 100.0,
            "outcome_side": "YES",
        }
    )
    stressed = layer.evaluate(
        {
            "confidence": 0.27,
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
