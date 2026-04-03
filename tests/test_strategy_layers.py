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
        }
    )

    assert result["allow"] is True
    assert result["ta_entry_ready"] is True
    assert result["ta_bias_conflicts"] is False


def test_entry_rule_vetoes_trade_when_trend_bias_conflicts():
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

    assert result["allow"] is False
    assert result["macro_veto"] is True
    assert result["ta_bias_conflicts"] is True
