from trade_quality import build_quality_context, infer_technical_regime_bucket


def test_infer_technical_regime_bucket_preserves_explicit_bucket():
    assert infer_technical_regime_bucket({"technical_regime_bucket": "bull_trending_impulse_calm"}) == "bull_trending_impulse_calm"


def test_infer_technical_regime_bucket_uses_richer_runtime_snapshot_fields():
    bucket = infer_technical_regime_bucket(
        {
            "btc_trend_bias": "LONG",
            "alligator_alignment": "BULLISH",
            "adx_value": 24.0,
            "adx_threshold": 18.0,
            "market_structure": "BULLISH",
            "trend_score": 0.82,
            "btc_momentum_regime": "BULLISH",
            "btc_momentum_confluence": 0.72,
            "btc_volatility_regime": "LOW",
            "btc_volatility_regime_score": 0.18,
        }
    )

    assert bucket == "bull_trending_impulse_calm"


def test_build_quality_context_derives_macro_bucket_when_intraday_bias_is_neutral():
    context = build_quality_context(
        {
            "market": "Will Bitcoin rise today?",
            "market_slug": "bitcoin-rise-today",
            "market_structure": "BEARISH",
            "trend_score": 0.18,
            "btc_momentum_regime": "BEARISH",
            "btc_momentum_confluence": 0.61,
            "btc_volatility_regime": "HIGH",
            "btc_volatility_regime_score": 0.74,
        }
    )

    assert context["technical_regime_bucket"] == "bear_macro_impulse_volatile"
