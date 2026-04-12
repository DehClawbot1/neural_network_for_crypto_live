import math


def _finite_float(value, default=None):
    try:
        num = float(value)
    except Exception:
        return default
    if not math.isfinite(num):
        return default
    return num


def _bounded_positive_metric(value, *, scale: float) -> float:
    metric = _finite_float(value, default=0.0)
    if metric is None or metric <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(metric) * float(scale)))


def _market_family(row: dict) -> str:
    return str(row.get("market_family", "") or "").strip().lower()


def _positive_gap(high_value, low_value) -> float:
    hi = _finite_float(high_value, default=0.0) or 0.0
    lo = _finite_float(low_value, default=0.0) or 0.0
    return max(0.0, hi - lo)


class PredictionLayer:
    """Placeholder interface for model outputs such as expected return or P(TP before SL)."""

    @staticmethod
    def select_signal_components(row: dict) -> dict:
        market_family = _market_family(row)
        confidence = _finite_float(row.get("confidence", 0.0), default=0.0)
        p_tp = _finite_float(
            row.get(
                "ensemble_probability",
                row.get("p_tp_before_sl", row.get("tp_before_sl_prob", 0.0)),
            ),
            default=0.0,
        )
        forecast_probability = _finite_float(
            row.get("weather_fair_probability_side", row.get("forecast_p_hit_interval", 0.0)),
            default=0.0,
        )
        market_probability = _finite_float(
            row.get("weather_market_probability", row.get("current_price", 0.0)),
            default=0.0,
        )

        # Probability signal: model p_tp is primary; confidence is a soft
        # fallback only when no model probability output is available.
        probability_signal = max(p_tp, 0.0)
        if probability_signal <= 0.0:
            probability_signal = max(confidence * 0.60, 0.0)

        profitability_signal = max(
            _bounded_positive_metric(row.get("edge_score", 0.0), scale=8.0),
            _bounded_positive_metric(row.get("hybrid_edge", 0.0), scale=8.0),
            _bounded_positive_metric(row.get("entry_ev", 0.0), scale=4.0),
            _bounded_positive_metric(row.get("risk_adjusted_ev", 0.0), scale=4.0),
            _bounded_positive_metric(row.get("calibrated_edge", 0.0), scale=250.0),
            _bounded_positive_metric(row.get("expected_return", 0.0), scale=4.0),
        )
        market_inefficiency_signal = 0.0

        if market_family.startswith("weather_temperature"):
            probability_signal = max(probability_signal, forecast_probability)
            market_inefficiency_signal = max(
                _bounded_positive_metric(_positive_gap(row.get("weather_fair_probability_side", forecast_probability), market_probability), scale=4.0),
                _bounded_positive_metric(row.get("weather_forecast_edge", 0.0), scale=6.0),
                _bounded_positive_metric(row.get("weather_forecast_margin_score", 0.0), scale=1.0),
                _bounded_positive_metric(row.get("weather_forecast_stability_score", 0.0), scale=1.0),
            )
        else:
            market_inefficiency_signal = max(
                _bounded_positive_metric(row.get("regime_blended_edge_score", 0.0), scale=8.0),
                _bounded_positive_metric(row.get("supervised_edge", 0.0), scale=8.0),
                _bounded_positive_metric(row.get("temporal_edge_score", 0.0), scale=8.0),
                _bounded_positive_metric(row.get("stage1_edge_score", 0.0), scale=8.0),
                _bounded_positive_metric(row.get("legacy_edge_score", 0.0), scale=8.0),
            )
        profitability_signal = max(profitability_signal, market_inefficiency_signal)

        # Profitability-first blend: profitability drives 70 % of the score,
        # probability (model p_tp / forecast) provides 30 % confirmation.
        if profitability_signal > 0.0 and probability_signal > 0.0:
            score = min(1.0, (profitability_signal * 0.70) + (probability_signal * 0.30))
        elif profitability_signal > 0.0:
            score = profitability_signal
        elif probability_signal > 0.0:
            score = probability_signal * 0.70  # no profitability backing → discount
        else:
            score = 0.0

        return {
            "score": float(max(0.0, min(1.0, score))),
            "probability_signal": float(max(0.0, min(1.0, probability_signal))),
            "profitability_signal": float(max(0.0, min(1.0, profitability_signal))),
            "market_inefficiency_signal": float(max(0.0, min(1.0, market_inefficiency_signal))),
            "market_family": market_family or "btc",
        }

    @staticmethod
    def select_signal_score(row: dict) -> float:
        return PredictionLayer.select_signal_components(row)["score"]


class EntryRuleLayer:
    """Entry filter separate from raw model prediction."""

    def __init__(self, min_score=0.25, max_spread=0.20, min_liquidity=5, min_liquidity_score=0.05):
        self.min_score = min_score
        self.max_spread = max_spread
        self.min_liquidity = min_liquidity
        self.min_liquidity_score = min_liquidity_score

    @staticmethod
    def _target_direction(row: dict) -> str:
        side = str(row.get("outcome_side", row.get("side", "UNKNOWN"))).strip().upper()
        if side in {"YES", "UP", "LONG", "BULLISH"}:
            return "LONG"
        if side in {"NO", "DOWN", "SHORT", "BEARISH"}:
            return "SHORT"
        return "NEUTRAL"

    def _evaluate_weather(self, row: dict) -> dict:
        decision_components = PredictionLayer.select_signal_components(row)
        score = decision_components["score"]
        score_relax = max(0.0, _finite_float(row.get("entry_score_relax", 0.0), default=0.0) or 0.0)
        dynamic_min_score = max(0.02, min(0.95, max(self.min_score, _finite_float(row.get("weather_min_score"), default=self.min_score) or self.min_score) - score_relax))
        spread = _finite_float(row.get("spread"), default=0.0) or 0.0
        spread_threshold = max(0.01, _finite_float(row.get("weather_max_spread"), default=self.max_spread) or self.max_spread)
        spread_ok = spread <= spread_threshold
        spread_penalty = 0.0 if spread_ok else min(0.12, max(0.0, spread - spread_threshold))

        liquidity_score = _finite_float(row.get("liquidity_score"), default=0.0) or 0.0
        liquidity_threshold = max(0.0, _finite_float(row.get("weather_min_liquidity_score"), default=self.min_liquidity_score) or self.min_liquidity_score)
        liquidity_metric = "liquidity_score"
        liquidity_ok = liquidity_score >= liquidity_threshold
        liquidity_penalty = 0.0 if liquidity_ok else min(0.10, liquidity_threshold - liquidity_score)

        wallet_gate_pass = bool(row.get("wallet_state_gate_pass", True))
        weather_parseable = bool(row.get("weather_parseable", False))
        forecast_ready = bool(row.get("forecast_ready", False))
        forecast_stale = bool(row.get("forecast_stale", True))
        forecast_confirms_direction = bool(row.get("weather_forecast_confirms_direction", False))
        analytics_only = bool(row.get("analytics_only", False))
        forecast_edge = _finite_float(row.get("weather_forecast_edge"), default=0.0) or 0.0
        min_forecast_edge = _finite_float(row.get("weather_min_forecast_edge"), default=0.08) or 0.08
        weather_cluster_conflict = bool(row.get("weather_threshold_conflict", False))

        macro_veto = (
            analytics_only
            or (not weather_parseable)
            or (not wallet_gate_pass)
            or (not forecast_ready)
            or forecast_stale
            or (not forecast_confirms_direction)
            or weather_cluster_conflict
        )
        score_threshold = min(
            0.98,
            dynamic_min_score + spread_penalty + liquidity_penalty + (0.03 if forecast_edge < min_forecast_edge else 0.0),
        )
        score_ok = score >= score_threshold and forecast_edge >= min_forecast_edge
        allow = score_ok and not macro_veto
        return {
            "allow": allow,
            "score": score,
            "score_threshold": score_threshold,
            "score_ok": score_ok,
            "spread": spread,
            "spread_threshold": spread_threshold,
            "spread_ok": spread_ok,
            "spread_penalty": spread_penalty,
            "liquidity_value": liquidity_score,
            "liquidity_threshold": liquidity_threshold,
            "liquidity_metric": liquidity_metric,
            "liquidity_ok": liquidity_ok,
            "liquidity_penalty": liquidity_penalty,
            "macro_veto": macro_veto,
            "weather_parseable": weather_parseable,
            "forecast_ready": forecast_ready,
            "forecast_stale": forecast_stale,
            "forecast_edge": forecast_edge,
            "min_forecast_edge": min_forecast_edge,
            "wallet_state_gate_pass": wallet_gate_pass,
            "analytics_only": analytics_only,
            "weather_cluster_conflict": weather_cluster_conflict,
            "portfolio_pressure_penalty": 0.0,
            "decision_probability_signal": decision_components["probability_signal"],
            "decision_profitability_signal": decision_components["profitability_signal"],
            "decision_market_inefficiency_signal": decision_components["market_inefficiency_signal"],
        }

    def evaluate(self, row: dict) -> dict:
        import logging
        market_family = str(row.get("market_family", "") or "").strip().lower()
        if market_family.startswith("weather_temperature"):
            return self._evaluate_weather(row)

        decision_components = PredictionLayer.select_signal_components(row)
        score = decision_components["score"]
        score_relax = max(0.0, _finite_float(row.get("entry_score_relax", 0.0), default=0.0) or 0.0)
        spread_relax = max(0.0, _finite_float(row.get("entry_spread_relax", 0.0), default=0.0) or 0.0)
        liquidity_relax_factor = _finite_float(row.get("entry_liquidity_relax_factor", 1.0), default=1.0)
        liquidity_relax_factor = max(0.0, min(1.0, liquidity_relax_factor if liquidity_relax_factor is not None else 1.0))
        
        # --- MACRO MOOD HUNTING LOGIC ---
        dynamic_min_score = self.min_score
        target_side = str(row.get("outcome_side", row.get("side", "UNKNOWN"))).upper()
        
        # 1. Protect against Long Squeezes
        is_overheated_long = row.get("is_overheated_long", False)
        macro_veto = False
        if is_overheated_long and target_side == "YES":
            macro_veto = True
            logging.warning(f"StrategyLayer: VETO! Market is Overheated Long. Blocking YES bet on {row.get('market_slug', 'market')}.")
            
        # 2. Aggressive Hunting (Trend Following)
        trend_score = _finite_float(row.get("trend_score", 0.5), default=0.5)
        fgi_value = row.get("fgi_value", 50)
        
        if trend_score > 0.75 and fgi_value >= 60:
            # Huge Bullish Conviction: Lower threshold for YES, raise for NO
            if target_side == "YES":
                dynamic_min_score = max(0.10, self.min_score - 0.15)
                # logging.info(f"StrategyLayer: HUNT MODE. Lowering YES threshold to {dynamic_min_score:.2f}")
            elif target_side == "NO":
                dynamic_min_score = min(0.95, self.min_score + 0.15)
                
        elif trend_score < 0.25 and fgi_value <= 40:
            # Huge Bearish Conviction: Lower threshold for NO, raise for YES
            if target_side == "NO":
                dynamic_min_score = max(0.10, self.min_score - 0.15)
            elif target_side == "YES":
                dynamic_min_score = min(0.95, self.min_score + 0.15)
        # --------------------------------

        ta_bias = str(row.get("btc_trend_bias", "NEUTRAL")).strip().upper()
        alligator_alignment = str(row.get("alligator_alignment", "NEUTRAL")).strip().upper()
        adx_value = _finite_float(row.get("adx_value"), default=0.0) or 0.0
        adx_threshold = _finite_float(row.get("adx_threshold"), default=18.0) or 18.0
        adx_trending = bool(row.get("adx_trending")) or adx_value >= adx_threshold
        price_above_anchored_vwap = bool(row.get("price_above_anchored_vwap"))
        price_below_anchored_vwap = bool(row.get("price_below_anchored_vwap"))
        target_direction = self._target_direction(row)
        trend_confluence = _finite_float(row.get("btc_trend_confluence"), default=0.0) or 0.0
        latest_bullish_fractal = _finite_float(row.get("latest_bullish_fractal"), default=None)
        latest_bearish_fractal = _finite_float(row.get("latest_bearish_fractal"), default=None)
        long_fractal_breakout = bool(row.get("long_fractal_breakout"))
        short_fractal_breakout = bool(row.get("short_fractal_breakout"))
        live_bias = str(row.get("btc_live_bias", "NEUTRAL") or "NEUTRAL").strip().upper()
        live_confluence = _finite_float(row.get("btc_live_confluence"), default=0.0) or 0.0
        live_quality_score = _finite_float(row.get("btc_live_source_quality_score"), default=0.0) or 0.0
        live_source_quality = str(row.get("btc_live_source_quality", "LOW") or "LOW").strip().upper()
        live_divergence_bps = abs(_finite_float(row.get("btc_live_source_divergence_bps"), default=0.0) or 0.0)
        live_mark_index_basis_bps = _finite_float(row.get("btc_live_mark_index_basis_bps"), default=0.0) or 0.0
        live_index_ready = bool(row.get("btc_live_index_ready", False))
        volatility_regime = str(row.get("btc_volatility_regime", "NORMAL") or "NORMAL").strip().upper()
        volatility_regime_score = _finite_float(row.get("btc_volatility_regime_score"), default=0.0) or 0.0
        trend_persistence = _finite_float(row.get("btc_trend_persistence"), default=0.0) or 0.0
        rsi_value = _finite_float(row.get("btc_rsi_14"), default=50.0) or 50.0
        rsi_divergence_score = _finite_float(row.get("btc_rsi_divergence_score"), default=0.0) or 0.0
        macd_hist = _finite_float(row.get("btc_macd_hist"), default=0.0) or 0.0
        macd_hist_slope = _finite_float(row.get("btc_macd_hist_slope"), default=0.0) or 0.0
        momentum_regime = str(row.get("btc_momentum_regime", "NEUTRAL") or "NEUTRAL").strip().upper()
        momentum_confluence = _finite_float(row.get("btc_momentum_confluence"), default=0.0) or 0.0
        open_positions_count = int(max(0.0, _finite_float(row.get("open_positions_count"), default=0.0) or 0.0))
        open_positions_unrealized_pnl_pct_total = _finite_float(
            row.get("open_positions_unrealized_pnl_pct_total"),
            default=0.0,
        ) or 0.0
        open_positions_avg_to_now_price_change_pct_mean = _finite_float(
            row.get("open_positions_avg_to_now_price_change_pct_mean"),
            default=0.0,
        ) or 0.0
        open_positions_winner_count = int(max(0.0, _finite_float(row.get("open_positions_winner_count"), default=0.0) or 0.0))
        open_positions_loser_count = int(max(0.0, _finite_float(row.get("open_positions_loser_count"), default=0.0) or 0.0))
        ta_bias_conflicts = (
            (ta_bias == "LONG" and target_direction == "SHORT")
            or (ta_bias == "SHORT" and target_direction == "LONG")
        )
        fractal_available = (
            (target_direction == "LONG" and latest_bullish_fractal is not None)
            or (target_direction == "SHORT" and latest_bearish_fractal is not None)
        )
        fractal_trigger_ready = (
            (target_direction == "LONG" and long_fractal_breakout)
            or (target_direction == "SHORT" and short_fractal_breakout)
            or target_direction == "NEUTRAL"
        )
        ta_entry_ready = (
            ta_bias in {"LONG", "SHORT"}
            and target_direction == ta_bias
            and alligator_alignment in {"BULLISH", "BEARISH"}
            and adx_trending
            and (price_above_anchored_vwap or price_below_anchored_vwap)
            and (not fractal_available or fractal_trigger_ready)
        )
        ta_trigger_blocked = False  # Fractal pending is now a soft penalty, not a hard block
        ta_fractal_pending = (
            ta_bias in {"LONG", "SHORT"}
            and target_direction == ta_bias
            and alligator_alignment in {"BULLISH", "BEARISH"}
            and adx_trending
            and (price_above_anchored_vwap or price_below_anchored_vwap)
            and fractal_available
            and not fractal_trigger_ready
        )
        if ta_bias_conflicts:
            # Soft penalty instead of hard veto — raise threshold significantly
            dynamic_min_score = min(0.95, dynamic_min_score + 0.10)
            logging.debug(
                "StrategyLayer: Trend conflict penalty +0.10. target=%s bias=%s market=%s",
                target_direction,
                ta_bias,
                row.get("market_slug", row.get("market_title", "market")),
            )
        elif ta_entry_ready:
            dynamic_min_score = max(0.05, dynamic_min_score - min(0.08, trend_confluence * 0.08))
        elif ta_fractal_pending:
            # Raise threshold slightly instead of blocking entirely
            dynamic_min_score = min(0.95, dynamic_min_score + 0.03)

        live_bias_aligns = live_bias in {"LONG", "SHORT"} and live_bias == target_direction
        live_bias_conflicts = live_bias in {"LONG", "SHORT"} and target_direction in {"LONG", "SHORT"} and live_bias != target_direction
        live_feed_unreliable = live_index_ready and (live_quality_score < 0.25 or live_divergence_bps >= 35.0)
        if live_feed_unreliable and abs(live_mark_index_basis_bps) >= 20.0 and target_direction in {"LONG", "SHORT"}:
            macro_veto = True
            logging.info(
                "StrategyLayer: Live BTC/index veto. target=%s quality=%s(%.2f) divergence_bps=%.2f basis_bps=%.2f market=%s",
                target_direction,
                live_source_quality,
                live_quality_score,
                live_divergence_bps,
                live_mark_index_basis_bps,
                row.get("market_slug", row.get("market_title", "market")),
            )
        elif live_bias_aligns and live_confluence >= 0.60 and live_quality_score >= 0.50:
            dynamic_min_score = max(0.05, dynamic_min_score - min(0.05, live_confluence * 0.05))
        elif live_bias_conflicts and live_confluence >= 0.70 and live_quality_score >= 0.55:
            dynamic_min_score = min(0.95, dynamic_min_score + min(0.08, live_confluence * 0.08))

        momentum_aligns = (
            (target_direction == "LONG" and momentum_regime in {"BULLISH", "OVERSOLD_EXHAUSTION"})
            or (target_direction == "SHORT" and momentum_regime in {"BEARISH", "OVERBOUGHT_EXHAUSTION"})
        )
        momentum_conflicts = (
            (target_direction == "LONG" and momentum_regime in {"BEARISH", "OVERBOUGHT_EXHAUSTION"})
            or (target_direction == "SHORT" and momentum_regime in {"BULLISH", "OVERSOLD_EXHAUSTION"})
        )
        if volatility_regime == "EXTREME" and volatility_regime_score >= 0.90 and momentum_conflicts and target_direction in {"LONG", "SHORT"}:
            # Soft penalty instead of hard veto
            dynamic_min_score = min(0.95, dynamic_min_score + 0.08)
            logging.debug(
                "StrategyLayer: Volatility/momentum penalty +0.08. target=%s vol_regime=%s momentum=%s market=%s",
                target_direction,
                volatility_regime,
                momentum_regime,
                row.get("market_slug", row.get("market_title", "market")),
            )
        elif momentum_aligns and momentum_confluence >= 0.45 and trend_persistence >= 0.55:
            dynamic_min_score = max(0.05, dynamic_min_score - min(0.05, momentum_confluence * 0.05))
        elif momentum_conflicts and momentum_confluence >= 0.45:
            dynamic_min_score = min(0.95, dynamic_min_score + min(0.07, momentum_confluence * 0.07))
        if target_direction == "LONG" and rsi_value >= 78 and rsi_divergence_score < -0.20 and macd_hist_slope <= 0:
            dynamic_min_score = min(0.95, dynamic_min_score + 0.08)
        elif target_direction == "SHORT" and rsi_value <= 22 and rsi_divergence_score > 0.20 and macd_hist_slope >= 0:
            dynamic_min_score = min(0.95, dynamic_min_score + 0.08)

        portfolio_pressure_penalty = 0.0
        if open_positions_count > 0:
            loser_ratio = open_positions_loser_count / max(open_positions_count, 1)
            if open_positions_unrealized_pnl_pct_total <= -0.06:
                portfolio_pressure_penalty += 0.05
            elif open_positions_unrealized_pnl_pct_total <= -0.02:
                portfolio_pressure_penalty += 0.02
            if loser_ratio >= 0.67:
                portfolio_pressure_penalty += 0.03
            elif open_positions_winner_count == open_positions_count and open_positions_unrealized_pnl_pct_total >= 0.08:
                portfolio_pressure_penalty -= 0.01
            if open_positions_avg_to_now_price_change_pct_mean <= -0.05:
                portfolio_pressure_penalty += 0.02
            dynamic_min_score = min(0.95, max(0.02, dynamic_min_score + portfolio_pressure_penalty))

        dynamic_min_score = max(0.02, dynamic_min_score - score_relax)

        spread = _finite_float(row.get("spread"), default=None)
        if spread is None:
            best_bid = _finite_float(row.get("best_bid"), default=None)
            best_ask = _finite_float(row.get("best_ask"), default=None)
            if best_bid is not None and best_ask is not None and best_ask >= best_bid:
                spread = best_ask - best_bid
            else:
                spread = 0.0

        liquidity_raw = _finite_float(row.get("liquidity", row.get("market_liquidity")), default=None)
        liquidity_score = _finite_float(row.get("liquidity_score"), default=None)

        has_raw_liquidity = liquidity_raw is not None and liquidity_raw > 0
        has_liquidity_score = liquidity_score is not None and liquidity_score > 0

        if has_raw_liquidity:
            liquidity_value = liquidity_raw
            liquidity_threshold = max(0.0, self.min_liquidity * liquidity_relax_factor)
            liquidity_metric = "liquidity"
            liquidity_ok = liquidity_raw >= liquidity_threshold
        elif has_liquidity_score:
            liquidity_value = liquidity_score
            liquidity_threshold = max(0.0, self.min_liquidity_score * liquidity_relax_factor)
            liquidity_metric = "liquidity_score"
            liquidity_ok = liquidity_score >= liquidity_threshold
        else:
            # If liquidity features are missing or encoded as non-positive placeholders,
            # defer hard filtering to orderbook guards.
            liquidity_value = None
            liquidity_threshold = None
            liquidity_metric = "missing"
            liquidity_ok = True

        spread_threshold = self.max_spread + spread_relax
        spread_ok = spread <= spread_threshold
        spread_penalty = 0.0
        if not spread_ok:
            spread_excess = max(0.0, spread - spread_threshold)
            spread_excess_ratio = spread_excess / max(spread_threshold, 0.01)
            spread_penalty = min(0.12, 0.04 + (spread_excess_ratio * 0.08))
            dynamic_min_score = min(0.95, dynamic_min_score + spread_penalty)

        liquidity_penalty = 0.0
        if not liquidity_ok and liquidity_value is not None and liquidity_threshold not in [None, 0]:
            liquidity_gap = max(0.0, float(liquidity_threshold) - float(liquidity_value))
            liquidity_gap_ratio = liquidity_gap / max(float(liquidity_threshold), 1e-9)
            liquidity_penalty = min(0.10, 0.04 + (liquidity_gap_ratio * 0.06))
            dynamic_min_score = min(0.95, dynamic_min_score + liquidity_penalty)

        score_ok = score >= dynamic_min_score
        allow = score_ok and not macro_veto and not ta_trigger_blocked

        return {
            "allow": allow,
            "score": score,
            "score_threshold": dynamic_min_score,
            "score_ok": score_ok,
            "spread": spread,
            "spread_threshold": spread_threshold,
            "spread_ok": spread_ok,
            "spread_penalty": spread_penalty,
            "liquidity_value": liquidity_value,
            "liquidity_threshold": liquidity_threshold,
            "liquidity_metric": liquidity_metric,
            "liquidity_ok": liquidity_ok,
            "liquidity_penalty": liquidity_penalty,
            "macro_veto": macro_veto,
            "score_relax": score_relax,
            "spread_relax": spread_relax,
            "liquidity_relax_factor": liquidity_relax_factor,
            "ta_bias": ta_bias,
            "target_direction": target_direction,
            "ta_entry_ready": ta_entry_ready,
            "ta_bias_conflicts": ta_bias_conflicts,
            "ta_trigger_blocked": ta_trigger_blocked,
            "alligator_alignment": alligator_alignment,
            "adx_value": adx_value,
            "adx_threshold": adx_threshold,
            "price_above_anchored_vwap": price_above_anchored_vwap,
            "price_below_anchored_vwap": price_below_anchored_vwap,
            "fractal_available": fractal_available,
            "long_fractal_breakout": long_fractal_breakout,
            "short_fractal_breakout": short_fractal_breakout,
            "btc_live_bias": live_bias,
            "btc_live_confluence": live_confluence,
            "btc_live_source_quality": live_source_quality,
            "btc_live_source_quality_score": live_quality_score,
            "btc_live_source_divergence_bps": live_divergence_bps,
            "btc_live_mark_index_basis_bps": live_mark_index_basis_bps,
            "btc_live_index_ready": live_index_ready,
            "btc_volatility_regime": volatility_regime,
            "btc_volatility_regime_score": volatility_regime_score,
            "btc_trend_persistence": trend_persistence,
            "btc_rsi_14": rsi_value,
            "btc_rsi_divergence_score": rsi_divergence_score,
            "btc_macd_hist": macd_hist,
            "btc_macd_hist_slope": macd_hist_slope,
            "btc_momentum_regime": momentum_regime,
            "btc_momentum_confluence": momentum_confluence,
            "open_positions_count": open_positions_count,
            "open_positions_unrealized_pnl_pct_total": open_positions_unrealized_pnl_pct_total,
            "open_positions_avg_to_now_price_change_pct_mean": open_positions_avg_to_now_price_change_pct_mean,
            "open_positions_winner_count": open_positions_winner_count,
            "open_positions_loser_count": open_positions_loser_count,
            "portfolio_pressure_penalty": portfolio_pressure_penalty,
            "decision_probability_signal": decision_components["probability_signal"],
            "decision_profitability_signal": decision_components["profitability_signal"],
            "decision_market_inefficiency_signal": decision_components["market_inefficiency_signal"],
        }

    def should_enter(self, row: dict) -> bool:
        return self.evaluate(row).get("allow", False)


class ExitRuleLayer:
    """Exit logic isolated from prediction logic for easier tuning."""

    def __init__(self, take_profit=5.0, stop_loss=-5.0, confidence_floor=0.45):
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.confidence_floor = confidence_floor

    def exit_reason(self, pnl: float, confidence: float, expected_return: float = 0.0, edge_score: float = 0.0) -> str | None:
        if pnl >= self.take_profit:
            return "take_profit"
        if pnl <= self.stop_loss:
            return "stop_loss"
        # Profitability-first: exit early if both expected_return and
        # edge have turned negative, even when confidence is still above floor.
        if expected_return < -0.01 and edge_score < -0.01:
            return "profitability_drop"
        if confidence < self.confidence_floor and expected_return <= 0:
            return "confidence_drop"
        return None
