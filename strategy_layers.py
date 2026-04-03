import math


def _finite_float(value, default=None):
    try:
        num = float(value)
    except Exception:
        return default
    if not math.isfinite(num):
        return default
    return num


class PredictionLayer:
    """Placeholder interface for model outputs such as expected return or P(TP before SL)."""

    @staticmethod
    def select_signal_score(row: dict) -> float:
        # Confidence remains the primary gate score when finite.
        confidence = _finite_float(row.get("confidence", 0.0), default=0.0)
        if confidence > 0:
            return confidence

        # Fallback: use p_tp if available and positive.
        p_tp = _finite_float(row.get("p_tp_before_sl", row.get("tp_before_sl_prob", 0.0)), default=0.0)
        if p_tp > 0:
            return p_tp

        # Last resort: expected return only if positive.
        er = _finite_float(row.get("expected_return", 0.0), default=0.0)
        return max(er, 0.0)


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

    def evaluate(self, row: dict) -> dict:
        import logging
        
        score = PredictionLayer.select_signal_score(row)
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
        ta_trigger_blocked = (
            ta_bias in {"LONG", "SHORT"}
            and target_direction == ta_bias
            and alligator_alignment in {"BULLISH", "BEARISH"}
            and adx_trending
            and (price_above_anchored_vwap or price_below_anchored_vwap)
            and fractal_available
            and not fractal_trigger_ready
        )
        if ta_bias_conflicts:
            macro_veto = True
            logging.info(
                "StrategyLayer: Trend veto. target=%s bias=%s alligator=%s adx=%.2f/%.2f avwap_above=%s avwap_below=%s market=%s",
                target_direction,
                ta_bias,
                alligator_alignment,
                adx_value,
                adx_threshold,
                price_above_anchored_vwap,
                price_below_anchored_vwap,
                row.get("market_slug", row.get("market_title", "market")),
            )
        elif ta_entry_ready:
            dynamic_min_score = max(0.05, dynamic_min_score - min(0.08, trend_confluence * 0.08))
        elif ta_trigger_blocked:
            logging.info(
                "StrategyLayer: Fractal trigger pending. target=%s bias=%s long_breakout=%s short_breakout=%s market=%s",
                target_direction,
                ta_bias,
                long_fractal_breakout,
                short_fractal_breakout,
                row.get("market_slug", row.get("market_title", "market")),
            )

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

        score_ok = score >= dynamic_min_score
        spread_threshold = self.max_spread + spread_relax
        spread_ok = spread <= spread_threshold
        allow = score_ok and spread_ok and liquidity_ok and not macro_veto and not ta_trigger_blocked

        return {
            "allow": allow,
            "score": score,
            "score_threshold": dynamic_min_score,
            "score_ok": score_ok,
            "spread": spread,
            "spread_threshold": spread_threshold,
            "spread_ok": spread_ok,
            "liquidity_value": liquidity_value,
            "liquidity_threshold": liquidity_threshold,
            "liquidity_metric": liquidity_metric,
            "liquidity_ok": liquidity_ok,
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
        }

    def should_enter(self, row: dict) -> bool:
        return self.evaluate(row).get("allow", False)


class ExitRuleLayer:
    """Exit logic isolated from prediction logic for easier tuning."""

    def __init__(self, take_profit=5.0, stop_loss=-5.0, confidence_floor=0.45):
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.confidence_floor = confidence_floor

    def exit_reason(self, pnl: float, confidence: float) -> str | None:
        if pnl >= self.take_profit:
            return "take_profit"
        if pnl <= self.stop_loss:
            return "stop_loss"
        if confidence < self.confidence_floor:
            return "confidence_drop"
        return None
