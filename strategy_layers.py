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

    def evaluate(self, row: dict) -> dict:
        score = PredictionLayer.select_signal_score(row)

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

        if liquidity_raw is not None:
            liquidity_value = liquidity_raw
            liquidity_threshold = self.min_liquidity
            liquidity_metric = "liquidity"
            liquidity_ok = liquidity_raw >= self.min_liquidity
        elif liquidity_score is not None:
            liquidity_value = liquidity_score
            liquidity_threshold = self.min_liquidity_score
            liquidity_metric = "liquidity_score"
            liquidity_ok = liquidity_score >= self.min_liquidity_score
        else:
            # If liquidity features are absent, defer hard filtering to orderbook guards.
            liquidity_value = None
            liquidity_threshold = None
            liquidity_metric = "missing"
            liquidity_ok = True

        score_ok = score >= self.min_score
        spread_ok = spread <= self.max_spread
        allow = score_ok and spread_ok and liquidity_ok

        return {
            "allow": allow,
            "score": score,
            "score_threshold": self.min_score,
            "score_ok": score_ok,
            "spread": spread,
            "spread_threshold": self.max_spread,
            "spread_ok": spread_ok,
            "liquidity_value": liquidity_value,
            "liquidity_threshold": liquidity_threshold,
            "liquidity_metric": liquidity_metric,
            "liquidity_ok": liquidity_ok,
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
