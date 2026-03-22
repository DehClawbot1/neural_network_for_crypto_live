class PredictionLayer:
    """Placeholder interface for model outputs such as expected return or P(TP before SL)."""

    @staticmethod
    def select_signal_score(row: dict) -> float:
        if "tp_before_sl_prob" in row:
            return float(row.get("tp_before_sl_prob", 0.0))
        if "expected_return" in row:
            return float(row.get("expected_return", 0.0))
        return float(row.get("confidence", 0.0))


class EntryRuleLayer:
    """Entry filter separate from raw model prediction."""

    def __init__(self, min_score=0.62, max_spread=0.03, min_liquidity=1000):
        self.min_score = min_score
        self.max_spread = max_spread
        self.min_liquidity = min_liquidity

    def should_enter(self, row: dict) -> bool:
        score = PredictionLayer.select_signal_score(row)
        spread = float(row.get("spread", 0.0) or 0.0)
        liquidity = float(row.get("liquidity", row.get("market_liquidity", 0.0)) or 0.0)
        return score >= self.min_score and spread <= self.max_spread and liquidity >= self.min_liquidity


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
