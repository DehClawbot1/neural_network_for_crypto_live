import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SignalEngine:
    """
    Safe research/paper-trading signal scorer.
    Outputs observation labels and confidence, not live betting instructions.
    """

    LABELS = {
        0: "IGNORE",
        1: "LOW-CONFIDENCE WATCH",
        2: "STRONG PAPER OPPORTUNITY",
        3: "HIGHEST-RANKED PAPER SIGNAL",
    }

    def score_row(self, row: dict):
        trader_win_rate = float(row.get("trader_win_rate", 0.5))
        normalized_trade_size = float(row.get("normalized_trade_size", 0.0))
        current_price = float(row.get("current_price", 0.5))
        time_left = float(row.get("time_left", 0.5))

        # Simple bounded heuristic for research-only scoring
        confidence = (
            trader_win_rate * 0.40
            + normalized_trade_size * 0.25
            + (1.0 - abs(current_price - 0.5) * 2.0) * 0.20
            + time_left * 0.15
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if confidence < 0.45:
            action_code = 0
        elif confidence < 0.60:
            action_code = 1
        elif confidence < 0.78:
            action_code = 2
        else:
            action_code = 3

        return {
            **row,
            "confidence": round(confidence, 4),
            "signal_label": self.LABELS[action_code],
            "action_code": action_code,
            "reason": self._build_reason(row, confidence),
        }

    def _build_reason(self, row: dict, confidence: float):
        return (
            f"wallet_win_rate={float(row.get('trader_win_rate', 0.5)):.2f}, "
            f"trade_size_score={float(row.get('normalized_trade_size', 0.0)):.2f}, "
            f"price={float(row.get('current_price', 0.5)):.2f}, "
            f"time_left={float(row.get('time_left', 0.5)):.2f}, "
            f"confidence={confidence:.2f}"
        )

    def score_features(self, features_df: pd.DataFrame):
        if features_df is None or features_df.empty:
            return pd.DataFrame()

        scored = [self.score_row(row.to_dict()) for _, row in features_df.iterrows()]
        scored_df = pd.DataFrame(scored)
        logging.info("Scored %s feature rows.", len(scored_df))
        return scored_df


if __name__ == "__main__":
    sample = pd.DataFrame(
        [
            {
                "trader_wallet": "0xabc",
                "market_title": "Will Bitcoin close above $100k?",
                "trader_win_rate": 0.72,
                "normalized_trade_size": 0.80,
                "current_price": 0.55,
                "time_left": 0.40,
            }
        ]
    )
    engine = SignalEngine()
    print(engine.score_features(sample)[["market_title", "signal_label", "confidence", "reason"]])
