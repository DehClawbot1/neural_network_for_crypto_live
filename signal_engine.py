import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SignalEngine:
    """
    Safe research/paper-trading signal scorer.
    Uses grouped feature sub-scores and outputs observation labels and confidence,
    not live betting instructions.
    """

    LABELS = {
        0: "IGNORE",
        1: "LOW-CONFIDENCE WATCH",
        2: "STRONG PAPER OPPORTUNITY",
        3: "HIGHEST-RANKED PAPER SIGNAL",
    }

    def score_row(self, row: dict):
        whale_pressure = float(row.get("whale_pressure", 0.5))
        market_structure_score = float(row.get("market_structure_score", 0.5))
        volatility_risk = float(row.get("volatility_risk", 0.5))
        time_decay_score = float(row.get("time_decay_score", 0.5))

        confidence = (
            whale_pressure * 0.40
            + market_structure_score * 0.35
            + (1.0 - volatility_risk) * 0.15
            + (1.0 - time_decay_score) * 0.10
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
            f"whale_pressure={float(row.get('whale_pressure', 0.5)):.2f}, "
            f"market_structure={float(row.get('market_structure_score', 0.5)):.2f}, "
            f"volatility_risk={float(row.get('volatility_risk', 0.5)):.2f}, "
            f"time_decay={float(row.get('time_decay_score', 0.5)):.2f}, "
            f"confidence={confidence:.2f}"
        )

    def score_features(self, features_df: pd.DataFrame):
        if features_df is None or features_df.empty:
            return pd.DataFrame()

        scored = [self.score_row(row.to_dict()) for _, row in features_df.iterrows()]
        scored_df = pd.DataFrame(scored)
        logging.info("Scored %s grouped feature rows.", len(scored_df))
        return scored_df


if __name__ == "__main__":
    sample = pd.DataFrame(
        [
            {
                "trader_wallet": "0xabc",
                "market_title": "Will Bitcoin close above $100k?",
                "whale_pressure": 0.72,
                "market_structure_score": 0.80,
                "volatility_risk": 0.35,
                "time_decay_score": 0.20,
            }
        ]
    )
    engine = SignalEngine()
    print(engine.score_features(sample)[["market_title", "signal_label", "confidence", "reason"]])
