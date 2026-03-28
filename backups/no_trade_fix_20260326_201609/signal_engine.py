import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SignalEngine:
    """
    Safe research/paper-trading signal scorer.
    Uses grouped feature sub-scores and outputs observation labels and confidence,
    not live betting instructions.

    FIX: Lowered thresholds from 0.45/0.60/0.78 to 0.35/0.50/0.70
    so early-stage signals with modest model scores still produce trades
    that the system can learn from.
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
        p_tp = float(row.get("p_tp_before_sl", 0.0) or 0.0)
        expected_return = float(row.get("expected_return", 0.0) or 0.0)
        edge_score = float(row.get("edge_score", 0.0) or 0.0)

        heuristic_confidence = (
            whale_pressure * 0.40
            + market_structure_score * 0.35
            + (1.0 - volatility_risk) * 0.15
            + (1.0 - time_decay_score) * 0.10
        )
        model_confidence = np.clip((p_tp * 0.75) + np.clip(expected_return * 5.0, -1.0, 1.0) * 0.10 + np.clip(edge_score * 8.0, -1.0, 1.0) * 0.15, 0.0, 1.0)
        confidence = float(np.clip(max(heuristic_confidence, model_confidence), 0.0, 1.0))

        # ── FIX: Lowered thresholds so trades actually happen
        if confidence < 0.35:
            action_code = 0
        elif confidence < 0.50:
            action_code = 1
        elif confidence < 0.70:
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
            f"p_tp={float(row.get('p_tp_before_sl', 0.0) or 0.0):.2f}, "
            f"expected_return={float(row.get('expected_return', 0.0) or 0.0):.4f}, "
            f"edge_score={float(row.get('edge_score', 0.0) or 0.0):.4f}, "
            f"whale_pressure={float(row.get('whale_pressure', 0.5)):.2f}, "
            f"market_structure={float(row.get('market_structure_score', 0.5)):.2f}, "
            f"confidence={confidence:.2f}"
        )

    def score_features(self, features_df: pd.DataFrame):
        if features_df is None or features_df.empty:
            return pd.DataFrame()

        scored = [self.score_row(row.to_dict()) for _, row in features_df.iterrows()]
        scored_df = pd.DataFrame(scored)
        logging.info("Scored %s grouped feature rows.", len(scored_df))

        # ── FIX: Log distribution so user can see what's happening
        if not scored_df.empty and "signal_label" in scored_df.columns:
            label_counts = scored_df["signal_label"].value_counts().to_dict()
            logging.info("Signal distribution: %s", label_counts)

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
