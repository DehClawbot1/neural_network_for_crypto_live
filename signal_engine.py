import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SignalEngine:
    """
    Safer signal scorer.

    Important changes:
    - blend heuristic and model confidence instead of taking the max
    - cap confidence when expected_return/edge is not actually trade-worthy
    - prevent weak or negative model outputs from graduating into strong signals
    """

    LABELS = {
        0: "IGNORE",
        1: "LOW-CONFIDENCE WATCH",
        2: "STRONG PAPER OPPORTUNITY",
        3: "HIGHEST-RANKED PAPER SIGNAL",
    }

    def score_row(self, row: dict):
        whale_pressure = float(row.get("whale_pressure", 0.5) or 0.5)
        market_structure_score = float(row.get("market_structure_score", 0.5) or 0.5)
        volatility_risk = float(row.get("volatility_risk", 0.5) or 0.5)
        time_decay_score = float(row.get("time_decay_score", 0.5) or 0.5)
        p_tp = float(row.get("p_tp_before_sl", 0.0) or 0.0)
        expected_return = float(row.get("expected_return", 0.0) or 0.0)
        edge_score = float(row.get("edge_score", 0.0) or 0.0)

        heuristic_confidence = (
            whale_pressure * 0.40
            + market_structure_score * 0.35
            + (1.0 - volatility_risk) * 0.15
            + (1.0 - time_decay_score) * 0.10
        )
        model_confidence = np.clip(
            (p_tp * 0.70)
            + np.clip(expected_return * 5.0, -1.0, 1.0) * 0.15
            + np.clip(edge_score * 8.0, -1.0, 1.0) * 0.15,
            0.0,
            1.0,
        )

        confidence = float(np.clip((heuristic_confidence * 0.45) + (model_confidence * 0.55), 0.0, 1.0))
        if expected_return == 0.0 and p_tp == 0.0: confidence = heuristic_confidence # BUG FIX 4: Restore 100% heuristic weight if AI is offline

        # If the model says the trade is weak, do not let the heuristic alone escalate it.
        if expected_return <= 0 or edge_score <= 0 or p_tp < 0.52:
            confidence = min(confidence, 0.59)
        if expected_return < 0 and p_tp < 0.48:
            confidence = min(confidence, 0.44)

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
            f"p_tp={float(row.get('p_tp_before_sl', 0.0) or 0.0):.2f}, "
            f"expected_return={float(row.get('expected_return', 0.0) or 0.0):.4f}, "
            f"edge_score={float(row.get('edge_score', 0.0) or 0.0):.4f}, "
            f"whale_pressure={float(row.get('whale_pressure', 0.5) or 0.5):.2f}, "
            f"market_structure={float(row.get('market_structure_score', 0.5) or 0.5):.2f}, "
            f"confidence={confidence:.2f}"
        )

    def score_features(self, features_df: pd.DataFrame):
        if features_df is None or features_df.empty:
            return pd.DataFrame()

        scored = [self.score_row(row.to_dict()) for _, row in features_df.iterrows()]
        scored_df = pd.DataFrame(scored)
        logging.info("Scored %s grouped feature rows.", len(scored_df))
        return scored_df
