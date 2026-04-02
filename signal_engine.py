import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _safe_float(value, default=0.0):
    try:
        num = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(num):
        return float(default)
    return float(num)


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
        whale_pressure = float(np.clip(_safe_float(row.get("whale_pressure", 0.5), default=0.5), 0.0, 1.0))
        market_structure_score = float(np.clip(_safe_float(row.get("market_structure_score", 0.5), default=0.5), 0.0, 1.0))
        volatility_risk = float(np.clip(_safe_float(row.get("volatility_risk", 0.5), default=0.5), 0.0, 1.0))
        time_decay_score = float(np.clip(_safe_float(row.get("time_decay_score", 0.5), default=0.5), 0.0, 1.0))
        network_activity_score = float(np.clip(_safe_float(row.get("btc_network_activity_score", 0.5), default=0.5), 0.0, 1.0))
        network_stress_score = float(np.clip(_safe_float(row.get("btc_network_stress_score", 0.5), default=0.5), 0.0, 1.0))
        p_tp = float(np.clip(_safe_float(row.get("p_tp_before_sl", 0.0), default=0.0), 0.0, 1.0))
        expected_return = _safe_float(row.get("expected_return", 0.0), default=0.0)
        edge_score = _safe_float(row.get("edge_score"), default=p_tp * expected_return)

        network_regime_bonus = 0.0
        if network_activity_score >= 0.55:
            network_regime_bonus += 0.03
        if network_stress_score >= 0.65 and whale_pressure >= 0.58 and market_structure_score >= 0.50:
            network_regime_bonus += 0.04
        elif network_stress_score <= 0.20:
            network_regime_bonus -= 0.02

        heuristic_confidence = (
            whale_pressure * 0.40
            + market_structure_score * 0.35
            + (1.0 - volatility_risk) * 0.15
            + (1.0 - time_decay_score) * 0.10
            + network_activity_score * 0.03
            + network_regime_bonus
        )
        model_confidence = np.clip(
            (p_tp * 0.70)
            + np.clip(expected_return * 5.0, -1.0, 1.0) * 0.15
            + np.clip(edge_score * 8.0, -1.0, 1.0) * 0.15,
            0.0,
            1.0,
        )

        confidence = float(np.clip((heuristic_confidence * 0.45) + (model_confidence * 0.55), 0.0, 1.0))
        if expected_return == 0.0 and p_tp == 0.0:
            confidence = heuristic_confidence  # BUG FIX 4: Restore 100% heuristic weight if AI is offline
        confidence = float(np.clip(_safe_float(confidence, default=0.0), 0.0, 1.0))

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
            f"p_tp={_safe_float(row.get('p_tp_before_sl', 0.0), default=0.0):.2f}, "
            f"expected_return={_safe_float(row.get('expected_return', 0.0), default=0.0):.4f}, "
            f"edge_score={_safe_float(row.get('edge_score', 0.0), default=0.0):.4f}, "
            f"whale_pressure={_safe_float(row.get('whale_pressure', 0.5), default=0.5):.2f}, "
            f"market_structure={_safe_float(row.get('market_structure_score', 0.5), default=0.5):.2f}, "
            f"network_stress={_safe_float(row.get('btc_network_stress_score', 0.5), default=0.5):.2f}, "
            f"confidence={_safe_float(confidence, default=0.0):.2f}"
        )

    def score_features(self, features_df: pd.DataFrame):
        if features_df is None or features_df.empty:
            return pd.DataFrame()

        scored = [self.score_row(row.to_dict()) for _, row in features_df.iterrows()]
        scored_df = pd.DataFrame(scored)
        logging.info("Scored %s grouped feature rows.", len(scored_df))
        return scored_df
