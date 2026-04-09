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


def _parse_reason_tokens(value) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    text = text.replace("|", ",")
    return {
        token.strip()
        for token in text.split(",")
        if token and token.strip()
    }


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

    @staticmethod
    def _target_direction(row: dict) -> str:
        side = str(row.get("outcome_side", row.get("side", "UNKNOWN"))).strip().upper()
        if side in {"YES", "UP", "LONG", "BULLISH"}:
            return "LONG"
        if side in {"NO", "DOWN", "SHORT", "BEARISH"}:
            return "SHORT"
        return "NEUTRAL"

    def score_row(self, row: dict):
        whale_pressure = float(np.clip(_safe_float(row.get("whale_pressure", 0.5), default=0.5), 0.0, 1.0))
        market_structure_score = float(np.clip(_safe_float(row.get("market_structure_score", 0.5), default=0.5), 0.0, 1.0))
        volatility_risk = float(np.clip(_safe_float(row.get("volatility_risk", 0.5), default=0.5), 0.0, 1.0))
        time_decay_score = float(np.clip(_safe_float(row.get("time_decay_score", 0.5), default=0.5), 0.0, 1.0))
        network_activity_score = float(np.clip(_safe_float(row.get("btc_network_activity_score", 0.5), default=0.5), 0.0, 1.0))
        network_stress_score = float(np.clip(_safe_float(row.get("btc_network_stress_score", 0.5), default=0.5), 0.0, 1.0))
        wallet_quality_score = float(np.clip(_safe_float(row.get("wallet_quality_score", 0.5), default=0.5), 0.0, 1.0))
        wallet_state_confidence = float(np.clip(_safe_float(row.get("wallet_state_confidence", 0.0), default=0.0), 0.0, 1.0))
        wallet_state_freshness_score = float(np.clip(_safe_float(row.get("wallet_state_freshness_score", 0.0), default=0.0), 0.0, 1.0))
        wallet_size_change_score = float(np.clip(_safe_float(row.get("wallet_size_change_score", 0.0), default=0.0), 0.0, 1.0))
        wallet_agreement_score = float(np.clip(_safe_float(row.get("wallet_agreement_score", 0.5), default=0.5), 0.0, 1.0))
        wallet_distance_score = float(np.clip(_safe_float(row.get("wallet_distance_score", 0.5), default=0.5), 0.0, 1.0))
        p_tp = float(np.clip(_safe_float(row.get("p_tp_before_sl", 0.0), default=0.0), 0.0, 1.0))
        expected_return = _safe_float(row.get("expected_return", 0.0), default=0.0)
        edge_score = _safe_float(row.get("edge_score"), default=p_tp * expected_return)
        ta_bias = str(row.get("btc_trend_bias", "NEUTRAL")).strip().upper()
        target_direction = self._target_direction(row)
        trend_confluence = float(np.clip(_safe_float(row.get("btc_trend_confluence", 0.0), default=0.0), 0.0, 1.0))
        long_fractal_breakout = bool(row.get("long_fractal_breakout"))
        short_fractal_breakout = bool(row.get("short_fractal_breakout"))
        fractal_trigger_ready = (
            (target_direction == "LONG" and long_fractal_breakout)
            or (target_direction == "SHORT" and short_fractal_breakout)
        )
        fractal_trigger_pending = ta_bias in {"LONG", "SHORT"} and ta_bias == target_direction and not fractal_trigger_ready
        ta_conflict = (
            (ta_bias == "LONG" and target_direction == "SHORT")
            or (ta_bias == "SHORT" and target_direction == "LONG")
        )
        ta_support = ta_bias in {"LONG", "SHORT"} and ta_bias == target_direction

        network_regime_bonus = 0.0
        if network_activity_score >= 0.55:
            network_regime_bonus += 0.03
        if network_stress_score >= 0.65 and whale_pressure >= 0.58 and market_structure_score >= 0.50:
            network_regime_bonus += 0.04
        elif network_stress_score <= 0.20:
            network_regime_bonus -= 0.02

        wallet_state_score = (
            wallet_quality_score * 0.28
            + wallet_state_freshness_score * 0.20
            + wallet_size_change_score * 0.18
            + wallet_agreement_score * 0.18
            + wallet_distance_score * 0.08
            + wallet_state_confidence * 0.08
        )

        heuristic_confidence = (
            whale_pressure * 0.40
            + market_structure_score * 0.35
            + (1.0 - volatility_risk) * 0.15
            + (1.0 - time_decay_score) * 0.10
            + network_activity_score * 0.03
            + network_regime_bonus
        )
        heuristic_confidence = float(np.clip((heuristic_confidence * 0.82) + (wallet_state_score * 0.18), 0.0, 1.0))
        if ta_support:
            heuristic_confidence += min(0.08, trend_confluence * 0.08)
            if fractal_trigger_ready:
                heuristic_confidence += 0.04
        elif ta_conflict:
            heuristic_confidence -= min(0.12, max(0.06, trend_confluence * 0.12))
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
        if ta_conflict:
            confidence = min(confidence, 0.39)
        if fractal_trigger_pending:
            confidence = min(confidence, 0.42)

        wallet_reason_tokens = _parse_reason_tokens(row.get("wallet_state_gate_reason"))
        position_event = str(row.get("source_wallet_position_event", "") or "").upper()
        scale_in_conflict_softened = (
            "conflict_with_stronger_wallet" in wallet_reason_tokens
            and position_event == "SCALE_IN"
            and bool(row.get("wallet_state_gate_soft_override", False))
        )
        if scale_in_conflict_softened:
            confidence = min(confidence * 0.88, 0.72)

        wallet_state_gate_pass = bool(row.get("wallet_state_gate_pass", True))
        entry_intent = str(row.get("entry_intent", "OPEN_LONG") or "OPEN_LONG").upper()
        wallet_entry_gate_fail = entry_intent == "OPEN_LONG" and not wallet_state_gate_pass
        if entry_intent == "CLOSE_LONG":
            confidence = max(confidence, 0.65 if bool(row.get("source_wallet_exit_signal", False)) else 0.50)

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
            "wallet_state_score": round(wallet_state_score, 4),
            "wallet_entry_gate_fail": bool(wallet_entry_gate_fail),
            "wallet_conflict_softened": bool(scale_in_conflict_softened),
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
            f"wallet_quality={_safe_float(row.get('wallet_quality_score', 0.5), default=0.5):.2f}, "
            f"wallet_state_score={_safe_float(row.get('wallet_state_score', 0.0), default=0.0):.2f}, "
            f"confidence={_safe_float(confidence, default=0.0):.2f}"
        )

    def score_features(self, features_df: pd.DataFrame):
        if features_df is None or features_df.empty:
            return pd.DataFrame()

        scored = [self.score_row(row.to_dict()) for _, row in features_df.iterrows()]
        scored_df = pd.DataFrame(scored)
        logging.info("Scored %s grouped feature rows.", len(scored_df))
        return scored_df
