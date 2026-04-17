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
    - removed heuristic confidence overrides (Phase 5 strict ML reliance)
    - cap confidence when expected_return/edge is not actually trade-worthy
    - prevent weak or negative model outputs from graduating into strong signals
    """

    LABELS = {
        0: "IGNORE",
        1: "LOW-CONFIDENCE WATCH",
        2: "STRONG PAPER OPPORTUNITY",
        3: "HIGHEST-RANKED PAPER SIGNAL",
    }

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = bool(strict_mode)
        # Optional per-family conformal services, set by the orchestrator.
        # Key: market_family ("btc" or "weather_temperature").
        # Value: fitted ConformalIntervalService or None.
        self.conformal_services: dict = {}
        # Optional fitted meta-learner. When provided and fitted, REPLACES
        # the hand-tuned 0.45/0.30/0.25 blend below with a data-fitted
        # L2-logistic prediction. When None or unfitted, we fall back to the
        # legacy formula — the caps still apply on top either way.
        self.confidence_stacker = None
        # Attempt to auto-load a persisted stacker artefact at construction
        # time. Silent failure is fine — legacy formula remains in charge.
        try:
            from services.confidence_stacker import ConfidenceStacker
            _cs = ConfidenceStacker()
            if _cs.load_if_exists():
                self.confidence_stacker = _cs
                logging.info("SignalEngine: loaded ConfidenceStacker %s",
                             self.confidence_stacker.summary())
        except Exception as _cs_exc:
            logging.debug("SignalEngine: stacker auto-load skipped: %s", _cs_exc)
        # Phase 16 cap calibrator: load data-fitted caps (falls back to defaults
        # if artefact is missing / stale). These replace the hand-picked 0.59 /
        # 0.44 / 0.39 / 0.42 magic numbers in score_row.
        try:
            from services.cap_calibrator import load_caps as _load_caps, DEFAULT_CAPS
            self._conf_caps = _load_caps()
        except Exception:
            self._conf_caps = {
                "profitability_weak": 0.59,
                "deep_negative": 0.44,
                "ta_conflict": 0.39,
                "fractal_pending": 0.42,
            }

    def reload_confidence_caps(self) -> dict:
        """Re-read the fitted caps from disk (called post-nightly refit)."""
        try:
            from services.cap_calibrator import load_caps as _load_caps
            self._conf_caps = _load_caps()
        except Exception:
            pass
        return dict(self._conf_caps)

    def set_conformal_services(self, services: dict) -> None:
        """Inject per-family ConformalIntervalService instances (or None values)."""
        self.conformal_services = dict(services or {})

    def set_confidence_stacker(self, stacker) -> None:
        """Inject a fitted ConfidenceStacker (or None to disable)."""
        self.confidence_stacker = stacker
        if stacker is not None:
            try:
                logging.info("SignalEngine: stacker plumbed %s", stacker.summary())
            except Exception:
                pass

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
        if getattr(self, "strict_mode", False):
            for _req in ("p_tp_before_sl", "expected_return"):
                _val = row.get(_req)
                if _val is None or (isinstance(_val, float) and not np.isfinite(_val)):
                    from services.types import DataFaultError
                    raise DataFaultError(f"SignalEngine strict_mode: missing/invalid {_req!r} in row")
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

        # Phase 16: Stacked meta-learner (L2 logistic) replaces the hand-tuned
        # 0.45/0.30/0.25 blend when a fitted artefact is available. Legacy
        # formula remains as fallback during cold-start or when prediction fails.
        stacker_conf = None
        _stacker = getattr(self, "confidence_stacker", None)
        if _stacker is not None and getattr(_stacker, "is_fitted", False):
            try:
                _pred = _stacker.predict_row(row)
                if _pred is not None:
                    _pred_f = float(_pred)
                    if np.isfinite(_pred_f):
                        stacker_conf = _pred_f
            except Exception:
                stacker_conf = None
        if stacker_conf is not None:
            model_confidence = float(np.clip(stacker_conf, 0.0, 1.0))
        else:
            model_confidence = np.clip(
                (p_tp * 0.45)
                + np.clip(expected_return * 5.0, -1.0, 1.0) * 0.30
                + np.clip(edge_score * 8.0, -1.0, 1.0) * 0.25,
                0.0,
                1.0,
            )

        confidence = float(model_confidence)
        if expected_return == 0.0 and p_tp == 0.0:
            confidence = 0.0  # Phase 5: No AI = No Trade (no heuristic recovery)

        # Profitability-first caps: if the model says return/edge is negative
        # the signal cannot graduate beyond WATCH regardless of heuristic score.
        _caps = getattr(self, "_conf_caps", None) or {
            "profitability_weak": 0.59, "deep_negative": 0.44,
            "ta_conflict": 0.39, "fractal_pending": 0.42,
        }
        if expected_return <= 0 or edge_score <= 0 or p_tp < 0.52:
            confidence = min(confidence, float(_caps.get("profitability_weak", 0.59)))
        if expected_return < 0 and p_tp < 0.48:
            confidence = min(confidence, float(_caps.get("deep_negative", 0.44)))
        if ta_conflict:
            confidence = min(confidence, float(_caps.get("ta_conflict", 0.39)))
        if fractal_trigger_pending:
            confidence = min(confidence, float(_caps.get("fractal_pending", 0.42)))

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

        # Action code: profitability-weighted score determines the tier.
        # Combine profitability (expected_return, edge) with the blended
        # confidence so that high-return trades can graduate even when
        # heuristic confidence is moderate.
        _profit_score = float(np.clip(
            np.clip(expected_return * 5.0, -1.0, 1.0) * 0.40
            + np.clip(edge_score * 8.0, -1.0, 1.0) * 0.30
            + confidence * 0.30,
            0.0, 1.0,
        ))
        if _profit_score < 0.25:
            action_code = 0
        elif _profit_score < 0.45:
            action_code = 1
        elif _profit_score < 0.70:
            action_code = 2
        else:
            action_code = 3

        # Phase 5: Near-resolved market gate. Cap action_code at 1 (LOW-CONFIDENCE WATCH)
        # if the market is effectively resolved (<0.15 or >0.86) to prevent tail-risk sizing.
        current_price = _safe_float(row.get("current_price", 0.5), default=0.5)
        if current_price < 0.15 or current_price > 0.86:
            action_code = min(action_code, 1)

        # Conformal uncertainty multiplier — scales position size downstream.
        # Returns 1.0 when no fitted service is registered for this family,
        # so behavior is identical to pre-plumbing defaults.
        um = 1.0
        try:
            fam = str(row.get("market_family") or "btc").lower()
            fam_key = "weather_temperature" if fam.startswith("weather") else "btc"
            svc = self.conformal_services.get(fam_key) if self.conformal_services else None
            if svc is not None and getattr(svc, "is_fitted", False):
                um = float(svc.size_multiplier(point_prediction=float(confidence)))
                um = max(0.0, min(1.0, um))
        except Exception:
            um = 1.0  # conformal never blocks scoring

        return {
            **row,
            "confidence": round(confidence, 4),
            "signal_label": self.LABELS[action_code],
            "action_code": action_code,
            "wallet_state_score": round(wallet_state_score, 4),
            "wallet_entry_gate_fail": bool(wallet_entry_gate_fail),
            "wallet_conflict_softened": bool(scale_in_conflict_softened),
            "uncertainty_multiplier": round(um, 4),
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
