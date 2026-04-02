import numpy as np

from config import TradingConfig


def _safe_float(value, default=0.0):
    try:
        num = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(num):
        return float(default)
    return float(num)


def apply_entry_cadence_boost(signal_row: dict, minutes_idle: float | None, target_minutes: float, candidate_rank: int) -> dict:
    boosted = dict(signal_row)
    if minutes_idle is None or minutes_idle < target_minutes:
        return boosted
    overtime = max(0.0, float(minutes_idle) - float(target_minutes))
    pressure = float(np.clip(overtime / max(float(target_minutes), 1.0), 0.0, 1.0))
    base_score_relax = float(getattr(TradingConfig, "ENTRY_INACTIVITY_SCORE_RELAX", 0.08))
    base_spread_relax = float(getattr(TradingConfig, "ENTRY_INACTIVITY_SPREAD_RELAX", 0.10))
    base_liquidity_relax = float(getattr(TradingConfig, "ENTRY_INACTIVITY_LIQUIDITY_RELAX_FACTOR", 0.35))
    base_confidence_boost = float(getattr(TradingConfig, "ENTRY_INACTIVITY_CONFIDENCE_BOOST", 0.08))
    rank_multiplier = 1.0 if candidate_rank <= 1 else 0.75 if candidate_rank == 2 else 0.55
    boosted["entry_score_relax"] = max(
        _safe_float(boosted.get("entry_score_relax", 0.0), default=0.0),
        base_score_relax * rank_multiplier * (1.0 + 0.5 * pressure),
    )
    boosted["entry_spread_relax"] = max(
        _safe_float(boosted.get("entry_spread_relax", 0.0), default=0.0),
        base_spread_relax * rank_multiplier * (1.0 + 0.5 * pressure),
    )
    boosted["entry_liquidity_relax_factor"] = min(
        _safe_float(boosted.get("entry_liquidity_relax_factor", 1.0), default=1.0),
        max(0.10, base_liquidity_relax * (1.0 - 0.25 * pressure)),
    )
    current_confidence = _safe_float(boosted.get("confidence", 0.0), default=0.0)
    confidence_boost = base_confidence_boost * rank_multiplier * (1.0 + 0.5 * pressure)
    boosted["confidence"] = float(np.clip(current_confidence + confidence_boost, 0.0, 0.78))
    boosted["activity_target_mode"] = True
    boosted["idle_minutes"] = round(float(minutes_idle), 3)
    if candidate_rank == 1:
        boosted["force_candidate"] = True
    return boosted
