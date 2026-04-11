import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

from brain_log_routing import append_csv_with_brain_mirrors
from pnl_engine import PNLEngine
from trade_quality import build_quality_context, resolve_entry_signal_label


class TradeState(str, Enum):
    NEW_SIGNAL = "NEW_SIGNAL"
    ENTERED = "ENTERED"
    OPEN = "OPEN"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    CLOSED = "CLOSED"
    RESOLVED = "RESOLVED"


def _json_safe_value(value):
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except Exception:
            pass
    return value


def serialize_signal_snapshot(signal_row: dict) -> tuple[str, int]:
    normalized = {str(key): _json_safe_value(value) for key, value in dict(signal_row or {}).items()}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True), len(normalized)


@dataclass
class TradeLifecycle:
    market: str
    token_id: str | None
    condition_id: str | None
    outcome_side: str
    logs_dir: str = "logs"
    state: TradeState = TradeState.NEW_SIGNAL
    size_usdc: float = 0.0
    shares: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    opened_at: str | None = None
    closed_at: str | None = None
    close_reason: str | None = None
    confidence_at_entry: float = 0.0
    signal_label: str = "UNKNOWN"
    entry_signal_snapshot_json: str = ""
    entry_signal_snapshot_feature_count: int = 0
    entry_signal_snapshot_version: int = 1
    source_wallet: str = ""
    source_wallet_direction_confidence: float = 0.0
    source_wallet_position_event: str = ""
    source_wallet_quality_score: float = 0.0
    entry_btc_trend_bias: str = "NEUTRAL"
    entry_btc_predicted_direction: int = 0
    entry_btc_predicted_return: float = 0.0
    entry_btc_forecast_confidence: float = 0.0
    entry_btc_price: float = 0.0
    entry_btc_mtf_agreement: float = 0.0
    entry_btc_mtf_source: str = ""
    exit_btc_price: float = 0.0
    entry_alligator_alignment: str = "NEUTRAL"
    entry_adx_value: float = 0.0
    entry_adx_threshold: float = 0.0
    entry_anchored_vwap: float = 0.0
    entry_fractal_trigger_direction: str = "NEUTRAL"
    entry_model_family: str = ""
    entry_model_version: str = ""
    performance_governor_level: int = 0
    market_family: str = "other"
    brain_id: str = ""
    active_model_group: str = ""
    active_model_kind: str = ""
    active_regime: str = ""
    horizon_bucket: str = "unknown"
    liquidity_bucket: str = "unknown"
    volatility_bucket: str = "unknown"
    technical_regime_bucket: str = "neutral"
    entry_context_complete: bool = False
    operational_close_flag: bool = False
    learning_eligible: bool = False
    exit_reason_family: str = "unknown"
    intended_exit_reason: str | None = None
    actual_execution_path: str | None = None
    exit_fill_latency_seconds: float = 0.0
    exit_cancel_count: int = 0
    exit_partial_fill_ratio: float = 0.0
    exit_realized_slippage_bps: float = 0.0
    max_adverse_excursion_pct: float = 0.0
    max_favorable_excursion_pct: float = 0.0
    max_drawdown_from_peak_pct: float = 0.0
    fast_adverse_move_count: int = 0
    last_fast_adverse_move_at: str | None = None
    ledger: list = field(default_factory=list)

    def _write_execution_event(self, payload: dict):
        logs_path = Path(self.logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        # Keep lifecycle events isolated from execution_log.csv so fill/trade logs
        # keep a stable schema for dashboards, dataset builders, and reconciliations.
        events_file = logs_path / "trade_events.csv"
        append_csv_with_brain_mirrors(
            events_file,
            pd.DataFrame([payload]),
            shared_logs_dir=logs_path,
            include_shared=True,
        )

    def on_signal(self, signal_row: dict):
        normalized_signal_row = dict(signal_row or {})
        normalized_signal_row["signal_label"] = resolve_entry_signal_label(normalized_signal_row)
        signal_snapshot_json, signal_snapshot_feature_count = serialize_signal_snapshot(normalized_signal_row)
        payload = {
            "event": "signal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market": self.market,
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "outcome_side": self.outcome_side,
            "signal_snapshot_json": signal_snapshot_json,
            "signal_snapshot_feature_count": signal_snapshot_feature_count,
            "signal_snapshot_version": self.entry_signal_snapshot_version,
        }
        self.ledger.append({**payload, "signal": normalized_signal_row})
        self._write_execution_event(payload)
        self.state = TradeState.NEW_SIGNAL
        self.confidence_at_entry = float(normalized_signal_row.get("confidence", 0.0) or 0.0)
        self.signal_label = str(normalized_signal_row.get("signal_label", "UNKNOWN") or "UNKNOWN")
        self.entry_signal_snapshot_json = signal_snapshot_json
        self.entry_signal_snapshot_feature_count = signal_snapshot_feature_count
        self.source_wallet = str(normalized_signal_row.get("trader_wallet", "") or "")
        self.source_wallet_direction_confidence = float(normalized_signal_row.get("source_wallet_direction_confidence", 0.0) or 0.0)
        self.source_wallet_position_event = str(normalized_signal_row.get("source_wallet_position_event", "") or "")
        self.source_wallet_quality_score = float(normalized_signal_row.get("wallet_quality_score", 0.0) or 0.0)
        self.entry_btc_trend_bias = str(normalized_signal_row.get("btc_trend_bias", "NEUTRAL") or "NEUTRAL")
        self.entry_btc_predicted_direction = int(normalized_signal_row.get("btc_predicted_direction", 0) or 0)
        self.entry_btc_predicted_return = float(normalized_signal_row.get("btc_predicted_return_15", 0.0) or 0.0)
        self.entry_btc_forecast_confidence = float(normalized_signal_row.get("btc_forecast_confidence", 0.0) or 0.0)
        self.entry_btc_price = float(normalized_signal_row.get("btc_live_price", 0.0) or 0.0)
        self.entry_btc_mtf_agreement = float(normalized_signal_row.get("btc_mtf_agreement", 0.0) or 0.0)
        self.entry_btc_mtf_source = str(normalized_signal_row.get("btc_mtf_source", "") or "")
        self.entry_alligator_alignment = str(normalized_signal_row.get("alligator_alignment", "NEUTRAL") or "NEUTRAL")
        self.entry_adx_value = float(normalized_signal_row.get("adx_value", 0.0) or 0.0)
        self.entry_adx_threshold = float(normalized_signal_row.get("adx_threshold", 0.0) or 0.0)
        self.entry_anchored_vwap = float(normalized_signal_row.get("anchored_vwap", 0.0) or 0.0)
        self.entry_fractal_trigger_direction = str(normalized_signal_row.get("fractal_trigger_direction", "NEUTRAL") or "NEUTRAL")
        self.entry_model_family = str(normalized_signal_row.get("entry_model_family", "") or "")
        self.entry_model_version = str(normalized_signal_row.get("entry_model_version", "") or "")
        self.performance_governor_level = int(normalized_signal_row.get("performance_governor_level", 0) or 0)
        quality_context = build_quality_context(normalized_signal_row)
        self.market_family = str(normalized_signal_row.get("market_family", quality_context.get("market_family", "other")) or "other")
        self.brain_id = str(normalized_signal_row.get("brain_id", self.brain_id or "") or "")
        self.active_model_group = str(normalized_signal_row.get("active_model_group", self.active_model_group or "") or "")
        self.active_model_kind = str(normalized_signal_row.get("active_model_kind", self.active_model_kind or "") or "")
        self.active_regime = str(normalized_signal_row.get("active_regime", self.active_regime or "") or "")
        self.horizon_bucket = str(normalized_signal_row.get("horizon_bucket", quality_context.get("horizon_bucket", "unknown")) or "unknown")
        self.liquidity_bucket = str(normalized_signal_row.get("liquidity_bucket", quality_context.get("liquidity_bucket", "unknown")) or "unknown")
        self.volatility_bucket = str(normalized_signal_row.get("volatility_bucket", quality_context.get("volatility_bucket", "unknown")) or "unknown")
        self.technical_regime_bucket = str(normalized_signal_row.get("technical_regime_bucket", quality_context.get("technical_regime_bucket", "neutral")) or "neutral")
        self.entry_context_complete = bool(
            normalized_signal_row.get("entry_context_complete", quality_context.get("entry_context_complete", False))
        )
        for key, value in normalized_signal_row.items():
            if str(key).startswith(("weather_", "forecast_")):
                setattr(self, key, value)

    def enter(self, size_usdc: float, entry_price: float):
        self.size_usdc = float(size_usdc)
        self.entry_price = float(entry_price)
        self.current_price = float(entry_price)
        self.shares = PNLEngine.shares_from_capital(size_usdc, entry_price)
        self.opened_at = datetime.now(timezone.utc).isoformat()
        self.state = TradeState.ENTERED
        self.peak_price = self.entry_price
        self.max_adverse_excursion_pct = 0.0
        self.max_favorable_excursion_pct = 0.0
        self.max_drawdown_from_peak_pct = 0.0
        self.fast_adverse_move_count = 0
        self.last_fast_adverse_move_at = None
        payload = {
            "event": "enter",
            "timestamp": self.opened_at,
            "market": self.market,
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "outcome_side": self.outcome_side,
            "size_usdc": self.size_usdc,
            "entry_price": self.entry_price,
            "shares": self.shares,
        }
        self.ledger.append(payload)
        self._write_execution_event(payload)

    def update_market(self, live_price: float):
        self.current_price = float(live_price)
        self.peak_price = max(float(getattr(self, "peak_price", self.entry_price or self.current_price)), self.current_price)
        self.unrealized_pnl = self.shares * (self.current_price - self.entry_price)
        return_pct = ((self.current_price - self.entry_price) / self.entry_price) if self.entry_price > 0 else 0.0
        self.max_adverse_excursion_pct = min(float(getattr(self, "max_adverse_excursion_pct", 0.0) or 0.0), return_pct)
        self.max_favorable_excursion_pct = max(float(getattr(self, "max_favorable_excursion_pct", 0.0) or 0.0), return_pct)
        drawdown_from_peak = ((self.peak_price - self.current_price) / self.peak_price) if self.peak_price > 0 else 0.0
        self.max_drawdown_from_peak_pct = max(float(getattr(self, "max_drawdown_from_peak_pct", 0.0) or 0.0), drawdown_from_peak)
        try:
            opened_dt = datetime.fromisoformat(str(self.opened_at)) if self.opened_at else None
        except Exception:
            opened_dt = None
        if opened_dt is not None:
            now_dt = datetime.now(timezone.utc)
            if opened_dt.tzinfo is None:
                opened_dt = opened_dt.replace(tzinfo=timezone.utc)
            minutes_open = max(0.0, (now_dt.astimezone(timezone.utc) - opened_dt.astimezone(timezone.utc)).total_seconds() / 60.0)
            if minutes_open <= 10 and return_pct <= -0.02:
                previous_mae = float(getattr(self, "_last_fast_adverse_recorded_mae", 0.0) or 0.0)
                if return_pct < previous_mae - 1e-9:
                    self.fast_adverse_move_count = int(getattr(self, "fast_adverse_move_count", 0) or 0) + 1
                    self.last_fast_adverse_move_at = now_dt.isoformat()
                    self._last_fast_adverse_recorded_mae = return_pct
        if self.state in [TradeState.ENTERED, TradeState.PARTIAL_EXIT]:
            self.state = TradeState.OPEN
        self.ledger.append({
            "event": "mark",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "live_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "return_pct": return_pct,
            "max_adverse_excursion_pct": self.max_adverse_excursion_pct,
            "max_drawdown_from_peak_pct": self.max_drawdown_from_peak_pct,
            "fast_adverse_move_count": self.fast_adverse_move_count,
        })
        return self.unrealized_pnl

    def partial_exit(self, fraction: float, exit_price: float):
        fraction = max(0.0, min(1.0, float(fraction)))
        if fraction <= 0 or self.shares <= 0:
            return 0.0
        exited_shares = self.shares * fraction
        pnl = exited_shares * (float(exit_price) - float(self.entry_price))
        self.realized_pnl += pnl
        self.shares = max(0.0, self.shares - exited_shares)
        self.size_usdc = max(0.0, self.shares * float(self.entry_price))
        self.current_price = float(exit_price)
        self.unrealized_pnl = self.shares * (self.current_price - self.entry_price)
        self.state = TradeState.PARTIAL_EXIT if self.shares > 0 else TradeState.CLOSED
        self.ledger.append({
            "event": "partial_exit",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fraction": fraction,
            "exit_price": exit_price,
            "remaining_shares": self.shares,
            "realized_pnl": pnl,
        })
        if self.shares <= 0:
            self.closed_at = datetime.now(timezone.utc).isoformat()
        return pnl

    def close(self, exit_price: float, reason: str = "policy_exit", exit_btc_price: float = 0.0):
        pnl = self.shares * (float(exit_price) - float(self.entry_price))
        self.realized_pnl += pnl
        self.current_price = float(exit_price)
        self.unrealized_pnl = 0.0
        self.closed_at = datetime.now(timezone.utc).isoformat()
        if exit_btc_price > 0:
            self.exit_btc_price = float(exit_btc_price)
        # self.shares = 0.0 # BUG 1 FIX: Keep shares intact so supervisor knows how much to sell
        # self.size_usdc = 0.0
        self.state = TradeState.CLOSED
        self.close_reason = reason
        quality_context = build_quality_context(
            {
                "close_reason": reason,
                "signal_label": self.signal_label,
                "confidence_at_entry": self.confidence_at_entry,
                "entry_model_family": self.entry_model_family,
                "entry_model_version": self.entry_model_version,
                "market_family": self.market_family,
                "horizon_bucket": self.horizon_bucket,
                "liquidity_bucket": self.liquidity_bucket,
                "volatility_bucket": self.volatility_bucket,
                "technical_regime_bucket": self.technical_regime_bucket,
            }
        )
        self.exit_reason_family = quality_context.get("exit_reason_family", "unknown")
        self.operational_close_flag = bool(quality_context.get("operational_close_flag", False))
        self.entry_context_complete = bool(quality_context.get("entry_context_complete", self.entry_context_complete))
        self.learning_eligible = bool(quality_context.get("learning_eligible", False))
        self.intended_exit_reason = self.intended_exit_reason or reason
        self.ledger.append({
            "event": "close",
            "timestamp": self.closed_at,
            "exit_price": exit_price,
            "realized_pnl": pnl,
            "close_reason": reason,
        })
        self._write_execution_event({
            "event": "close",
            "timestamp": self.closed_at,
            "market": self.market,
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "outcome_side": self.outcome_side,
            "exit_price": exit_price,
            "realized_pnl": self.realized_pnl,
            "close_reason": reason,
        })
        return pnl

    def resolve(self, token_won: bool):
        pnl = PNLEngine.resolution_pnl(self.size_usdc, self.entry_price, token_won)
        self.realized_pnl = pnl
        self.unrealized_pnl = 0.0
        self.state = TradeState.RESOLVED
        self.closed_at = datetime.now(timezone.utc).isoformat()
        self.ledger.append({
            "event": "resolve",
            "timestamp": self.closed_at,
            "token_won": token_won,
            "realized_pnl": self.realized_pnl,
        })
        return self.realized_pnl
