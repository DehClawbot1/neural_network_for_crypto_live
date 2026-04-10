from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from balance_normalization import maybe_trace_allowance_payload
from csv_utils import safe_csv_append
from db import Database
from trade_lifecycle import TradeLifecycle, TradeState
from trade_quality import build_quality_context, classify_exit_reason_family, resolve_entry_signal_label
from config import TradingConfig

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Manages a collection of TradeLifecycle instances, acting as the central
    runtime trade manager for the supervisor.

    BUG FIXES APPLIED:
      C - persist confidence_at_entry + signal_label to positions.csv
      D - pass real close_reason (not hardcoded "policy_exit")
      E - TradeLifecycle.close() now accepts reason param
      F - write both realized_pnl AND net_realized_pnl to closed_positions.csv
      H - consistent ISO timestamp format
    """

    def __init__(self, logs_dir="logs"):
        self.active_trades: Dict[str, TradeLifecycle] = {}
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.positions_file = self.logs_dir / "positions.csv"
        self.closed_file = self.logs_dir / "closed_positions.csv"
        self.db = Database(self.logs_dir / "trading.db")
        self._empty_reconciled_streak = 0
        self._max_empty_reconciled_streak = max(
            1,
            int(os.getenv("MAX_EMPTY_RECONCILED_STREAK", "3") or 3),
        )
        logger.info("[+] Initialized TradeManager.")

    def _env_float(self, name: str, default: float, minimum: float = 0.0, maximum: float = 1_000_000.0) -> float:
        try:
            value = float(os.getenv(name, str(default)) or default)
        except Exception:
            value = float(default)
        value = max(float(minimum), value)
        value = min(float(maximum), value)
        return value

    def _env_int(self, name: str, default: int, minimum: int = 0, maximum: int = 1_000_000) -> int:
        try:
            value = int(os.getenv(name, str(default)) or default)
        except Exception:
            value = int(default)
        value = max(int(minimum), value)
        value = min(int(maximum), value)
        return value

    @staticmethod
    def _load_entry_snapshot(snapshot_json: str) -> dict:
        text = str(snapshot_json or "").strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _hydrate_trade_from_entry_snapshot(self, trade: TradeLifecycle, snapshot_json: str):
        snapshot = self._load_entry_snapshot(snapshot_json)
        if not snapshot:
            return
        trade.confidence_at_entry = float(snapshot.get("confidence", trade.confidence_at_entry) or trade.confidence_at_entry or 0.0)
        trade.signal_label = str(snapshot.get("signal_label", trade.signal_label or "UNKNOWN") or "UNKNOWN")
        trade.source_wallet = str(snapshot.get("trader_wallet", trade.source_wallet or "") or "")
        trade.source_wallet_direction_confidence = float(
            snapshot.get("source_wallet_direction_confidence", trade.source_wallet_direction_confidence) or trade.source_wallet_direction_confidence or 0.0
        )
        trade.source_wallet_position_event = str(snapshot.get("source_wallet_position_event", trade.source_wallet_position_event or "") or "")
        trade.source_wallet_quality_score = float(
            snapshot.get("wallet_quality_score", trade.source_wallet_quality_score) or trade.source_wallet_quality_score or 0.0
        )
        trade.entry_btc_trend_bias = str(snapshot.get("btc_trend_bias", trade.entry_btc_trend_bias or "NEUTRAL") or "NEUTRAL")
        trade.entry_btc_predicted_direction = int(snapshot.get("btc_predicted_direction", trade.entry_btc_predicted_direction) or trade.entry_btc_predicted_direction or 0)
        trade.entry_btc_predicted_return = float(
            snapshot.get("btc_predicted_return_15", trade.entry_btc_predicted_return) or trade.entry_btc_predicted_return or 0.0
        )
        trade.entry_btc_forecast_confidence = float(
            snapshot.get("btc_forecast_confidence", trade.entry_btc_forecast_confidence) or trade.entry_btc_forecast_confidence or 0.0
        )
        trade.entry_btc_price = float(snapshot.get("btc_live_price", trade.entry_btc_price) or trade.entry_btc_price or 0.0)
        trade.entry_btc_mtf_agreement = float(snapshot.get("btc_mtf_agreement", trade.entry_btc_mtf_agreement) or trade.entry_btc_mtf_agreement or 0.0)
        trade.entry_btc_mtf_source = str(snapshot.get("btc_mtf_source", trade.entry_btc_mtf_source or "") or "")
        trade.entry_alligator_alignment = str(snapshot.get("alligator_alignment", trade.entry_alligator_alignment or "NEUTRAL") or "NEUTRAL")
        trade.entry_adx_value = float(snapshot.get("adx_value", trade.entry_adx_value) or trade.entry_adx_value or 0.0)
        trade.entry_adx_threshold = float(snapshot.get("adx_threshold", trade.entry_adx_threshold) or trade.entry_adx_threshold or 0.0)
        trade.entry_anchored_vwap = float(snapshot.get("anchored_vwap", trade.entry_anchored_vwap) or trade.entry_anchored_vwap or 0.0)
        trade.entry_fractal_trigger_direction = str(snapshot.get("fractal_trigger_direction", trade.entry_fractal_trigger_direction or "NEUTRAL") or "NEUTRAL")
        trade.entry_model_family = str(snapshot.get("entry_model_family", trade.entry_model_family or "") or "")
        trade.entry_model_version = str(snapshot.get("entry_model_version", trade.entry_model_version or "") or "")
        trade.performance_governor_level = int(snapshot.get("performance_governor_level", trade.performance_governor_level) or trade.performance_governor_level or 0)
        trade.market_family = str(snapshot.get("market_family", trade.market_family or "other") or "other")
        trade.horizon_bucket = str(snapshot.get("horizon_bucket", trade.horizon_bucket or "unknown") or "unknown")
        trade.liquidity_bucket = str(snapshot.get("liquidity_bucket", trade.liquidity_bucket or "unknown") or "unknown")
        trade.volatility_bucket = str(snapshot.get("volatility_bucket", trade.volatility_bucket or "unknown") or "unknown")
        trade.technical_regime_bucket = str(snapshot.get("technical_regime_bucket", trade.technical_regime_bucket or "neutral") or "neutral")
        trade.entry_context_complete = bool(snapshot.get("entry_context_complete", trade.entry_context_complete))
        for key, value in snapshot.items():
            if str(key).startswith(("weather_", "forecast_")):
                setattr(trade, key, value)

    def _resolve_exit_thresholds(self) -> dict:
        open_count = len(self.get_open_positions())

        thresholds = {
            "tp_roi": self._env_float("EXIT_TP_ROI", 0.06, minimum=0.001, maximum=1.0),
            "tp_delta": self._env_float("EXIT_TP_DELTA", 0.03, minimum=0.001, maximum=1.0),
            "sl_delta": self._env_float("EXIT_SL_DELTA", 0.02, minimum=0.001, maximum=1.0),
            "time_stop_minutes": self._env_int("EXIT_TIME_STOP_MINUTES", 90, minimum=1, maximum=10_000),
            "trailing_stop": self._env_float("EXIT_TRAILING_STOP", 0.05, minimum=0.001, maximum=1.0),
            "full_book_threshold": self._env_int("FULL_BOOK_POSITION_COUNT", 5, minimum=1, maximum=1_000),
            "full_book_mode": False,
        }

        if open_count >= thresholds["full_book_threshold"]:
            thresholds["full_book_mode"] = True
            thresholds["tp_roi"] = min(
                thresholds["tp_roi"],
                self._env_float("FULL_BOOK_EXIT_TP_ROI", 0.045, minimum=0.001, maximum=1.0),
            )
            thresholds["tp_delta"] = min(
                thresholds["tp_delta"],
                self._env_float("FULL_BOOK_EXIT_TP_DELTA", 0.025, minimum=0.001, maximum=1.0),
            )
            thresholds["sl_delta"] = min(
                thresholds["sl_delta"],
                self._env_float("FULL_BOOK_EXIT_SL_DELTA", 0.018, minimum=0.001, maximum=1.0),
            )
            thresholds["time_stop_minutes"] = min(
                thresholds["time_stop_minutes"],
                self._env_int("FULL_BOOK_EXIT_TIME_STOP_MINUTES", 60, minimum=1, maximum=10_000),
            )
            thresholds["trailing_stop"] = min(
                thresholds["trailing_stop"],
                self._env_float("FULL_BOOK_EXIT_TRAILING_STOP", 0.035, minimum=0.001, maximum=1.0),
            )

        thresholds["open_count"] = open_count
        return thresholds

    @staticmethod
    def _trade_direction(outcome_side: str | None) -> str:
        side = str(outcome_side or "").strip().upper()
        if side in {"YES", "UP", "LONG", "BULLISH"}:
            return "LONG"
        if side in {"NO", "DOWN", "SHORT", "BEARISH"}:
            return "SHORT"
        return "NEUTRAL"

    def _apply_closed_trade_metadata(self, trade: TradeLifecycle):
        quality_context = build_quality_context(
            {
                "market": trade.market,
                "signal_label": getattr(trade, "signal_label", None),
                "confidence_at_entry": getattr(trade, "confidence_at_entry", None),
                "entry_model_family": getattr(trade, "entry_model_family", None),
                "entry_model_version": getattr(trade, "entry_model_version", None),
                "market_family": getattr(trade, "market_family", None),
                "horizon_bucket": getattr(trade, "horizon_bucket", None),
                "liquidity_bucket": getattr(trade, "liquidity_bucket", None),
                "volatility_bucket": getattr(trade, "volatility_bucket", None),
                "technical_regime_bucket": getattr(trade, "technical_regime_bucket", None),
                "close_reason": getattr(trade, "close_reason", None),
            }
        )
        trade.market_family = quality_context.get("market_family", getattr(trade, "market_family", "other"))
        trade.horizon_bucket = quality_context.get("horizon_bucket", getattr(trade, "horizon_bucket", "unknown"))
        trade.liquidity_bucket = quality_context.get("liquidity_bucket", getattr(trade, "liquidity_bucket", "unknown"))
        trade.volatility_bucket = quality_context.get("volatility_bucket", getattr(trade, "volatility_bucket", "unknown"))
        trade.technical_regime_bucket = quality_context.get("technical_regime_bucket", getattr(trade, "technical_regime_bucket", "neutral"))
        trade.exit_reason_family = quality_context.get("exit_reason_family", classify_exit_reason_family(getattr(trade, "close_reason", None)))
        trade.operational_close_flag = bool(quality_context.get("operational_close_flag", False))
        trade.entry_context_complete = bool(quality_context.get("entry_context_complete", False))
        trade.learning_eligible = bool(quality_context.get("learning_eligible", False))
        return trade

    def _technical_exit_reason(self, trade: TradeLifecycle, technical_context: dict | None, minutes_open: float) -> str | None:
        if not technical_context:
            return None

        min_minutes = self._env_int("TECHNICAL_EXIT_MIN_MINUTES", 10, minimum=0, maximum=10_000)
        if minutes_open < min_minutes:
            return None

        direction = self._trade_direction(getattr(trade, "outcome_side", None))
        if direction == "NEUTRAL":
            return None

        alligator_alignment = str(technical_context.get("alligator_alignment", "NEUTRAL") or "NEUTRAL").strip().upper()
        price_above_vwap = bool(technical_context.get("price_above_anchored_vwap", False))
        price_below_vwap = bool(technical_context.get("price_below_anchored_vwap", False))

        try:
            adx_value = float(technical_context.get("adx_value", 0.0) or 0.0)
        except Exception:
            adx_value = 0.0
        try:
            adx_threshold = float(technical_context.get("adx_threshold", 0.0) or 0.0)
        except Exception:
            adx_threshold = 0.0
        try:
            entry_adx_value = float(getattr(trade, "entry_adx_value", 0.0) or 0.0)
        except Exception:
            entry_adx_value = 0.0

        opposite_alligator = (
            (direction == "LONG" and alligator_alignment == "BEARISH")
            or (direction == "SHORT" and alligator_alignment == "BULLISH")
        )
        vwap_recross = (
            (direction == "LONG" and price_below_vwap)
            or (direction == "SHORT" and price_above_vwap)
        )
        adx_floor = max(adx_threshold, entry_adx_value * 0.85 if entry_adx_value > 0 else 0.0)
        adx_weakening = adx_value > 0 and adx_floor > 0 and adx_value < adx_floor

        if opposite_alligator:
            return "technical_alligator_reversal"
        if vwap_recross:
            return "technical_vwap_recross"
        if adx_weakening:
            return "technical_adx_weakening"
        return None

    def _compose_trade_key(self, token_id=None, condition_id=None, outcome_side=None, market=None) -> Optional[str]:
        token_id = str(token_id).strip() if token_id not in [None, ""] else ""
        condition_id = str(condition_id).strip() if condition_id not in [None, ""] else ""
        outcome_side = str(outcome_side).strip() if outcome_side not in [None, ""] else ""
        market = str(market).strip() if market not in [None, ""] else ""
        if token_id or condition_id:
            return f"{token_id}|{condition_id}|{outcome_side}"
        if market and outcome_side:
            return f"{market}|{outcome_side}"
        return None

    def _prune_local_dust_trades(self, min_notional=0.01) -> int:
        removed = 0
        for key, trade in list(self.active_trades.items()):
            try:
                shares = float(getattr(trade, "shares", 0.0) or 0.0)
                px = float(getattr(trade, "current_price", 0.0) or getattr(trade, "entry_price", 0.0) or 0.0)
                if shares <= 0 or (shares * px) < float(min_notional):
                    trade.state = TradeState.CLOSED
                    trade.close_reason = trade.close_reason or "local_dust_pruned"
                    self.active_trades.pop(key, None)
                    removed += 1
            except Exception:
                continue
        return removed
    def _get_trade_key(self, signal_row: pd.Series) -> Optional[str]:
        if str(signal_row.get("action", "BUY")).upper() != "BUY":
            return None # BUG FIX 3: Prevent opening new trades on EXIT signals

        market = signal_row.get("market_title") or signal_row.get("market")
        outcome_side = signal_row.get("outcome_side") or signal_row.get("side")
        token_id = signal_row.get("token_id")
        condition_id = signal_row.get("condition_id")
        trade_key = self._compose_trade_key(
            token_id=token_id,
            condition_id=condition_id,
            outcome_side=outcome_side,
            market=market,
        )
        if not trade_key:
            logger.warning("Missing token/condition/market or outcome_side in signal: %s", dict(signal_row) if hasattr(signal_row, 'items') else signal_row)
            return None
        return trade_key


    def handle_signal(self, signal_row: pd.Series, confidence: float, size_usdc: float, entry_price_override: float | None = None) -> Optional[TradeLifecycle]:
        signal_row = signal_row.copy()
        signal_row["signal_label"] = resolve_entry_signal_label(
            signal_row.to_dict() if hasattr(signal_row, "to_dict") else dict(signal_row)
        )
        trade_key = self._get_trade_key(signal_row)
        if trade_key is None:
            return None

        if str(signal_row.get("action", "BUY")).upper() != "BUY":
            return None # BUG FIX 3: Prevent opening new trades on EXIT signals

        market = signal_row.get("market_title") or signal_row.get("market")
        outcome_side = signal_row.get("outcome_side") or signal_row.get("side")
        token_id = signal_row.get("token_id")
        condition_id = signal_row.get("condition_id")

        entry_price = entry_price_override if entry_price_override is not None else (
            signal_row.get("current_price")
            or signal_row.get("entry_price")
            or signal_row.get("market_last_trade_price")
            or signal_row.get("price")
        )
        if entry_price is None or float(entry_price) <= 0:
            logger.warning("Cannot open trade — entry_price is missing or zero for %s", market)
            return None
        entry_price = float(entry_price)

        if trade_key in self.active_trades:
            trade = self.active_trades[trade_key]
            logger.info("[!] Trade already open for %s (%s). Skipping new entry.", market, outcome_side)
            return trade
        else:
            trade = TradeLifecycle(
                market=market,
                token_id=token_id,
                condition_id=condition_id,
                outcome_side=outcome_side,
            )
            trade.on_signal(signal_row.to_dict() if hasattr(signal_row, 'to_dict') else dict(signal_row))
            trade.confidence_at_entry = float(confidence)
            trade.signal_label = str(signal_row.get("signal_label", "UNKNOWN") or "UNKNOWN")
            trade.enter(size_usdc=size_usdc, entry_price=entry_price)
            self.active_trades[trade_key] = trade
            logger.info("[+] New trade initiated for %s (%s) with %s USDC at %.4f.",
                        market, outcome_side, size_usdc, entry_price)
            return trade

    def update_markets(self, market_prices: Dict[str, float]):
        for trade_key, trade in list(self.active_trades.items()):
            if trade.market in market_prices:
                live_price = market_prices[trade.market]
                trade.update_market(live_price)

    def process_exits(
        self,
        current_timestamp: datetime,
        alerts_df: pd.DataFrame = None,
        execution_client=None,
        persist_closed: bool = True,
        predictive_exit_targets: Dict[str, float] | None = None,
        trajectory_metrics: Dict[str, dict] | None = None,
        technical_context: dict | None = None,
    ):
        closed_trades: List[TradeLifecycle] = []
        close_reasons: Dict[str, str] = {}
        exit_thresholds = self._resolve_exit_thresholds()
        logger.info(
            "Exit policy active: open_positions=%s full_book=%s tp_roi=%.4f tp_delta=%.4f sl_delta=%.4f trailing=%.4f time_stop=%sm",
            exit_thresholds["open_count"],
            exit_thresholds["full_book_mode"],
            exit_thresholds["tp_roi"],
            exit_thresholds["tp_delta"],
            exit_thresholds["sl_delta"],
            exit_thresholds["trailing_stop"],
            exit_thresholds["time_stop_minutes"],
        )

        for trade_key, trade in list(self.active_trades.items()):
            if trade.state == TradeState.CLOSED:
                closed_trades.append(trade)
                close_reasons[trade_key] = trade.close_reason or "already_closed"
                continue

            if not trade.opened_at:
                continue

            try:
                opened_dt = datetime.fromisoformat(trade.opened_at)
            except (ValueError, TypeError):
                continue

            if opened_dt.tzinfo is not None:
                current_ts = current_timestamp if current_timestamp.tzinfo is not None else current_timestamp.replace(tzinfo=timezone.utc)
                opened_dt = opened_dt.astimezone(timezone.utc)
                current_ts = current_ts.astimezone(timezone.utc)
            else:
                current_ts = current_timestamp.replace(tzinfo=timezone.utc) if current_timestamp.tzinfo is None else current_timestamp # BUG FIX 8: Enforce UTC safely
            if opened_dt.tzinfo is None: opened_dt = opened_dt.replace(tzinfo=timezone.utc)

            entry_price = float(trade.entry_price or 0)
            current_price = float(trade.current_price or entry_price)
            if entry_price <= 0:
                continue

            roi = (current_price - entry_price) / entry_price
            minutes_open = (current_ts - opened_dt).total_seconds() / 60.0
            
            # PATCHED: Calculate true trailing stop using peak price
            if not hasattr(trade, 'peak_price'):
                trade.peak_price = entry_price
            if current_price > trade.entry_price: # BUG FIX 9: Only raise peak on real profit, ignoring wide spreads at entry
                trade.peak_price = max(trade.peak_price, current_price)
            trailing_drop = (trade.peak_price - current_price) / trade.peak_price if trade.peak_price > 0 else 0

            close_reason = None

            # Model-driven take-profit target (if provided by supervisor).
            # Key format matches _compose_trade_key.
            predicted_target_price = None
            if predictive_exit_targets:
                predicted_target_price = predictive_exit_targets.get(trade_key)
                if predicted_target_price is not None:
                    try:
                        predicted_target_price = float(predicted_target_price)
                    except Exception:
                        predicted_target_price = None
            trajectory_signal = trajectory_metrics.get(trade_key, {}) if trajectory_metrics else {}

            # --- STRICT SYNC: Verify external Polymarket balance ---
            # If you manually closed this trade on the Polymarket website, 
            # this will detect the missing shares and force the bot to close it locally.
            try:
                # Dynamically find the execution client reference
                client_ref = getattr(self, 'exec_client', getattr(self, 'client', getattr(self, 'execution_client', getattr(self, 'api', None))))
                if client_ref and hasattr(trade, 'token_id') and trade.token_id:
                    raw_bal = client_ref.get_balance_allowance(asset_type="CONDITIONAL", token_id=trade.token_id)
                    bal_raw_val = None
                    if isinstance(raw_bal, dict):
                        for key in ("balance", "available", "available_balance", "amount"):
                            if raw_bal.get(key) is not None:
                                bal_raw_val = raw_bal.get(key)
                                break
                    elif raw_bal is not None:
                        bal_raw_val = raw_bal

                    bal_val = 0.0
                    if bal_raw_val is not None:
                        try:
                            normalizer = getattr(client_ref, "_normalize_allowance_balance", None)
                            if callable(normalizer):
                                bal_val = float(normalizer(bal_raw_val, asset_type="CONDITIONAL"))
                            else:
                                bal_val = float(bal_raw_val)
                        except Exception:
                            try:
                                bal_val = float(bal_raw_val)
                            except Exception:
                                bal_val = 0.0
                    maybe_trace_allowance_payload(
                        logs_dir=self.logs_dir,
                        source="trade_manager.strict_sync",
                        asset_type="CONDITIONAL",
                        token_id=trade.token_id,
                        payload=raw_bal,
                        normalized_balance=bal_val,
                        local_balance=getattr(trade, "shares", 0.0),
                        note=f"close_eps={float(os.getenv('EXTERNAL_CLOSE_BALANCE_EPS_SHARES', '1e-6') or 1e-6)}",
                    )

                    # Treat near-zero exchange inventory as external/manual close.
                    # Use an explicit share epsilon instead of a hardcoded raw-unit threshold.
                    close_eps = float(os.getenv("EXTERNAL_CLOSE_BALANCE_EPS_SHARES", "1e-6") or 1e-6)
                    if bal_raw_val is not None and bal_val <= close_eps:
                        close_reason = "external_manual_close"
                        if hasattr(trade, 'unrealized_pnl'):
                            trade.unrealized_pnl = 0.0
            except Exception:
                pass # Safely ignore if API rate limits or client ref not found
            # -------------------------------------------------------

            hard_emergency_delta = self._env_float("EXIT_HARD_EMERGENCY_DELTA", 0.035, minimum=0.001, maximum=1.0)
            if close_reason is None and (entry_price - current_price) >= hard_emergency_delta:
                close_reason = "hard_emergency_stop"
            elif close_reason is None and bool(trajectory_signal.get("panic_exit_signal")):
                close_reason = "trajectory_panic_exit"
            elif close_reason is None and (entry_price - current_price) >= exit_thresholds["sl_delta"]:
                close_reason = "stop_loss"
            elif close_reason is None and bool(trajectory_signal.get("reversal_exit_signal")):
                close_reason = "trajectory_reversal_exit"
            elif close_reason is None:
                close_reason = self._technical_exit_reason(trade, technical_context, minutes_open)
            if close_reason is None and minutes_open >= exit_thresholds["time_stop_minutes"]:
                close_reason = "time_stop"
            elif close_reason is None and roi >= exit_thresholds["tp_roi"]:
                close_reason = "take_profit_roi"
            elif close_reason is None and (current_price - entry_price) >= exit_thresholds["tp_delta"]:
                close_reason = "take_profit_price_move"
            elif close_reason is None and predicted_target_price is not None and current_price >= predicted_target_price:
                close_reason = "take_profit_model_target"
            elif close_reason is None and trailing_drop >= exit_thresholds["trailing_stop"] and minutes_open > 15:
                close_reason = "trailing_stop"
            elif close_reason is None and roi > 0 and bool(trajectory_signal.get("liquidity_stress_signal")):
                close_reason = "trajectory_liquidity_stress"
            elif close_reason is None and roi > 0 and bool(trajectory_signal.get("profit_lock_signal")):
                close_reason = "trajectory_profit_lock"

            if close_reason:
                trade.intended_exit_reason = close_reason
                trade.actual_execution_path = trade.actual_execution_path or "rule_exit_pending_execution"
                trade.close(current_price, reason=close_reason)
                self._apply_closed_trade_metadata(trade)
                logger.info("[->] Closed trade for %s (%s). Reason: %s. PnL: %.4f",
                            trade.market, trade.outcome_side, close_reason, trade.realized_pnl)
                closed_trades.append(trade)
                close_reasons[trade_key] = close_reason

        for trade in closed_trades:
            key = self._compose_trade_key(token_id=trade.token_id, condition_id=trade.condition_id, outcome_side=trade.outcome_side, market=trade.market)
            if key:
                self.active_trades.pop(key, None)

        if closed_trades and persist_closed:
            self._append_closed_trades(closed_trades)

        return closed_trades

    def persist_closed_trades(self, closed_trades: List[TradeLifecycle]):
        self._append_closed_trades(closed_trades)

    def _maybe_load_reconciled_positions(self):
        try:
            from live_position_book import LivePositionBook
        except Exception:
            return

        try:
            live_book = LivePositionBook(logs_dir=str(self.logs_dir))
            live_book.rebuild_from_db()
            reconciled_positions_df = live_book.get_open_positions()
        except Exception as exc:
            logger.debug("Unable to load reconciled positions into TradeManager: %s", exc)
            return

        if reconciled_positions_df is None or reconciled_positions_df.empty:
            self._empty_reconciled_streak += 1
            if self.active_trades:
                removed = self._prune_local_dust_trades(min_notional=0.01)
                if removed > 0:
                    logger.info("Pruned %s local dust trades while reconciled positions were empty.", removed)
                if self.active_trades:
                    if self._empty_reconciled_streak >= self._max_empty_reconciled_streak:
                        logger.error(
                            "Reconciled positions empty for %s consecutive checks. Closing %s stale local trades to prevent ghost loops.",
                            self._empty_reconciled_streak,
                            len(self.active_trades),
                        )
                        for trade_key, trade in list(self.active_trades.items()):
                            trade.state = TradeState.CLOSED
                            trade.close_reason = trade.close_reason or "exchange_reconciliation_empty_streak"
                            self.active_trades.pop(trade_key, None)
                    else:
                        logger.warning("Reconciled positions empty. Keeping %s active_trades in memory to prevent amnesia.", len(self.active_trades))
            return
        self._empty_reconciled_streak = 0

        reconciled_count = len(reconciled_positions_df.index)
        current_open_count = sum(
            1 for trade in self.active_trades.values()
            if getattr(trade, "state", None) != TradeState.CLOSED
        )

        if current_open_count == reconciled_count and current_open_count > 0:
            current_keys = {
                self._compose_trade_key(
                    token_id=getattr(trade, "token_id", None),
                    condition_id=getattr(trade, "condition_id", None),
                    outcome_side=getattr(trade, "outcome_side", None),
                    market=getattr(trade, "market", None),
                )
                for trade in self.active_trades.values()
                if getattr(trade, "state", None) != TradeState.CLOSED
            }
            reconciled_keys = {
                self._compose_trade_key(
                    token_id=row.get("token_id"),
                    condition_id=row.get("condition_id"),
                    outcome_side=row.get("outcome_side") or row.get("side"),
                    market=row.get("market") or row.get("market_title"),
                )
                for _, row in reconciled_positions_df.iterrows()
            }
            if current_keys == reconciled_keys:
                return

        self.reconcile_live_positions(reconciled_positions_df=reconciled_positions_df)

    def get_open_positions(self) -> List[TradeLifecycle]:
        self._maybe_load_reconciled_positions()
        return [t for t in self.active_trades.values() if t.state != TradeState.CLOSED]

    def get_metrics(self) -> Dict[str, any]:
        open_trades = self.get_open_positions()
        total_unrealized_pnl = sum(trade.unrealized_pnl for trade in open_trades)
        return {
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_open_positions": len(open_trades),
            "last_reconciled_at": datetime.now().isoformat(),
        }

    def _trade_to_dict(self, trade: TradeLifecycle) -> dict:
        negotiated_value_usdc = float(getattr(trade, "size_usdc", 0.0) or 0.0)
        shares = float(getattr(trade, "shares", 0.0) or 0.0)
        entry_price = float(getattr(trade, "entry_price", 0.0) or 0.0)
        current_price = float(getattr(trade, "current_price", 0.0) or 0.0)
        market_value = shares * current_price if current_price else 0.0
        unrealized_pnl = float(getattr(trade, "unrealized_pnl", 0.0) or 0.0)
        max_payout_usdc = shares
        avg_to_now_price_change = current_price - entry_price if entry_price or current_price else 0.0
        avg_to_now_price_change_pct = (avg_to_now_price_change / entry_price) if entry_price > 0 else 0.0
        unrealized_pnl_pct = (unrealized_pnl / negotiated_value_usdc) if negotiated_value_usdc > 0 else 0.0
        row = {
            "position_id": self._canonical_position_id(
                token_id=trade.token_id,
                condition_id=trade.condition_id,
                outcome_side=trade.outcome_side,
                opened_at=trade.opened_at,
                market=trade.market,
            ),
            "market": trade.market,
            "market_title": trade.market,
            "token_id": trade.token_id,
            "condition_id": trade.condition_id,
            "outcome_side": trade.outcome_side,
            "order_side": "BUY",
            "entry_price": entry_price,
            "current_price": current_price,
            "size_usdc": negotiated_value_usdc,
            "negotiated_value_usdc": negotiated_value_usdc,
            "shares": shares,
            "max_payout_usdc": max_payout_usdc,
            "market_value": market_value,
            "current_value_usdc": market_value,
            "unrealized_pnl": round(unrealized_pnl, 4),
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 6),
            "avg_to_now_price_change": round(avg_to_now_price_change, 6),
            "avg_to_now_price_change_pct": round(avg_to_now_price_change_pct, 6),
            "realized_pnl": round(trade.realized_pnl, 4),
            "net_realized_pnl": round(trade.realized_pnl, 4),
            "opened_at": trade.opened_at,
            "closed_at": trade.closed_at,
            "status": trade.state.value if hasattr(trade.state, 'value') else str(trade.state),
            "confidence": trade.confidence_at_entry,
            "confidence_at_entry": trade.confidence_at_entry,
            "signal_label": trade.signal_label,
            "entry_signal_snapshot_json": getattr(trade, "entry_signal_snapshot_json", ""),
            "entry_signal_snapshot_feature_count": getattr(trade, "entry_signal_snapshot_feature_count", 0),
            "entry_signal_snapshot_version": getattr(trade, "entry_signal_snapshot_version", 1),
            "entry_btc_predicted_direction": getattr(trade, "entry_btc_predicted_direction", 0),
            "entry_btc_predicted_return": getattr(trade, "entry_btc_predicted_return", 0.0),
            "entry_btc_forecast_confidence": getattr(trade, "entry_btc_forecast_confidence", 0.0),
            "entry_btc_price": getattr(trade, "entry_btc_price", 0.0),
            "entry_btc_mtf_agreement": getattr(trade, "entry_btc_mtf_agreement", 0.0),
            "entry_btc_mtf_source": getattr(trade, "entry_btc_mtf_source", ""),
            "exit_btc_price": getattr(trade, "exit_btc_price", 0.0),
            "entry_model_family": getattr(trade, "entry_model_family", ""),
            "entry_model_version": getattr(trade, "entry_model_version", ""),
            "performance_governor_level": getattr(trade, "performance_governor_level", 0),
            "market_family": getattr(trade, "market_family", "other"),
            "horizon_bucket": getattr(trade, "horizon_bucket", "unknown"),
            "liquidity_bucket": getattr(trade, "liquidity_bucket", "unknown"),
            "volatility_bucket": getattr(trade, "volatility_bucket", "unknown"),
            "technical_regime_bucket": getattr(trade, "technical_regime_bucket", "neutral"),
            "entry_context_complete": getattr(trade, "entry_context_complete", False),
            "learning_eligible": getattr(trade, "learning_eligible", False),
            "operational_close_flag": getattr(trade, "operational_close_flag", False),
            "reconciliation_close_flag": str(getattr(trade, "close_reason", "") or "").strip().lower() == "external_manual_close",
            "exit_reason_family": getattr(trade, "exit_reason_family", "unknown"),
            "intended_exit_reason": getattr(trade, "intended_exit_reason", None),
            "actual_execution_path": getattr(trade, "actual_execution_path", None),
            "exit_fill_latency_seconds": getattr(trade, "exit_fill_latency_seconds", 0.0),
            "exit_cancel_count": getattr(trade, "exit_cancel_count", 0),
            "exit_partial_fill_ratio": getattr(trade, "exit_partial_fill_ratio", 0.0),
            "exit_realized_slippage_bps": getattr(trade, "exit_realized_slippage_bps", 0.0),
            "mark_price": trade.current_price,
            "best_bid": None,
            "best_ask": None,
            "spread": None,
            "mid_price": None,
            "spread_pct": None,
            "mark_source": "trade_manager_memory",
            "trajectory_state": None,
            "drawdown_from_peak": None,
            "recent_return_3": None,
            "runup_from_entry": None,
            "volatility_short": None,
            "fallback_ratio": None,
            "max_adverse_excursion_pct": getattr(trade, "max_adverse_excursion_pct", 0.0),
            "max_favorable_excursion_pct": getattr(trade, "max_favorable_excursion_pct", 0.0),
            "max_drawdown_from_peak_pct": getattr(trade, "max_drawdown_from_peak_pct", 0.0),
            "fast_adverse_move_count": getattr(trade, "fast_adverse_move_count", 0),
            "last_fast_adverse_move_at": getattr(trade, "last_fast_adverse_move_at", None),
            "close_reason": getattr(trade, "close_reason", None),
            "exit_price": getattr(trade, "current_price", None) if getattr(trade, "state", None) == TradeState.CLOSED else None,
            "close_fingerprint": self._closed_trade_fingerprint(trade) if getattr(trade, "state", None) == TradeState.CLOSED else None,
            "is_reconciliation_close": str(getattr(trade, "close_reason", "") or "").strip().lower() == "external_manual_close",
            "lifecycle_source": "trade_manager_memory",
        }
        for key, value in getattr(trade, "__dict__", {}).items():
            if str(key).startswith(("weather_", "forecast_")) and key not in row:
                row[key] = value
        return row

    def _sync_trade_from_rebuilt(self, existing_trade: TradeLifecycle, rebuilt_trade: TradeLifecycle):
        existing_trade.market = rebuilt_trade.market or existing_trade.market
        existing_trade.token_id = rebuilt_trade.token_id or existing_trade.token_id
        existing_trade.condition_id = rebuilt_trade.condition_id or existing_trade.condition_id
        existing_trade.outcome_side = rebuilt_trade.outcome_side or existing_trade.outcome_side
        if rebuilt_trade.entry_price > 0:
            existing_trade.entry_price = rebuilt_trade.entry_price
        if rebuilt_trade.current_price > 0:
            existing_trade.current_price = rebuilt_trade.current_price
        existing_trade.shares = float(rebuilt_trade.shares or 0.0)
        existing_trade.size_usdc = float(rebuilt_trade.size_usdc or (existing_trade.shares * max(existing_trade.entry_price, 0.0)))
        existing_trade.realized_pnl = float(rebuilt_trade.realized_pnl or 0.0)
        existing_trade.unrealized_pnl = float(rebuilt_trade.unrealized_pnl or 0.0)
        existing_trade.opened_at = rebuilt_trade.opened_at or existing_trade.opened_at
        existing_trade.state = TradeState.OPEN
        existing_trade.entry_signal_snapshot_json = (
            getattr(rebuilt_trade, "entry_signal_snapshot_json", None)
            or getattr(existing_trade, "entry_signal_snapshot_json", "")
        )
        existing_trade.entry_signal_snapshot_feature_count = int(
            getattr(rebuilt_trade, "entry_signal_snapshot_feature_count", 0)
            or getattr(existing_trade, "entry_signal_snapshot_feature_count", 0)
            or 0
        )
        existing_trade.entry_signal_snapshot_version = int(
            getattr(rebuilt_trade, "entry_signal_snapshot_version", 1)
            or getattr(existing_trade, "entry_signal_snapshot_version", 1)
            or 1
        )
        self._hydrate_trade_from_entry_snapshot(existing_trade, existing_trade.entry_signal_snapshot_json)
        for key, value in getattr(rebuilt_trade, "__dict__", {}).items():
            if str(key).startswith(("weather_", "forecast_")):
                setattr(existing_trade, key, value)

    def _closed_trade_fingerprint(self, trade: TradeLifecycle) -> str:
        return "|".join(
            [
                str(getattr(trade, "token_id", "") or ""),
                str(getattr(trade, "condition_id", "") or ""),
                str(getattr(trade, "outcome_side", "") or ""),
                str(getattr(trade, "opened_at", "") or ""),
                str(getattr(trade, "close_reason", "") or ""),
                f"{float(getattr(trade, 'entry_price', 0.0) or 0.0):.6f}",
                f"{float(getattr(trade, 'current_price', 0.0) or 0.0):.6f}",
                f"{float(getattr(trade, 'shares', 0.0) or 0.0):.6f}",
            ]
        )

    def _canonical_position_id(
        self,
        *,
        token_id=None,
        condition_id=None,
        outcome_side=None,
        opened_at=None,
        market=None,
        close_fingerprint=None,
    ) -> str:
        if close_fingerprint:
            stamp = re.sub(r"[^0-9]", "", str(opened_at or ""))[:14] or "unknown"
            return f"{stamp}|{str(close_fingerprint)}"
        open_stamp = re.sub(r"[^0-9]", "", str(opened_at or ""))[:14] or "unknown"
        token_part = str(token_id or market or "unknown").strip() or "unknown"
        condition_part = str(condition_id or "").strip() or "na"
        outcome_part = str(outcome_side or "").strip() or "na"
        return f"{token_part}|{condition_part}|{outcome_part}|{open_stamp}"

    def _build_closed_trade_row(self, trade: TradeLifecycle) -> dict:
        close_fingerprint = self._closed_trade_fingerprint(trade)
        row = self._trade_to_dict(trade)
        row["position_id"] = self._canonical_position_id(
            token_id=trade.token_id,
            condition_id=trade.condition_id,
            outcome_side=trade.outcome_side,
            opened_at=trade.opened_at,
            market=trade.market,
            close_fingerprint=close_fingerprint,
        )
        row["close_reason"] = trade.close_reason or "policy_exit"
        row["exit_price"] = trade.current_price
        row["realized_pnl"] = round(trade.realized_pnl, 4)
        row["net_realized_pnl"] = round(trade.realized_pnl, 4)
        row["status"] = "CLOSED"
        row["close_fingerprint"] = close_fingerprint
        row["is_reconciliation_close"] = str(trade.close_reason or "").strip().lower() == "external_manual_close"
        row["lifecycle_source"] = "trade_manager_closed"
        row["entry_model_family"] = getattr(trade, "entry_model_family", "")
        row["entry_model_version"] = getattr(trade, "entry_model_version", "")
        row["performance_governor_level"] = getattr(trade, "performance_governor_level", 0)
        row["market_family"] = getattr(trade, "market_family", "other")
        row["horizon_bucket"] = getattr(trade, "horizon_bucket", "unknown")
        row["liquidity_bucket"] = getattr(trade, "liquidity_bucket", "unknown")
        row["volatility_bucket"] = getattr(trade, "volatility_bucket", "unknown")
        row["technical_regime_bucket"] = getattr(trade, "technical_regime_bucket", "neutral")
        row["entry_context_complete"] = getattr(trade, "entry_context_complete", False)
        row["learning_eligible"] = getattr(trade, "learning_eligible", False)
        row["operational_close_flag"] = getattr(trade, "operational_close_flag", False)
        row["reconciliation_close_flag"] = str(trade.close_reason or "").strip().lower() == "external_manual_close"
        row["exit_reason_family"] = getattr(trade, "exit_reason_family", classify_exit_reason_family(trade.close_reason))
        row["intended_exit_reason"] = getattr(trade, "intended_exit_reason", trade.close_reason)
        row["actual_execution_path"] = getattr(trade, "actual_execution_path", "local_rule_close")
        row["exit_fill_latency_seconds"] = getattr(trade, "exit_fill_latency_seconds", 0.0)
        row["exit_cancel_count"] = getattr(trade, "exit_cancel_count", 0)
        row["exit_partial_fill_ratio"] = getattr(trade, "exit_partial_fill_ratio", 0.0)
        row["exit_realized_slippage_bps"] = getattr(trade, "exit_realized_slippage_bps", 0.0)
        return row

    def _ledger_presence_key(self, row: dict) -> str:
        token_id = str(row.get("token_id") or "").strip()
        condition_id = str(row.get("condition_id") or "").strip()
        outcome_side = str(row.get("outcome_side") or "").strip()
        if token_id or condition_id or outcome_side:
            return f"{token_id}|{condition_id}|{outcome_side}"
        return str(row.get("position_id") or "").strip()

    def _closed_row_fingerprint(self, row: dict) -> str:
        close_fingerprint = str(row.get("close_fingerprint") or "").strip()
        if close_fingerprint:
            return close_fingerprint
        return "|".join(
            [
                str(row.get("token_id", "") or ""),
                str(row.get("condition_id", "") or ""),
                str(row.get("outcome_side", "") or ""),
                str(row.get("opened_at", "") or ""),
                str(row.get("close_reason", "") or ""),
                f"{float(row.get('entry_price', 0.0) or 0.0):.6f}",
                f"{float(row.get('exit_price', row.get('current_price', 0.0)) or 0.0):.6f}",
                f"{float(row.get('shares', 0.0) or 0.0):.6f}",
            ]
        )

    def _refresh_lifecycle_audit_reports(self):
        try:
            from trade_lifecycle_audit import TradeLifecycleAuditor

            TradeLifecycleAuditor(logs_dir=str(self.logs_dir)).build_reports()
        except Exception as exc:
            logger.warning("[~] Failed to refresh trade lifecycle audit after close sync: %s", exc)

    def _append_closed_rows(self, rows: List[dict]):
        if not rows:
            return 0
        existing_fingerprints = set()
        existing_df = pd.DataFrame()
        if self.closed_file.exists():
            try:
                existing_df = pd.read_csv(self.closed_file, engine="python", on_bad_lines="skip")
                if not existing_df.empty and "close_fingerprint" in existing_df.columns:
                    existing_fingerprints = {
                        str(value)
                        for value in existing_df["close_fingerprint"].dropna().astype(str).tolist()
                        if str(value).strip()
                    }
            except Exception:
                existing_fingerprints = set()

        deduped_rows = []
        for row in rows:
            row = dict(row)
            row["close_fingerprint"] = self._closed_row_fingerprint(row)
            if row["close_fingerprint"] in existing_fingerprints:
                continue
            deduped_rows.append(row)
            existing_fingerprints.add(row["close_fingerprint"])

        if not deduped_rows:
            return 0

        append_df = pd.DataFrame(deduped_rows)
        ordered_cols = list(existing_df.columns)
        for column in append_df.columns:
            if column not in ordered_cols:
                ordered_cols.append(column)
        schema_changed = bool(existing_df.columns.tolist()) and existing_df.columns.tolist() != ordered_cols
        append_df = append_df.reindex(columns=ordered_cols)
        existing_df = existing_df.reindex(columns=ordered_cols)
        if not self.closed_file.exists() or self.closed_file.stat().st_size == 0:
            append_df.to_csv(self.closed_file, index=False)
        elif schema_changed:
            existing_df.to_csv(self.closed_file, index=False)
            safe_csv_append(self.closed_file, append_df)
        else:
            safe_csv_append(self.closed_file, append_df)
        self._upsert_position_rows_to_db(deduped_rows)
        self._refresh_lifecycle_audit_reports()
        return len(deduped_rows)

    def _close_absent_ledger_positions(self, open_snapshot_df: pd.DataFrame | None, close_reason: str = "external_manual_close") -> int:
        open_snapshot_df = open_snapshot_df if open_snapshot_df is not None else pd.DataFrame()
        snapshot_keys = set()
        if not open_snapshot_df.empty:
            for _, row in open_snapshot_df.iterrows():
                key = self._ledger_presence_key(row.to_dict())
                if key:
                    snapshot_keys.add(key)

        open_rows = self.db.query_all(
            """
            SELECT
                position_id, market, market_title, token_id, condition_id, outcome_side, order_side,
                status, entry_price, current_price, size_usdc, shares, market_value, realized_pnl,
                net_realized_pnl, unrealized_pnl, confidence, confidence_at_entry, signal_label,
                entry_signal_snapshot_json, entry_signal_snapshot_feature_count, entry_signal_snapshot_version,
                close_reason, exit_price, close_fingerprint, is_reconciliation_close, lifecycle_source,
                entry_model_family, entry_model_version, performance_governor_level, market_family,
                horizon_bucket, liquidity_bucket, volatility_bucket, technical_regime_bucket,
                entry_context_complete, learning_eligible, operational_close_flag, reconciliation_close_flag, exit_reason_family,
                intended_exit_reason, actual_execution_path, exit_fill_latency_seconds, exit_cancel_count,
                exit_partial_fill_ratio, exit_realized_slippage_bps, market_slug, opened_at, closed_at
            FROM positions
            WHERE UPPER(COALESCE(status, '')) = 'OPEN'
            """
        )
        if not open_rows:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        rows_to_close = []
        for row in open_rows:
            presence_key = self._ledger_presence_key(row)
            if presence_key in snapshot_keys:
                continue
            closed_row = dict(row)
            closed_row["status"] = "CLOSED"
            closed_row["close_reason"] = close_reason
            closed_row["closed_at"] = now
            closed_row["exit_price"] = row.get("exit_price") if row.get("exit_price") is not None else row.get("current_price", row.get("entry_price"))
            closed_row["current_price"] = row.get("current_price", row.get("entry_price"))
            closed_row["is_reconciliation_close"] = True
            closed_row["reconciliation_close_flag"] = True
            closed_row["lifecycle_source"] = "trade_manager_reconciled_closed"
            closed_row["close_fingerprint"] = self._closed_row_fingerprint(closed_row)
            rows_to_close.append(closed_row)

        return self._append_closed_rows(rows_to_close)

    def _upsert_position_rows_to_db(self, rows: List[dict]):
        for row in rows or []:
            self.db.execute(
                """
                INSERT OR REPLACE INTO positions (
                    position_id, market, market_title, token_id, condition_id, outcome_side, order_side,
                    status, entry_price, current_price, size_usdc, negotiated_value_usdc, shares, max_payout_usdc,
                    market_value, current_value_usdc, realized_pnl, net_realized_pnl, unrealized_pnl, unrealized_pnl_pct,
                    avg_to_now_price_change, avg_to_now_price_change_pct, confidence, confidence_at_entry, signal_label,
                    entry_signal_snapshot_json, entry_signal_snapshot_feature_count, entry_signal_snapshot_version,
                    close_reason, exit_price, close_fingerprint, is_reconciliation_close, lifecycle_source,
                    entry_model_family, entry_model_version, performance_governor_level, market_family,
                    horizon_bucket, liquidity_bucket, volatility_bucket, technical_regime_bucket,
                    entry_context_complete, learning_eligible, operational_close_flag, reconciliation_close_flag, exit_reason_family,
                    intended_exit_reason, actual_execution_path, exit_fill_latency_seconds, exit_cancel_count,
                    exit_partial_fill_ratio, exit_realized_slippage_bps, market_slug, opened_at, closed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("position_id"),
                    row.get("market"),
                    row.get("market_title"),
                    row.get("token_id"),
                    row.get("condition_id"),
                    row.get("outcome_side"),
                    row.get("order_side"),
                    row.get("status"),
                    row.get("entry_price"),
                    row.get("current_price"),
                    row.get("size_usdc"),
                    row.get("negotiated_value_usdc"),
                    row.get("shares"),
                    row.get("max_payout_usdc"),
                    row.get("market_value"),
                    row.get("current_value_usdc"),
                    row.get("realized_pnl"),
                    row.get("net_realized_pnl"),
                    row.get("unrealized_pnl"),
                    row.get("unrealized_pnl_pct"),
                    row.get("avg_to_now_price_change"),
                    row.get("avg_to_now_price_change_pct"),
                    row.get("confidence"),
                    row.get("confidence_at_entry"),
                    row.get("signal_label"),
                    row.get("entry_signal_snapshot_json"),
                    int(row.get("entry_signal_snapshot_feature_count", 0) or 0),
                    int(row.get("entry_signal_snapshot_version", 1) or 1),
                    row.get("close_reason"),
                    row.get("exit_price"),
                    row.get("close_fingerprint"),
                    int(bool(row.get("is_reconciliation_close"))) if row.get("is_reconciliation_close") is not None else 0,
                    row.get("lifecycle_source"),
                    row.get("entry_model_family"),
                    row.get("entry_model_version"),
                    int(row.get("performance_governor_level", 0) or 0),
                    row.get("market_family"),
                    row.get("horizon_bucket"),
                    row.get("liquidity_bucket"),
                    row.get("volatility_bucket"),
                    row.get("technical_regime_bucket"),
                    int(bool(row.get("entry_context_complete"))) if row.get("entry_context_complete") is not None else 0,
                    int(bool(row.get("learning_eligible"))) if row.get("learning_eligible") is not None else 0,
                    int(bool(row.get("operational_close_flag"))) if row.get("operational_close_flag") is not None else 0,
                    int(bool(row.get("reconciliation_close_flag"))) if row.get("reconciliation_close_flag") is not None else 0,
                    row.get("exit_reason_family"),
                    row.get("intended_exit_reason"),
                    row.get("actual_execution_path"),
                    row.get("exit_fill_latency_seconds"),
                    row.get("exit_cancel_count"),
                    row.get("exit_partial_fill_ratio"),
                    row.get("exit_realized_slippage_bps"),
                    row.get("market_slug"),
                    row.get("opened_at"),
                    row.get("closed_at"),
                ),
            )

    def _empty_positions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "position_id", "market", "market_title", "token_id",
                "condition_id", "outcome_side", "order_side",
                "entry_price", "current_price", "size_usdc", "negotiated_value_usdc", "shares", "max_payout_usdc",
                "market_value", "current_value_usdc", "unrealized_pnl", "unrealized_pnl_pct", "realized_pnl",
                "avg_to_now_price_change", "avg_to_now_price_change_pct",
                "net_realized_pnl", "opened_at", "status",
                "confidence", "confidence_at_entry", "signal_label",
                "entry_signal_snapshot_json", "entry_signal_snapshot_feature_count", "entry_signal_snapshot_version",
                "entry_model_family", "entry_model_version", "performance_governor_level",
                "market_family", "horizon_bucket", "liquidity_bucket", "volatility_bucket", "technical_regime_bucket",
                "entry_context_complete", "learning_eligible", "operational_close_flag", "reconciliation_close_flag", "exit_reason_family",
                "intended_exit_reason", "actual_execution_path", "exit_fill_latency_seconds", "exit_cancel_count",
                "exit_partial_fill_ratio", "exit_realized_slippage_bps",
                "mark_price", "best_bid", "best_ask", "spread", "mid_price", "spread_pct", "mark_source",
                "trajectory_state", "drawdown_from_peak", "recent_return_3", "runup_from_entry",
                "volatility_short", "fallback_ratio",
                "max_adverse_excursion_pct", "max_favorable_excursion_pct", "max_drawdown_from_peak_pct",
                "fast_adverse_move_count", "last_fast_adverse_move_at",
                "close_reason", "exit_price", "close_fingerprint", "is_reconciliation_close", "lifecycle_source",
            ]
        )

    def _normalize_reconciled_positions_for_csv(self, reconciled_positions_df: pd.DataFrame) -> pd.DataFrame:
        if reconciled_positions_df is None or reconciled_positions_df.empty:
            return self._empty_positions_frame()
        df = reconciled_positions_df.loc[:, ~reconciled_positions_df.columns.duplicated()].copy()
        if "token_id" in df.columns:
            df["token_id"] = df["token_id"].astype(str)
        if "entry_price" not in df.columns and "avg_entry_price" in df.columns:
            df["entry_price"] = df["avg_entry_price"]
        if "current_price" not in df.columns:
            for col in ("mark_price", "best_bid", "entry_price", "avg_entry_price"):
                if col in df.columns:
                    df["current_price"] = df[col]
                    break
        if "market" not in df.columns and "market_title" in df.columns:
            df["market"] = df["market_title"]
        if "market_title" not in df.columns and "market" in df.columns:
            df["market_title"] = df["market"]
        if "position_id" not in df.columns:
            if "position_key" in df.columns:
                df["position_id"] = df["position_key"]
            else:
                df["position_id"] = (
                    df.get("token_id", pd.Series(index=df.index, dtype=str)).astype(str)
                    + "|"
                    + df.get("condition_id", pd.Series(index=df.index, dtype=str)).astype(str)
                    + "|"
                    + df.get("outcome_side", pd.Series(index=df.index, dtype=str)).astype(str)
                )
        if "order_side" not in df.columns:
            df["order_side"] = "BUY"
        if "entry_signal_snapshot_json" not in df.columns:
            df["entry_signal_snapshot_json"] = ""
        if "entry_signal_snapshot_feature_count" not in df.columns:
            df["entry_signal_snapshot_feature_count"] = 0
        if "entry_signal_snapshot_version" not in df.columns:
            df["entry_signal_snapshot_version"] = 1
        if "shares" not in df.columns:
            df["shares"] = 0.0
        if "entry_price" not in df.columns:
            df["entry_price"] = 0.0
        if "current_price" not in df.columns:
            df["current_price"] = df["entry_price"]
        if "mark_price" not in df.columns:
            df["mark_price"] = df["current_price"]
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce").fillna(0.0)
        df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce").fillna(df["entry_price"])
        df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce").fillna(df["current_price"])
        for col in ["best_bid", "best_ask", "spread", "mid_price", "spread_pct", "drawdown_from_peak", "recent_return_3", "runup_from_entry", "volatility_short", "fallback_ratio", "max_adverse_excursion_pct", "max_favorable_excursion_pct", "max_drawdown_from_peak_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "fast_adverse_move_count" in df.columns:
            df["fast_adverse_move_count"] = pd.to_numeric(df["fast_adverse_move_count"], errors="coerce").fillna(0).astype(int)
        df["entry_signal_snapshot_feature_count"] = pd.to_numeric(
            df["entry_signal_snapshot_feature_count"], errors="coerce"
        ).fillna(0).astype(int)
        df["entry_signal_snapshot_version"] = pd.to_numeric(
            df["entry_signal_snapshot_version"], errors="coerce"
        ).fillna(1).astype(int)
        if "size_usdc" not in df.columns:
            df["size_usdc"] = df["shares"] * df["entry_price"]
        if "negotiated_value_usdc" not in df.columns:
            df["negotiated_value_usdc"] = pd.to_numeric(df["size_usdc"], errors="coerce").fillna(df["shares"] * df["entry_price"])
        else:
            df["negotiated_value_usdc"] = pd.to_numeric(df["negotiated_value_usdc"], errors="coerce").fillna(df["shares"] * df["entry_price"])
        if "max_payout_usdc" not in df.columns:
            df["max_payout_usdc"] = df["shares"]
        else:
            df["max_payout_usdc"] = pd.to_numeric(df["max_payout_usdc"], errors="coerce").fillna(df["shares"])
        if "market_value" not in df.columns:
            df["market_value"] = df["shares"] * df["current_price"]
        else:
            df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(df["shares"] * df["current_price"])
        if "current_value_usdc" not in df.columns:
            df["current_value_usdc"] = df["market_value"]
        else:
            df["current_value_usdc"] = pd.to_numeric(df["current_value_usdc"], errors="coerce").fillna(df["market_value"])
        if "unrealized_pnl" not in df.columns:
            df["unrealized_pnl"] = df["shares"] * (df["current_price"] - df["entry_price"])
        else:
            df["unrealized_pnl"] = pd.to_numeric(df["unrealized_pnl"], errors="coerce").fillna(df["shares"] * (df["current_price"] - df["entry_price"]))
        fallback_unrealized_pnl_pct = pd.Series(
            np.where(df["negotiated_value_usdc"] > 0, df["unrealized_pnl"] / df["negotiated_value_usdc"], 0.0),
            index=df.index,
        )
        if "unrealized_pnl_pct" not in df.columns:
            df["unrealized_pnl_pct"] = fallback_unrealized_pnl_pct
        else:
            df["unrealized_pnl_pct"] = pd.to_numeric(df["unrealized_pnl_pct"], errors="coerce").fillna(fallback_unrealized_pnl_pct)
        if "avg_to_now_price_change" not in df.columns:
            df["avg_to_now_price_change"] = df["current_price"] - df["entry_price"]
        else:
            df["avg_to_now_price_change"] = pd.to_numeric(df["avg_to_now_price_change"], errors="coerce").fillna(df["current_price"] - df["entry_price"])
        fallback_avg_to_now_price_change_pct = pd.Series(
            np.where(df["entry_price"] > 0, (df["current_price"] - df["entry_price"]) / df["entry_price"], 0.0),
            index=df.index,
        )
        if "avg_to_now_price_change_pct" not in df.columns:
            df["avg_to_now_price_change_pct"] = fallback_avg_to_now_price_change_pct
        else:
            df["avg_to_now_price_change_pct"] = pd.to_numeric(df["avg_to_now_price_change_pct"], errors="coerce").fillna(fallback_avg_to_now_price_change_pct)
        if "realized_pnl" not in df.columns:
            realized_series = df["realized_pnl"] if "realized_pnl" in df.columns else pd.Series(0.0, index=df.index)
            df["realized_pnl"] = pd.to_numeric(realized_series, errors="coerce").fillna(0.0)
        if "net_realized_pnl" not in df.columns:
            df["net_realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
        if "opened_at" not in df.columns:
            df["opened_at"] = df.get("first_fill_at", df.get("last_fill_at", datetime.now(timezone.utc).isoformat()))
        else:
            df["opened_at"] = df["opened_at"].fillna(df.get("first_fill_at", df.get("last_fill_at", datetime.now(timezone.utc).isoformat())))
        if "status" not in df.columns:
            df["status"] = "OPEN"
        if "confidence" not in df.columns:
            df["confidence"] = None
        if "confidence_at_entry" not in df.columns:
            df["confidence_at_entry"] = df["confidence"]
        if "signal_label" not in df.columns:
            df["signal_label"] = None
        if "entry_model_family" not in df.columns:
            df["entry_model_family"] = ""
        if "entry_model_version" not in df.columns:
            df["entry_model_version"] = ""
        if "performance_governor_level" not in df.columns:
            df["performance_governor_level"] = 0
        if "market_family" not in df.columns:
            df["market_family"] = "other"
        if "horizon_bucket" not in df.columns:
            df["horizon_bucket"] = "unknown"
        if "liquidity_bucket" not in df.columns:
            df["liquidity_bucket"] = "unknown"
        if "volatility_bucket" not in df.columns:
            df["volatility_bucket"] = "unknown"
        if "technical_regime_bucket" not in df.columns:
            df["technical_regime_bucket"] = "neutral"
        if "entry_context_complete" not in df.columns:
            df["entry_context_complete"] = False
        if "learning_eligible" not in df.columns:
            df["learning_eligible"] = False
        if "operational_close_flag" not in df.columns:
            df["operational_close_flag"] = False
        if "reconciliation_close_flag" not in df.columns:
            df["reconciliation_close_flag"] = False
        if "exit_reason_family" not in df.columns:
            df["exit_reason_family"] = "unknown"
        if "intended_exit_reason" not in df.columns:
            df["intended_exit_reason"] = None
        if "actual_execution_path" not in df.columns:
            df["actual_execution_path"] = None
        if "exit_fill_latency_seconds" not in df.columns:
            df["exit_fill_latency_seconds"] = 0.0
        if "exit_cancel_count" not in df.columns:
            df["exit_cancel_count"] = 0
        if "exit_partial_fill_ratio" not in df.columns:
            df["exit_partial_fill_ratio"] = 0.0
        if "exit_realized_slippage_bps" not in df.columns:
            df["exit_realized_slippage_bps"] = 0.0
        if "close_reason" not in df.columns:
            df["close_reason"] = None
        if "exit_price" not in df.columns:
            df["exit_price"] = None
        if "close_fingerprint" not in df.columns:
            df["close_fingerprint"] = None
        if "is_reconciliation_close" not in df.columns:
            df["is_reconciliation_close"] = False
        if "lifecycle_source" not in df.columns:
            df["lifecycle_source"] = "trade_manager_reconciled_open"
        if "max_adverse_excursion_pct" not in df.columns:
            df["max_adverse_excursion_pct"] = 0.0
        if "max_favorable_excursion_pct" not in df.columns:
            df["max_favorable_excursion_pct"] = 0.0
        if "max_drawdown_from_peak_pct" not in df.columns:
            df["max_drawdown_from_peak_pct"] = 0.0
        if "fast_adverse_move_count" not in df.columns:
            df["fast_adverse_move_count"] = 0
        if "last_fast_adverse_move_at" not in df.columns:
            df["last_fast_adverse_move_at"] = None

        keep = self._empty_positions_frame().columns.tolist()
        for col in keep:
            if col not in df.columns:
                df[col] = None
        return df[keep]

    def persist_open_positions(self, reconciled_positions_df: pd.DataFrame | None = None):
        if reconciled_positions_df is not None:
            out_df = self._normalize_reconciled_positions_for_csv(reconciled_positions_df)
            out_df.to_csv(self.positions_file, index=False)
            self._close_absent_ledger_positions(out_df, close_reason="external_manual_close")
            self._upsert_position_rows_to_db(out_df.to_dict("records"))
            return

        open_trades = self.get_open_positions()
        if not open_trades:
            self._empty_positions_frame().to_csv(self.positions_file, index=False)
            return

        rows = [self._trade_to_dict(t) for t in open_trades]
        out_df = pd.DataFrame(rows)
        out_df.to_csv(self.positions_file, index=False)
        self._upsert_position_rows_to_db(rows)

    def _append_closed_trades(self, closed_trades: List[TradeLifecycle]):
        if not closed_trades:
            return
        rows = []
        for trade in closed_trades:
            self._apply_closed_trade_metadata(trade)
            rows.append(self._build_closed_trade_row(trade))
        self._append_closed_rows(rows)

    def backfill_closed_positions_db_from_csv(self):
        if not self.closed_file.exists():
            return {"db_rows_upserted": 0, "csv_rows": 0}
        try:
            closed_df = pd.read_csv(self.closed_file, engine="python", on_bad_lines="skip")
        except Exception:
            return {"db_rows_upserted": 0, "csv_rows": 0}
        if closed_df.empty:
            return {"db_rows_upserted": 0, "csv_rows": 0}

        rows = []
        for _, row in closed_df.iterrows():
            data = row.to_dict()
            close_fingerprint = str(data.get("close_fingerprint") or "").strip()
            if not close_fingerprint:
                close_fingerprint = "|".join(
                    [
                        str(data.get("token_id", "") or ""),
                        str(data.get("condition_id", "") or ""),
                        str(data.get("outcome_side", "") or ""),
                        str(data.get("opened_at", "") or ""),
                        str(data.get("close_reason", "") or ""),
                        f"{float(data.get('entry_price', 0.0) or 0.0):.6f}",
                        f"{float(data.get('exit_price', data.get('current_price', 0.0)) or 0.0):.6f}",
                        f"{float(data.get('shares', 0.0) or 0.0):.6f}",
                    ]
                )
            data["close_fingerprint"] = close_fingerprint
            data["position_id"] = self._canonical_position_id(
                token_id=data.get("token_id"),
                condition_id=data.get("condition_id"),
                outcome_side=data.get("outcome_side"),
                opened_at=data.get("opened_at"),
                market=data.get("market"),
                close_fingerprint=close_fingerprint,
            )
            data["status"] = "CLOSED"
            data["exit_price"] = data.get("exit_price", data.get("current_price"))
            data["lifecycle_source"] = data.get("lifecycle_source") or "closed_positions_csv_backfill"
            rows.append(data)

        self._upsert_position_rows_to_db(rows)
        return {"db_rows_upserted": len(rows), "csv_rows": len(closed_df.index)}
    def reconcile_live_positions(self, execution_client=None, reconciled_positions_df: pd.DataFrame | None = None):
        if reconciled_positions_df is None:
            try:
                from live_position_book import LivePositionBook
                live_book = LivePositionBook(logs_dir=str(self.logs_dir))
                live_book.rebuild_from_db()
                reconciled_positions_df = live_book.get_open_positions()
            except Exception as exc:
                logger.warning("[~] Live reconciliation skipped because positions could not be loaded: %s", exc)
                return

        if reconciled_positions_df is None or reconciled_positions_df.empty:
            if self.active_trades:
                removed = self._prune_local_dust_trades(min_notional=0.01)
                if removed > 0:
                    logger.info("[~] Pruned %s local dust trades after empty live reconciliation.", removed)
                if self.active_trades:
                    logger.warning("[~] Reconciled live positions came back empty. Keeping %s local trades in memory to avoid ghost closes after a transient DB/API failure.", len(self.active_trades))
            return

        min_reconciled_notional = float(os.getenv("MIN_RECONCILED_POSITION_NOTIONAL_USDC", "0.01") or 0.01)
        rebuilt_trades: Dict[str, TradeLifecycle] = {}
        for _, row in reconciled_positions_df.iterrows():
            market = row.get("market") or row.get("market_title") or str(row.get("condition_id") or row.get("token_id") or "unknown_market")
            outcome_side = row.get("outcome_side") or row.get("side") or "UNKNOWN"
            trade_key = self._compose_trade_key(
                token_id=row.get("token_id"),
                condition_id=row.get("condition_id"),
                outcome_side=outcome_side,
                market=market,
            )
            if not trade_key:
                continue
            trade = TradeLifecycle(
                market=str(market),
                token_id=row.get("token_id"),
                condition_id=row.get("condition_id"),
                outcome_side=str(outcome_side),
            )
            trade.entry_price = float(row.get("avg_entry_price", row.get("entry_price", 0.0)) or 0.0)
            trade.current_price = float(row.get("mark_price", row.get("current_price", trade.entry_price)) or trade.entry_price)
            trade.shares = float(row.get("shares", 0.0) or 0.0)
            notional = max(
                trade.shares * max(trade.current_price, 0.0),
                trade.shares * max(trade.entry_price, 0.0),
            )
            if trade.shares <= 0 or notional < min_reconciled_notional:
                continue
            trade.size_usdc = trade.shares * trade.entry_price
            trade.realized_pnl = float(row.get("realized_pnl", 0.0) or 0.0)
            trade.unrealized_pnl = float(row.get("unrealized_pnl", 0.0) or 0.0)
            trade.entry_signal_snapshot_json = str(row.get("entry_signal_snapshot_json", "") or "")
            trade.entry_signal_snapshot_feature_count = int(row.get("entry_signal_snapshot_feature_count", 0) or 0)
            trade.entry_signal_snapshot_version = int(row.get("entry_signal_snapshot_version", 1) or 1)
            for key, value in row.items():
                if str(key).startswith(("weather_", "forecast_")):
                    setattr(trade, key, value)
            trade.opened_at = row.get("opened_at") or row.get("first_fill_at") or row.get("last_fill_at") or datetime.now().isoformat()
            self._hydrate_trade_from_entry_snapshot(trade, trade.entry_signal_snapshot_json)
            trade.state = TradeState.OPEN
            rebuilt_trades[trade_key] = trade

        # BUG 2 FIX: Merge to avoid erasing newly opened un-indexed trades
        cutoff = datetime.now(timezone.utc).timestamp() - 60
        for key, rebuilt_trade in rebuilt_trades.items():
            if key in self.active_trades:
                self._sync_trade_from_rebuilt(self.active_trades[key], rebuilt_trade)
            else:
                self.active_trades[key] = rebuilt_trade
        for key in list(self.active_trades.keys()):
            if key not in rebuilt_trades:
                try:
                    open_ts = datetime.fromisoformat(self.active_trades[key].opened_at).timestamp()
                    if open_ts < cutoff:
                        self.active_trades.pop(key)
                except Exception:
                    pass
        removed_dust = self._prune_local_dust_trades(min_notional=min_reconciled_notional)
        if removed_dust > 0:
            logger.info(
                "[~] Pruned %s local dust trades during live reconciliation (<$%.4f).",
                removed_dust,
                min_reconciled_notional,
            )
        try:
            self.persist_open_positions(reconciled_positions_df=reconciled_positions_df)
        except Exception as exc:
            logger.warning("[~] Failed to persist reconciled live positions snapshot: %s", exc)
        logger.info("[~] Reconciled %s live positions into TradeManager.", len(self.active_trades))
