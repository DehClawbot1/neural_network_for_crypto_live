from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trade_lifecycle import TradeLifecycle, TradeState
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
        logger.info("[+] Initialized TradeManager.")

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
    ):
        closed_trades: List[TradeLifecycle] = []
        close_reasons: Dict[str, str] = {}

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
                            normalizer = getattr(client_ref, "_normalize_usdc_balance", None)
                            if callable(normalizer):
                                bal_val = float(normalizer(bal_raw_val))
                            else:
                                bal_val = float(bal_raw_val)
                        except Exception:
                            try:
                                bal_val = float(bal_raw_val)
                            except Exception:
                                bal_val = 0.0

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

            # Keep external/manual-close reason sticky: do not overwrite with rule exits.
            if close_reason is None and roi >= TradingConfig.PAPER_TP_ROI:
                close_reason = "take_profit_roi"
            elif close_reason is None and (current_price - entry_price) >= TradingConfig.SHADOW_TP_DELTA:
                close_reason = "take_profit_price_move"
            elif close_reason is None and predicted_target_price is not None and current_price >= predicted_target_price:
                close_reason = "take_profit_model_target"
            elif close_reason is None and (entry_price - current_price) >= TradingConfig.SHADOW_SL_DELTA:
                close_reason = "stop_loss"
            elif close_reason is None and minutes_open >= getattr(TradingConfig, 'TIME_STOP_MINUTES', 120):
                close_reason = "time_stop"
            elif close_reason is None and trailing_drop >= TradingConfig.PAPER_TRAILING_STOP and minutes_open > 15:
                close_reason = "trailing_stop"

            if close_reason:
                trade.close(current_price, reason=close_reason)
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
            if self.active_trades:
                removed = self._prune_local_dust_trades(min_notional=0.01)
                if removed > 0:
                    logger.info("Pruned %s local dust trades while reconciled positions were empty.", removed)
                if self.active_trades:
                    logger.warning("Reconciled positions empty. Keeping %s active_trades in memory to prevent amnesia.", len(self.active_trades))
            return

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
        return {
            "position_id": f"{trade.market}-{trade.outcome_side}",
            "market": trade.market,
            "market_title": trade.market,
            "token_id": trade.token_id,
            "condition_id": trade.condition_id,
            "outcome_side": trade.outcome_side,
            "order_side": "BUY",
            "entry_price": trade.entry_price,
            "current_price": trade.current_price,
            "size_usdc": trade.size_usdc,
            "shares": trade.shares,
            "market_value": trade.shares * trade.current_price if trade.current_price else 0.0,
            "unrealized_pnl": round(trade.unrealized_pnl, 4),
            "realized_pnl": round(trade.realized_pnl, 4),
            "net_realized_pnl": round(trade.realized_pnl, 4),
            "opened_at": trade.opened_at,
            "closed_at": trade.closed_at,
            "status": trade.state.value if hasattr(trade.state, 'value') else str(trade.state),
            "confidence": trade.confidence_at_entry,
            "confidence_at_entry": trade.confidence_at_entry,
            "signal_label": trade.signal_label,
        }

    def _empty_positions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "position_id", "market", "market_title", "token_id",
                "condition_id", "outcome_side", "order_side",
                "entry_price", "current_price", "size_usdc", "shares",
                "market_value", "unrealized_pnl", "realized_pnl",
                "net_realized_pnl", "opened_at", "status",
                "confidence", "confidence_at_entry", "signal_label",
            ]
        )

    def _normalize_reconciled_positions_for_csv(self, reconciled_positions_df: pd.DataFrame) -> pd.DataFrame:
        if reconciled_positions_df is None or reconciled_positions_df.empty:
            return self._empty_positions_frame()
        df = reconciled_positions_df.copy()
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
        if "shares" not in df.columns:
            df["shares"] = 0.0
        if "entry_price" not in df.columns:
            df["entry_price"] = 0.0
        if "current_price" not in df.columns:
            df["current_price"] = df["entry_price"]
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce").fillna(0.0)
        df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce").fillna(df["entry_price"])
        if "size_usdc" not in df.columns:
            df["size_usdc"] = df["shares"] * df["entry_price"]
        if "market_value" not in df.columns:
            df["market_value"] = df["shares"] * df["current_price"]
        if "unrealized_pnl" not in df.columns:
            df["unrealized_pnl"] = df["shares"] * (df["current_price"] - df["entry_price"])
        if "realized_pnl" not in df.columns:
            df["realized_pnl"] = pd.to_numeric(df.get("realized_pnl", 0.0), errors="coerce").fillna(0.0)
        if "net_realized_pnl" not in df.columns:
            df["net_realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
        if "opened_at" not in df.columns:
            df["opened_at"] = df.get("last_fill_at", datetime.now(timezone.utc).isoformat())
        if "status" not in df.columns:
            df["status"] = "OPEN"
        if "confidence" not in df.columns:
            df["confidence"] = None
        if "confidence_at_entry" not in df.columns:
            df["confidence_at_entry"] = df["confidence"]
        if "signal_label" not in df.columns:
            df["signal_label"] = None

        keep = self._empty_positions_frame().columns.tolist()
        for col in keep:
            if col not in df.columns:
                df[col] = None
        return df[keep]

    def persist_open_positions(self, reconciled_positions_df: pd.DataFrame | None = None):
        if reconciled_positions_df is not None:
            out_df = self._normalize_reconciled_positions_for_csv(reconciled_positions_df)
            out_df.to_csv(self.positions_file, index=False)
            return

        open_trades = self.get_open_positions()
        if not open_trades:
            self._empty_positions_frame().to_csv(self.positions_file, index=False)
            return

        rows = [self._trade_to_dict(t) for t in open_trades]
        pd.DataFrame(rows).to_csv(self.positions_file, index=False)

    def _append_closed_trades(self, closed_trades: List[TradeLifecycle]):
        if not closed_trades:
            return

        rows = []
        for trade in closed_trades:
            row = self._trade_to_dict(trade)
            row["close_reason"] = trade.close_reason or "policy_exit"
            row["exit_price"] = trade.current_price
            row["realized_pnl"] = round(trade.realized_pnl, 4)
            row["net_realized_pnl"] = round(trade.realized_pnl, 4)
            row["status"] = "CLOSED"
            rows.append(row)

        pd.DataFrame(rows).to_csv(
            self.closed_file, mode="a",
            header=not self.closed_file.exists(), index=False,
        )
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
            trade.opened_at = row.get("last_fill_at") or datetime.now().isoformat()
            trade.state = TradeState.OPEN
            rebuilt_trades[trade_key] = trade

        # BUG 2 FIX: Merge to avoid erasing newly opened un-indexed trades
        cutoff = datetime.now(timezone.utc).timestamp() - 60
        for key, rebuilt_trade in rebuilt_trades.items():
            if key in self.active_trades:
                self.active_trades[key].current_price = rebuilt_trade.current_price
                self.active_trades[key].unrealized_pnl = rebuilt_trade.unrealized_pnl
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
        logger.info("[~] Reconciled %s live positions into TradeManager.", len(self.active_trades))
