from __future__ import annotations

import logging
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
    def _get_trade_key(self, signal_row: pd.Series) -> Optional[str]:
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

    def process_exits(self, current_timestamp: datetime, alerts_df: pd.DataFrame = None):
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
                current_ts = current_timestamp.replace(tzinfo=None) if current_timestamp.tzinfo is not None else current_timestamp

            entry_price = float(trade.entry_price or 0)
            current_price = float(trade.current_price or entry_price)
            if entry_price <= 0:
                continue

            roi = (current_price - entry_price) / entry_price
            minutes_open = (current_ts - opened_dt).total_seconds() / 60.0
            
            # PATCHED: Calculate true trailing stop using peak price
            if not hasattr(trade, 'peak_price'):
                trade.peak_price = entry_price
            trade.peak_price = max(trade.peak_price, current_price)
            trailing_drop = (trade.peak_price - current_price) / trade.peak_price if trade.peak_price > 0 else 0

            close_reason = None

            # --- STRICT SYNC: Verify external Polymarket balance ---
            # If you manually closed this trade on the Polymarket website, 
            # this will detect the missing shares and force the bot to close it locally.
            try:
                # Dynamically find the execution client reference
                client_ref = getattr(self, 'exec_client', getattr(self, 'client', getattr(self, 'execution_client', getattr(self, 'api', None))))
                if client_ref and hasattr(trade, 'token_id') and trade.token_id:
                    raw_bal = client_ref.get_balance_allowance(asset_type="CONDITIONAL", token_id=trade.token_id)
                    
                    bal_val = 0.0
                    if isinstance(raw_bal, dict) and 'balance' in raw_bal:
                        bal_val = float(raw_bal['balance'])
                    elif raw_bal is not None:
                        bal_val = float(raw_bal)
                    
                    # Polymarket returns microdollars. < 10000 = less than 0.01 shares (dust)
                    if bal_val < 10000:
                        close_reason = "external_manual_close"
                        # Zero out unrealized PnL so the dashboard doesn't skew
                        if hasattr(trade, 'unrealized_pnl'): trade.unrealized_pnl = 0.0 
            except Exception:
                pass # Safely ignore if API rate limits or client ref not found
            # -------------------------------------------------------

            if roi >= TradingConfig.PAPER_TP_ROI:
                close_reason = "take_profit_roi"
            elif (current_price - entry_price) >= TradingConfig.SHADOW_TP_DELTA:
                close_reason = "take_profit_price_move"
            elif (entry_price - current_price) >= TradingConfig.SHADOW_SL_DELTA:
                close_reason = "stop_loss"
            elif minutes_open >= getattr(TradingConfig, 'TIME_STOP_MINUTES', 120):
                close_reason = "time_stop"
            elif trailing_drop >= TradingConfig.PAPER_TRAILING_STOP and minutes_open > 15:
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

        if closed_trades:
            self._append_closed_trades(closed_trades)

        return closed_trades

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

    def persist_open_positions(self):
        open_trades = self.get_open_positions()
        if not open_trades:
            pd.DataFrame(columns=[
                "position_id", "market", "market_title", "token_id",
                "condition_id", "outcome_side", "order_side",
                "entry_price", "current_price", "size_usdc", "shares",
                "market_value", "unrealized_pnl", "realized_pnl",
                "net_realized_pnl", "opened_at", "status",
                "confidence", "confidence_at_entry", "signal_label",
            ]).to_csv(self.positions_file, index=False)
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
                logger.warning("[~] Exchange shows no open positions. Clearing %s local live trades.", len(self.active_trades))
            self.active_trades = {}
            return

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
            trade.size_usdc = trade.shares * trade.entry_price
            trade.realized_pnl = float(row.get("realized_pnl", 0.0) or 0.0)
            trade.unrealized_pnl = float(row.get("unrealized_pnl", 0.0) or 0.0)
            trade.opened_at = row.get("last_fill_at") or datetime.now().isoformat()
            trade.state = TradeState.OPEN
            rebuilt_trades[trade_key] = trade

        self.active_trades = rebuilt_trades
        logger.info("[~] Reconciled %s live positions into TradeManager.", len(self.active_trades))
