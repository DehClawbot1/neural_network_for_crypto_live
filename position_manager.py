import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from config import TradingConfig
from pnl_engine import PNLEngine
from market_price_service import MarketPriceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PositionManager:
    """
    Manages paper positions, unrealized PnL, and simulated closes.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs", max_open_positions=10, max_positions_per_token=1, max_positions_per_condition=2, max_positions_per_wallet=2, cooldown_minutes=30, take_profit_price_move=0.25, take_profit_roi_pct=TradingConfig.PAPER_TP_ROI, trailing_stop_pct=TradingConfig.PAPER_TRAILING_STOP, time_stop_minutes=180, max_spread_to_exit=0.05, min_bid_size_to_exit=0, fee_rate=0.0, slippage_rate=0.005):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.positions_file = self.logs_dir / "positions.csv"
        self.closed_file = self.logs_dir / "closed_positions.csv"
        self.episode_file = self.logs_dir / "episode_log.csv"
        self.price_service = MarketPriceService()
        self.max_open_positions = max_open_positions
        self.max_positions_per_token = max_positions_per_token
        self.max_positions_per_condition = max_positions_per_condition
        self.max_positions_per_wallet = max_positions_per_wallet
        self.cooldown_minutes = cooldown_minutes
        self.take_profit_price_move = take_profit_price_move
        self.take_profit_roi_pct = take_profit_roi_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.time_stop_minutes = time_stop_minutes
        self.max_spread_to_exit = max_spread_to_exit
        self.min_bid_size_to_exit = min_bid_size_to_exit
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def _read_positions(self):
        if not self.positions_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.positions_file)
        except Exception:
            return pd.DataFrame()

    def _write_positions(self, df: pd.DataFrame):
        df.to_csv(self.positions_file, index=False)

    def open_position(self, signal_row: dict, size_usdc: float, fill_price: float):
        df = self._read_positions()
        market = signal_row.get("market_title", signal_row.get("market", "Unknown Market"))
        outcome_side = str(signal_row.get("outcome_side", signal_row.get("side", "UNKNOWN"))).upper()
        order_side = str(signal_row.get("order_side", signal_row.get("trade_side", "BUY"))).upper()
        wallet = signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown"))
        token_id = str(signal_row.get("token_id", "") or "")
        condition_id = str(signal_row.get("condition_id", "") or "")
        shares = PNLEngine.shares_from_capital(size_usdc, fill_price)

        if len(df) >= self.max_open_positions:
            logging.info("Position rejected: max open positions reached.")
            return False
        if token_id and "token_id" in df.columns and (df["token_id"].astype(str) == token_id).sum() >= self.max_positions_per_token:
            logging.info("Position rejected: token exposure cap reached for %s", token_id)
            return False
        if condition_id and "condition_id" in df.columns and (df["condition_id"].astype(str) == condition_id).sum() >= self.max_positions_per_condition:
            logging.info("Position rejected: condition exposure cap reached for %s", condition_id)
            return False
        if "wallet_copied" in df.columns and (df["wallet_copied"].astype(str) == str(wallet)).sum() >= self.max_positions_per_wallet:
            logging.info("Position rejected: wallet exposure cap reached for %s", wallet)
            return False

        signal_bucket = str(signal_row.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M")))[:16]
        idempotency_key = f"{signal_bucket}|{token_id}|{wallet}|ENTER"
        if "idempotency_key" in df.columns and (df["idempotency_key"].astype(str) == idempotency_key).any():
            logging.info("Position rejected: duplicate signal idempotency key %s", idempotency_key)
            return False

        recent_closed = self.get_closed_positions()
        if not recent_closed.empty and token_id and "token_id" in recent_closed.columns and "closed_at" in recent_closed.columns:
            recent_closed["closed_at"] = pd.to_datetime(recent_closed["closed_at"], errors="coerce")
            recent_token = recent_closed[recent_closed["token_id"].astype(str) == token_id].sort_values("closed_at")
            if not recent_token.empty:
                last_closed = recent_token.iloc[-1].get("closed_at")
                if pd.notna(last_closed):
                    minutes_since = (pd.Timestamp.now() - last_closed).total_seconds() / 60.0
                    if minutes_since < self.cooldown_minutes:
                        logging.info("Position rejected: token cooldown active for %s", token_id)
                        return False

        opened_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "position_id": f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "idempotency_key": idempotency_key,
            "opened_at": opened_at,
            "market": market,
            "condition_id": signal_row.get("condition_id"),
            "token_id": signal_row.get("token_id"),
            "wallet_copied": wallet,
            "order_side": order_side,
            "trade_side": order_side,
            "outcome_side": outcome_side,
            "entry_intent": signal_row.get("entry_intent", "OPEN_LONG"),
            "position_action": "ENTER",
            "signal_label": signal_row.get("signal_label", "UNKNOWN"),
            "confidence": signal_row.get("confidence", 0.0),
            "confidence_at_entry": signal_row.get("confidence", 0.0),
            "size_usdc": size_usdc,
            "shares": shares,
            "entry_price": fill_price,
            "current_price": fill_price,
            "market_value": size_usdc,
            "peak_price": fill_price,
            "fees_paid": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "status": "OPEN",
        }

        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        self._write_positions(df)
        logging.info("Opened paper position on %s", market)
        return True

    def update_mark_to_market(self, scored_df: pd.DataFrame | None = None):
        positions = self._read_positions()
        if positions.empty:
            return positions

        latest_conf = {}
        if scored_df is not None and not scored_df.empty:
            for _, row in scored_df.iterrows():
                token_id = row.get("token_id")
                if token_id:
                    latest_conf[str(token_id)] = float(row.get("confidence", 0.0))

        token_ids = [str(x) for x in positions.get("token_id", pd.Series(dtype=str)).dropna().tolist()]
        latest_quotes = self.price_service.get_batch_prices(token_ids) if token_ids else {}

        for idx, row in positions.iterrows():
            token_id = str(row.get("token_id", ""))
            quote = latest_quotes.get(token_id) or {}
            current_price = quote.get("midpoint") or quote.get("last_trade_price") or quote.get("price")
            if current_price is None:
                current_price = float(row.get("current_price", row.get("entry_price", 0.5)))
            entry_price = float(row.get("entry_price", current_price))
            shares = float(row.get("shares", 0.0))
            fees_paid = float(row.get("fees_paid", 0.0))
            market_value = shares * float(current_price)
            unrealized_pnl = PNLEngine.mark_to_market_pnl(float(row.get("size_usdc", 0.0)), entry_price, current_price, fees=fees_paid)

            positions.at[idx, "current_price"] = current_price
            positions.at[idx, "spread"] = float(quote.get("spread", 0.0) or 0.0)
            positions.at[idx, "bid_size"] = float(quote.get("best_bid_size", 0.0) or 0.0)
            positions.at[idx, "market_value"] = round(float(market_value), 4)
            positions.at[idx, "unrealized_pnl"] = round(float(unrealized_pnl), 4)
            prior_peak = float(row.get("peak_price", entry_price) or entry_price)
            positions.at[idx, "peak_price"] = max(prior_peak, float(current_price))
            if token_id in latest_conf:
                positions.at[idx, "confidence"] = latest_conf[token_id]

        self._write_positions(positions)
        return positions

    def reduce_position(self, position_row: dict, fraction=0.5, exit_price=None, filled_shares=None):
        positions = self._read_positions()
        if positions.empty or "position_id" not in positions.columns:
            return positions
        position_id = position_row.get("position_id")
        mask = positions["position_id"].astype(str) == str(position_id)
        if not mask.any():
            return positions

        idx = positions[mask].index[0]
        shares = float(positions.at[idx, "shares"] or 0.0)
        size_usdc = float(positions.at[idx, "size_usdc"] or 0.0)
        entry_price = float(positions.at[idx, "entry_price"] or 0.0)
        token_id = str(positions.at[idx, "token_id"] or "") if "token_id" in positions.columns else ""
        if exit_price is None:
            quote = self.price_service.get_quote(token_id) if token_id else {}
            live_price = quote.get("best_bid") or quote.get("midpoint") or quote.get("last_trade_price") or quote.get("price")
            exit_price = float(live_price if live_price is not None else positions.at[idx, "current_price"] or entry_price)
        else:
            exit_price = float(exit_price)

        shares_closed = float(filled_shares) if filled_shares is not None else shares * fraction
        shares_closed = min(shares, max(0.0, shares_closed))
        shares_remaining = shares - shares_closed
        effective_exit_price = exit_price * (1.0 - self.slippage_rate)
        gross_realized_pnl = shares_closed * (effective_exit_price - entry_price)
        fees_paid_exit = shares_closed * effective_exit_price * self.fee_rate
        net_realized_pnl = gross_realized_pnl - fees_paid_exit

        positions.at[idx, "shares"] = shares_remaining
        positions.at[idx, "size_usdc"] = size_usdc * (1.0 - fraction)
        positions.at[idx, "current_price"] = exit_price
        positions.at[idx, "realized_pnl"] = float(positions.at[idx, "realized_pnl"] or 0.0) + net_realized_pnl
        positions.at[idx, "position_action"] = "REDUCE"
        positions.at[idx, "shares_closed"] = shares_closed
        positions.at[idx, "shares_remaining"] = shares_remaining
        positions.at[idx, "gross_realized_pnl"] = gross_realized_pnl
        positions.at[idx, "fees_paid_exit"] = fees_paid_exit
        positions.at[idx, "net_realized_pnl"] = net_realized_pnl
        self._write_positions(positions)
        return positions

    def close_position(self, position_row: dict, reason="policy_exit", exit_price=None, filled_shares=None):
        positions = self._read_positions()
        if positions.empty or "position_id" not in positions.columns:
            return pd.DataFrame()
        position_id = position_row.get("position_id")
        mask = positions["position_id"].astype(str) == str(position_id)
        if not mask.any():
            return pd.DataFrame()

        row = positions[mask].iloc[0].to_dict()
        token_id = str(row.get("token_id", "") or "")
        if exit_price is None:
            quote = self.price_service.get_quote(token_id) if token_id else {}
            live_price = quote.get("best_bid") or quote.get("midpoint") or quote.get("last_trade_price") or quote.get("price")
            exit_price = float(live_price if live_price is not None else row.get("current_price", row.get("entry_price", 0.5)))
        else:
            exit_price = float(exit_price)
        entry_price = float(row.get("entry_price", exit_price))
        size_usdc = float(row.get("size_usdc", 0.0) or 0.0)
        fees_paid = float(row.get("fees_paid", 0.0) or 0.0)
        shares = float(row.get("shares", 0.0) or 0.0)
        shares = min(shares, max(0.0, float(filled_shares) if filled_shares is not None else shares))
        effective_exit_price = exit_price * (1.0 - self.slippage_rate)
        market_value = shares * effective_exit_price
        gross_realized_pnl = shares * (effective_exit_price - entry_price)
        fees_paid_exit = shares * effective_exit_price * self.fee_rate
        net_realized_pnl = gross_realized_pnl - fees_paid_exit - fees_paid

        row["current_price"] = exit_price
        row["exit_price"] = exit_price
        row["market_value"] = round(float(market_value), 4)
        row["gross_realized_pnl"] = round(float(gross_realized_pnl), 4)
        row["fees_paid_exit"] = round(float(fees_paid_exit), 4)
        row["net_realized_pnl"] = round(float(net_realized_pnl), 4)
        row["realized_pnl"] = round(float(net_realized_pnl), 4)
        row["unrealized_pnl"] = 0.0
        row["shares_closed"] = shares
        row["shares_remaining"] = 0.0
        row["exit_order_side"] = "SELL"
        row["filled_shares"] = shares
        row["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row["position_action"] = "EXIT"
        row["close_reason"] = reason
        row["status"] = "CLOSED"

        remaining = positions[~mask]
        self._write_positions(remaining)
        closed_df = pd.DataFrame([row])
        closed_df.to_csv(self.closed_file, mode="a", header=not self.closed_file.exists(), index=False)
        closed_df.to_csv(self.episode_file, mode="a", header=not self.episode_file.exists(), index=False)
        return closed_df

    def get_exit_reason(self, row: dict, alerts_df: pd.DataFrame | None = None):
        confidence = float(row.get("confidence", 0.0))
        market = str(row.get("market", ""))
        entry_price = float(row.get("entry_price", 0.0) or 0.0)
        current_price = float(row.get("current_price", entry_price) or entry_price)
        peak_price = float(row.get("peak_price", entry_price) or entry_price)
        roi_pct = ((current_price - entry_price) / entry_price) if entry_price else 0.0
        trailing_floor = peak_price * (1.0 - self.trailing_stop_pct)
        opened_at = pd.to_datetime(row.get("opened_at"), errors="coerce")
        minutes_open = (pd.Timestamp.now() - opened_at).total_seconds() / 60.0 if pd.notna(opened_at) else 0.0
        spread = float(row.get("spread", 0.0) or 0.0)
        bid_size = float(row.get("bid_size", self.min_bid_size_to_exit) or self.min_bid_size_to_exit)

        alert_markets = set()
        if alerts_df is not None and not alerts_df.empty and "market" in alerts_df.columns:
            alert_markets = set(alerts_df["market"].dropna().astype(str).tolist())

        if (current_price - entry_price) >= self.take_profit_price_move:
            return "take_profit_price_move"
        if roi_pct >= self.take_profit_roi_pct:
            return "take_profit_roi"
        if peak_price > entry_price and current_price <= trailing_floor:
            return "trailing_stop"
        if minutes_open >= self.time_stop_minutes:
            return "time_stop"
        if spread > self.max_spread_to_exit:
            return None
        if bid_size < self.min_bid_size_to_exit:
            return None
        if confidence < 0.45:
            return "confidence_drop"
        if market in alert_markets:
            return "market_alert"
        return None

    def apply_exit_rules(self, alerts_df: pd.DataFrame | None = None):
        positions = self._read_positions()
        if positions.empty:
            return pd.DataFrame()

        closed = []
        remaining_rows = []

        for _, row in positions.iterrows():
            close_reason = self.get_exit_reason(row.to_dict(), alerts_df=alerts_df)

            if close_reason:
                closed_row = row.to_dict()
                shares = float(closed_row.get("shares", 0.0) or 0.0)
                entry_price = float(closed_row.get("entry_price", 0.0) or 0.0)
                exit_price = float(closed_row.get("current_price", entry_price) or entry_price)
                effective_exit_price = exit_price * (1.0 - self.slippage_rate)
                gross_realized_pnl = shares * (effective_exit_price - entry_price)
                fees_paid_exit = shares * effective_exit_price * self.fee_rate
                net_realized_pnl = gross_realized_pnl - fees_paid_exit - float(closed_row.get("fees_paid", 0.0) or 0.0)
                closed_row["exit_price"] = exit_price
                closed_row["gross_realized_pnl"] = gross_realized_pnl
                closed_row["fees_paid_exit"] = fees_paid_exit
                closed_row["net_realized_pnl"] = net_realized_pnl
                closed_row["realized_pnl"] = net_realized_pnl
                closed_row["shares_closed"] = shares
                closed_row["shares_remaining"] = 0.0
                closed_row["filled_shares"] = shares
                closed_row["exit_order_side"] = "SELL"
                closed_row["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                closed_row["position_action"] = "EXIT"
                closed_row["close_reason"] = close_reason
                closed_row["status"] = "CLOSED"
                closed.append(closed_row)
            else:
                remaining_rows.append(row.to_dict())

        self._write_positions(pd.DataFrame(remaining_rows))

        if closed:
            closed_df = pd.DataFrame(closed)
            closed_df.to_csv(self.closed_file, mode="a", header=not self.closed_file.exists(), index=False)
            closed_df.to_csv(self.episode_file, mode="a", header=not self.episode_file.exists(), index=False)
            logging.info("Closed %s paper positions.", len(closed_df))
            return closed_df

        return pd.DataFrame()

    def get_open_positions(self):
        positions = self._read_positions()
        if positions.empty:
            return positions
        return positions[positions["status"] == "OPEN"] if "status" in positions.columns else positions

    def get_closed_positions(self):
        if not self.closed_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.closed_file)
        except Exception:
            return pd.DataFrame()

