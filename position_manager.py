import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from pnl_engine import PNLEngine
from market_price_service import MarketPriceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PositionManager:
    """
    Manages paper positions, unrealized PnL, and simulated closes.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs", max_open_positions=10, max_positions_per_token=1, max_positions_per_condition=2, max_positions_per_wallet=2, cooldown_minutes=30):
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
            "size_usdc": size_usdc,
            "shares": shares,
            "entry_price": fill_price,
            "current_price": fill_price,
            "market_value": size_usdc,
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
        latest_prices = self.price_service.get_latest_prices(token_ids) if token_ids else {}

        for idx, row in positions.iterrows():
            token_id = str(row.get("token_id", ""))
            current_price = latest_prices.get(token_id)
            if current_price is None:
                current_price = float(row.get("current_price", row.get("entry_price", 0.5)))
            entry_price = float(row.get("entry_price", current_price))
            shares = float(row.get("shares", 0.0))
            fees_paid = float(row.get("fees_paid", 0.0))
            market_value = shares * float(current_price)
            unrealized_pnl = PNLEngine.mark_to_market_pnl(float(row.get("size_usdc", 0.0)), entry_price, current_price, fees=fees_paid)

            positions.at[idx, "current_price"] = current_price
            positions.at[idx, "market_value"] = round(float(market_value), 4)
            positions.at[idx, "unrealized_pnl"] = round(float(unrealized_pnl), 4)
            if token_id in latest_conf:
                positions.at[idx, "confidence"] = latest_conf[token_id]

        self._write_positions(positions)
        return positions

    def reduce_position(self, position_row: dict, fraction=0.5):
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
        positions.at[idx, "shares"] = shares * (1.0 - fraction)
        positions.at[idx, "size_usdc"] = size_usdc * (1.0 - fraction)
        positions.at[idx, "position_action"] = "REDUCE"
        self._write_positions(positions)
        return positions

    def close_position(self, position_row: dict, reason="policy_exit"):
        positions = self._read_positions()
        if positions.empty or "position_id" not in positions.columns:
            return pd.DataFrame()
        position_id = position_row.get("position_id")
        mask = positions["position_id"].astype(str) == str(position_id)
        if not mask.any():
            return pd.DataFrame()

        row = positions[mask].iloc[0].to_dict()
        token_id = str(row.get("token_id", "") or "")
        live_price = self.price_service.get_latest_price(token_id) if token_id else None
        exit_price = float(live_price if live_price is not None else row.get("current_price", row.get("entry_price", 0.5)))
        entry_price = float(row.get("entry_price", exit_price))
        size_usdc = float(row.get("size_usdc", 0.0) or 0.0)
        fees_paid = float(row.get("fees_paid", 0.0) or 0.0)
        shares = float(row.get("shares", 0.0) or 0.0)
        market_value = shares * exit_price
        realized_pnl = PNLEngine.mark_to_market_pnl(size_usdc, entry_price, exit_price, fees=fees_paid)

        row["current_price"] = exit_price
        row["market_value"] = round(float(market_value), 4)
        row["realized_pnl"] = round(float(realized_pnl), 4)
        row["unrealized_pnl"] = 0.0
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

    def apply_exit_rules(self, alerts_df: pd.DataFrame | None = None):
        positions = self._read_positions()
        if positions.empty:
            return pd.DataFrame()

        closed = []
        remaining_rows = []

        alert_markets = set()
        if alerts_df is not None and not alerts_df.empty and "market" in alerts_df.columns:
            alert_markets = set(alerts_df["market"].dropna().astype(str).tolist())

        for _, row in positions.iterrows():
            confidence = float(row.get("confidence", 0.0))
            pnl = float(row.get("unrealized_pnl", 0.0))
            market = str(row.get("market", ""))

            close_reason = None
            if pnl >= 5.0:
                close_reason = "take_profit"
            elif pnl <= -5.0:
                close_reason = "stop_loss"
            elif pnl > 1.0 and confidence < 0.55:
                close_reason = "profit_protection"
            elif confidence < 0.45:
                close_reason = "confidence_drop"
            elif market in alert_markets:
                close_reason = "market_alert"

            if close_reason:
                closed_row = row.to_dict()
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
