import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PositionManager:
    """
    Manages paper positions, unrealized PnL, and simulated closes.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.positions_file = self.logs_dir / "positions.csv"
        self.closed_file = self.logs_dir / "closed_positions.csv"

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
        side = signal_row.get("side", "UNKNOWN")
        wallet = signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown"))

        record = {
            "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market": market,
            "wallet_copied": wallet,
            "side": side,
            "signal_label": signal_row.get("signal_label", "UNKNOWN"),
            "confidence": signal_row.get("confidence", 0.0),
            "size_usdc": size_usdc,
            "entry_price": fill_price,
            "current_price": fill_price,
            "unrealized_pnl": 0.0,
            "status": "OPEN",
        }

        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        self._write_positions(df)
        logging.info("Opened paper position on %s", market)

    def update_mark_to_market(self, scored_df: pd.DataFrame):
        positions = self._read_positions()
        if positions.empty or scored_df is None or scored_df.empty:
            return positions

        latest_prices = {}
        latest_conf = {}
        for _, row in scored_df.iterrows():
            market = row.get("market_title", row.get("market", "Unknown Market"))
            latest_prices[market] = float(row.get("current_price", 0.5))
            latest_conf[market] = float(row.get("confidence", 0.0))

        for idx, row in positions.iterrows():
            market = row.get("market")
            if market not in latest_prices:
                continue
            current_price = latest_prices[market]
            entry_price = float(row.get("entry_price", current_price))
            side = str(row.get("side", "BUY")).upper()
            size = float(row.get("size_usdc", 0.0))

            if side == "BUY":
                pnl = (current_price - entry_price) * size
            else:
                pnl = (entry_price - current_price) * size

            positions.at[idx, "current_price"] = current_price
            positions.at[idx, "unrealized_pnl"] = round(float(pnl), 4)
            positions.at[idx, "confidence"] = latest_conf.get(market, row.get("confidence", 0.0))

        self._write_positions(positions)
        return positions

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
            elif confidence < 0.45:
                close_reason = "confidence_drop"
            elif market in alert_markets:
                close_reason = "market_alert"

            if close_reason:
                closed_row = row.to_dict()
                closed_row["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                closed_row["close_reason"] = close_reason
                closed_row["status"] = "CLOSED"
                closed.append(closed_row)
            else:
                remaining_rows.append(row.to_dict())

        self._write_positions(pd.DataFrame(remaining_rows))

        if closed:
            closed_df = pd.DataFrame(closed)
            closed_df.to_csv(self.closed_file, mode="a", header=not self.closed_file.exists(), index=False)
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
