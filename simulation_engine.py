import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SimulationEngine:
    """
    Richer paper-trading simulation support: tracks open/closed simulated positions.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.positions_file = self.logs_dir / "positions.csv"

    def open_position(self, signal_row: dict, size_usdc: float, fill_price: float):
        record = {
            "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market": signal_row.get("market_title", signal_row.get("market", "Unknown Market")),
            "wallet_copied": signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown")),
            "side": signal_row.get("side", "UNKNOWN"),
            "signal_label": signal_row.get("signal_label", "UNKNOWN"),
            "confidence": signal_row.get("confidence", 0.0),
            "size_usdc": size_usdc,
            "entry_price": fill_price,
            "status": "OPEN",
        }
        df = pd.DataFrame([record])
        df.to_csv(self.positions_file, mode="a", header=not self.positions_file.exists(), index=False)
        logging.info("Opened simulated position on %s", record["market"])

    def summarize_open_positions(self):
        if not self.positions_file.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.positions_file)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        return df[df["status"] == "OPEN"]
