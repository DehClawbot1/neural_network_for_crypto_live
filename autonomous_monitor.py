import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AutonomousMonitor:
    """
    Writes lightweight health/status summaries for the project.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.logs_dir / "system_health.csv"

    def write_status(self, signals_df=None, trades_df=None, alerts_df=None, open_positions_df=None):
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signal_rows": 0 if signals_df is None else len(signals_df),
            "trade_rows": 0 if trades_df is None else len(trades_df),
            "alert_rows": 0 if alerts_df is None else len(alerts_df),
            "open_positions": 0 if open_positions_df is None else len(open_positions_df),
        }
        df = pd.DataFrame([record])
        df.to_csv(self.output_file, mode="a", header=not self.output_file.exists(), index=False)
        logging.info("Saved autonomous health status to %s", self.output_file)
