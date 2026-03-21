import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AlertsEngine:
    """
    Detects notable public-data changes such as probability moves and whale clustering.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs", probability_move_threshold=0.08):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_file = self.logs_dir / "alerts.csv"
        self.probability_move_threshold = probability_move_threshold

    def _append_alert(self, record: dict):
        df = pd.DataFrame([record])
        df.to_csv(self.alerts_file, mode="a", header=not self.alerts_file.exists(), index=False)

    def detect_probability_moves(self, current_markets_df: pd.DataFrame, previous_markets_df: pd.DataFrame | None):
        alerts = []
        if current_markets_df is None or current_markets_df.empty:
            return alerts
        if previous_markets_df is None or previous_markets_df.empty:
            return alerts

        prev_lookup = previous_markets_df.set_index("market_id").to_dict("index")
        for _, row in current_markets_df.iterrows():
            market_id = row.get("market_id")
            if market_id not in prev_lookup:
                continue

            prev = prev_lookup[market_id]
            current_price = float(row.get("last_trade_price", 0.0) or 0.0)
            prev_price = float(prev.get("last_trade_price", 0.0) or 0.0)
            move = current_price - prev_price

            if abs(move) >= self.probability_move_threshold:
                alerts.append(
                    {
                        "alert_type": "PROBABILITY_MOVE",
                        "market_id": market_id,
                        "market": row.get("question"),
                        "previous_price": prev_price,
                        "current_price": current_price,
                        "move": round(move, 4),
                    }
                )
        return alerts

    def detect_whale_clustering(self, signals_df: pd.DataFrame):
        alerts = []
        if signals_df is None or signals_df.empty:
            return alerts

        grouped = signals_df.groupby("market_title")
        for market_title, group in grouped:
            unique_wallets = group["trader_wallet"].nunique() if "trader_wallet" in group.columns else 0
            total_signals = len(group)
            if unique_wallets >= 3:
                alerts.append(
                    {
                        "alert_type": "WHALE_CLUSTER",
                        "market": market_title,
                        "unique_wallets": int(unique_wallets),
                        "signal_count": int(total_signals),
                    }
                )
        return alerts

    def process_alerts(self, current_markets_df: pd.DataFrame, previous_markets_df: pd.DataFrame | None, signals_df: pd.DataFrame):
        alerts = []
        alerts.extend(self.detect_probability_moves(current_markets_df, previous_markets_df))
        alerts.extend(self.detect_whale_clustering(signals_df))

        for alert in alerts:
            self._append_alert(alert)
            logging.info("ALERT -> %s", alert)

        return pd.DataFrame(alerts)
