import logging
import uuid
from pathlib import Path
from datetime import datetime

import pandas as pd
from incident_manager import IncidentManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AlertsEngine:
    """
    Detects notable public-data changes such as probability moves and whale clustering.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs", probability_move_threshold=0.03):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_file = self.logs_dir / "alerts.csv"
        self.probability_move_threshold = probability_move_threshold
        self.incident_manager = IncidentManager(self.logs_dir)

    def _normalize_alert(self, record: dict):
        alert_type = str(record.get("alert_type", "UNKNOWN"))
        severity = record.get("severity")
        if severity is None:
            if alert_type == "PROBABILITY_MOVE":
                severity = "warning"
            elif alert_type == "WHALE_CLUSTER":
                severity = "info"
            else:
                severity = "info"
        message = record.get("message")
        if not message:
            if alert_type == "PROBABILITY_MOVE":
                message = f"Probability moved from {record.get('previous_price')} to {record.get('current_price')}"
            elif alert_type == "WHALE_CLUSTER":
                message = f"{record.get('unique_wallets', 0)} wallets clustered in {record.get('market', 'unknown market')}"
            else:
                message = alert_type
        normalized = {
            "alert_id": record.get("alert_id", str(uuid.uuid4())),
            "timestamp": record.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
            "alert_type": alert_type,
            "severity": severity,
            "status": record.get("status", "open"),
            "source_module": record.get("source_module", "alerts_engine"),
            "message": message,
        }
        normalized.update(record)
        return normalized

    def _append_alert(self, record: dict):
        normalized = self._normalize_alert(record)
        if self.alerts_file.exists():
            try:
                existing = pd.read_csv(self.alerts_file)
                dedupe_cols = [c for c in ["alert_type", "market", "message", "status"] if c in existing.columns and c in normalized]
                if dedupe_cols and not existing.empty:
                    latest = existing.tail(50)
                    mask = pd.Series(True, index=latest.index)
                    for col in dedupe_cols:
                        mask &= latest[col].astype(str) == str(normalized.get(col))
                    if mask.any():
                        return
            except Exception:
                pass
        df = pd.DataFrame([normalized])
        df.to_csv(self.alerts_file, mode="a", header=not self.alerts_file.exists(), index=False)
        self.incident_manager.raise_incident(
            dedupe_key=f"{normalized.get('alert_type')}|{normalized.get('market')}|{normalized.get('message')}",
            source_module=normalized.get("source_module", "alerts_engine"),
            severity=str(normalized.get("severity", "info")),
            message=str(normalized.get("message", normalized.get("alert_type", "alert"))),
            status=str(normalized.get("status", "open")),
        )

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
                        "severity": "warning" if abs(move) < (self.probability_move_threshold * 2) else "critical",
                        "status": "open",
                        "source_module": "alerts_engine.probability_moves",
                        "market_id": market_id,
                        "market": row.get("question"),
                        "previous_price": prev_price,
                        "current_price": current_price,
                        "move": round(move, 4),
                        "message": f"Market moved by {round(move, 4)} from {prev_price} to {current_price}",
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
            if unique_wallets >= 2:
                alerts.append(
                    {
                        "alert_type": "WHALE_CLUSTER",
                        "severity": "warning" if unique_wallets < 4 else "critical",
                        "status": "open",
                        "source_module": "alerts_engine.whale_cluster",
                        "market": market_title,
                        "unique_wallets": int(unique_wallets),
                        "signal_count": int(total_signals),
                        "message": f"{int(unique_wallets)} wallets clustered with {int(total_signals)} signals",
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

