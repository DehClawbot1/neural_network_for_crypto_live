from datetime import datetime
from pathlib import Path

import pandas as pd

from csv_utils import safe_csv_append


class IncidentManager:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.incidents_file = self.logs_dir / "incidents.csv"

    def _load(self):
        if not self.incidents_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.incidents_file)
        except Exception:
            return pd.DataFrame()

    def raise_incident(self, dedupe_key: str, source_module: str, severity: str, message: str, status: str = "open"):
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        df = self._load()
        if not df.empty and "dedupe_key" in df.columns:
            mask = (df["dedupe_key"].astype(str) == str(dedupe_key)) & (df.get("status", pd.Series(dtype=str)).astype(str) != "resolved")
            if mask.any():
                df.loc[mask, "last_seen"] = now
                self.incidents_file.write_text(df.to_csv(index=False), encoding="utf-8")
                return
        incident_id = f"inc-{int(datetime.utcnow().timestamp())}-{abs(hash(dedupe_key)) % 100000}"
        row = {
            "incident_id": incident_id,
            "dedupe_key": dedupe_key,
            "source_module": source_module,
            "severity": severity,
            "status": status,
            "first_seen": now,
            "last_seen": now,
            "message": message,
        }
        safe_csv_append(self.incidents_file, pd.DataFrame([row]))

    def resolve_incident(self, dedupe_key: str):
        df = self._load()
        if df.empty or "dedupe_key" not in df.columns:
            return
        mask = df["dedupe_key"].astype(str) == str(dedupe_key)
        if mask.any():
            df.loc[mask, "status"] = "resolved"
            df.loc[mask, "last_seen"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            self.incidents_file.write_text(df.to_csv(index=False), encoding="utf-8")

