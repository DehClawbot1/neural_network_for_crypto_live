from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from db import Database


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _event_id(prefix: str, row: dict) -> str:
    payload = "|".join(str(row.get(k, "")) for k in sorted(row.keys()))
    digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}:{digest}"


def sync_ops_state_to_db(logs_dir="logs"):
    logs_path = Path(logs_dir)
    db = Database(logs_path / "trading.db")

    incidents_df = _read_csv(logs_path / "incidents.csv")
    for _, row in incidents_df.iterrows():
        record = row.to_dict()
        incident_id = _event_id("incident", record)
        created_at = record.get("timestamp") or record.get("created_at")
        category = record.get("category") or record.get("type")
        severity = record.get("severity") or record.get("level")
        summary = record.get("summary") or record.get("title") or record.get("message")
        detail = record.get("detail") or record.get("description") or record.get("message")
        db.execute(
            "INSERT OR REPLACE INTO incidents (incident_id, category, severity, summary, detail, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (incident_id, category, severity, summary, detail, created_at),
        )

    heartbeats_df = _read_csv(logs_path / "service_heartbeats.csv")
    for _, row in heartbeats_df.iterrows():
        record = row.to_dict()
        heartbeat_id = _event_id("heartbeat", record)
        created_at = record.get("timestamp") or record.get("created_at")
        service = record.get("service") or record.get("component") or record.get("source")
        status = record.get("status") or record.get("heartbeat") or record.get("state")
        detail = record.get("detail") or record.get("message")
        db.execute(
            "INSERT OR REPLACE INTO service_heartbeats (heartbeat_id, service, status, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (heartbeat_id, service, status, detail, created_at),
        )

    health_df = _read_csv(logs_path / "system_health.csv")
    for _, row in health_df.iterrows():
        record = row.to_dict()
        health_id = _event_id("health", record)
        created_at = record.get("timestamp") or record.get("created_at")
        component = record.get("component") or record.get("service") or record.get("source")
        status = record.get("status") or record.get("state")
        detail = record.get("detail") or record.get("message")
        db.execute(
            "INSERT OR REPLACE INTO system_health (health_id, component, status, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (health_id, component, status, detail, created_at),
        )

    return {
        "incidents": len(incidents_df),
        "service_heartbeats": len(heartbeats_df),
        "system_health": len(health_df),
    }


def sync_ops_csv_to_db(logs_dir="logs"):
    """Backward-compatible alias for older imports."""
    return sync_ops_state_to_db(logs_dir)
