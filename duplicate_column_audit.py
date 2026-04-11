from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from csv_utils import safe_csv_append_with_schema


def _scalarize(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return str(value)


def describe_duplicate_columns(frame: pd.DataFrame | None) -> list[dict]:
    if frame is None or not hasattr(frame, "columns"):
        return []
    counts = Counter(str(column) for column in frame.columns.tolist())
    return [
        {"column": column, "count": count}
        for column, count in counts.items()
        if count > 1
    ]


def audit_duplicate_columns(
    frame: pd.DataFrame | None,
    *,
    step_name: str,
    cycle_id: str,
    report_path: str | Path = "logs/duplicate_column_audit.csv",
    extra: dict | None = None,
) -> list[dict]:
    duplicates = describe_duplicate_columns(frame)
    if not duplicates:
        return []

    duplicate_columns = [item["column"] for item in duplicates]
    duplicate_counts = {item["column"]: item["count"] for item in duplicates}
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle_id": cycle_id,
        "step_name": step_name,
        "row_count": int(len(frame.index)) if frame is not None and hasattr(frame, "index") else 0,
        "column_count": int(len(frame.columns)) if frame is not None and hasattr(frame, "columns") else 0,
        "duplicate_column_count": len(duplicates),
        "duplicate_total_instances": int(sum(max(0, item["count"] - 1) for item in duplicates)),
        "duplicate_columns": ",".join(duplicate_columns),
        "duplicate_counts_json": json.dumps(duplicate_counts, sort_keys=True),
    }
    for key, value in (extra or {}).items():
        row[key] = _scalarize(value)

    safe_csv_append_with_schema(report_path, pd.DataFrame([row]))
    logging.warning(
        "Duplicate column audit: cycle=%s step=%s duplicates=%s",
        cycle_id,
        step_name,
        duplicate_columns,
    )
    return duplicates
