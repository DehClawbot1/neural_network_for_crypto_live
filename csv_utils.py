"""Atomic CSV append utility — prevents race conditions on header writes."""

import os
import tempfile
from pathlib import Path

import pandas as pd


def safe_csv_append(path, df):
    """Append a DataFrame to a CSV file without header duplication races.

    Writes to a temporary file first, then appends its content to the target.
    The header-needed check and write happen on the same file handle, so
    concurrent callers cannot both decide to write headers.
    """
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".csv.tmp")
    try:
        os.close(fd)
        df.to_csv(tmp, mode="w", header=needs_header, index=False, encoding="utf-8")
        with open(str(path), "a", encoding="utf-8") as dst, open(tmp, "r", encoding="utf-8") as src:
            dst.write(src.read())
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def safe_csv_append_with_schema(path, df):
    """Append a DataFrame to CSV and widen the header when new columns appear."""
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        return

    if not path.exists() or path.stat().st_size == 0:
        safe_csv_append(path, df)
        return

    try:
        existing_header = pd.read_csv(path, nrows=0, engine="python", on_bad_lines="skip")
        existing_columns = existing_header.columns.tolist()
    except Exception:
        existing_columns = []

    ordered_columns = list(existing_columns)
    for column in df.columns:
        if column not in ordered_columns:
            ordered_columns.append(column)

    append_df = df.reindex(columns=ordered_columns)
    if existing_columns and ordered_columns != existing_columns:
        try:
            existing_df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            existing_df = pd.DataFrame(columns=existing_columns)
        existing_df = existing_df.reindex(columns=ordered_columns)
        existing_df.to_csv(path, index=False, encoding="utf-8")
    safe_csv_append(path, append_df)
