from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _parse_logged_timestamp_series(series: pd.Series) -> pd.Series:
    local_tz = os.getenv("BOT_LOG_LOCAL_TIMEZONE", "Europe/Lisbon")
    raw = pd.to_datetime(series, errors="coerce", utc=False, format="mixed")
    if getattr(raw.dt, "tz", None) is None:
        try:
            return raw.dt.tz_localize(local_tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
        except Exception:
            return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    return raw.dt.tz_convert("UTC")


def _parse_snapshot_json(value) -> dict:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off", ""}
    values = []
    for value in series.tolist():
        if value is None:
            values.append(bool(default))
            continue
        try:
            if pd.isna(value):
                values.append(bool(default))
                continue
        except Exception:
            pass
        if isinstance(value, bool):
            values.append(value)
            continue
        if isinstance(value, (int, float)):
            values.append(float(value) != 0.0)
            continue
        text = str(value).strip().lower()
        if text in truthy:
            values.append(True)
        elif text in falsy:
            values.append(False)
        else:
            values.append(bool(default))
    return pd.Series(values, index=series.index, dtype=bool)


def _normalize_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "market_title" not in out.columns and "market" in out.columns:
        out["market_title"] = out["market"]
    elif "market_title" in out.columns and "market" not in out.columns:
        out["market"] = out["market_title"]
    elif "market_title" in out.columns and "market" in out.columns:
        out["market_title"] = out["market_title"].fillna(out["market"])
        out["market"] = out["market"].fillna(out["market_title"])
    return out


def _event_key(frame: pd.DataFrame) -> pd.Series:
    work = frame.copy()
    timestamp_series = work.get("timestamp", pd.Series(pd.NaT, index=work.index))
    if not pd.api.types.is_datetime64_any_dtype(timestamp_series):
        timestamp_series = _parse_logged_timestamp_series(timestamp_series)
    timestamp_key = timestamp_series.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
    outcome_side = work.get("outcome_side")
    if outcome_side is None:
        outcome_side = work.get("side", pd.Series("", index=work.index))
    else:
        outcome_side = outcome_side.fillna(work.get("side", pd.Series("", index=work.index)))
    token_id = work.get("token_id", pd.Series("", index=work.index)).fillna("").astype(str).str.strip()
    condition_id = work.get("condition_id", pd.Series("", index=work.index)).fillna("").astype(str).str.strip()
    primary_id = token_id.where(token_id != "", condition_id)
    return (
        primary_id
        + "|"
        + outcome_side.fillna("").astype(str).str.strip().str.upper()
        + "|"
        + work.get("market_title", pd.Series("", index=work.index)).fillna("").astype(str).str.strip()
        + "|"
        + timestamp_key
    )


def load_entry_snapshot_frame(logs_dir="logs") -> pd.DataFrame:
    logs_path = Path(logs_dir)
    source_specs = [
        ("trade_events.csv", "signal_snapshot_json", "timestamp", "trade_events_signal"),
        ("positions.csv", "entry_signal_snapshot_json", "opened_at", "positions_open"),
        ("closed_positions.csv", "entry_signal_snapshot_json", "opened_at", "closed_positions"),
    ]
    rows: list[dict] = []
    for filename, json_col, fallback_ts_col, source_name in source_specs:
        df = _safe_read_csv(logs_path / filename)
        if df.empty or json_col not in df.columns:
            continue
        if filename == "trade_events.csv" and "event" in df.columns:
            df = df[df["event"].astype(str).str.lower() == "signal"].copy()
        for _, row in df.iterrows():
            snapshot = _parse_snapshot_json(row.get(json_col))
            if not snapshot:
                continue
            record = dict(snapshot)
            record["timestamp"] = record.get("timestamp") or row.get(fallback_ts_col)
            record["token_id"] = record.get("token_id") or row.get("token_id")
            record["condition_id"] = record.get("condition_id") or row.get("condition_id")
            record["outcome_side"] = record.get("outcome_side") or row.get("outcome_side") or row.get("side")
            record["trader_wallet"] = record.get("trader_wallet") or row.get("trader_wallet") or row.get("wallet_copied")
            record["market_title"] = record.get("market_title") or record.get("market") or row.get("market_title") or row.get("market")
            record["market"] = record.get("market") or record.get("market_title") or row.get("market") or row.get("market_title")
            record["_snapshot_source"] = source_name
            rows.append(record)
    if not rows:
        return pd.DataFrame()

    snapshots = pd.DataFrame(rows)
    snapshots = _normalize_market_columns(snapshots)
    if "timestamp" not in snapshots.columns:
        snapshots["timestamp"] = pd.NaT
    snapshots["timestamp"] = _parse_logged_timestamp_series(snapshots["timestamp"])
    snapshots = snapshots.loc[snapshots["timestamp"].notna()].copy()
    snapshots["_event_key"] = _event_key(snapshots)
    snapshots = snapshots.drop_duplicates(subset=["_event_key"], keep="last").reset_index(drop=True)
    return snapshots


def enrich_frame_with_entry_snapshots(base_df: pd.DataFrame, logs_dir="logs") -> pd.DataFrame:
    snapshots = load_entry_snapshot_frame(logs_dir=logs_dir)
    if snapshots.empty:
        return base_df.copy() if base_df is not None else pd.DataFrame()

    base = base_df.copy() if base_df is not None and not base_df.empty else pd.DataFrame()
    if base.empty:
        out = snapshots.copy()
        out["entry_snapshot_backfilled"] = True
        return out

    base = _normalize_market_columns(base)
    if "timestamp" not in base.columns:
        base["timestamp"] = pd.NaT
    base["timestamp"] = _parse_logged_timestamp_series(base["timestamp"])
    base["_event_key"] = _event_key(base)
    snapshots = snapshots.copy()
    snapshots["entry_snapshot_backfilled"] = True

    merged = base.merge(
        snapshots,
        on="_event_key",
        how="left",
        suffixes=("", "__snapshot"),
    )

    base_cols = list(base.columns)
    for snapshot_col in snapshots.columns:
        if snapshot_col == "_event_key":
            continue
        merged_col = f"{snapshot_col}__snapshot"
        if merged_col not in merged.columns:
            continue
        if snapshot_col not in merged.columns:
            merged[snapshot_col] = merged[merged_col]
            continue
        left = merged[snapshot_col]
        right = merged[merged_col]
        if pd.api.types.is_numeric_dtype(left):
            numeric_left = pd.to_numeric(left, errors="coerce")
            numeric_right = pd.to_numeric(right, errors="coerce")
            merged[snapshot_col] = numeric_left.where(numeric_left.notna(), numeric_right)
        else:
            merged[snapshot_col] = left.where(left.notna() & (left.astype(str) != ""), right)

    event_keys = set(base["_event_key"].dropna().astype(str))
    appended = snapshots[~snapshots["_event_key"].isin(event_keys)].copy()
    backfilled_series = merged.get("entry_snapshot_backfilled")
    if isinstance(backfilled_series, pd.Series):
        merged["entry_snapshot_backfilled"] = _coerce_bool_series(backfilled_series, default=False)
    else:
        merged["entry_snapshot_backfilled"] = False

    drop_cols = [col for col in merged.columns if col.endswith("__snapshot")]
    merged = merged.drop(columns=drop_cols)
    if not appended.empty:
        appended_aligned = appended.reindex(columns=merged.columns)
        merged_records = merged.to_dict("records")
        merged_records.extend(appended_aligned.to_dict("records"))
        merged = pd.DataFrame(merged_records, columns=merged.columns)
    return merged.drop(columns=["_event_key"], errors="ignore")
