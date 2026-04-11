from __future__ import annotations

from pathlib import Path

import pandas as pd

from brain_paths import BTC_FAMILY, WEATHER_FAMILY, ensure_market_family_column, filter_frame_for_brain, list_brain_contexts, resolve_brain_context
from csv_utils import safe_csv_append_with_schema


def _normalize_path(path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _brain_log_path(path: Path, context) -> Path:
    return context.logs_dir / path.name


def _write_overwrite(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame() if df is None else df.copy()
    out.to_csv(path, index=False)


def _write_append(path: Path, df: pd.DataFrame):
    if df is None or df.empty:
        return
    safe_csv_append_with_schema(path, df)


def _split_for_brains(df: pd.DataFrame, *, shared_logs_dir="logs", shared_weights_dir="weights") -> dict[str, pd.DataFrame]:
    if df is None:
        df = pd.DataFrame()
    if df.empty:
        return {
            context.brain_id: pd.DataFrame(columns=df.columns)
            for context in list_brain_contexts(shared_logs_dir=shared_logs_dir, shared_weights_dir=shared_weights_dir)
        }
    normalized = ensure_market_family_column(df)
    outputs: dict[str, pd.DataFrame] = {}
    for context in list_brain_contexts(shared_logs_dir=shared_logs_dir, shared_weights_dir=shared_weights_dir):
        outputs[context.brain_id] = filter_frame_for_brain(normalized, context)
    return outputs


def append_csv_with_brain_mirrors(
    path,
    df: pd.DataFrame,
    *,
    family_hint: str | None = None,
    include_shared: bool = True,
    shared_logs_dir="logs",
    shared_weights_dir="weights",
):
    csv_path = _normalize_path(path)
    if df is None or df.empty:
        return
    if include_shared:
        _write_append(csv_path, df)
    if family_hint:
        context = resolve_brain_context(
            family_hint,
            shared_logs_dir=shared_logs_dir,
            shared_weights_dir=shared_weights_dir,
        )
        _write_append(_brain_log_path(csv_path, context), df)
        return
    split = _split_for_brains(df, shared_logs_dir=shared_logs_dir, shared_weights_dir=shared_weights_dir)
    for context in list_brain_contexts(shared_logs_dir=shared_logs_dir, shared_weights_dir=shared_weights_dir):
        frame = split.get(context.brain_id)
        if frame is None or frame.empty:
            continue
        _write_append(_brain_log_path(csv_path, context), frame)


def overwrite_csv_with_brain_mirrors(
    path,
    df: pd.DataFrame,
    *,
    family_hint: str | None = None,
    include_shared: bool = True,
    shared_logs_dir="logs",
    shared_weights_dir="weights",
):
    csv_path = _normalize_path(path)
    if include_shared:
        _write_overwrite(csv_path, pd.DataFrame() if df is None else df)
    if family_hint:
        context = resolve_brain_context(
            family_hint,
            shared_logs_dir=shared_logs_dir,
            shared_weights_dir=shared_weights_dir,
        )
        _write_overwrite(_brain_log_path(csv_path, context), pd.DataFrame() if df is None else df)
        return
    split = _split_for_brains(pd.DataFrame() if df is None else df, shared_logs_dir=shared_logs_dir, shared_weights_dir=shared_weights_dir)
    for context in list_brain_contexts(shared_logs_dir=shared_logs_dir, shared_weights_dir=shared_weights_dir):
        frame = split.get(context.brain_id)
        _write_overwrite(_brain_log_path(csv_path, context), frame if frame is not None else pd.DataFrame())

