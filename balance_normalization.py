from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


_TRACED_ALLOWANCE_KEYS: set[str] = set()


def _is_micro_balance_mode() -> bool:
    try:
        from config import TradingConfig

        return bool(getattr(TradingConfig, "BALANCE_IS_MICRODOLLARS", True))
    except Exception:
        return True


def normalize_allowance_balance(raw_balance, asset_type="COLLATERAL") -> float:
    if raw_balance is None:
        return 0.0
    raw_text = str(raw_balance).strip()
    if not raw_text:
        return 0.0
    try:
        val = float(raw_text)
    except (TypeError, ValueError):
        return 0.0

    # Polymarket allowance payloads for collateral and conditional balances
    # are represented in 6-decimal raw units when returned as integer-like strings.
    if _is_micro_balance_mode():
        if re.fullmatch(r"-?\d+", raw_text):
            try:
                return int(raw_text) / 1e6
            except Exception:
                return val
        if abs(val - round(val)) < 1e-9 and abs(val) >= 1_000_000:
            return val / 1e6
    return val


def extract_balance_value(payload):
    if isinstance(payload, dict):
        for key in ("balance", "available", "available_balance", "amount"):
            if payload.get(key) is not None:
                return key, payload.get(key)
    elif payload is not None:
        return "scalar", payload
    return None, None


def maybe_trace_allowance_payload(
    *,
    logs_dir="logs",
    source: str,
    asset_type: str,
    token_id: str | None,
    payload,
    normalized_balance,
    local_balance=None,
    note: str | None = None,
):
    trace_all = str(os.getenv("TRACE_ALLOWANCE_PAYLOADS", "false")).strip().lower() in {"1", "true", "yes", "on"}
    trace_once = str(os.getenv("TRACE_CONDITIONAL_BALANCE_ONCE", "false")).strip().lower() in {"1", "true", "yes", "on"}
    if not (trace_all or trace_once):
        return

    asset_label = str(asset_type or "").upper()
    trace_key = f"{source}|{asset_label}|{str(token_id or '')}"
    if trace_once and not trace_all:
        if trace_key in _TRACED_ALLOWANCE_KEYS:
            return
        _TRACED_ALLOWANCE_KEYS.add(trace_key)

    picked_key, raw_value = extract_balance_value(payload)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "asset_type": asset_label,
        "token_id": str(token_id or ""),
        "picked_key": picked_key,
        "raw_value": raw_value,
        "normalized_balance": float(normalized_balance or 0.0),
        "local_balance": None if local_balance is None else float(local_balance),
        "note": note or "",
        "payload": payload,
    }
    trace_path = Path(logs_dir) / "allowance_balance_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
