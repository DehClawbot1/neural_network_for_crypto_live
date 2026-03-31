from __future__ import annotations

from datetime import datetime, timezone
import threading


_LOCK = threading.Lock()
_ERRORS: list[dict] = []


def reset_cycle() -> None:
    with _LOCK:
        _ERRORS.clear()


def report_error(stage: str, error: Exception, context: str | None = None) -> None:
    payload = {
        "stage": str(stage or "unknown"),
        "error_type": type(error).__name__,
        "message": str(error),
        "context": str(context or ""),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with _LOCK:
        _ERRORS.append(payload)


def has_errors() -> bool:
    with _LOCK:
        return len(_ERRORS) > 0


def get_errors(limit: int | None = None) -> list[dict]:
    with _LOCK:
        rows = list(_ERRORS)
    if limit is None:
        return rows
    try:
        lim = max(0, int(limit))
    except Exception:
        lim = 0
    if lim <= 0:
        return []
    return rows[:lim]

