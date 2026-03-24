from __future__ import annotations

from functools import lru_cache
from typing import Any

from execution_client import ExecutionClient
from polymarket_capabilities import SUPPORTED_POLYMARKET_EXAMPLE_METHODS


@lru_cache(maxsize=1)
def _build_execution_client() -> ExecutionClient:
    return ExecutionClient()


def get_execution_client(force_new: bool = False) -> ExecutionClient:
    if force_new:
        _build_execution_client.cache_clear()
    return _build_execution_client()


def list_polymarket_capabilities() -> list[str]:
    return list(SUPPORTED_POLYMARKET_EXAMPLE_METHODS)


def has_polymarket_capability(capability: str) -> bool:
    return capability in SUPPORTED_POLYMARKET_EXAMPLE_METHODS


def call_polymarket_capability(capability: str, **kwargs: Any) -> Any:
    if not has_polymarket_capability(capability):
        raise AttributeError(f"Unsupported Polymarket capability: {capability}")
    client = get_execution_client()
    method = getattr(client, capability, None)
    if method is None or not callable(method):
        raise AttributeError(f"ExecutionClient does not expose capability: {capability}")
    return method(**kwargs)
