from __future__ import annotations

from execution_client import ExecutionClient
from polymarket_capabilities import SUPPORTED_POLYMARKET_EXAMPLE_METHODS


class RepositoryPolymarketClient(ExecutionClient):
    """Repository-native execution client with the Polymarket example capability surface."""

    def list_capabilities(self):
        return list(SUPPORTED_POLYMARKET_EXAMPLE_METHODS)

    def supports_capability(self, capability_name: str):
        return capability_name in SUPPORTED_POLYMARKET_EXAMPLE_METHODS and callable(getattr(self, capability_name, None))

    def call_capability(self, capability_name: str, **kwargs):
        method = getattr(self, capability_name, None)
        if method is None or not callable(method):
            raise AttributeError(f"Unsupported Polymarket capability: {capability_name}")
        return method(**kwargs)
