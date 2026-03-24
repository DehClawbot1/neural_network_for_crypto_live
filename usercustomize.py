"""User-level runtime patches loaded automatically by Python when enabled.

This hook extends the live-test ExecutionClient with a broader Polymarket
example-compatible capability surface without forcing call-site rewrites.
"""

from __future__ import annotations

try:
    from polymarket_capabilities import apply_execution_client_patch
except Exception:
    apply_execution_client_patch = None

if apply_execution_client_patch is not None:
    try:
        apply_execution_client_patch()
    except Exception:
        pass
