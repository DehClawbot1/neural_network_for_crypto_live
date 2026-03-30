"""User-level runtime patches loaded automatically by Python when enabled.

This hook applies repo-local runtime patches for Polymarket capability wiring
and retraining/runtime overrides without forcing broad call-site rewrites.
"""

from __future__ import annotations

try:
    from usercustomize_patches import apply_all_runtime_patches
except Exception:
    apply_all_runtime_patches = None

if apply_all_runtime_patches is not None:
    try:
        apply_all_runtime_patches()
    except Exception:
        pass
else:
    try:
        from polymarket_capabilities import apply_execution_client_patch
    except Exception:
        apply_execution_client_patch = None

    if apply_execution_client_patch is not None:
        try:
            apply_execution_client_patch()
        except Exception:
            pass
