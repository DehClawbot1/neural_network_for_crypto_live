from __future__ import annotations

try:
    from polymarket_capabilities import apply_execution_client_patch
except Exception:
    apply_execution_client_patch = None

try:
    from retrainer_runtime_patch import apply_retrainer_runtime_patch
except Exception:
    apply_retrainer_runtime_patch = None


def apply_all_runtime_patches() -> None:
    if apply_execution_client_patch is not None:
        try:
            apply_execution_client_patch()
        except Exception:
            pass
    if apply_retrainer_runtime_patch is not None:
        try:
            apply_retrainer_runtime_patch()
        except Exception:
            pass
