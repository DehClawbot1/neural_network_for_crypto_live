from polymarket_capabilities import SUPPORTED_POLYMARKET_EXAMPLE_METHODS, apply_execution_client_patch


def test_execution_client_exposes_polymarket_example_surface():
    apply_execution_client_patch()

    from execution_client import ExecutionClient

    missing = [
        name
        for name in SUPPORTED_POLYMARKET_EXAMPLE_METHODS
        if not hasattr(ExecutionClient, name)
    ]
    assert missing == []
