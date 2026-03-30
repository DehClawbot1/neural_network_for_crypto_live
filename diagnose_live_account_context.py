import os
from pprint import pprint

from execution_client import ExecutionClient
from polymarket_profile_client import PolymarketProfileClient


def _safe_env(name: str):
    value = os.getenv(name)
    return value if value else None


def _derive_eoa(private_key: str | None):
    if not private_key:
        return None
    try:
        from eth_account import Account
        return Account.from_key(private_key).address
    except Exception as exc:
        return f"<derive_failed: {exc}>"


def _value_number(payload):
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first.get("value")
    if isinstance(payload, dict):
        return payload.get("value")
    return payload


def inspect_address(label: str, address: str | None, profile_client: PolymarketProfileClient):
    print(f"\n=== {label} ===")
    print("address:", address)
    if not address or str(address).startswith("<derive_failed"):
        return
    try:
        total_value = profile_client.get_total_value(address)
        print("total_value:", _value_number(total_value))
    except Exception as exc:
        print("total_value_error:", exc)
    try:
        positions = profile_client.get_positions(user=address, limit=10, offset=0, size_threshold=0)
        count = len(positions) if isinstance(positions, list) else "?"
        print("positions_count:", count)
        if isinstance(positions, list):
            preview = positions[:3]
            pprint(preview)
    except Exception as exc:
        print("positions_error:", exc)
    try:
        trades = profile_client.get_trades(user=address, limit=10, offset=0)
        count = len(trades) if isinstance(trades, list) else "?"
        print("recent_trades_count:", count)
        if isinstance(trades, list):
            preview = trades[:3]
            pprint(preview)
    except Exception as exc:
        print("trades_error:", exc)


def main():
    funder = _safe_env("POLYMARKET_FUNDER")
    public_address = _safe_env("POLYMARKET_PUBLIC_ADDRESS")
    signature_type = _safe_env("POLYMARKET_SIGNATURE_TYPE")
    private_key = _safe_env("PRIVATE_KEY")
    derived_eoa = _derive_eoa(private_key)

    print("=== EFFECTIVE LIVE ACCOUNT CONTEXT ===")
    print("TRADING_MODE:", _safe_env("TRADING_MODE"))
    print("POLYMARKET_FUNDER:", funder)
    print("POLYMARKET_PUBLIC_ADDRESS:", public_address)
    print("POLYMARKET_SIGNATURE_TYPE:", signature_type)
    print("PRIVATE_KEY_DERIVED_EOA:", derived_eoa)

    client = ExecutionClient()
    profile_client = PolymarketProfileClient()

    try:
        collateral = client.get_balance_allowance(asset_type="COLLATERAL")
        print("\nCLOB_COLLATERAL_BALANCE_ALLOWANCE:")
        pprint(collateral)
    except Exception as exc:
        print("\nCLOB_COLLATERAL_ERROR:", exc)

    inspect_address("FUNDER", funder, profile_client)
    if public_address and public_address != funder:
        inspect_address("PUBLIC_ADDRESS", public_address, profile_client)
    if derived_eoa and derived_eoa not in {funder, public_address} and not str(derived_eoa).startswith("<derive_failed"):
        inspect_address("DERIVED_EOA", derived_eoa, profile_client)


if __name__ == "__main__":
    main()
