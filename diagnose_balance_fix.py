"""
diagnose_balance_fix.py
========================
Run this AFTER applying the fix to verify your balance is being read correctly.

Usage:
    python diagnose_balance_fix.py

Expected output: Shows both raw API response and normalized $ balance.
If you see money on the website but $0.00 here, the issue is deeper
(approval/allowance problem or wrong wallet).
"""

from pprint import pprint

from execution_client import ExecutionClient


def main():
    print("=" * 60)
    print("BALANCE DIAGNOSTIC")
    print("=" * 60)

    client = ExecutionClient()
    print(f"\nCredential source: {client.credential_source}")
    print(f"Funder wallet: {client.funder}")

    # Step 1: Raw API response
    print("\n--- RAW API RESPONSE ---")
    try:
        raw = client.get_balance_allowance(asset_type="COLLATERAL")
        pprint(raw)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raw = {}

    # Step 2: Normalized balance
    print("\n--- NORMALIZED BALANCE ---")
    if isinstance(raw, dict):
        for key in ["balance", "available", "available_balance", "amount"]:
            if raw.get(key) is not None:
                raw_val = raw[key]
                normalized = client._normalize_usdc_balance(raw_val)
                print(f"  Key '{key}': raw={raw_val} -> normalized=${normalized:.6f}")
    else:
        print(f"  API returned non-dict: {type(raw)}")

    # Step 3: get_available_balance (includes on-chain fallback)
    print("\n--- AVAILABLE BALANCE (with on-chain fallback) ---")
    try:
        available = client.get_available_balance(asset_type="COLLATERAL")
        print(f"  Available to trade: ${available:.2f}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # Step 4: On-chain USDC
    print("\n--- ON-CHAIN USDC ---")
    try:
        onchain = client.get_onchain_collateral_balance()
        pprint(onchain)
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # Step 5: Test small market order feasibility
    print("\n--- MARKET ORDER TEST (dry run) ---")
    if isinstance(raw, dict):
        raw_balance = 0.0
        for key in ["balance", "available", "available_balance", "amount"]:
            if raw.get(key) is not None:
                raw_balance = client._normalize_usdc_balance(raw.get(key))
                break
        print(f"  Normalized CLOB balance: ${raw_balance:.2f}")
        if raw_balance >= 1.0:
            print(f"  ✅ Can place a $1.00 market order")
        elif raw_balance > 0:
            print(f"  ⚠️  Balance is ${raw_balance:.2f} — can only bet up to that amount")
        else:
            print(f"  ❌ CLOB balance is $0 — check if funds need approval")
            print(f"     On polymarket.com, make sure you have deposited USDC")
            print(f"     and that it's available for trading (not just in your wallet)")

    print("\n" + "=" * 60)
    print("If balance shows $0 but you have funds on polymarket.com:")
    print("  1. Check that POLYMARKET_FUNDER matches your deposit address")
    print("  2. Try making a small trade on the website first")
    print("  3. Check if you need to approve USDC spending for CLOB")
    print("=" * 60)


if __name__ == "__main__":
    main()
