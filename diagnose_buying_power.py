import sys
from pprint import pprint

from execution_client import ExecutionClient


def main():
    test_notional = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    client = ExecutionClient()

    clob_payload = client.get_balance_allowance(asset_type="COLLATERAL")
    clob_balance = 0.0
    if isinstance(clob_payload, dict):
        for key in ["balance", "amount", "available", "available_balance"]:
            if clob_payload.get(key) is not None:
                clob_balance = float(clob_payload.get(key))
                break

    onchain = client.get_onchain_collateral_balance()
    onchain_total = float((onchain or {}).get("total", 0.0) or 0.0)

    print("TEST_NOTIONAL:", test_notional)
    print("\nCLOB_COLLATERAL:")
    pprint(clob_payload)
    print("\nONCHAIN_COLLATERAL:")
    pprint(onchain)

    if clob_balance >= test_notional:
        source = "CLOB/API"
        allowed = True
    elif onchain_total >= test_notional:
        source = "On-chain USDC fallback"
        allowed = True
    else:
        source = "Insufficient funds"
        allowed = False

    print("\nDECISION:")
    print({
        "allowed": allowed,
        "source": source,
        "clob_balance": clob_balance,
        "onchain_total": onchain_total,
        "test_notional": test_notional,
    })


if __name__ == "__main__":
    main()
