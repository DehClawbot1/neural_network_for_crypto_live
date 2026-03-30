import os
from pprint import pprint

import requests

from execution_client import ExecutionClient

RPC = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
WALLET = "0x672c1b45553aac41e9dccdf68a65be6a401c3176"
TOKENS = {
    "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    "USDC.e": "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",
}


def rpc_call(method, params):
    response = requests.post(
        RPC,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    return payload.get("result")


def encode_balance_of(wallet):
    selector = "70a08231"
    wallet_hex = wallet.lower().replace("0x", "")
    return "0x" + selector + ("0" * 24) + wallet_hex


def token_decimals(token):
    result = rpc_call("eth_call", [{"to": token, "data": "0x313ce567"}, "latest"])
    return int(result, 16)


def token_balance(token, wallet):
    data = encode_balance_of(wallet)
    result = rpc_call("eth_call", [{"to": token, "data": data}, "latest"])
    return int(result, 16)


def main():
    print("RPC:", RPC)
    print("wallet:", WALLET)

    for name, token in TOKENS.items():
        try:
            decimals = token_decimals(token)
            raw_balance = token_balance(token, WALLET)
            human_balance = raw_balance / (10 ** decimals)
            print(f"{name} token={token} raw={raw_balance} balance={human_balance}")
        except Exception as exc:
            print(f"{name} token={token} error={exc}")

    try:
        client = ExecutionClient()
        payload = client.get_balance_allowance(asset_type="COLLATERAL")
        print("CLOB collateral payload:")
        pprint(payload)
    except Exception as exc:
        print("CLOB collateral error:", exc)


if __name__ == "__main__":
    main()
