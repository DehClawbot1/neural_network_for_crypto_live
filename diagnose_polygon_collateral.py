import os
from pprint import pprint

from web3 import Web3
from execution_client import ExecutionClient

RPC = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
WALLET = "0x672c1b45553aac41e9dccdf68a65be6a401c3176"
TOKENS = {
    "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    "USDC.e": "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",
}
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def main():
    print("RPC:", RPC)
    print("wallet:", WALLET)
    w3 = Web3(Web3.HTTPProvider(RPC))
    checksum_wallet = Web3.to_checksum_address(WALLET)

    for name, token in TOKENS.items():
        try:
            contract = w3.eth.contract(address=Web3.to_checksum_address(token), abi=ERC20_ABI)
            decimals = contract.functions.decimals().call()
            raw_balance = contract.functions.balanceOf(checksum_wallet).call()
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
