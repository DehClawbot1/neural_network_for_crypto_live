from pprint import pprint

from execution_client import ExecutionClient


def main():
    client = ExecutionClient()

    print("BEFORE:")
    pprint(client.get_balance_allowance(asset_type="COLLATERAL"))

    print("\nUPDATE:")
    try:
        pprint(client.update_balance_allowance(asset_type="COLLATERAL"))
    except Exception as exc:
        print("UPDATE_ERROR:", exc)

    print("\nAFTER:")
    pprint(client.get_balance_allowance(asset_type="COLLATERAL"))


if __name__ == "__main__":
    main()
