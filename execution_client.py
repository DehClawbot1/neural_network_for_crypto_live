import os
from dataclasses import asdict


class ExecutionClient:
    """
    Live-test execution wrapper around Polymarket's py-clob-client.
    Intended only for the isolated live-test branch.
    """

    def __init__(self, host=None, chain_id=None, private_key=None, funder=None, signature_type=None):
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
            from py_clob_client.credentials import ApiCreds
        except Exception as exc:
            raise ImportError("py-clob-client is required for live-test execution_client.py") from exc

        self.ClobClient = ClobClient
        self.OrderArgs = OrderArgs
        self.OrderType = OrderType
        self.BUY = BUY
        self.SELL = SELL
        self.ApiCreds = ApiCreds

        self.host = host or os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
        self.chain_id = int(chain_id or os.getenv("POLYMARKET_CHAIN_ID", "137"))
        self.private_key = private_key or os.getenv("PRIVATE_KEY")
        self.funder = funder or os.getenv("POLYMARKET_FUNDER")
        self.signature_type = int(signature_type or os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))
        self.api_key = os.getenv("POLYMARKET_API_KEY")
        self.api_secret = os.getenv("POLYMARKET_API_SECRET")
        self.api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE")

        if not self.private_key:
            raise ValueError("PRIVATE_KEY is required for live-test execution client")

        if self.api_key and self.api_secret and self.api_passphrase:
            self.api_creds = self.ApiCreds(self.api_key, self.api_secret, self.api_passphrase)
        else:
            temp_client = self.ClobClient(self.host, key=self.private_key, chain_id=self.chain_id)
            self.api_creds = temp_client.create_or_derive_api_creds()
            print(
                "SAVE THESE TO .ENV:\n"
                f"POLYMARKET_API_KEY={getattr(self.api_creds, 'api_key', getattr(self.api_creds, 'key', ''))}\n"
                f"POLYMARKET_API_SECRET={getattr(self.api_creds, 'api_secret', getattr(self.api_creds, 'secret', ''))}\n"
                f"POLYMARKET_API_PASSPHRASE={getattr(self.api_creds, 'api_passphrase', getattr(self.api_creds, 'passphrase', ''))}"
            )

        self.client = self.ClobClient(
            self.host,
            key=self.private_key,
            chain_id=self.chain_id,
            creds=self.api_creds,
            signature_type=self.signature_type,
            funder=self.funder,
        )

    def create_and_post_order(self, token_id, price, size, side="BUY", order_type="GTC", options=None):
        side_const = self.BUY if str(side).upper() == "BUY" else self.SELL
        order_type_const = getattr(self.OrderType, str(order_type).upper())
        args = self.OrderArgs(token_id=token_id, price=float(price), size=float(size), side=side_const, order_type=order_type_const)
        return self.client.create_and_post_order(args, options=options or {})

    def create_and_post_market_order(self, token_id, amount, side="BUY"):
        side_const = self.BUY if str(side).upper() == "BUY" else self.SELL
        return self.client.create_and_post_market_order(token_id=token_id, amount=float(amount), side=side_const)

    def cancel_order(self, order_id):
        return self.client.cancel(order_id)

    def get_order(self, order_id):
        return self.client.get_order(order_id)

    def get_open_orders(self):
        return self.client.get_orders()

    def get_trades(self):
        return self.client.get_trades()

    def get_balance_allowance(self, asset_type=None):
        kwargs = {}
        if asset_type is not None:
            kwargs["asset_type"] = asset_type
        return self.client.get_balance_allowance(**kwargs)
