import os
import logging


class ExecutionClient:
    """
    Live-test execution wrapper around Polymarket's py-clob-client.
    Intended only for the isolated live-test branch.
    """

    def __init__(self, host=None, chain_id=None, private_key=None, funder=None, signature_type=None):
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType, MarketOrderArgs, BalanceAllowanceParams, AssetType, ApiCreds
            from py_clob_client.order_builder.constants import BUY, SELL
        except Exception as exc:
            raise ImportError("py-clob-client is required for live-test execution_client.py") from exc

        self.ClobClient = ClobClient
        self.OrderArgs = OrderArgs
        self.MarketOrderArgs = MarketOrderArgs
        self.BalanceAllowanceParams = BalanceAllowanceParams
        self.AssetType = AssetType
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

        self.client = self.ClobClient(
            self.host,
            key=self.private_key,
            chain_id=self.chain_id,
            signature_type=self.signature_type,
            funder=self.funder,
        )
        self.api_creds = None
        self.credential_source = None
        self._initialize_credentials()

    def _build_stored_creds(self):
        if self.api_key and self.api_secret and self.api_passphrase:
            return self.ApiCreds(self.api_key, self.api_secret, self.api_passphrase)
        return None

    def _validate_current_creds(self):
        params = self.BalanceAllowanceParams(asset_type=self.AssetType.COLLATERAL)
        return self.client.get_balance_allowance(params=params)

    def _initialize_credentials(self):
        attempts = []
        stored_creds = self._build_stored_creds()
        if stored_creds is not None:
            attempts.append(("stored_env", stored_creds))
        try:
            derived_creds = self.client.create_or_derive_api_creds()
            attempts.append(("derived", derived_creds))
        except Exception as exc:
            logging.warning("ExecutionClient: failed to derive API creds: %s", exc)

        last_exc = None
        seen = set()
        for source, creds in attempts:
            if creds is None:
                continue
            key_tuple = (
                getattr(creds, "api_key", getattr(creds, "key", None)),
                getattr(creds, "api_secret", getattr(creds, "secret", None)),
                getattr(creds, "api_passphrase", getattr(creds, "passphrase", None)),
            )
            if key_tuple in seen:
                continue
            seen.add(key_tuple)
            try:
                self.client.set_api_creds(creds)
                self._validate_current_creds()
                self.api_creds = creds
                self.credential_source = source
                if source == "derived":
                    print(
                        "SAVE THESE TO .ENV:\n"
                        f"POLYMARKET_API_KEY={getattr(creds, 'api_key', getattr(creds, 'key', ''))}\n"
                        f"POLYMARKET_API_SECRET={getattr(creds, 'api_secret', getattr(creds, 'secret', ''))}\n"
                        f"POLYMARKET_API_PASSPHRASE={getattr(creds, 'api_passphrase', getattr(creds, 'passphrase', ''))}"
                    )
                return
            except Exception as exc:
                last_exc = exc
                logging.warning("ExecutionClient: credential attempt '%s' failed: %s", source, exc)

        if last_exc is not None:
            raise last_exc
        raise ValueError("Unable to initialize Polymarket API credentials")

    def create_and_post_order(self, token_id, price, size, side="BUY", order_type="GTC", options=None):
        side_const = self.BUY if str(side).upper() == "BUY" else self.SELL
        order_type_const = getattr(self.OrderType, str(order_type).upper())
        args = self.OrderArgs(token_id=token_id, price=float(price), size=float(size), side=side_const)
        signed_order = self.client.create_order(args, options=options or {})
        return self.client.post_order(signed_order, order_type_const)

    def create_and_post_market_order(self, token_id, amount, side="BUY", order_type="FOK"):
        side_const = self.BUY if str(side).upper() == "BUY" else self.SELL
        order_type_const = getattr(self.OrderType, str(order_type).upper())
        args = self.MarketOrderArgs(token_id=token_id, amount=float(amount), side=side_const)
        signed_order = self.client.create_market_order(args)
        return self.client.post_order(signed_order, order_type_const)

    def cancel_order(self, order_id):
        return self.client.cancel(order_id)

    def get_order(self, order_id):
        return self.client.get_order(order_id)

    def get_open_orders(self):
        return self.client.get_orders()

    def get_trades(self):
        return self.client.get_trades()

    def get_balance_allowance(self, asset_type="COLLATERAL"):
        target_asset = self.AssetType.COLLATERAL if str(asset_type).upper() == "COLLATERAL" else self.AssetType.CONDITIONAL
        params = self.BalanceAllowanceParams(asset_type=target_asset)
        return self.client.get_balance_allowance(params=params)

    def get_available_balance(self, asset_type=None):
        payload = self.get_balance_allowance(asset_type=asset_type)
        if isinstance(payload, dict):
            for key in ["balance", "available", "available_balance", "amount"]:
                if payload.get(key) is not None:
                    return float(payload[key])
        return 0.0

