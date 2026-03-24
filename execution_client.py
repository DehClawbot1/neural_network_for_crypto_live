import os
import logging
from pathlib import Path

import requests

from dotenv import set_key


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
        env_signature_type = os.getenv("POLYMARKET_SIGNATURE_TYPE", "").strip()
        if signature_type is not None:
            self.signature_type = int(signature_type)
        elif env_signature_type:
            self.signature_type = int(env_signature_type)
        else:
            self.signature_type = 1 if self.funder else 0
        self.api_key = os.getenv("POLYMARKET_API_KEY")
        self.api_secret = os.getenv("POLYMARKET_API_SECRET")
        self.api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE")

        if not self.private_key:
            raise ValueError("PRIVATE_KEY is required for live-test execution client")
        if self.funder and not str(self.funder).startswith("0x"):
            logging.warning("ExecutionClient: POLYMARKET_FUNDER does not look like a wallet address: %s", self.funder)
        if self.signature_type == 1 and not self.funder:
            logging.warning("ExecutionClient: signature_type=1 without POLYMARKET_FUNDER may be incorrect for proxy-wallet accounts.")

        self.client = self.ClobClient(
            self.host,
            key=self.private_key,
            chain_id=self.chain_id,
            signature_type=self.signature_type,
            funder=self.funder,
        )
        self.api_creds = None
        self.credential_source = None
        self.env_file = Path(".env")
        self._initialize_credentials()

    def _build_stored_creds(self):
        if self.api_key and self.api_secret and self.api_passphrase:
            return self.ApiCreds(self.api_key, self.api_secret, self.api_passphrase)
        return None

    def _validate_current_creds(self):
        params = self.BalanceAllowanceParams(asset_type=self.AssetType.COLLATERAL)
        return self.client.get_balance_allowance(params=params)

    def _creds_tuple(self, creds):
        return (
            getattr(creds, "api_key", getattr(creds, "key", None)),
            getattr(creds, "api_secret", getattr(creds, "secret", None)),
            getattr(creds, "api_passphrase", getattr(creds, "passphrase", None)),
        )

    def _persist_creds_to_env(self, creds):
        if not self.env_file.exists():
            self.env_file.touch()
        set_key(str(self.env_file), "POLYMARKET_API_KEY", getattr(creds, "api_key", getattr(creds, "key", "")))
        set_key(str(self.env_file), "POLYMARKET_API_SECRET", getattr(creds, "api_secret", getattr(creds, "secret", "")))
        set_key(str(self.env_file), "POLYMARKET_API_PASSPHRASE", getattr(creds, "api_passphrase", getattr(creds, "passphrase", "")))

    def _initialize_credentials(self):
        stored_creds = self._build_stored_creds()
        if stored_creds is not None:
            try:
                self.client.set_api_creds(stored_creds)
                self._validate_current_creds()
                self.api_creds = stored_creds
                self.credential_source = "stored_env"
                return
            except Exception as exc:
                logging.warning("ExecutionClient: stored L2 creds failed validation: %s", exc)

        try:
            derived_creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(derived_creds)
            self._validate_current_creds()
            self.api_creds = derived_creds
            self.credential_source = "derived_refreshed_env"
            self._persist_creds_to_env(derived_creds)
            print("REFRESHED .ENV WITH DERIVED L2 CREDS (values persisted, hidden in logs).")
            return
        except Exception as exc:
            raise exc

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

    def _build_balance_params(self, asset_type="COLLATERAL", token_id=None):
        target_asset = self.AssetType.COLLATERAL if str(asset_type).upper() == "COLLATERAL" else self.AssetType.CONDITIONAL
        kwargs = {"asset_type": target_asset}
        if target_asset == self.AssetType.CONDITIONAL and token_id is not None:
            kwargs["token_id"] = str(token_id)
        return self.BalanceAllowanceParams(**kwargs)

    def update_balance_allowance(self, asset_type="COLLATERAL", token_id=None):
        params = self._build_balance_params(asset_type=asset_type, token_id=token_id)
        if hasattr(self.client, "update_balance_allowance"):
            return self.client.update_balance_allowance(params=params)
        return None

    def get_balance_allowance(self, asset_type="COLLATERAL", token_id=None):
        params = self._build_balance_params(asset_type=asset_type, token_id=token_id)
        return self.client.get_balance_allowance(params=params)

    def _rpc_call(self, rpc_url, method, params):
        response = requests.post(
            rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        return payload.get("result")

    def _erc20_balance(self, token_address, wallet_address, rpc_url=None):
        wallet_address = str(wallet_address or "")
        if not wallet_address.startswith("0x"):
            return 0.0
        rpc_url = rpc_url or os.getenv("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com")
        decimals_result = self._rpc_call(rpc_url, "eth_call", [{"to": token_address, "data": "0x313ce567"}, "latest"])
        decimals = int(decimals_result, 16)
        balance_data = "0x70a08231" + ("0" * 24) + wallet_address.lower().replace("0x", "")
        balance_result = self._rpc_call(rpc_url, "eth_call", [{"to": token_address, "data": balance_data}, "latest"])
        raw_balance = int(balance_result, 16)
        return raw_balance / (10 ** decimals)

    def get_onchain_collateral_balance(self, wallet_address=None):
        wallet_address = wallet_address or self.funder or os.getenv("POLYMARKET_PUBLIC_ADDRESS")
        token_candidates = [
            "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
            "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",
        ]
        balances = {}
        total = 0.0
        for token in token_candidates:
            try:
                balance = float(self._erc20_balance(token, wallet_address))
            except Exception:
                balance = 0.0
            balances[token] = balance
            total += balance
        return {
            "wallet": wallet_address,
            "balances": balances,
            "total": total,
        }

    def get_available_balance(self, asset_type=None):
        payload = self.get_balance_allowance(asset_type=asset_type)
        api_balance = 0.0
        if isinstance(payload, dict):
            for key in ["balance", "available", "available_balance", "amount"]:
                if payload.get(key) is not None:
                    api_balance = float(payload[key])
                    break
        if str(asset_type or "COLLATERAL").upper() == "COLLATERAL":
            try:
                onchain = self.get_onchain_collateral_balance()
                onchain_total = float((onchain or {}).get("total", 0.0) or 0.0)
                return max(api_balance, onchain_total)
            except Exception:
                return api_balance
        return api_balance

