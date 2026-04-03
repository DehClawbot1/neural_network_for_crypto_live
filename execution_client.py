import os
import logging
import re
from pathlib import Path

import requests
from balance_normalization import normalize_allowance_balance

# Only import set_key if we'll actually write to .env
try:
    from dotenv import set_key
except ImportError:
    set_key = None


def _is_interactive():
    return os.environ.get("_INTERACTIVE_MODE") == "1"



# --- DUST POLLING BYPASS ---
# Intercepts py_clob_client to instantly resolve fake dust orders during polling
import py_clob_client.client
if not hasattr(py_clob_client.client.ClobClient, '_dust_patched'):
    _orig_get_order = py_clob_client.client.ClobClient.get_order
    def _patched_get_order(self, order_id, *args, **kwargs):
        # If the bot asks about our synthetic dust order, instantly feed it a FILLED response
        if order_id and "dust_clear" in str(order_id):
            return {"status": "FILLED", "id": order_id}
        return _orig_get_order(self, order_id, *args, **kwargs)
        
    _orig_cancel = py_clob_client.client.ClobClient.cancel
    def _patched_cancel(self, order_id, *args, **kwargs):
        # If the bot tries to cancel a synthetic dust order, silently approve it
        if order_id and "dust_clear" in str(order_id):
            return {"status": "canceled"}
        return _orig_cancel(self, order_id, *args, **kwargs)
        
    py_clob_client.client.ClobClient.get_order = _patched_get_order
    py_clob_client.client.ClobClient.cancel = _patched_cancel
    py_clob_client.client.ClobClient._dust_patched = True
# ---------------------------

class ExecutionClient:
    """
    Live-test execution wrapper around Polymarket's py-clob-client.
    Intended only for the isolated live-test branch.

    In interactive mode, derived credentials are kept in memory only
    and never persisted to .env files.

    FIX: Aligned authentication flow with official Polymarket tutorial:
      - Uses derive_api_key() as primary credential derivation (matching tutorial)
      - Falls back to create_or_derive_api_creds() for older client versions
      - Proper USDC balance decimal handling (balance / 1e6 when raw integer)
      - Cleaner order creation matching tutorial patterns

    FIX: signature_type defaults:
      - Email/Magic/Google login → signature_type=1 (proxy wallet)
      - MetaMask/Rabby browser wallet → signature_type=2 (proxy wallet)
      - Direct EOA (no Polymarket account) → signature_type=0
      If you see $0.00 balance, the most common cause is wrong signature_type.
    """

    def __init__(self, host=None, chain_id=None, private_key=None, funder=None, signature_type=None):
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType, MarketOrderArgs, BalanceAllowanceParams, AssetType, ApiCreds, OpenOrderParams, BookParams
            from py_clob_client.order_builder.constants import BUY, SELL
        except Exception as exc:
            raise ImportError("py-clob-client is required for live-test execution_client.py") from exc

        self.ClobClient = ClobClient
        self.OrderArgs = OrderArgs
        self.MarketOrderArgs = MarketOrderArgs
        self.BalanceAllowanceParams = BalanceAllowanceParams
        self.AssetType = AssetType
        self.OrderType = OrderType
        self.OpenOrderParams = OpenOrderParams
        self.BookParams = BookParams
        self.BUY = BUY
        self.SELL = SELL
        self.ApiCreds = ApiCreds

        self.host = host or os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
        self.chain_id = int(chain_id or os.getenv("POLYMARKET_CHAIN_ID", "137"))
        self.private_key = private_key or os.getenv("PRIVATE_KEY")
        raw_funder = funder or os.getenv("POLYMARKET_FUNDER")
        self.funder = str(raw_funder).strip() if raw_funder not in (None, "") else None
        if self.funder:
            sanitized_funder = self.funder.rstrip(".,;: ")
            if sanitized_funder != self.funder:
                logging.warning(
                    "ExecutionClient: POLYMARKET_FUNDER had trailing punctuation; using sanitized address %s",
                    sanitized_funder,
                )
                self.funder = sanitized_funder
        env_signature_type = os.getenv("POLYMARKET_SIGNATURE_TYPE", "").strip()

        # ── FIX: Signature type resolution with clear priority ──
        # Priority: explicit arg > env var > smart default
        if signature_type is not None:
            self.signature_type = int(signature_type)
        elif env_signature_type:
            self.signature_type = int(env_signature_type)
        else:
            # Default to proxy mode for compatibility with existing bot/test flows.
            # Explicit env/arg values still win when the operator really intends to
            # use a different signature type.
            self.signature_type = 1

        self.api_key = os.getenv("POLYMARKET_API_KEY")
        self.api_secret = os.getenv("POLYMARKET_API_SECRET")
        self.api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE")
        self.api_creds_signature_type = os.getenv("POLYMARKET_API_CREDS_SIGNATURE_TYPE", "").strip()
        self.api_creds_funder = os.getenv("POLYMARKET_API_CREDS_FUNDER", "").strip()

        if not self.private_key:
            raise ValueError("PRIVATE_KEY is required for live-test execution client")

        # ── FIX: Warn about common misconfigurations ──
        if self.funder and not str(self.funder).startswith("0x"):
            logging.warning("ExecutionClient: POLYMARKET_FUNDER does not look like a wallet address: %s", self.funder)

        if self.signature_type == 0:
            logging.error("CRITICAL: signature_type=0 (EOA/MetaMask) detected.")
            logging.error("EOA wallets require on-chain USDC/conditional token allowances to trade.")
            logging.error("This bot does not currently execute on-chain allowance transactions.")
            logging.warning(
                "ExecutionClient: continuing with signature_type=0 for compatibility, "
                "but live trading may fail unless allowances are managed externally."
            )

        if self.signature_type == 1 and not self.funder:
            logging.warning(
                "ExecutionClient: signature_type=1 (email/Magic proxy) but POLYMARKET_FUNDER is not set. "
                "Set POLYMARKET_FUNDER to your Polymarket wallet address (shown in your Polymarket profile)."
            )

        logging.info(
            "ExecutionClient: signature_type=%d (%s), funder=%s",
            self.signature_type,
            {0: "EOA", 1: "email/Magic proxy", 2: "MetaMask/Rabby proxy"}.get(self.signature_type, "unknown"),
            self.funder[:16] + "..." if self.funder and len(self.funder) > 16 else self.funder,
        )

        # FIX: Initialize client matching tutorial pattern exactly:
        #   client = ClobClient(CLOB_API, key=PRIVATE_KEY, chain_id=137,
        #                       signature_type=SIGNATURE_TYPE, funder=FUNDER_ADDRESS)
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
        """Save derived credentials.

        In interactive mode: store in os.environ only (memory).
        In .env mode: persist to .env file on disk.
        """
        key = getattr(creds, "api_key", getattr(creds, "key", ""))
        secret = getattr(creds, "api_secret", getattr(creds, "secret", ""))
        passphrase = getattr(creds, "api_passphrase", getattr(creds, "passphrase", ""))

        # Always update in-memory env vars so subsequent calls work
        os.environ["POLYMARKET_API_KEY"] = key
        os.environ["POLYMARKET_API_SECRET"] = secret
        os.environ["POLYMARKET_API_PASSPHRASE"] = passphrase
        os.environ["POLYMARKET_API_CREDS_SIGNATURE_TYPE"] = str(self.signature_type)
        os.environ["POLYMARKET_API_CREDS_FUNDER"] = str(self.funder or "")

        if _is_interactive():
            logging.info("Derived L2 API creds stored in memory (interactive mode — not written to disk).")
            return

        if set_key is None:
            logging.warning("python-dotenv not available; cannot persist creds to .env.")
            return
        if not self.env_file.exists():
            self.env_file.touch()
        set_key(str(self.env_file), "POLYMARKET_API_KEY", key)
        set_key(str(self.env_file), "POLYMARKET_API_SECRET", secret)
        set_key(str(self.env_file), "POLYMARKET_API_PASSPHRASE", passphrase)
        set_key(str(self.env_file), "POLYMARKET_API_CREDS_SIGNATURE_TYPE", str(self.signature_type))
        set_key(str(self.env_file), "POLYMARKET_API_CREDS_FUNDER", str(self.funder or ""))

    def _clear_stored_creds_env(self):
        for key in [
            "POLYMARKET_API_KEY",
            "POLYMARKET_API_SECRET",
            "POLYMARKET_API_PASSPHRASE",
            "POLYMARKET_API_CREDS_SIGNATURE_TYPE",
            "POLYMARKET_API_CREDS_FUNDER",
        ]:
            os.environ.pop(key, None)
        self.api_key = None
        self.api_secret = None
        self.api_passphrase = None
        self.api_creds_signature_type = ""
        self.api_creds_funder = ""

    def _stored_creds_match_context(self):
        stored_sig = str(self.api_creds_signature_type or "").strip()
        stored_funder = str(self.api_creds_funder or "").strip().lower()
        current_sig = str(self.signature_type)
        current_funder = str(self.funder or "").strip().lower()
        if not stored_sig and not stored_funder:
            return True
        return stored_sig == current_sig and stored_funder == current_funder

    def _derive_creds(self):
        """Aligned with official documentation: exclusively uses create_or_derive_api_creds."""
        if hasattr(self.client, "create_or_derive_api_creds"):
            derived_creds = self.client.create_or_derive_api_creds()
            logging.info("Derived credentials using create_or_derive_api_creds()")
            return derived_creds
        raise RuntimeError("No credential derivation method available on ClobClient")

    def _initialize_credentials(self):
        # Step 1: Try stored L2 API credentials from env
        stored_creds = self._build_stored_creds()
        if stored_creds is not None:
            if not self._stored_creds_match_context():
                logging.warning(
                    "ExecutionClient: stored L2 creds were derived for a different funder/signature_type. "
                    "Clearing them and deriving fresh creds for signature_type=%s funder=%s.",
                    self.signature_type,
                    self.funder,
                )
                self._clear_stored_creds_env()
                stored_creds = None
            else:
                try:
                    self.client.set_api_creds(stored_creds)
                    self._validate_current_creds()
                    self.api_creds = stored_creds
                    self.credential_source = "stored_env"
                    logging.info("ExecutionClient: Using stored L2 API credentials.")
                    return
                except Exception as exc:
                    logging.warning("ExecutionClient: stored L2 creds failed validation: %s", exc)

        # Step 2: Derive fresh credentials (FIX: uses tutorial-compatible method)
        try:
            derived_creds = self._derive_creds()
            self.client.set_api_creds(derived_creds)
            self._validate_current_creds()
            self.api_creds = derived_creds
            self.credential_source = "derived_refreshed_env"
            self._persist_creds_to_env(derived_creds)
            if _is_interactive():
                print("DERIVED L2 CREDS (stored in memory only, not on disk).")
            else:
                print("REFRESHED .ENV WITH DERIVED L2 CREDS (values persisted, hidden in logs).")
            return
        except Exception as exc:
            raise exc

    def create_and_post_order(self, token_id, price, size, side="BUY", order_type="GTC", options=None):
        """FIX: Aligned with tutorial limit order pattern:
            limit_order = OrderArgs(token_id=yes_token_id, price=0.001, size=10.0, side=BUY)
            signed_order = auth_client.create_order(limit_order)
            response = auth_client.post_order(signed_order, OrderType.GTC)
        """
        side_const = self.BUY if str(side).upper() == "BUY" else self.SELL
        order_type_const = getattr(self.OrderType, str(order_type).upper())
        args = self.OrderArgs(token_id=token_id, price=float(price), size=float(size), side=side_const)

        # BUG FIX: py_clob_client.create_order expects PartialCreateOrderOptions or None,
        # not a raw dict.
        post_only = False
        create_options = None
        if isinstance(options, dict):
            post_only = bool(options.pop("post_only", False))
            tick_size = options.get("tick_size")
            neg_risk = options.get("neg_risk")
            if tick_size is not None or neg_risk is not None:
                try:
                    from py_clob_client.clob_types import PartialCreateOrderOptions
                    create_options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
                except Exception:
                    create_options = None # BUG FIX 7: Log the failure to prevent silent erasure
                    logging.warning("Failed to map PartialCreateOrderOptions. V5 tick sizing may be dropped.")
        elif options is not None:
            create_options = options

        signed_order = self.client.create_order(args, options=create_options)
        return self.client.post_order(signed_order, order_type_const)

    def create_and_post_market_order(self, token_id, amount, side="BUY", order_type="FOK"):
        """FIX: Aligned with tutorial market order pattern:
            market_order = MarketOrderArgs(token_id=yes_token_id, amount=5.0, side=BUY)
            signed_market_order = auth_client.create_market_order(market_order)
            response = auth_client.post_order(signed_market_order, OrderType.FOK)

        Note: 'amount' is the dollar amount to spend, NOT shares.
        """
        side_const = self.BUY if str(side).upper() == "BUY" else self.SELL
        order_type_const = getattr(self.OrderType, str(order_type).upper())
        args = self.MarketOrderArgs(token_id=token_id, amount=float(amount), side=side_const)
        signed_order = self.client.create_market_order(args)
        return self.client.post_order(signed_order, order_type_const)

    def cancel_order(self, order_id):
        return self.client.cancel(order_id)

    def cancel_all(self):
        """FIX: Added cancel_all matching tutorial pattern."""
        return self.client.cancel_all()

    def get_order(self, order_id):
        return self.client.get_order(order_id)

    def get_open_orders(self):
        """Aligned with tutorial: fetches open orders using OpenOrderParams()"""
        return self.client.get_orders(self.OpenOrderParams())

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

    def _normalize_usdc_balance(self, raw_balance):
        """FIX: Tutorial shows balance needs /1e6 conversion for raw USDC:
            balance = auth_client.get_balance_allowance(...)
            usdc_balance = int(balance['balance']) / 1e6

        Some API responses return raw integer (microdollars), others return
        float dollars. This normalizer handles both cases.
        """
        return normalize_allowance_balance(raw_balance, asset_type="COLLATERAL")

    def _normalize_allowance_balance(self, raw_balance, asset_type="COLLATERAL"):
        return normalize_allowance_balance(raw_balance, asset_type=asset_type)

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
        """Return spendable balance for trading decisions.

        For COLLATERAL, this must come from the CLOB/API allowance, not from raw
        on-chain wallet balance. On-chain USDC is diagnostic only unless explicit
        fallback is enabled.
        """
        payload = self.get_balance_allowance(asset_type=asset_type)
        api_balance = 0.0
        if isinstance(payload, dict):
            for key in ["balance", "available", "available_balance", "amount"]:
                if payload.get(key) is not None:
                    api_balance = self._normalize_allowance_balance(payload[key], asset_type=asset_type)
                    break
        if str(asset_type or "COLLATERAL").upper() == "COLLATERAL":
            allow_onchain_fallback = os.getenv("ALLOW_ONCHAIN_BALANCE_FALLBACK", "false").strip().lower() in {"1", "true", "yes", "on"}
            if allow_onchain_fallback and api_balance <= 0:
                try:
                    onchain = self.get_onchain_collateral_balance()
                    onchain_total = float((onchain or {}).get("total", 0.0) or 0.0)
                    logging.warning(
                        "ExecutionClient: using on-chain fallback balance because CLOB/API balance is zero. "
                        "This is diagnostic-only behavior and can still fail at order placement."
                    )
                    return onchain_total
                except Exception:
                    return api_balance
            return api_balance
        return api_balance

    # ── Tutorial-compatible convenience methods ──

    def get_order_book(self, token_id):
        """FIX: Direct access to order book matching tutorial:
            book = client.get_order_book(yes_token_id)
        """
        return self.client.get_order_book(str(token_id))

    def get_midpoint(self, token_id):
        """FIX: Midpoint matching tutorial:
            mid = client.get_midpoint(yes_token_id)
        """
        return self.client.get_midpoint(str(token_id))

    def get_price(self, token_id, side="BUY"):
        """FIX: Price matching tutorial:
            buy_price = client.get_price(yes_token_id, side="BUY")
        """
        return self.client.get_price(str(token_id), str(side).upper())

    def get_spread(self, token_id):
        """FIX: Spread matching tutorial:
            spread = client.get_spread(yes_token_id)
        """
        return self.client.get_spread(str(token_id))

    def get_simplified_markets(self):
        """Fetches available markets (read-only)."""
        return self.client.get_simplified_markets()

    def get_last_trade_price(self, token_id):
        """Fetches the last traded price for a specific token."""
        return self.client.get_last_trade_price(str(token_id))
        
    def get_order_books(self, token_id):
        """Fetches orderbooks using BookParams array."""
        return self.client.get_order_books([self.client.BookParams(token_id=str(token_id))])
