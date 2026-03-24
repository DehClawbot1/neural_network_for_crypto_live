from __future__ import annotations

import os
import time
from typing import Any, Iterable, Optional

import requests


SUPPORTED_POLYMARKET_EXAMPLE_METHODS = [
    "GTD_order",
    "gtd_order",
    "are_orders_scoring",
    "cancel_all",
    "cancel_market_orders",
    "cancel_order",
    "cancel_orders",
    "create_api_key",
    "create_readonly_api_key",
    "delete_readonly_api_key",
    "derive_api_key",
    "drop_notifications",
    "get_api_keys",
    "get_balance_allowance",
    "get_builder_trades",
    "get_closed_only_mode",
    "get_last_trade_price",
    "get_last_trades_prices",
    "get_market_trades_events",
    "get_markets",
    "get_mid_market_price",
    "get_mid_markets_prices",
    "get_notifications",
    "get_ok",
    "get_open_orders_with_readonly_key",
    "get_order",
    "get_orderbook",
    "get_orderbooks",
    "get_orders",
    "get_price",
    "get_prices",
    "get_readonly_api_keys",
    "get_server_time",
    "get_spread",
    "get_spreads",
    "get_trades",
    "is_order_scoring",
    "market_buy_order",
    "market_sell_order",
    "order",
    "orders",
    "place_builder_order",
    "post_heartbeat",
    "post_only_order",
    "rfq_accept_quote",
    "rfq_approve_order",
    "rfq_cancel_quote",
    "rfq_cancel_request",
    "rfq_config",
    "rfq_create_quote",
    "rfq_create_request",
    "rfq_full_flow",
    "rfq_get_best_quote",
    "rfq_get_quotes",
    "rfq_get_requests",
    "update_balance_allowance",
]


class PolymarketCapabilityMixin:
    """
    Adapter that exposes the Polymarket py-clob-client example surface as reusable
    methods on this repo's live-test ExecutionClient.
    """

    def _import_clob_types(self):
        from py_clob_client.clob_types import (
            BookParams,
            OpenOrderParams,
            OrderArgs,
            OrderScoringParams,
            OrdersScoringParams,
            PartialCreateOrderOptions,
            PostOrdersArgs,
            TradeParams,
        )

        return {
            "BookParams": BookParams,
            "OpenOrderParams": OpenOrderParams,
            "OrderArgs": OrderArgs,
            "OrderScoringParams": OrderScoringParams,
            "OrdersScoringParams": OrdersScoringParams,
            "PartialCreateOrderOptions": PartialCreateOrderOptions,
            "PostOrdersArgs": PostOrdersArgs,
            "TradeParams": TradeParams,
        }

    def _import_rfq_types(self):
        from py_clob_client.rfq import (
            AcceptQuoteParams,
            ApproveOrderParams,
            CancelRfqQuoteParams,
            CancelRfqRequestParams,
            GetRfqBestQuoteParams,
            GetRfqQuotesParams,
            GetRfqRequestsParams,
            RfqUserQuote,
            RfqUserRequest,
        )

        return {
            "AcceptQuoteParams": AcceptQuoteParams,
            "ApproveOrderParams": ApproveOrderParams,
            "CancelRfqQuoteParams": CancelRfqQuoteParams,
            "CancelRfqRequestParams": CancelRfqRequestParams,
            "GetRfqBestQuoteParams": GetRfqBestQuoteParams,
            "GetRfqQuotesParams": GetRfqQuotesParams,
            "GetRfqRequestsParams": GetRfqRequestsParams,
            "RfqUserQuote": RfqUserQuote,
            "RfqUserRequest": RfqUserRequest,
        }

    def _coerce_side(self, side: str) -> str:
        return self.BUY if str(side).upper() == "BUY" else self.SELL

    def _coerce_order_type(self, order_type: str):
        return getattr(self.OrderType, str(order_type).upper())

    def _book_params(self, token_ids: Iterable[str], sides: Optional[Iterable[str]] = None):
        clob = self._import_clob_types()
        BookParams = clob["BookParams"]
        token_ids = list(token_ids)
        if sides is None:
            return [BookParams(token_id=str(token_id)) for token_id in token_ids]
        sides = list(sides)
        if len(sides) == 1 and len(token_ids) > 1:
            sides = sides * len(token_ids)
        return [
            BookParams(token_id=str(token_id), side=str(side).upper())
            for token_id, side in zip(token_ids, sides)
        ]

    def _open_order_params(self, id: Optional[str] = None, market: Optional[str] = None, asset_id: Optional[str] = None):
        clob = self._import_clob_types()
        OpenOrderParams = clob["OpenOrderParams"]
        if id is None and market is None and asset_id is None:
            return None
        return OpenOrderParams(id=id, market=market, asset_id=asset_id)

    def _trade_params(
        self,
        id: Optional[str] = None,
        maker_address: Optional[str] = None,
        market: Optional[str] = None,
        asset_id: Optional[str] = None,
        before: Optional[int] = None,
        after: Optional[int] = None,
    ):
        clob = self._import_clob_types()
        TradeParams = clob["TradeParams"]
        if all(value is None for value in [id, maker_address, market, asset_id, before, after]):
            return None
        return TradeParams(
            id=id,
            maker_address=maker_address,
            market=market,
            asset_id=asset_id,
            before=before,
            after=after,
        )

    def _normalize_creds(self):
        if getattr(self, "api_creds", None) is not None:
            return self.api_creds
        if hasattr(self, "_build_stored_creds"):
            creds = self._build_stored_creds()
            if creds is not None:
                return creds
        return None

    def _builder_client(self):
        from py_builder_signing_sdk.config import BuilderApiKeyCreds, BuilderConfig

        builder_key = os.getenv("BUILDER_API_KEY")
        builder_secret = os.getenv("BUILDER_SECRET")
        builder_passphrase = os.getenv("BUILDER_PASS_PHRASE")
        if not builder_key or not builder_secret or not builder_passphrase:
            raise ValueError(
                "Builder credentials are missing. Set BUILDER_API_KEY, BUILDER_SECRET, and BUILDER_PASS_PHRASE."
            )

        builder_config = BuilderConfig(
            local_builder_creds=BuilderApiKeyCreds(
                key=builder_key,
                secret=builder_secret,
                passphrase=builder_passphrase,
            )
        )

        creds = self._normalize_creds()
        if creds is None:
            raise ValueError("Level-2 API credentials are required for builder-authenticated requests.")

        return self.ClobClient(
            self.host,
            key=self.private_key,
            chain_id=self.chain_id,
            creds=creds,
            signature_type=self.signature_type,
            funder=self.funder,
            builder_config=builder_config,
        )

    def _extract_id(self, payload: Any, *candidate_keys: str):
        if isinstance(payload, dict):
            for key in candidate_keys:
                if payload.get(key):
                    return payload.get(key)
        return None

    def get_ok(self):
        return self.client.get_ok()

    def get_server_time(self):
        return self.client.get_server_time()

    def create_api_key(self, nonce: Optional[int] = None):
        return self.client.create_api_key(nonce=nonce)

    def derive_api_key(self, nonce: Optional[int] = None):
        return self.client.derive_api_key(nonce=nonce)

    def get_api_keys(self):
        return self.client.get_api_keys()

    def get_closed_only_mode(self):
        return self.client.get_closed_only_mode()

    def create_readonly_api_key(self):
        return self.client.create_readonly_api_key()

    def get_readonly_api_keys(self):
        return self.client.get_readonly_api_keys()

    def delete_readonly_api_key(self, key: str):
        return self.client.delete_readonly_api_key(key)

    def get_notifications(self):
        return self.client.get_notifications()

    def drop_notifications(self, ids: Optional[Iterable[str]] = None):
        if ids is None:
            return self.client.drop_notifications()
        from py_clob_client.clob_types import DropNotificationParams

        return self.client.drop_notifications(DropNotificationParams(ids=list(ids)))

    def get_last_trade_price(self, token_id: str):
        return self.client.get_last_trade_price(str(token_id))

    def get_last_trades_prices(self, token_ids: Iterable[str]):
        return self.client.get_last_trades_prices(self._book_params(token_ids))

    def get_market_trades_events(self, condition_id: str):
        return self.client.get_market_trades_events(str(condition_id))

    def get_markets(self, next_cursor: str = "MA==", collect_all: bool = False, max_pages: Optional[int] = None):
        if not collect_all:
            return self.client.get_markets(next_cursor=next_cursor)

        results = []
        cursor = next_cursor or "MA=="
        page_count = 0
        while cursor:
            page = self.client.get_markets(next_cursor=cursor)
            if isinstance(page, dict):
                results.extend(page.get("data", []))
                next_value = page.get("next_cursor")
            else:
                break
            page_count += 1
            if not next_value or next_value == "LTE=":
                break
            if max_pages is not None and page_count >= int(max_pages):
                break
            cursor = next_value

        return {"data": results, "next_cursor": cursor}

    def get_mid_market_price(self, token_id: str):
        return self.client.get_midpoint(str(token_id))

    def get_mid_markets_prices(self, token_ids: Iterable[str]):
        return self.client.get_midpoints(self._book_params(token_ids))

    def get_orderbook(self, token_id: str):
        return self.client.get_order_book(str(token_id))

    def get_orderbooks(self, token_ids: Iterable[str]):
        return self.client.get_order_books(self._book_params(token_ids))

    def get_price(self, token_id: str, side: str = "BUY"):
        return self.client.get_price(str(token_id), str(side).upper())

    def get_prices(self, token_ids: Iterable[str], side: str = "BUY", sides: Optional[Iterable[str]] = None):
        if sides is None:
            sides = [side]
        return self.client.get_prices(self._book_params(token_ids, sides=sides))

    def get_spread(self, token_id: str):
        return self.client.get_spread(str(token_id))

    def get_spreads(self, token_ids: Iterable[str]):
        return self.client.get_spreads(self._book_params(token_ids))

    def get_orders(self, id: Optional[str] = None, market: Optional[str] = None, asset_id: Optional[str] = None):
        return self.client.get_orders(self._open_order_params(id=id, market=market, asset_id=asset_id))

    def get_order(self, order_id: str):
        return self.client.get_order(order_id)

    def get_trades(
        self,
        id: Optional[str] = None,
        maker_address: Optional[str] = None,
        market: Optional[str] = None,
        asset_id: Optional[str] = None,
        before: Optional[int] = None,
        after: Optional[int] = None,
    ):
        return self.client.get_trades(
            self._trade_params(
                id=id,
                maker_address=maker_address,
                market=market,
                asset_id=asset_id,
                before=before,
                after=after,
            )
        )

    def get_builder_trades(
        self,
        id: Optional[str] = None,
        maker_address: Optional[str] = None,
        market: Optional[str] = None,
        asset_id: Optional[str] = None,
        before: Optional[int] = None,
        after: Optional[int] = None,
    ):
        builder_client = self._builder_client()
        return builder_client.get_builder_trades(
            self._trade_params(
                id=id,
                maker_address=maker_address,
                market=market,
                asset_id=asset_id,
                before=before,
                after=after,
            )
        )

    def get_open_orders_with_readonly_key(
        self,
        address: Optional[str] = None,
        readonly_api_key: Optional[str] = None,
        maker_address: Optional[str] = None,
    ):
        from py_clob_client.endpoints import ORDERS

        poly_address = (
            address
            or os.getenv("POLY_ADDRESS")
            or getattr(self, "funder", None)
            or (self.client.get_address() if hasattr(self.client, "get_address") else None)
        )
        if not poly_address:
            raise ValueError("Address is required. Provide address or set POLY_ADDRESS / POLYMARKET_FUNDER.")

        key = readonly_api_key or os.getenv("CLOB_READONLY_API_KEY")
        if not key:
            raise ValueError("Readonly API key is required. Provide readonly_api_key or set CLOB_READONLY_API_KEY.")

        response = requests.get(
            f"{self.host}{ORDERS}",
            headers={
                "POLY_READONLY_API_KEY": key,
                "POLY_ADDRESS": poly_address,
                "Content-Type": "application/json",
            },
            params={"maker_address": maker_address or poly_address},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def is_order_scoring(self, order_id: str):
        clob = self._import_clob_types()
        return self.client.is_order_scoring(clob["OrderScoringParams"](orderId=order_id))

    def are_orders_scoring(self, order_ids: Iterable[str]):
        clob = self._import_clob_types()
        return self.client.are_orders_scoring(clob["OrdersScoringParams"](orderIds=list(order_ids)))

    def _build_order_args(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str = "BUY",
        expiration: int = 0,
        fee_rate_bps: int = 0,
        nonce: int = 0,
        taker: Optional[str] = None,
    ):
        args_cls = self._import_clob_types()["OrderArgs"]
        kwargs = {
            "token_id": str(token_id),
            "price": float(price),
            "size": float(size),
            "side": self._coerce_side(side),
            "expiration": int(expiration or 0),
            "fee_rate_bps": int(fee_rate_bps or 0),
            "nonce": int(nonce or 0),
        }
        if taker is not None:
            kwargs["taker"] = taker
        return args_cls(**kwargs)

    def _build_create_options(self, tick_size: Optional[float] = None, neg_risk: Optional[bool] = None):
        if tick_size is None and neg_risk is None:
            return None
        clob = self._import_clob_types()
        return clob["PartialCreateOrderOptions"](tick_size=tick_size, neg_risk=neg_risk)

    def order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str = "BUY",
        order_type: str = "GTC",
        post_only: bool = False,
        expiration: int = 0,
        fee_rate_bps: int = 0,
        nonce: int = 0,
        taker: Optional[str] = None,
        tick_size: Optional[float] = None,
        neg_risk: Optional[bool] = None,
    ):
        args = self._build_order_args(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            expiration=expiration,
            fee_rate_bps=fee_rate_bps,
            nonce=nonce,
            taker=taker,
        )
        signed_order = self.client.create_order(args, options=self._build_create_options(tick_size=tick_size, neg_risk=neg_risk))
        return self.client.post_order(
            signed_order,
            self._coerce_order_type(order_type),
            post_only=bool(post_only),
        )

    def GTD_order(
        self,
        token_id: str,
        price: float,
        size: float,
        expiration: int,
        side: str = "BUY",
        post_only: bool = False,
    ):
        return self.order(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            order_type="GTD",
            post_only=post_only,
            expiration=expiration,
        )

    def gtd_order(self, *args, **kwargs):
        return self.GTD_order(*args, **kwargs)

    def post_only_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str = "BUY",
        order_type: str = "GTC",
        expiration: int = 0,
    ):
        return self.order(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            order_type=order_type,
            post_only=True,
            expiration=expiration,
        )

    def market_buy_order(self, token_id: str, amount: float, order_type: str = "FOK"):
        return self.create_and_post_market_order(token_id=token_id, amount=amount, side="BUY", order_type=order_type)

    def market_sell_order(self, token_id: str, amount: float, order_type: str = "FOK"):
        return self.create_and_post_market_order(token_id=token_id, amount=amount, side="SELL", order_type=order_type)

    def orders(self, order_specs: Iterable[dict]):
        clob = self._import_clob_types()
        PostOrdersArgs = clob["PostOrdersArgs"]
        posts = []
        for spec in order_specs:
            args = self._build_order_args(
                token_id=spec["token_id"],
                price=spec["price"],
                size=spec["size"],
                side=spec.get("side", "BUY"),
                expiration=spec.get("expiration", 0),
                fee_rate_bps=spec.get("fee_rate_bps", 0),
                nonce=spec.get("nonce", 0),
                taker=spec.get("taker"),
            )
            signed_order = self.client.create_order(
                args,
                options=self._build_create_options(
                    tick_size=spec.get("tick_size"),
                    neg_risk=spec.get("neg_risk"),
                ),
            )
            posts.append(
                PostOrdersArgs(
                    order=signed_order,
                    orderType=self._coerce_order_type(spec.get("order_type", "GTC")),
                    postOnly=bool(spec.get("post_only", False)),
                )
            )
        return self.client.post_orders(posts)

    def place_builder_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str = "BUY",
        order_type: str = "GTC",
        post_only: bool = False,
        expiration: int = 0,
    ):
        builder_client = self._builder_client()
        args = self._build_order_args(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            expiration=expiration,
        )
        signed_order = builder_client.create_order(args)
        return builder_client.post_order(
            signed_order,
            self._coerce_order_type(order_type),
            post_only=bool(post_only),
        )

    def cancel_orders(self, order_ids: Iterable[str]):
        return self.client.cancel_orders(list(order_ids))

    def cancel_market_orders(self, market: str = "", asset_id: str = ""):
        return self.client.cancel_market_orders(market=market, asset_id=asset_id)

    def cancel_all(self):
        return self.client.cancel_all()

    def post_heartbeat(self, heartbeat_id: Optional[str] = None):
        return self.client.post_heartbeat(heartbeat_id)

    def rfq_config(self):
        return self.client.rfq.rfq_config()

    def rfq_create_request(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str = "BUY",
        tick_size: Optional[float] = None,
    ):
        rfq = self._import_rfq_types()
        user_request = rfq["RfqUserRequest"](
            token_id=str(token_id),
            price=float(price),
            side=self._coerce_side(side),
            size=float(size),
        )
        return self.client.rfq.create_rfq_request(
            user_request,
            options=self._build_create_options(tick_size=tick_size),
        )

    def rfq_create_quote(
        self,
        request_id: str,
        token_id: str,
        price: float,
        size: float,
        side: str = "SELL",
        tick_size: Optional[float] = None,
    ):
        rfq = self._import_rfq_types()
        user_quote = rfq["RfqUserQuote"](
            request_id=str(request_id),
            token_id=str(token_id),
            price=float(price),
            side=self._coerce_side(side),
            size=float(size),
        )
        return self.client.rfq.create_rfq_quote(
            user_quote,
            options=self._build_create_options(tick_size=tick_size),
        )

    def rfq_get_requests(self, **filters):
        rfq = self._import_rfq_types()
        params = rfq["GetRfqRequestsParams"](**filters) if filters else None
        return self.client.rfq.get_rfq_requests(params)

    def rfq_get_quotes(self, **filters):
        rfq = self._import_rfq_types()
        params = rfq["GetRfqQuotesParams"](**filters) if filters else None
        return {
            "requester": self.client.rfq.get_rfq_requester_quotes(params),
            "quoter": self.client.rfq.get_rfq_quoter_quotes(params),
        }

    def rfq_get_best_quote(self, request_id: str):
        rfq = self._import_rfq_types()
        return self.client.rfq.get_rfq_best_quote(rfq["GetRfqBestQuoteParams"](request_id=str(request_id)))

    def rfq_cancel_request(self, request_id: str):
        rfq = self._import_rfq_types()
        return self.client.rfq.cancel_rfq_request(rfq["CancelRfqRequestParams"](request_id=str(request_id)))

    def rfq_cancel_quote(self, quote_id: str):
        rfq = self._import_rfq_types()
        return self.client.rfq.cancel_rfq_quote(rfq["CancelRfqQuoteParams"](quote_id=str(quote_id)))

    def rfq_accept_quote(self, request_id: str, quote_id: str, expiration: Optional[int] = None):
        rfq = self._import_rfq_types()
        expiration = int(expiration or (time.time() + 3600))
        return self.client.rfq.accept_rfq_quote(
            rfq["AcceptQuoteParams"](
                request_id=str(request_id),
                quote_id=str(quote_id),
                expiration=expiration,
            )
        )

    def rfq_approve_order(self, request_id: str, quote_id: str, expiration: Optional[int] = None):
        rfq = self._import_rfq_types()
        expiration = int(expiration or (time.time() + 3600))
        return self.client.rfq.approve_rfq_order(
            rfq["ApproveOrderParams"](
                request_id=str(request_id),
                quote_id=str(quote_id),
                expiration=expiration,
            )
        )

    def rfq_full_flow(
        self,
        token_id: str,
        request_price: float,
        request_size: float,
        request_side: str = "BUY",
        quote_price: Optional[float] = None,
        quote_size: Optional[float] = None,
        quote_side: Optional[str] = None,
        tick_size: Optional[float] = None,
        accept_best_quote: bool = False,
        approve_created_quote: bool = False,
        expiration: Optional[int] = None,
    ):
        results = {
            "request": self.rfq_create_request(
                token_id=token_id,
                price=request_price,
                size=request_size,
                side=request_side,
                tick_size=tick_size,
            )
        }

        request_id = self._extract_id(results["request"], "requestID", "requestId", "id")
        if request_id:
            results["best_quote"] = self.rfq_get_best_quote(request_id)

        if quote_price is not None and request_id:
            results["quote"] = self.rfq_create_quote(
                request_id=request_id,
                token_id=token_id,
                price=quote_price,
                size=quote_size if quote_size is not None else request_size,
                side=quote_side or ("SELL" if str(request_side).upper() == "BUY" else "BUY"),
                tick_size=tick_size,
            )
            quote_id = self._extract_id(results["quote"], "quoteID", "quoteId", "id")
            if approve_created_quote and quote_id:
                results["approved"] = self.rfq_approve_order(
                    request_id=request_id,
                    quote_id=quote_id,
                    expiration=expiration,
                )
            if accept_best_quote and quote_id:
                results["accepted"] = self.rfq_accept_quote(
                    request_id=request_id,
                    quote_id=quote_id,
                    expiration=expiration,
                )

        return results


def apply_execution_client_patch() -> None:
    try:
        import execution_client as execution_client_module
    except Exception:
        return

    target = getattr(execution_client_module, "ExecutionClient", None)
    if target is None:
        return

    setattr(target, "SUPPORTED_POLYMARKET_EXAMPLE_METHODS", SUPPORTED_POLYMARKET_EXAMPLE_METHODS)
    for name, value in PolymarketCapabilityMixin.__dict__.items():
        if name.startswith("__"):
            continue
        if callable(value):
            setattr(target, name, value)
