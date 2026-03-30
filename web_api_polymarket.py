from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from repository_polymarket_service import (
    call_polymarket_capability,
    get_execution_client,
    list_polymarket_capabilities,
)

router = APIRouter(prefix="/polymarket", tags=["polymarket"])


class CapabilityRequest(BaseModel):
    capability: str = Field(..., description="Capability name from the Polymarket adapter surface")
    kwargs: dict[str, Any] = Field(default_factory=dict)
    force_new_client: bool = False


class ScoreCheckRequest(BaseModel):
    order_ids: list[str] = Field(default_factory=list)


class OrderRequest(BaseModel):
    token_id: str
    price: float
    size: float
    side: str = "BUY"
    order_type: str = "GTC"
    expiration: int = 0


class CancelMarketOrdersRequest(BaseModel):
    market: str = ""
    asset_id: str = ""


class MarketOrderRequest(BaseModel):
    token_id: str
    amount: float
    order_type: str = "FOK"


class CancelOrdersRequest(BaseModel):
    order_ids: list[str] = Field(default_factory=list)


class BatchOrdersRequest(BaseModel):
    order_specs: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/capabilities")
def get_capabilities():
    return {"capabilities": list_polymarket_capabilities()}


@router.get("/status")
def get_status():
    try:
        client = get_execution_client()
        return {
            "ok": True,
            "host": getattr(client, "host", None),
            "chain_id": getattr(client, "chain_id", None),
            "credential_source": getattr(client, "credential_source", None),
            "supports": list_polymarket_capabilities(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "supports": list_polymarket_capabilities()}


@router.get("/health")
def get_health():
    try:
        client = get_execution_client()
        return {
            "ok": True,
            "exchange_ok": client.get_ok(),
            "server_time": client.get_server_time(),
            "credential_source": getattr(client, "credential_source", None),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/call")
def call_capability(request: CapabilityRequest):
    try:
        if request.force_new_client:
            get_execution_client(force_new=True)
        result = call_polymarket_capability(request.capability, **request.kwargs)
        return {"ok": True, "capability": request.capability, "result": result}
    except AttributeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/balance")
def get_collateral_balance(token_id: Optional[str] = None):
    try:
        client = get_execution_client()
        asset_type = "CONDITIONAL" if token_id else "COLLATERAL"
        balance = client.get_balance_allowance(asset_type=asset_type, token_id=token_id)
        return {"ok": True, "asset_type": asset_type, "token_id": token_id, "result": balance}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/balance/refresh")
def refresh_collateral_balance(token_id: Optional[str] = None):
    try:
        client = get_execution_client()
        asset_type = "CONDITIONAL" if token_id else "COLLATERAL"
        refresh = client.update_balance_allowance(asset_type=asset_type, token_id=token_id)
        balance = client.get_balance_allowance(asset_type=asset_type, token_id=token_id)
        return {
            "ok": True,
            "asset_type": asset_type,
            "token_id": token_id,
            "refresh_result": refresh,
            "balance": balance,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/markets/current")
def get_current_markets(next_cursor: str = "MA=="):
    try:
        client = get_execution_client()
        result = client.get_markets(next_cursor=next_cursor)
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/orderbook/{token_id}")
def get_orderbook(token_id: str):
    try:
        client = get_execution_client()
        result = client.get_orderbook(token_id)
        return {"ok": True, "token_id": token_id, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/price/{token_id}")
def get_price(token_id: str, side: str = "BUY"):
    try:
        client = get_execution_client()
        result = client.get_price(token_id, side=side)
        return {"ok": True, "token_id": token_id, "side": side, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/spread/{token_id}")
def get_spread(token_id: str):
    try:
        client = get_execution_client()
        result = client.get_spread(token_id)
        return {"ok": True, "token_id": token_id, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/orders/open")
def get_open_orders(market: Optional[str] = None, asset_id: Optional[str] = None):
    try:
        client = get_execution_client()
        result = client.get_orders(market=market, asset_id=asset_id)
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/order/{order_id}")
def get_order(order_id: str):
    try:
        client = get_execution_client()
        result = client.get_order(order_id)
        return {"ok": True, "order_id": order_id, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/order/{order_id}/cancel")
def cancel_order(order_id: str):
    try:
        client = get_execution_client()
        result = client.cancel_order(order_id)
        return {"ok": True, "order_id": order_id, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/orders/cancel-all")
def cancel_all_orders():
    try:
        client = get_execution_client()
        result = client.cancel_all()
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/orders/cancel-batch")
def cancel_batch_orders(request: CancelOrdersRequest):
    try:
        client = get_execution_client()
        result = client.cancel_orders(request.order_ids)
        return {"ok": True, "order_ids": request.order_ids, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/orders/cancel-market")
def cancel_orders_for_market(request: CancelMarketOrdersRequest):
    try:
        client = get_execution_client()
        result = client.cancel_market_orders(market=request.market, asset_id=request.asset_id)
        return {"ok": True, "market": request.market, "asset_id": request.asset_id, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/order/post-only")
def post_only_order(request: OrderRequest):
    try:
        client = get_execution_client()
        result = client.post_only_order(
            token_id=request.token_id,
            price=request.price,
            size=request.size,
            side=request.side,
            order_type=request.order_type,
            expiration=request.expiration,
        )
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/order/gtd")
def gtd_order(request: OrderRequest):
    try:
        if request.expiration <= 0:
            raise HTTPException(status_code=400, detail="expiration must be a positive unix timestamp for GTD orders")
        client = get_execution_client()
        result = client.GTD_order(
            token_id=request.token_id,
            price=request.price,
            size=request.size,
            expiration=request.expiration,
            side=request.side,
        )
        return {"ok": True, "result": result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/order/market-buy")
def market_buy_order(request: MarketOrderRequest):
    try:
        client = get_execution_client()
        result = client.market_buy_order(token_id=request.token_id, amount=request.amount, order_type=request.order_type)
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/order/market-sell")
def market_sell_order(request: MarketOrderRequest):
    try:
        client = get_execution_client()
        result = client.market_sell_order(token_id=request.token_id, amount=request.amount, order_type=request.order_type)
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/orders/batch")
def batch_orders(request: BatchOrdersRequest):
    try:
        client = get_execution_client()
        result = client.orders(request.order_specs)
        return {"ok": True, "count": len(request.order_specs), "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/order-scoring/{order_id}")
def get_order_scoring(order_id: str):
    try:
        client = get_execution_client()
        result = client.is_order_scoring(order_id)
        return {"ok": True, "order_id": order_id, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/order-scoring/batch")
def get_orders_scoring(request: ScoreCheckRequest):
    try:
        client = get_execution_client()
        result = client.are_orders_scoring(request.order_ids)
        return {"ok": True, "order_ids": request.order_ids, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
