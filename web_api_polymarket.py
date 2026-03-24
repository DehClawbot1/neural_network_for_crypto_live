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


@router.get("/orders/open")
def get_open_orders(market: Optional[str] = None, asset_id: Optional[str] = None):
    try:
        client = get_execution_client()
        result = client.get_orders(market=market, asset_id=asset_id)
        return {"ok": True, "result": result}
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
