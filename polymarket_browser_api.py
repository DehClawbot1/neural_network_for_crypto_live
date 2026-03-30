from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from polymarket_profile_client import PolymarketProfileClient

app = FastAPI(title="Polymarket Browser API", version="1.0.0")
client = PolymarketProfileClient()


def _raise_http_error(exc: Exception) -> None:
    raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/")
def root() -> dict:
    return {
        "name": "Polymarket Browser API",
        "mode": "public profile and data browser",
        "endpoints": [
            "/health",
            "/profile",
            "/positions",
            "/closed-positions",
            "/activity",
            "/trades",
            "/value",
            "/traded",
            "/market-positions",
            "/summary",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/profile")
def profile(address: str = Query(..., description="Wallet address")):
    try:
        return client.get_public_profile(address)
    except Exception as exc:
        _raise_http_error(exc)


@app.get("/positions")
def positions(
    user: str = Query(..., description="Wallet address"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0, le=10000),
    title: Optional[str] = None,
):
    try:
        return client.get_positions(user=user, limit=limit, offset=offset, title=title)
    except Exception as exc:
        _raise_http_error(exc)


@app.get("/closed-positions")
def closed_positions(
    user: str = Query(..., description="Wallet address"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0, le=10000),
):
    try:
        return client.get_closed_positions(user=user, limit=limit, offset=offset)
    except Exception as exc:
        _raise_http_error(exc)


@app.get("/activity")
def activity(
    user: str = Query(..., description="Wallet address"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0, le=10000),
):
    try:
        return client.get_activity(user=user, limit=limit, offset=offset)
    except Exception as exc:
        _raise_http_error(exc)


@app.get("/trades")
def trades(
    user: Optional[str] = Query(None, description="Wallet address"),
    market: Optional[str] = Query(None, description="Condition ID"),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0, le=10000),
):
    try:
        return client.get_trades(user=user, market=market, limit=limit, offset=offset)
    except Exception as exc:
        _raise_http_error(exc)


@app.get("/value")
def value(user: str = Query(..., description="Wallet address")):
    try:
        return client.get_value(user)
    except Exception as exc:
        _raise_http_error(exc)


@app.get("/traded")
def traded(user: str = Query(..., description="Wallet address")):
    try:
        return client.get_traded(user)
    except Exception as exc:
        _raise_http_error(exc)


@app.get("/market-positions")
def market_positions(
    market: str = Query(..., description="Condition ID"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0, le=10000),
):
    try:
        return client.get_market_positions(market=market, limit=limit, offset=offset)
    except Exception as exc:
        _raise_http_error(exc)


@app.get("/summary")
def summary(user: str = Query(..., description="Wallet address")):
    try:
        profile_data = client.get_public_profile(user)
        positions_data = client.get_positions(user=user, limit=100, offset=0)
        closed_positions_data = client.get_closed_positions(user=user, limit=100, offset=0)
        activity_data = client.get_activity(user=user, limit=100, offset=0)
        trades_data = client.get_trades(user=user, limit=100, offset=0)
        value_data = client.get_value(user)
        traded_data = client.get_traded(user)
        return {
            "profile": profile_data,
            "positions": positions_data,
            "closed_positions": closed_positions_data,
            "activity": activity_data,
            "trades": trades_data,
            "value": value_data,
            "traded": traded_data,
        }
    except Exception as exc:
        _raise_http_error(exc)

