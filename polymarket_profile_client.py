from typing import Any, Dict, Iterable, Optional

import requests


def prompt_polymarket_runtime() -> Dict[str, str]:
    from getpass import getpass

    trading_mode = input("TRADING_MODE [paper/live]: ").strip().lower() or "paper"
    wallet = input("POLYMARKET_WALLET_ADDRESS: ").strip()
    cfg: Dict[str, str] = {
        "trading_mode": trading_mode,
        "wallet": wallet,
    }
    if trading_mode == "live":
        cfg["private_key"] = getpass("PRIVATE_KEY: ").strip()
        cfg["funder"] = input("POLYMARKET_FUNDER: ").strip()
        cfg["api_key"] = input("POLYMARKET_API_KEY (optional): ").strip()
        cfg["api_secret"] = getpass("POLYMARKET_API_SECRET (optional): ").strip()
        cfg["api_passphrase"] = getpass("POLYMARKET_API_PASSPHRASE (optional): ").strip()
    return cfg


class PolymarketProfileClient:
    GAMMA_BASE = "https://gamma-api.polymarket.com"
    DATA_BASE = "https://data-api.polymarket.com"

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout
        self.session = requests.Session()

    def _clean_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        if not params:
            return clean
        for key, value in params.items():
            if value is None or value == "":
                continue
            if isinstance(value, (list, tuple, set)):
                clean[key] = ",".join(str(v) for v in value)
            elif isinstance(value, bool):
                clean[key] = str(value).lower()
            else:
                clean[key] = value
        return clean

    def _get(self, base: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        response = self.session.get(
            f"{base}{path}",
            params=self._clean_params(params),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_public_profile(self, address: str) -> Any:
        return self._get(self.GAMMA_BASE, "/public-profile", {"address": address})

    def get_positions(
        self,
        user: str,
        market: Optional[Iterable[str]] = None,
        event_id: Optional[Iterable[int]] = None,
        size_threshold: float = 1,
        redeemable: bool = False,
        mergeable: bool = False,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "TOKENS",
        sort_direction: str = "DESC",
        title: Optional[str] = None,
    ) -> Any:
        return self._get(
            self.DATA_BASE,
            "/positions",
            {
                "user": user,
                "market": market,
                "eventId": event_id,
                "sizeThreshold": size_threshold,
                "redeemable": redeemable,
                "mergeable": mergeable,
                "limit": limit,
                "offset": offset,
                "sortBy": sort_by,
                "sortDirection": sort_direction,
                "title": title,
            },
        )

    def get_closed_positions(
        self,
        user: str,
        market: Optional[Iterable[str]] = None,
        event_id: Optional[Iterable[int]] = None,
        title: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "REALIZEDPNL",
        sort_direction: str = "DESC",
    ) -> Any:
        return self._get(
            self.DATA_BASE,
            "/closed-positions",
            {
                "user": user,
                "market": market,
                "eventId": event_id,
                "title": title,
                "limit": limit,
                "offset": offset,
                "sortBy": sort_by,
                "sortDirection": sort_direction,
            },
        )

    def get_activity(
        self,
        user: str,
        market: Optional[Iterable[str]] = None,
        event_id: Optional[Iterable[int]] = None,
        activity_type: Optional[Iterable[str]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        side: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "TIMESTAMP",
        sort_direction: str = "DESC",
    ) -> Any:
        return self._get(
            self.DATA_BASE,
            "/activity",
            {
                "user": user,
                "market": market,
                "eventId": event_id,
                "type": activity_type,
                "start": start,
                "end": end,
                "side": side,
                "limit": limit,
                "offset": offset,
                "sortBy": sort_by,
                "sortDirection": sort_direction,
            },
        )

    def get_trades(
        self,
        user: Optional[str] = None,
        market: Optional[Iterable[str]] = None,
        event_id: Optional[Iterable[int]] = None,
        side: Optional[str] = None,
        taker_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        return self._get(
            self.DATA_BASE,
            "/trades",
            {
                "user": user,
                "market": market,
                "eventId": event_id,
                "side": side,
                "takerOnly": taker_only,
                "limit": limit,
                "offset": offset,
            },
        )

    def get_total_value(self, user: str, market: Optional[Iterable[str]] = None) -> Any:
        return self._get(self.DATA_BASE, "/value", {"user": user, "market": market})

    def get_value(self, user: str, market: Optional[Iterable[str]] = None) -> Any:
        return self.get_total_value(user=user, market=market)

    def get_total_markets_traded(self, user: str) -> Any:
        return self._get(self.DATA_BASE, "/traded", {"user": user})

    def get_traded(self, user: str) -> Any:
        return self.get_total_markets_traded(user=user)

    def get_market_positions(
        self,
        market: str,
        user: Optional[str] = None,
        status: str = "ALL",
        sort_by: str = "TOTAL_PNL",
        sort_direction: str = "DESC",
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        return self._get(
            self.DATA_BASE,
            "/v1/market-positions",
            {
                "market": market,
                "user": user,
                "status": status,
                "sortBy": sort_by,
                "sortDirection": sort_direction,
                "limit": limit,
                "offset": offset,
            },
        )

