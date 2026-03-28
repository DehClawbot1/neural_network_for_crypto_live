"""
pages/1_Account_Profile.py — FIXED

FIXES:
  1. Uses safe_load_dotenv() instead of load_dotenv()
  2. Uses cached ExecutionClient from dashboard_auth
  3. Works in paper mode too (shows available log data)
  4. No longer crashes when live credentials are missing
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ── FIX 1: Use shared auth utility ──
from dashboard_auth import (
    safe_load_dotenv,
    get_trading_mode,
    is_live_mode,
    is_interactive_mode,
    get_execution_client_cached,
    get_wallet_address,
    get_balance_info,
)

safe_load_dotenv()

try:
    from polymarket_profile_client import PolymarketProfileClient
except Exception:
    PolymarketProfileClient = None

st.set_page_config(page_title="Account Profile", page_icon="N", layout="wide")


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _first_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    return {}


def _profile_name(profile: Dict[str, Any]) -> str:
    for key in ["name", "pseudonym", "xUsername", "proxyWallet"]:
        value = profile.get(key)
        if value:
            return str(value)
    return "Unknown profile"


def _load_local_live_activity() -> Dict[str, pd.DataFrame]:
    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / "logs"
    orders_path = logs_dir / "live_orders.csv"
    fills_path = logs_dir / "live_fills.csv"

    def safe_read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    return {
        "orders": safe_read(orders_path),
        "fills": safe_read(fills_path),
    }


def _latest_tradability_snapshot(orders_df: pd.DataFrame) -> Dict[str, Any]:
    if orders_df is None or orders_df.empty:
        return {"tradable_status": "N/A", "tradable_reason": None}
    df = orders_df.copy()
    if "timestamp" in df.columns:
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values(by="_ts")
    row = df.tail(1).to_dict(orient="records")[0]
    tradable = row.get("tradable")
    if pd.isna(tradable):
        tradable = None
    return {
        "tradable_status": "YES" if tradable is True else "NO" if tradable is False else "N/A",
        "tradable_reason": row.get("reason") or row.get("orderbook_error"),
        "bid_levels": row.get("bid_levels"),
        "ask_levels": row.get("ask_levels"),
        "quoted_price": row.get("quoted_price"),
        "quoted_spread": row.get("quoted_spread"),
    }


# ── FIX 2: Rewritten to use cached client ──
def _read_live_client_state(local_activity: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "live_mode": is_live_mode(),
        "client_ok": False,
        "exchange_ok": False,
        "client_error": None,
        "server_time": None,
        "funder": get_wallet_address() or None,
        "address_source": "dashboard_auth",
        "balance_source": "none",
        "collateral_balance": None,
        "collateral_allowance": None,
        "onchain_wallet_balance": None,
        "balance_mismatch": False,
    }

    if local_activity is not None:
        result.update(_latest_tradability_snapshot(local_activity.get("orders", pd.DataFrame())))

    if not result["live_mode"]:
        return result

    # ── FIX: Use cached client ──
    client = get_execution_client_cached()
    if client is None:
        result["client_error"] = "ExecutionClient not available"
        return result

    try:
        result["funder"] = getattr(client, "funder", None) or result["funder"]
        result["address_source"] = "client_funder" if getattr(client, "funder", None) else "env_fallback"

        raw_client = getattr(client, "client", None)
        if raw_client is not None and hasattr(raw_client, "get_ok"):
            try:
                result["exchange_ok"] = bool(raw_client.get_ok())
            except Exception:
                pass
        if raw_client is not None and hasattr(raw_client, "get_server_time"):
            try:
                result["server_time"] = raw_client.get_server_time()
            except Exception:
                pass

        balance_info = get_balance_info()
        result["client_ok"] = balance_info.get("error") is None
        result["collateral_balance"] = balance_info["clob_balance"]
        result["onchain_wallet_balance"] = balance_info["onchain_balance"]
        result["balance_source"] = f"api + onchain ({balance_info['source']})"
        result["balance_mismatch"] = balance_info["onchain_balance"] > 0 and balance_info["clob_balance"] <= 0

    except Exception as exc:
        result["client_error"] = str(exc)

    return result


@st.cache_data(show_spinner=False, ttl=30)
def _fetch_profile_bundle(address: str) -> Dict[str, Any]:
    if PolymarketProfileClient is None:
        return {"error": "PolymarketProfileClient is not available."}

    client = PolymarketProfileClient()
    bundle: Dict[str, Any] = {
        "address": address,
        "profile": {},
        "value": None,
        "markets_traded": None,
        "positions": [],
        "activity": [],
        "errors": [],
    }

    try:
        bundle["profile"] = _first_dict(client.get_public_profile(address))
    except Exception as exc:
        bundle["errors"].append(f"public_profile: {exc}")

    try:
        value_payload = client.get_total_value(address)
        value_row = _first_dict(value_payload)
        bundle["value"] = _safe_float(value_row.get("value"))
    except Exception as exc:
        bundle["errors"].append(f"total_value: {exc}")

    try:
        traded_payload = client.get_total_markets_traded(address)
        if isinstance(traded_payload, list) and traded_payload:
            traded_payload = traded_payload[0]
        if isinstance(traded_payload, dict):
            bundle["markets_traded"] = traded_payload.get("traded") or traded_payload.get("count") or traded_payload.get("total")
        else:
            bundle["markets_traded"] = traded_payload
    except Exception as exc:
        bundle["errors"].append(f"total_markets_traded: {exc}")

    try:
        positions_payload = client.get_positions(address, limit=20)
        bundle["positions"] = positions_payload if isinstance(positions_payload, list) else []
    except Exception as exc:
        bundle["errors"].append(f"positions: {exc}")

    try:
        activity_payload = client.get_activity(address, limit=20)
        bundle["activity"] = activity_payload if isinstance(activity_payload, list) else []
    except Exception as exc:
        bundle["errors"].append(f"activity: {exc}")

    return bundle


def _render_top_status(live_state: Dict[str, Any], address: Optional[str], bundle: Optional[Dict[str, Any]]) -> None:
    status_bits: List[str] = []
    if live_state.get("live_mode"):
        if live_state.get("client_ok"):
            status_bits.append("[OK] live client connected")
        elif live_state.get("client_error"):
            status_bits.append("[FAIL] live client failed")
        else:
            status_bits.append("[--] live mode active")
    else:
        status_bits.append("[i] paper mode")

    if is_interactive_mode():
        status_bits.append("[i] interactive auth")

    if live_state.get("exchange_ok"):
        status_bits.append("[OK] exchange reachable")

    if address:
        status_bits.append(f"[OK] wallet: {address[:10]}...")
    else:
        status_bits.append("[FAIL] wallet address missing")

    if bundle is not None:
        if bundle.get("profile"):
            status_bits.append("[OK] public profile fetched")
        if bundle.get("positions"):
            status_bits.append(f"[OK] {len(bundle['positions'])} positions")

    if live_state.get("client_ok"):
        st.success(" | ".join(status_bits))
    else:
        st.warning(" | ".join(status_bits))


def main() -> None:
    st.title("Polymarket Account Profile")
    st.caption("Live truth view: real client connectivity, API-fetched collateral balance, address source, and latest local tradability signal.")

    local_activity = _load_local_live_activity()
    live_state = _read_live_client_state(local_activity=local_activity)
    default_address = live_state.get("funder") or ""

    # ── FIX 8: Don't block paper mode entirely — show what we can ──
    if not live_state.get("live_mode"):
        st.warning("Live mode is not active. Some features require TRADING_MODE=live.")
        st.caption("Showing available local data and public profile lookup.")

    address = st.text_input(
        "Wallet / profile address",
        value=default_address,
        help="Enter your Polymarket wallet address to view public profile data.",
    ).strip()

    bundle = None
    if address:
        bundle = _fetch_profile_bundle(address)

    _render_top_status(live_state, address, bundle)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trading mode", get_trading_mode())
    c2.metric("Client connected", "YES" if live_state.get("client_ok") else "NO")
    c3.metric("Exchange reachable", "YES" if live_state.get("exchange_ok") else "N/A")
    c4.metric("Balance source", live_state.get("balance_source") or "none")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Wallet address", address[:16] + "..." if len(address) > 16 else (address or "missing"))
    c6.metric("Address source", live_state.get("address_source") or "missing")
    c7.metric("CLOB/API Balance", f"${live_state['collateral_balance']:.2f}" if live_state.get("collateral_balance") is not None else "N/A")
    c8.metric("On-chain USDC", f"${live_state['onchain_wallet_balance']:.2f}" if live_state.get("onchain_wallet_balance") is not None else "N/A")

    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Local live orders", len(local_activity.get("orders", pd.DataFrame())))
    lc2.metric("Local live fills", len(local_activity.get("fills", pd.DataFrame())))
    lc3.metric("Latest tradability", live_state.get("tradable_status") or "N/A")
    lc4.metric("Auth mode", "interactive" if is_interactive_mode() else ".env file" if os.getenv("PRIVATE_KEY") else "none")

    if live_state.get("balance_mismatch"):
        st.warning("On-chain USDC is present, but Available to Trade from the CLOB/API is still zero.")
    if live_state.get("server_time") is not None:
        st.caption(f"Server time: {live_state['server_time']}")
    if live_state.get("client_error"):
        st.error(f"Live client error: {live_state['client_error']}")

    if not address:
        st.info("No address available. Set POLYMARKET_PUBLIC_ADDRESS or POLYMARKET_FUNDER, then refresh.")
        return

    if bundle is None:
        st.info("Enter an address and press Enter to fetch profile data.")
        return

    if bundle.get("error"):
        st.error(bundle["error"])
        return

    profile = bundle.get("profile") or {}
    positions = bundle.get("positions") or []
    activity = bundle.get("activity") or []

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Profile summary")
        name = _profile_name(profile)
        profile_rows = [
            {"field": "Display name", "value": name},
            {"field": "Verified", "value": "Yes" if profile.get("verifiedBadge") else "No"},
            {"field": "Proxy wallet", "value": profile.get("proxyWallet") or "N/A"},
            {"field": "X username", "value": profile.get("xUsername") or "N/A"},
            {"field": "Bio", "value": profile.get("bio") or "N/A"},
            {"field": "Total value", "value": bundle.get("value") if bundle.get("value") is not None else "N/A"},
            {"field": "Markets traded", "value": bundle.get("markets_traded") if bundle.get("markets_traded") is not None else "N/A"},
            {"field": "Open positions", "value": len(positions)},
            {"field": "Recent activity", "value": len(activity)},
        ]
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Confirmation")
        if profile:
            st.success("Public profile endpoint returned data.")
        else:
            st.warning("No public profile payload returned.")
        if positions:
            st.success("Positions Data API returned data.")
        else:
            st.info("No open positions found.")
        if activity:
            st.success("Activity endpoint returned rows.")
        else:
            st.info("No recent activity rows.")
        if bundle.get("errors"):
            st.warning("Some endpoint calls failed:")
            for item in bundle["errors"]:
                st.caption(item)

    if positions:
        st.subheader("Open positions (Data API)")
        pos_df = pd.DataFrame(positions)
        keep_cols = [c for c in ["title", "outcome", "size", "avgPrice", "curPrice", "currentValue", "cashPnl", "percentPnl", "realizedPnl", "endDate"] if c in pos_df.columns]
        st.dataframe(pos_df[keep_cols] if keep_cols else pos_df, use_container_width=True, hide_index=True)

    if activity:
        st.subheader("Recent activity")
        act_df = pd.DataFrame(activity)
        keep_cols = [c for c in ["type", "title", "side", "size", "price", "timestamp", "outcome"] if c in act_df.columns]
        st.dataframe(act_df[keep_cols] if keep_cols else act_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
