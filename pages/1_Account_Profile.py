import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

try:
    from execution_client import ExecutionClient
except Exception:
    ExecutionClient = None

try:
    from polymarket_profile_client import PolymarketProfileClient
except Exception:
    PolymarketProfileClient = None


st.set_page_config(page_title="Account Profile", page_icon="🪪", layout="wide")


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


def _get_candidate_address() -> Optional[str]:
    for key in ["POLYMARKET_PUBLIC_ADDRESS", "POLYMARKET_FUNDER"]:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _read_live_client_state() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "live_mode": os.getenv("TRADING_MODE", "paper").strip().lower() == "live",
        "client_ok": False,
        "client_error": None,
        "funder": os.getenv("POLYMARKET_FUNDER", "").strip() or None,
        "collateral_balance": None,
        "collateral_allowance": None,
        "conditional_allowance": None,
    }
    if ExecutionClient is None or not result["live_mode"]:
        return result
    try:
        client = ExecutionClient()
        result["client_ok"] = True
        result["funder"] = getattr(client, "funder", None) or result["funder"]
        collat = client.get_balance_allowance(asset_type="COLLATERAL")
        cond = client.get_balance_allowance(asset_type="CONDITIONAL")
        if isinstance(collat, dict):
            result["collateral_balance"] = _safe_float(collat.get("balance", collat.get("amount")))
            result["collateral_allowance"] = _safe_float(collat.get("allowance"))
        if isinstance(cond, dict):
            result["conditional_allowance"] = _safe_float(cond.get("allowance"))
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
            status_bits.append("✅ live trading client connected")
        elif live_state.get("client_error"):
            status_bits.append("❌ live trading client failed")
        else:
            status_bits.append("⚪ live mode active")
    else:
        status_bits.append("ℹ️ paper mode")

    if address:
        status_bits.append("✅ wallet address available")
    else:
        status_bits.append("❌ wallet address missing")

    if bundle is not None:
        if bundle.get("profile"):
            status_bits.append("✅ public profile fetched")
        else:
            status_bits.append("⚠️ public profile not found")
        if bundle.get("positions"):
            status_bits.append("✅ positions fetched")
        if bundle.get("activity"):
            status_bits.append("✅ recent activity fetched")

    st.success(" | ".join(status_bits))


def main() -> None:
    st.title("🪪 Polymarket Account Profile")
    st.caption("Use this page to visually confirm that your wallet/profile data is reachable after setup and auth.")

    live_state = _read_live_client_state()
    default_address = live_state.get("funder") or _get_candidate_address() or ""

    address = st.text_input(
        "Wallet / profile address",
        value=default_address,
        help="Uses POLYMARKET_PUBLIC_ADDRESS first, then POLYMARKET_FUNDER, then the live client funder when available.",
    ).strip()

    bundle = None
    if address:
        bundle = _fetch_profile_bundle(address)

    _render_top_status(live_state, address, bundle)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trading mode", "live" if live_state.get("live_mode") else "paper")
    c2.metric("Wallet address", address if address else "missing")
    c3.metric("Collateral balance", f"${live_state['collateral_balance']:.2f}" if live_state.get("collateral_balance") is not None else "N/A")
    c4.metric("Collateral allowance", f"{live_state['collateral_allowance']:.2f}" if live_state.get("collateral_allowance") is not None else "N/A")

    if live_state.get("conditional_allowance") is not None:
        st.caption(f"Conditional allowance: {live_state['conditional_allowance']:.2f}")
    if live_state.get("client_error"):
        st.error(f"Live client error: {live_state['client_error']}")

    if not address:
        st.warning("No address available yet. Set POLYMARKET_PUBLIC_ADDRESS or POLYMARKET_FUNDER, then refresh this page.")
        return

    if bundle is None:
        st.info("No profile request has been made yet.")
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
        verified = bool(profile.get("verifiedBadge"))
        x_username = profile.get("xUsername")
        bio = profile.get("bio")
        proxy_wallet = profile.get("proxyWallet")
        profile_rows = [
            {"field": "Display name", "value": name},
            {"field": "Verified", "value": "Yes" if verified else "No"},
            {"field": "Proxy wallet", "value": proxy_wallet or "N/A"},
            {"field": "X username", "value": x_username or "N/A"},
            {"field": "Bio", "value": bio or "N/A"},
            {"field": "Total value", "value": bundle.get("value") if bundle.get("value") is not None else "N/A"},
            {"field": "Markets traded", "value": bundle.get("markets_traded") if bundle.get("markets_traded") is not None else "N/A"},
            {"field": "Open positions count", "value": len(positions)},
            {"field": "Recent activity count", "value": len(activity)},
        ]
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Confirmation")
        if profile:
            st.success("Public profile endpoint returned data for this address.")
        else:
            st.warning("No public profile payload was returned for this address.")
        if positions:
            st.success("Positions endpoint returned data.")
        else:
            st.info("No open positions found, or positions endpoint returned no rows.")
        if activity:
            st.success("Activity endpoint returned recent rows.")
        else:
            st.info("No recent activity rows returned.")
        if bundle.get("errors"):
            st.warning("Some endpoint calls failed:")
            for item in bundle["errors"]:
                st.caption(item)

    if positions:
        st.subheader("Open positions")
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
