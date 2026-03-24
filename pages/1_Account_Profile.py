import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from execution_client import ExecutionClient
except Exception:
    ExecutionClient = None

try:
    from polymarket_profile_client import PolymarketProfileClient
except Exception:
    PolymarketProfileClient = None


load_dotenv()

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


def _derive_address_from_private_key() -> Optional[str]:
    private_key = os.getenv("PRIVATE_KEY", "").strip()
    if not private_key:
        return None
    try:
        from eth_account import Account
        acct = Account.from_key(private_key)
        return str(acct.address)
    except Exception:
        return None


def _get_candidate_address_with_source() -> tuple[Optional[str], str]:
    public_address = os.getenv("POLYMARKET_PUBLIC_ADDRESS", "").strip()
    if public_address:
        return public_address, "env:POLYMARKET_PUBLIC_ADDRESS"
    funder = os.getenv("POLYMARKET_FUNDER", "").strip()
    if funder:
        return funder, "env:POLYMARKET_FUNDER"
    derived = _derive_address_from_private_key()
    if derived:
        return derived, "derived_from_private_key"
    return None, "missing"


def _read_live_client_state() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "live_mode": os.getenv("TRADING_MODE", "paper").strip().lower() == "live",
        "client_ok": False,
        "client_error": None,
        "funder": os.getenv("POLYMARKET_FUNDER", "").strip() or None,
        "address_source": "missing",
        "collateral_balance": None,
        "collateral_allowance": None,
        "conditional_allowance": None,
    }
    fallback_address, fallback_source = _get_candidate_address_with_source()
    result["address_source"] = fallback_source
    if ExecutionClient is None or not result["live_mode"]:
        result["funder"] = result["funder"] or fallback_address
        return result
    try:
        client = ExecutionClient()
        client_funder = getattr(client, "funder", None)
        result["funder"] = client_funder or result["funder"] or fallback_address
        result["address_source"] = "client_funder" if client_funder else fallback_source
        collat = client.get_balance_allowance(asset_type="COLLATERAL")
        result["client_ok"] = True
        if isinstance(collat, dict):
            result["collateral_balance"] = _safe_float(collat.get("balance", collat.get("amount")))
            result["collateral_allowance"] = _safe_float(collat.get("allowance"))
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


def _render_top_status(live_state: Dict[str, Any], address: Optional[str], bundle: Optional[Dict[str, Any]]) -> None:
    status_bits: List[str] = []
    if live_state.get("live_mode"):
        if live_state.get("client_ok"):
            status_bits.append("✅ live client connected (real API call succeeded)")
        elif live_state.get("client_error"):
            status_bits.append("❌ live client failed")
        else:
            status_bits.append("⚪ live mode active")
    else:
        status_bits.append("ℹ️ paper mode")

    address_source = live_state.get("address_source", "missing")
    if address:
        if address_source == "client_funder":
            status_bits.append("✅ wallet address from live client")
        else:
            status_bits.append(f"⚠️ wallet address fallback ({address_source})")
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

    if live_state.get("client_ok"):
        st.success(" | ".join(status_bits))
    else:
        st.warning(" | ".join(status_bits))


def main() -> None:
    st.title("🪪 Polymarket Account Profile")
    st.caption("Use this page to visually confirm that your wallet/profile data is reachable after setup and auth.")

    live_state = _read_live_client_state()
    local_activity = _load_local_live_activity()
    default_address = live_state.get("funder") or ""

    if not live_state.get("live_mode"):
        st.error("Live mode is not active. Set TRADING_MODE=live and restart the dashboard.")
        return

    address = st.text_input(
        "Wallet / profile address",
        value=default_address,
        help="Strict mode: connection status is based on real live client calls. Address may still come from client/env/derived fallback and is labeled below.",
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

    lc1, lc2 = st.columns(2)
    lc1.metric("Local live orders", len(local_activity.get("orders", pd.DataFrame())))
    lc2.metric("Local live fills", len(local_activity.get("fills", pd.DataFrame())))

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
            st.success("Positions Data API endpoint returned data.")
        else:
            st.info("No open positions found, or Data API /positions returned no rows.")
        if activity:
            st.success("Activity endpoint returned recent rows.")
        else:
            st.info("No recent activity rows returned.")
        if bundle.get("errors"):
            st.warning("Some endpoint calls failed:")
            for item in bundle["errors"]:
                st.caption(item)

    if positions:
        st.subheader("Open positions (Data API /positions)")
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
