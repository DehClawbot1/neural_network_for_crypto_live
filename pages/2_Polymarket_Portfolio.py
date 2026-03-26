"""
pages/2_Polymarket_Portfolio.py — FIXED

FIXES:
  1. Removed module-level apply_execution_client_patch() — crashed before Streamlit loaded
  2. Removed module-level get_execution_client() — triggered auth on every page import
  3. Uses dashboard_auth.safe_load_dotenv() instead of load_dotenv()
  4. Client creation deferred to @st.cache_data functions
  5. Graceful fallback when live client is unavailable
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# ── FIX 1: Use shared auth utility ──
from dashboard_auth import (
    safe_load_dotenv,
    is_live_mode,
    get_execution_client_cached,
    get_wallet_address,
)

safe_load_dotenv()

# ── FIX 3: Lazy import profile client ──
try:
    from polymarket_profile_client import PolymarketProfileClient
except Exception:
    PolymarketProfileClient = None

# ── FIX 2: DO NOT call apply_execution_client_patch() at module level ──
# It will be called lazily when the client is first used.

st.set_page_config(page_title="Polymarket Portfolio", page_icon="N", layout="wide")

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _money(value: Optional[float]) -> str:
    return f"${float(value or 0):,.2f}"


def _pct(value: Optional[float]) -> str:
    return f"{float(value or 0) * 100:.2f}%"


@st.cache_data(show_spinner=False, ttl=30)
def _load_local_closed() -> pd.DataFrame:
    path = LOGS_DIR / "closed_positions.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


# ── FIX 4: Client creation deferred and cached ──
@st.cache_data(show_spinner=False, ttl=20)
def _live_bundle() -> dict:
    out = {"ok": False, "balance": 0.0, "available": 0.0, "onchain_total": 0.0, "orders": [], "trades": [], "address": get_wallet_address(), "error": None}

    if not is_live_mode():
        out["error"] = "Paper mode — live data unavailable"
        return out

    client = get_execution_client_cached()
    if client is None:
        out["error"] = "ExecutionClient not available"
        return out

    try:
        # ── FIX: Apply capability patch lazily ──
        try:
            from polymarket_capabilities import apply_execution_client_patch
            apply_execution_client_patch()
        except Exception:
            pass

        bal = client.get_balance_allowance(asset_type="COLLATERAL")
        out["ok"] = True
        out["address"] = getattr(client, "funder", None) or out["address"]
        if isinstance(bal, dict):
            out["balance"] = _safe_float(bal.get("balance", bal.get("amount")), 0.0)
        try:
            onchain = client.get_onchain_collateral_balance(wallet_address=out["address"])
            out["onchain_total"] = _safe_float((onchain or {}).get("total"), 0.0) or 0.0
        except Exception:
            out["onchain_total"] = 0.0
        out["available"] = out["onchain_total"] if out["onchain_total"] else out["balance"]

        # ── FIX 5: Graceful fallback for orders/trades ──
        try:
            orders = client.get_open_orders() if hasattr(client, "get_open_orders") else []
            out["orders"] = orders if isinstance(orders, list) else []
        except Exception:
            out["orders"] = []
        try:
            trades = client.get_trades() if hasattr(client, "get_trades") else []
            out["trades"] = trades if isinstance(trades, list) else []
        except Exception:
            out["trades"] = []
    except Exception as exc:
        out["error"] = str(exc)
    return out


@st.cache_data(show_spinner=False, ttl=30)
def _profile_bundle(address: str) -> dict:
    if PolymarketProfileClient is None:
        return {"value": 0.0, "positions": [], "closed": [], "activity": [], "errors": ["PolymarketProfileClient not available"]}

    client = PolymarketProfileClient()
    out = {"value": None, "positions": [], "closed": [], "activity": [], "errors": []}
    try:
        value = client.get_total_value(address)
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, dict):
            out["value"] = _safe_float(value.get("value"), 0.0)
    except Exception as exc:
        out["errors"].append(f"value: {exc}")
    try:
        positions = client.get_positions(address, limit=100)
        out["positions"] = positions if isinstance(positions, list) else []
    except Exception as exc:
        out["errors"].append(f"positions: {exc}")
    try:
        closed = client.get_closed_positions(address, limit=100)
        out["closed"] = closed if isinstance(closed, list) else []
    except Exception as exc:
        out["errors"].append(f"closed: {exc}")
    try:
        activity = client.get_activity(address, limit=100)
        out["activity"] = activity if isinstance(activity, list) else []
    except Exception as exc:
        out["errors"].append(f"activity: {exc}")
    return out


def _positions_df(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty:
        return df
    df = df.rename(columns={"title": "Market", "outcome": "Outcome", "size": "Traded", "avgPrice": "Avg", "curPrice": "Now", "currentValue": "Value", "cashPnl": "Cash PnL", "percentPnl": "Percent PnL"})
    for col in ["Traded", "Avg", "Now", "Value", "Cash PnL", "Percent PnL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"Avg", "Now"}.issubset(df.columns):
        df["AVG -> NOW"] = df.apply(lambda r: f"{r['Avg']:.2f} -> {r['Now']:.2f}" if pd.notna(r['Avg']) and pd.notna(r['Now']) else "N/A", axis=1)
    cols = [c for c in ["Market", "AVG -> NOW", "Traded", "Value", "Cash PnL", "Percent PnL", "Outcome"] if c in df.columns]
    return df[cols] if cols else df


def _orders_df(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty:
        return df
    df = df.rename(columns={"market": "Market", "asset_id": "Asset", "token_id": "Token", "side": "Side", "price": "Price", "size": "Size", "status": "Status", "created_at": "Created"})
    cols = [c for c in ["Market", "Token", "Side", "Price", "Size", "Status", "Created"] if c in df.columns]
    return df[cols] if cols else df


def _history_df(activity: list[dict], trades: list[dict]) -> pd.DataFrame:
    if activity:
        df = pd.DataFrame(activity)
        df = df.rename(columns={"timestamp": "Timestamp", "title": "Market", "type": "Type", "side": "Side", "price": "Price", "size": "Size", "outcome": "Outcome"})
        cols = [c for c in ["Timestamp", "Market", "Type", "Side", "Price", "Size", "Outcome"] if c in df.columns]
        return df[cols] if cols else df
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    df = df.rename(columns={"timestamp": "Timestamp", "market": "Market", "side": "Side", "price": "Price", "size": "Size", "status": "Status"})
    cols = [c for c in ["Timestamp", "Market", "Side", "Price", "Size", "Status"] if c in df.columns]
    return df[cols] if cols else df


def _pnl_chart_df(remote_closed: list[dict], local_closed: pd.DataFrame, range_key: str) -> pd.DataFrame:
    frames = []
    if remote_closed:
        frames.append(pd.DataFrame(remote_closed))
    if local_closed is not None and not local_closed.empty:
        frames.append(local_closed.copy())
    if not frames:
        return pd.DataFrame({"timestamp": pd.date_range(end=pd.Timestamp.utcnow(), periods=24, freq="H"), "pnl": [0.0] * 24})
    df = pd.concat(frames, ignore_index=True, sort=False)
    tcol = next((c for c in ["closed_at", "timestamp", "updated_at", "created_at"] if c in df.columns), None)
    pcol = next((c for c in ["net_realized_pnl", "realizedPnl", "realized_pnl", "cashPnl", "net_pnl"] if c in df.columns), None)
    if not tcol or not pcol:
        return pd.DataFrame({"timestamp": pd.date_range(end=pd.Timestamp.utcnow(), periods=24, freq="H"), "pnl": [0.0] * 24})
    df["timestamp"] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    df["pnl"] = pd.to_numeric(df[pcol], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    now = pd.Timestamp.utcnow()
    if range_key == "1D":
        cutoff, freq = now - pd.Timedelta(days=1), "H"
    elif range_key == "1W":
        cutoff, freq = now - pd.Timedelta(days=7), "D"
    elif range_key == "1M":
        cutoff, freq = now - pd.Timedelta(days=30), "D"
    else:
        cutoff, freq = df["timestamp"].min(), "D"
    df = df[df["timestamp"] >= cutoff]
    if df.empty:
        return pd.DataFrame({"timestamp": pd.date_range(end=now, periods=24, freq="H"), "pnl": [0.0] * 24})
    return df.set_index("timestamp").resample(freq)["pnl"].sum().cumsum().reset_index()


def _search(df: pd.DataFrame, term: str) -> pd.DataFrame:
    if df.empty or not term:
        return df
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        mask = mask | df[col].astype(str).str.contains(term, case=False, na=False)
    return df[mask]


st.title("Polymarket Portfolio")
st.caption("Portfolio, available balance, positions, open orders, and history.")

live = _live_bundle()
address = live.get("address")
profile = _profile_bundle(address) if address else {"value": 0.0, "positions": [], "closed": [], "activity": [], "errors": ["missing address"]}
local_closed = _load_local_closed()

portfolio_value = profile.get("value") if profile.get("value") is not None else live.get("balance", 0.0)
available = live.get("available", live.get("balance", 0.0))
range_key = st.radio("Range", ["1D", "1W", "1M", "ALL"], horizontal=True, index=0)
chart_df = _pnl_chart_df(profile.get("closed", []), local_closed, range_key)
current_pnl = float(chart_df["pnl"].iloc[-1]) if not chart_df.empty else 0.0
pnl_pct = (current_pnl / float(portfolio_value)) if portfolio_value else 0.0

left, right = st.columns(2)
with left:
    c1, c2 = st.columns(2)
    c1.metric("Portfolio", _money(portfolio_value), f"{_money(current_pnl)} ({_pct(pnl_pct)}) past {range_key.lower()}")
    c2.metric("Available to trade", _money(available))
    st.caption(f"CLOB/API: {_money(live.get('balance', 0.0))} | On-chain: {_money(live.get('onchain_total', 0.0))}")
with right:
    st.metric("Profit/Loss", _money(current_pnl), f"Past {range_key}")
    fig = px.line(chart_df, x="timestamp", y="pnl")
    fig.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

if address:
    st.caption(f"Wallet: {address}")
if live.get("error"):
    st.warning(f"Live client: {live['error']}")
for note in profile.get("errors", []):
    st.caption(f"Data note: {note}")

search = st.text_input("Search", "")
sort_key = st.selectbox("Sort", ["Current value", "Market", "Traded"])

positions_df = _positions_df(profile.get("positions", []))
orders_df = _orders_df(live.get("orders", []))
history_df = _history_df(profile.get("activity", []), live.get("trades", []))

positions_tab, orders_tab, history_tab = st.tabs(["Positions", "Open orders", "History"])

with positions_tab:
    df = _search(positions_df, search)
    if not df.empty:
        if sort_key == "Current value" and "Value" in df.columns:
            df = df.sort_values("Value", ascending=False)
        elif sort_key == "Market" and "Market" in df.columns:
            df = df.sort_values("Market", ascending=True)
        elif sort_key == "Traded" and "Traded" in df.columns:
            df = df.sort_values("Traded", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No positions found.")

with orders_tab:
    df = _search(orders_df, search)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No open orders found.")

with history_tab:
    df = _search(history_df, search)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No history found.")
