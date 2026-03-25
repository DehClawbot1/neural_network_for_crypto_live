from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from polymarket_capabilities import apply_execution_client_patch
from repository_polymarket_service import get_execution_client
from polymarket_profile_client import PolymarketProfileClient

apply_execution_client_patch()
load_dotenv()
st.set_page_config(page_title="Polymarket Portfolio Styled", page_icon="💼", layout="wide")

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

st.markdown(
    """
    <style>
    .pm-wrap {padding-top: .2rem;}
    .pm-box {border:1px solid rgba(148,163,184,.22); border-radius:18px; padding:18px 20px; background:#ffffff;}
    .pm-top {display:flex; justify-content:space-between; gap:12px; align-items:flex-start;}
    .pm-label {color:#7c8798; font-size:.95rem; margin-bottom:6px;}
    .pm-big {font-size:2.05rem; font-weight:700; line-height:1.05; color:#1f2a44;}
    .pm-sub {color:#00a63e; font-size:1rem; margin-top:8px;}
    .pm-muted {color:#7c8798; font-size:.95rem;}
    .pm-history-toolbar {display:flex; gap:10px; align-items:center; margin:8px 0 18px 0;}
    .pm-chip {border:1px solid rgba(148,163,184,.28); background:#fff; border-radius:12px; padding:8px 14px; font-size:.92rem; color:#1f2a44;}
    .pm-card {border:1px solid rgba(148,163,184,.18); background:#fbfbfc; border-radius:16px; padding:14px 16px; margin-bottom:12px;}
    .pm-row {display:grid; grid-template-columns: 90px 1fr 120px 90px; gap:14px; align-items:center;}
    .pm-activity {font-weight:600; color:#1f2a44;}
    .pm-market {font-weight:600; color:#111827; font-size:1.02rem;}
    .pm-small {color:#7c8798; font-size:.9rem;}
    .pm-green {color:#16a34a; font-weight:700;}
    .pm-red {color:#dc2626; font-weight:700;}
    .pm-time {color:#6b7280; text-align:left;}
    .stTabs [data-baseweb="tab-list"] {gap:28px;}
    .stTabs [data-baseweb="tab"] {padding-left:0; padding-right:0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _money(value: Optional[float]) -> str:
    return f"${float(value or 0):,.2f}"


def _pct(value: Optional[float]) -> str:
    return f"{float(value or 0) * 100:.2f}%"


def _addr() -> Optional[str]:
    for key in ["POLYMARKET_PUBLIC_ADDRESS", "POLY_ADDRESS", "POLYMARKET_FUNDER"]:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


@st.cache_data(show_spinner=False, ttl=20)
def _live() -> dict:
    out = {"balance": 0.0, "available": 0.0, "onchain_total": 0.0, "orders": [], "trades": [], "address": _addr(), "error": None}
    try:
        client = get_execution_client()
        bal = client.get_balance_allowance(asset_type="COLLATERAL")
        out["address"] = getattr(client, "funder", None) or out["address"]
        if isinstance(bal, dict):
            out["balance"] = _safe_float(bal.get("balance", bal.get("amount")), 0.0)
        try:
            onchain = client.get_onchain_collateral_balance(wallet_address=out["address"])
            out["onchain_total"] = _safe_float((onchain or {}).get("total"), 0.0) or 0.0
        except Exception:
            out["onchain_total"] = 0.0
        out["available"] = out["onchain_total"] if out["onchain_total"] else out["balance"]
        out["orders"] = client.get_orders() or []
        out["trades"] = client.get_trades() or []
    except Exception as exc:
        out["error"] = str(exc)
    return out


@st.cache_data(show_spinner=False, ttl=30)
def _profile(address: str) -> dict:
    client = PolymarketProfileClient()
    out = {"value": 0.0, "positions": [], "closed": [], "activity": [], "errors": []}
    try:
        value = client.get_total_value(address)
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, dict):
            out["value"] = _safe_float(value.get("value"), 0.0)
    except Exception as exc:
        out["errors"].append(f"value: {exc}")
    try:
        out["positions"] = client.get_positions(address, limit=100) or []
    except Exception as exc:
        out["errors"].append(f"positions: {exc}")
    try:
        out["closed"] = client.get_closed_positions(address, limit=100) or []
    except Exception as exc:
        out["errors"].append(f"closed: {exc}")
    try:
        out["activity"] = client.get_activity(address, limit=100) or []
    except Exception as exc:
        out["errors"].append(f"activity: {exc}")
    return out


@st.cache_data(show_spinner=False, ttl=30)
def _local_closed() -> pd.DataFrame:
    path = LOGS_DIR / "closed_positions.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _curve(remote_closed: list[dict], local_closed: pd.DataFrame, span: str) -> pd.DataFrame:
    frames = [pd.DataFrame(remote_closed)] if remote_closed else []
    if not local_closed.empty:
        frames.append(local_closed.copy())
    if not frames:
        return pd.DataFrame({"timestamp": pd.date_range(end=pd.Timestamp.utcnow(), periods=24, freq="H"), "pnl": [0.0] * 24})
    df = pd.concat(frames, ignore_index=True, sort=False)
    tcol = next((c for c in ["closed_at", "timestamp", "updated_at"] if c in df.columns), None)
    pcol = next((c for c in ["net_realized_pnl", "realizedPnl", "realized_pnl", "cashPnl", "net_pnl"] if c in df.columns), None)
    if not tcol or not pcol:
        return pd.DataFrame({"timestamp": pd.date_range(end=pd.Timestamp.utcnow(), periods=24, freq="H"), "pnl": [0.0] * 24})
    df["timestamp"] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    df["pnl"] = pd.to_numeric(df[pcol], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    now = pd.Timestamp.utcnow()
    if span == "1D":
        cutoff, freq = now - pd.Timedelta(days=1), "H"
    elif span == "1W":
        cutoff, freq = now - pd.Timedelta(days=7), "D"
    elif span == "1M":
        cutoff, freq = now - pd.Timedelta(days=30), "D"
    else:
        cutoff, freq = df["timestamp"].min(), "D"
    df = df[df["timestamp"] >= cutoff]
    if df.empty:
        return pd.DataFrame({"timestamp": pd.date_range(end=now, periods=24, freq="H"), "pnl": [0.0] * 24})
    return df.set_index("timestamp").resample(freq)["pnl"].sum().cumsum().reset_index()


def _positions_df(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty:
        return df
    df = df.rename(columns={"title": "Market", "outcome": "Outcome", "size": "Traded", "avgPrice": "Avg", "curPrice": "Now", "currentValue": "Value"})
    for col in ["Traded", "Avg", "Now", "Value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"Avg", "Now"}.issubset(df.columns):
        df["AVG → NOW"] = df.apply(lambda r: f"{r['Avg']:.2f} → {r['Now']:.2f}" if pd.notna(r['Avg']) and pd.notna(r['Now']) else "N/A", axis=1)
    cols = [c for c in ["Market", "AVG → NOW", "Traded", "Value", "Outcome"] if c in df.columns]
    return df[cols] if cols else df


def _orders_df(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty:
        return df
    df = df.rename(columns={"market": "Market", "token_id": "Token", "side": "Side", "price": "Price", "size": "Size", "status": "Status", "created_at": "Created"})
    cols = [c for c in ["Market", "Token", "Side", "Price", "Size", "Status", "Created"] if c in df.columns]
    return df[cols] if cols else df


def _search(df: pd.DataFrame, term: str) -> pd.DataFrame:
    if df.empty or not term:
        return df
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        mask = mask | df[col].astype(str).str.contains(term, case=False, na=False)
    return df[mask]


def _history_cards(items: list[dict], term: str, newest_first: bool = True, type_filter: str = "All") -> None:
    df = pd.DataFrame(items)
    if df.empty:
        st.info("No history found.")
        return
    if term:
        mask = pd.Series(False, index=df.index)
        for col in df.columns:
            mask = mask | df[col].astype(str).str.contains(term, case=False, na=False)
        df = df[mask]
    if type_filter != "All" and "type" in df.columns:
        df = df[df["type"].astype(str).str.upper() == type_filter.upper()]
    if newest_first and "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)

    for _, row in df.head(30).iterrows():
        activity = str(row.get("type", "TRADE")).title()
        side = str(row.get("side", "")).title()
        market = str(row.get("title", row.get("market", "Unknown market")))
        outcome = str(row.get("outcome", ""))
        shares = _safe_float(row.get("size"), 0.0)
        price = _safe_float(row.get("price"), 0.0)
        value = shares * price if shares is not None and price is not None else 0.0
        ts = pd.to_datetime(row.get("timestamp"), errors="coerce", utc=True)
        if pd.notna(ts):
            delta = pd.Timestamp.utcnow() - ts
            days = int(delta.total_seconds() // 86400)
            time_text = f"{days}d ago" if days > 0 else "today"
        else:
            time_text = "-"
        value_cls = "pm-green" if side.upper() == "SELL" else "pm-red" if side.upper() == "BUY" else "pm-green"
        sign = "+" if side.upper() == "SELL" else "-" if side.upper() == "BUY" else ""
        st.markdown(
            f"""
            <div class="pm-card">
                <div class="pm-row">
                    <div class="pm-activity">{activity}</div>
                    <div>
                        <div class="pm-market">{market}</div>
                        <div class="pm-small">{outcome} {price:.3f}c &nbsp; {shares:,.1f} shares</div>
                    </div>
                    <div class="{value_cls}">{sign}{_money(value)}</div>
                    <div class="pm-time">{time_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown('<div class="pm-wrap"></div>', unsafe_allow_html=True)
st.title("💼 Polymarket Portfolio")
st.caption("Styled portfolio view using your merged Polymarket integration.")

live = _live()
address = live.get("address")
profile = _profile(address) if address else {"value": 0.0, "positions": [], "closed": [], "activity": [], "errors": ["missing address"]}
curve_span = st.radio("", ["1D", "1W", "1M", "ALL"], horizontal=True, index=0)
curve_df = _curve(profile.get("closed", []), _local_closed(), curve_span)
portfolio_value = profile.get("value") or live.get("balance", 0.0)
available = live.get("available", live.get("balance", 0.0))
current_pnl = float(curve_df["pnl"].iloc[-1]) if not curve_df.empty else 0.0
pnl_pct = (current_pnl / float(portfolio_value)) if portfolio_value else 0.0

c1, c2 = st.columns(2)
with c1:
    st.markdown(f'<div class="pm-box"><div class="pm-top"><div><div class="pm-label">Portfolio</div><div class="pm-big">{_money(portfolio_value)}</div><div class="pm-sub">{_money(current_pnl)} ({_pct(pnl_pct)}) past {curve_span.lower()}</div></div><div><div class="pm-label">Available to trade</div><div class="pm-big">{_money(available)}</div></div></div></div>', unsafe_allow_html=True)
    st.caption(f"CLOB/API collateral: {_money(live.get('balance', 0.0))} | On-chain USDC: {_money(live.get('onchain_total', 0.0))}")
    d1, d2 = st.columns(2)
    d1.button("Deposit", use_container_width=True, disabled=True)
    d2.button("Withdraw", use_container_width=True, disabled=True)
with c2:
    st.markdown(f'<div class="pm-box"><div class="pm-top"><div><div class="pm-label">Profit/Loss</div><div class="pm-big">{_money(current_pnl)}</div><div class="pm-sub">Past {curve_span}</div></div></div></div>', unsafe_allow_html=True)
    fig = px.line(curve_df, x="timestamp", y="pnl")
    fig.update_layout(height=190, margin=dict(l=0, r=0, t=8, b=0), xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

if address:
    st.caption(f"Wallet: {address}")
if live.get("error"):
    st.warning(f"Live client: {live['error']}")
for note in profile.get("errors", []):
    st.caption(f"Data note: {note}")

search = st.text_input("Search", "", placeholder="Search")
sort_key = st.selectbox("Sort", ["Current value", "Market", "Traded"])
positions_tab, orders_tab, history_tab = st.tabs(["Positions", "Open orders", "History"])

with positions_tab:
    df = _search(_positions_df(profile.get("positions", [])), search)
    if not df.empty:
        if sort_key == "Current value" and "Value" in df.columns:
            df = df.sort_values("Value", ascending=False)
        elif sort_key == "Market" and "Market" in df.columns:
            df = df.sort_values("Market")
        elif sort_key == "Traded" and "Traded" in df.columns:
            df = df.sort_values("Traded", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No positions found.")

with orders_tab:
    df = _search(_orders_df(live.get("orders", [])), search)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No open orders found.")

with history_tab:
    bar1, bar2, bar3, bar4 = st.columns([3, 1, 1, 1])
    with bar1:
        history_search = st.text_input("", search, placeholder="Search", key="history_search")
    with bar2:
        history_type = st.selectbox("", ["All", "Trade", "Redeem"], key="history_type")
    with bar3:
        newest = st.selectbox("", ["Newest", "Oldest"], key="history_sort") == "Newest"
    with bar4:
        st.download_button("Export", data=pd.DataFrame(profile.get("activity", [])).to_csv(index=False).encode("utf-8"), file_name="polymarket_history.csv", mime="text/csv", use_container_width=True)
    _history_cards(profile.get("activity", []), history_search, newest_first=newest, type_filter=history_type)
