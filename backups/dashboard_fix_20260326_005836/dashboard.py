"""
Neural Network for Crypto — Dashboard v2 (FIXED)
Full-featured trading terminal with all original panels preserved.

FIXES APPLIED:
  1. Conditional load_dotenv() — respects _INTERACTIVE_MODE from start.py
  2. Cached ExecutionClient — no longer re-created on every Streamlit rerun
  3. Live sidebar uses shared auth utility
  4. Auto-refresh missing-package warning
  5. Graceful handling of empty DataFrames throughout
  6. Schema normalization applied consistently
"""

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── FIX 1: Use conditional dotenv loading ──
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
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

from schema import normalize_dataframe_columns
from log_loader import load_execution_history as shared_load_execution_history

BASE = Path(__file__).resolve().parent
LOGS = BASE / "logs"
WEIGHTS = BASE / "weights"

FILES = {
    "signals": LOGS / "signals.csv",
    "execution": LOGS / "execution_log.csv",
    "legacy_execution": LOGS / "daily_summary.txt",
    "episode_log": LOGS / "episode_log.csv",
    "markets": LOGS / "markets.csv",
    "whales": LOGS / "whales.csv",
    "alerts": LOGS / "alerts.csv",
    "positions": LOGS / "positions.csv",
    "closed": LOGS / "closed_positions.csv",
    "model_status": LOGS / "model_status.csv",
    "health": LOGS / "system_health.csv",
    "heartbeats": LOGS / "service_heartbeats.csv",
    "supervised_eval": LOGS / "supervised_eval.csv",
    "time_split": LOGS / "time_split_eval.csv",
    "replay": LOGS / "path_replay_backtest.csv",
    "wallet_backtest": LOGS / "backtest_by_wallet.csv",
    "registry": WEIGHTS / "model_registry.csv",
    "shadow": LOGS / "shadow_results.csv",
    "distribution": LOGS / "market_distribution.csv",
}

st.set_page_config(page_title="NNC Trading Terminal", page_icon="N", layout="wide", initial_sidebar_state="expanded")


def inject_theme():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    :root{--bg-card:#111113;--border:rgba(255,255,255,0.06);--border-active:rgba(255,255,255,0.12);--text-primary:#fafafa;--text-secondary:#a1a1aa;--text-muted:#52525b;--green:#22c55e;--green-bg:rgba(34,197,94,0.08);--red:#ef4444;--red-bg:rgba(239,68,68,0.08);--amber:#f59e0b;--amber-bg:rgba(245,158,11,0.08);--blue:#3b82f6;--blue-bg:rgba(59,130,246,0.08);--cyan:#06b6d4;}
    .main .block-container{padding-top:1.5rem;max-width:1440px;}
    html,body,[class*="st-"]{font-family:'Outfit',sans-serif!important;}
    .terminal-header{display:flex;justify-content:space-between;align-items:center;padding:0.8rem 0;margin-bottom:1.2rem;border-bottom:1px solid var(--border);}
    .terminal-title{font-size:1.4rem;font-weight:800;letter-spacing:-0.03em;color:var(--text-primary);}
    .terminal-title span{color:var(--cyan);}
    .terminal-status{display:flex;align-items:center;gap:8px;font-size:0.82rem;color:var(--text-secondary);font-family:'JetBrains Mono',monospace;}
    .pulse{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px rgba(34,197,94,0.6);animation:pg 2s ease-in-out infinite;}
    @keyframes pg{0%,100%{opacity:1;}50%{opacity:0.4;}}
    .metric-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:1.2rem;}
    .m-card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;transition:border-color 0.2s;}
    .m-card:hover{border-color:var(--border-active);}
    .m-label{font-size:0.7rem;font-weight:500;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;}
    .m-value{font-size:1.3rem;font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:-0.02em;}
    .m-sub{font-size:0.73rem;color:var(--text-muted);margin-top:2px;font-family:'JetBrains Mono',monospace;}
    .clr-green{color:var(--green);}.clr-red{color:var(--red);}.clr-amber{color:var(--amber);}.clr-blue{color:var(--blue);}.clr-cyan{color:var(--cyan);}.clr-default{color:var(--text-primary);}
    .sig-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;margin-bottom:1rem;}
    .sig-card{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:16px 18px;transition:all 0.2s;}
    .sig-card:hover{border-color:var(--border-active);transform:translateY(-1px);}
    .sig-market{font-size:0.95rem;font-weight:600;color:var(--text-primary);margin-bottom:8px;line-height:1.3;}
    .sig-badges{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;}
    .badge{display:inline-block;padding:3px 10px;border-radius:100px;font-size:0.7rem;font-weight:600;letter-spacing:0.03em;}
    .badge-top{background:var(--blue-bg);color:var(--blue);border:1px solid rgba(59,130,246,0.15);}
    .badge-strong{background:var(--green-bg);color:var(--green);border:1px solid rgba(34,197,94,0.20);}
    .badge-watch{background:var(--amber-bg);color:var(--amber);border:1px solid rgba(245,158,11,0.15);}
    .badge-ignore{background:rgba(82,82,91,0.15);color:var(--text-muted);border:1px solid rgba(82,82,91,0.2);}
    .sig-row{display:flex;justify-content:space-between;font-size:0.8rem;color:var(--text-secondary);padding:3px 0;font-family:'JetBrains Mono',monospace;}
    .sig-row span:first-child{color:var(--text-muted);}
    .conf-track{width:100%;height:4px;background:rgba(255,255,255,0.06);border-radius:100px;margin:10px 0 4px;overflow:hidden;}
    .conf-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--cyan),var(--green));transition:width 0.4s ease;}
    .health-grid{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1rem;}
    .health-pill{display:flex;align-items:center;gap:6px;background:var(--bg-card);border:1px solid var(--border);border-radius:100px;padding:6px 14px;font-size:0.78rem;color:var(--text-secondary);}
    .dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
    .dot-ok{background:var(--green);box-shadow:0 0 6px rgba(34,197,94,0.5);}.dot-warn{background:var(--amber);}.dot-err{background:var(--red);}.dot-off{background:var(--text-muted);}
    .sec-title{font-size:0.85rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin:1.4rem 0 0.7rem;padding-bottom:0.5rem;border-bottom:1px solid var(--border);}
    .reason-box{margin-top:0.5rem;color:#94a3b8;font-size:0.82rem;padding:8px 10px;border-radius:10px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.04);}
    .action-group-title{font-size:0.9rem;font-weight:700;margin:1rem 0 0.4rem;color:var(--text-primary);}
    .stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--border);}
    .stTabs [data-baseweb="tab"]{font-family:'Outfit',sans-serif;font-weight:600;font-size:0.85rem;padding:10px 20px;}
    .stDataFrame{border-radius:10px;overflow:hidden;}
    </style>""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=15)
def load(key):
    path = FILES.get(key)
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return normalize_dataframe_columns(pd.read_csv(str(path), engine="python", on_bad_lines="skip"))
    except Exception:
        return pd.DataFrame()


def load_trades():
    try:
        return normalize_dataframe_columns(shared_load_execution_history(str(LOGS)))
    except Exception:
        return pd.DataFrame()


def latest_ts(df, cols=None):
    if df is None or df.empty:
        return None
    for c in (cols or ["timestamp", "updated_at", "created_at", "closed_at", "opened_at"]):
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=True).dropna()
            if not ts.empty:
                return ts.max()
    return None


def age_seconds(df, cols=None):
    ts = latest_ts(df, cols)
    return max(0, int((pd.Timestamp.utcnow() - ts).total_seconds())) if ts else None


def fmt_age(s):
    if s is None:
        return "unknown"
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h"


def fmt_pct(v):
    try:
        if pd.isna(v):
            return "N/A"
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return "N/A"


def fmt_num(v, d=4):
    try:
        if pd.isna(v):
            return "N/A"
        return f"{float(v):.{d}f}"
    except Exception:
        return "N/A"


def fmt_money(v):
    try:
        if pd.isna(v):
            return "N/A"
        val = float(v)
        sign = "+" if val > 0 else ""
        return f"{sign}${val:,.2f}"
    except Exception:
        return "N/A"


def optional_number(v, d=3):
    try:
        if v is None or pd.isna(v):
            return "N/A"
        return round(float(v), d)
    except Exception:
        return "N/A"


def pnl_color(v):
    try:
        val = float(v)
        if val > 0:
            return "clr-green"
        if val < 0:
            return "clr-red"
    except Exception:
        pass
    return "clr-default"


def badge_cls(l):
    l = str(l).upper()
    if "HIGHEST" in l:
        return "badge-top"
    if "STRONG" in l:
        return "badge-strong"
    if "WATCH" in l:
        return "badge-watch"
    return "badge-ignore"


def freshness_status(a):
    if a is None:
        return "missing", "dot-off"
    if a < 120:
        return "fresh", "dot-ok"
    if a <= 600:
        return "delayed", "dot-warn"
    return "stale", "dot-err"


def ensure_safe(df):
    if df is None or getattr(df, "empty", False):
        return df
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].apply(lambda x: "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
    return out


def apply_filters(df, market_search="", wallet_search="", min_confidence=0.0, signal_label="All", side_filter="All", min_edge=-1.0, only_actionable=False, time_hours=None):
    if df is None or df.empty:
        return df
    out = df.copy()
    if market_search:
        mc = next((c for c in ["market_title", "market"] if c in out.columns), None)
        if mc:
            out = out[out[mc].astype(str).str.contains(market_search, case=False, na=False)]
    if wallet_search:
        wc = next((c for c in ["trader_wallet", "wallet_copied"] if c in out.columns), None)
        if wc:
            out = out[out[wc].astype(str).str.contains(wallet_search, case=False, na=False)]
    if "confidence" in out.columns:
        out = out[pd.to_numeric(out["confidence"], errors="coerce").fillna(0) >= min_confidence]
    if min_edge > -1.0 and "edge_score" in out.columns:
        out = out[pd.to_numeric(out["edge_score"], errors="coerce").fillna(0) >= min_edge]
    if signal_label != "All" and "signal_label" in out.columns:
        out = out[out["signal_label"].astype(str) == signal_label]
    if side_filter != "All":
        sc = next((c for c in ["outcome_side", "side"] if c in out.columns), None)
        if sc:
            out = out[out[sc].astype(str).str.upper() == side_filter]
    if only_actionable and "signal_label" in out.columns:
        out = out[~out["signal_label"].astype(str).str.upper().isin(["IGNORE", "NO_ACTION"])]
    if time_hours is not None:
        for c in ["timestamp", "closed_at", "updated_at"]:
            if c in out.columns:
                ts = pd.to_datetime(out[c], errors="coerce", utc=True)
                out = out[ts >= (pd.Timestamp.utcnow() - pd.Timedelta(hours=float(time_hours)))]
                break
    return out


PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Outfit,sans-serif", color="#a1a1aa", size=12), margin=dict(l=0, r=0, t=32, b=0), xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"), legend=dict(bgcolor="rgba(0,0,0,0)"))
COLORS = ["#06b6d4", "#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#ec4899"]


def sfig(fig, h=320):
    fig.update_layout(**PL, height=h)
    return fig


# ── Render functions ─────────────────────────────────────────────────────────

def render_header(signals_df, markets_df):
    ages = [a for a in [age_seconds(signals_df), age_seconds(markets_df)] if a is not None]
    mode = get_trading_mode().upper()
    mc = "clr-green" if mode == "LIVE" else "clr-amber"
    fr = fmt_age(min(ages)) if ages else "unknown"
    # ── FIX: Show interactive mode indicator ──
    mode_label = f"{mode}"
    if is_interactive_mode():
        mode_label += " (interactive)"
    st.markdown(f'<div class="terminal-header"><div class="terminal-title"><span>NNC</span> Trading Terminal</div><div class="terminal-status"><div class="pulse"></div><span class="{mc}">{mode_label}</span><span>|</span><span>Updated {fr} ago</span></div></div>', unsafe_allow_html=True)
    st.info("Real-time public-data market tracker, whale tracker, and paper-trading dashboard. In LIVE mode, connects to Polymarket for balances and execution.")


def render_metrics(signals_df, positions_df, closed_df, alerts_df, markets_df):
    now = pd.Timestamp.utcnow()
    st_th = 120
    sig_count = 0
    top_conf = "N/A"
    if not signals_df.empty:
        if "timestamp" in signals_df.columns:
            ts = pd.to_datetime(signals_df["timestamp"], errors="coerce", utc=True)
            sig_count = int((ts >= (now - pd.Timedelta(hours=1))).fillna(False).sum())
        if "confidence" in signals_df.columns:
            c = pd.to_numeric(signals_df["confidence"], errors="coerce").dropna()
            # ── FIX 6: Guard against empty series before .max() ──
            if not c.empty:
                top_conf = f"{c.max():.2f}"
    n_open = len(positions_df) if not positions_df.empty else 0
    upnl = float(pd.to_numeric(positions_df.get("unrealized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not positions_df.empty and "unrealized_pnl" in positions_df.columns else 0.0
    rpnl = 0.0
    wr = "N/A"
    n_closed = len(closed_df) if not closed_df.empty else 0
    if not closed_df.empty:
        pc = next((c for c in ["net_realized_pnl", "realized_pnl"] if c in closed_df.columns), None)
        if pc:
            ps = pd.to_numeric(closed_df[pc], errors="coerce").fillna(0)
            rpnl = float(ps.sum())
            if len(ps):
                wr = f"{(ps > 0).mean() * 100:.1f}%"
    nc = 0
    if not alerts_df.empty:
        sc = next((c for c in ["severity", "level"] if c in alerts_df.columns), None)
        if sc:
            nc = int(alerts_df[sc].astype(str).str.contains("critical", case=False, na=False).sum())
    nm = int(markets_df["market_id"].nunique()) if not markets_df.empty and "market_id" in markets_df.columns else len(markets_df)
    fa = {k: age_seconds(v) for k, v in [("signals", signals_df), ("markets", markets_df), ("alerts", alerts_df), ("positions", positions_df)]}
    av = [a for a in fa.values() if a is not None]
    ns = sum(1 for a in av if a > st_th)
    sa = fa["signals"]
    pa = fa["positions"]
    cards = [
        ("SIGNALS LAST HOUR", str(sig_count) if sa and sa <= st_th else ("N/A" if sa is None else str(sig_count)), "clr-cyan", ""),
        ("OPEN POSITIONS", str(n_open), "clr-blue", ""),
        ("REALIZED PNL", fmt_money(rpnl), pnl_color(rpnl), f"Win rate: {wr}"),
        ("UNREALIZED PNL", fmt_money(upnl), pnl_color(upnl), ""),
        ("CRITICAL ALERTS", str(nc), "clr-red" if nc else "clr-green", ""),
        ("STALE FEEDS", str(ns), "clr-red" if ns else "clr-green", f"of {len(av)} monitored"),
        ("TOP CONFIDENCE", top_conf, "clr-cyan", ""),
        ("MARKETS WATCHED", str(nm), "clr-default", ""),
        ("CLOSED POSITIONS", str(n_closed), "clr-default", ""),
        ("MAX FRESHNESS AGE", fmt_age(max(av)) if av else "N/A", "clr-amber" if av and max(av) > st_th else "clr-green", ""),
    ]
    h = '<div class="metric-strip">'
    for l, v, c, s in cards:
        h += f'<div class="m-card"><div class="m-label">{l}</div><div class="m-value {c}">{v}</div>'
        if s:
            h += f'<div class="m-sub">{s}</div>'
        h += '</div>'
    h += '</div>'
    st.markdown(h, unsafe_allow_html=True)
    st.caption(f"Feed freshness — {' | '.join(f'{k}: {fmt_age(v)}' for k, v in fa.items())} | stale threshold: {st_th}s")


def render_freshness(source_frames):
    st.markdown('<div class="sec-title">Data Freshness</div>', unsafe_allow_html=True)
    now = pd.Timestamp.utcnow()
    rows = []
    for label, path, df in source_frames:
        p = Path(path)
        fm = pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC") if p.exists() else None
        lt = latest_ts(df)
        best = max([t for t in [fm, lt] if t], default=None)
        a = int((now - best).total_seconds()) if best else None
        s, _ = freshness_status(a)
        rows.append({"source": label, "latest_row": lt.strftime('%Y-%m-%d %H:%M:%S') if lt else "N/A", "file_modified": fm.strftime('%Y-%m-%d %H:%M:%S') if fm else "N/A", "age": f"{a}s" if a and a < 120 else (f"{round(a / 60, 1)}m" if a else "N/A"), "status": s})
    st.dataframe(ensure_safe(pd.DataFrame(rows)), use_container_width=True, hide_index=True)


def render_health(signals_df, markets_df, positions_df, model_status_df, replay_df, health_df):
    st.markdown('<div class="sec-title">Pipeline Health</div>', unsafe_allow_html=True)

    def chk(df, ma=600):
        a = age_seconds(df)
        if a is None:
            return "dot-off"
        return "dot-ok" if a <= ma else "dot-warn" if a <= 1800 else "dot-err"

    items = [("Market monitor", chk(markets_df)), ("Whale tracker", chk(load("whales"))), ("Signal engine", chk(signals_df)), ("Order simulation", chk(positions_df)), ("Model status", chk(model_status_df, 1800)), ("System health", chk(health_df)), ("Replay available", "dot-ok" if replay_df is not None and not replay_df.empty else "dot-off"), ("Signals growing", "dot-ok" if signals_df is not None and len(signals_df) > 0 else "dot-off")]
    h = '<div class="health-grid">'
    for n, d in items:
        h += f'<div class="health-pill"><div class="dot {d}"></div>{n}</div>'
    h += '</div>'
    st.markdown(h, unsafe_allow_html=True)


def render_attention(signals_df, trades_df, alerts_df, positions_df, model_status_df, replay_df, health_df):
    st.markdown('<div class="sec-title">Attention Needed</div>', unsafe_allow_html=True)
    w = []
    now = pd.Timestamp.utcnow()
    st_ = latest_ts(signals_df)
    if st_ is None or (now - st_).total_seconds() > 1800:
        w.append("No signals in last 30 min")
    pt = latest_ts(positions_df)
    if pt is None or (now - pt).total_seconds() > 600:
        w.append("Positions file stale")
    if alerts_df is None or alerts_df.empty:
        w.append("Alerts file missing or empty")
    if model_status_df is None or model_status_df.empty:
        w.append("Model outputs missing")
    if replay_df is None or replay_df.empty:
        w.append("No replay outputs available")
    ht = latest_ts(health_df)
    if ht is None or (now - ht).total_seconds() > 600:
        w.append("System health feed stale")
    if w:
        for x in w:
            st.warning(x)
    else:
        st.success("No immediate incidents detected.")


def render_perf_charts(trades_df, closed_df, alerts_df, wbt_df, reg_df, positions_df):
    st.markdown('<div class="sec-title">Performance Charts</div>', unsafe_allow_html=True)
    if not closed_df.empty:
        pc = next((c for c in ["net_realized_pnl", "realized_pnl"] if c in closed_df.columns), None)
        tc = next((c for c in ["closed_at", "timestamp"] if c in closed_df.columns), None)
        if pc and tc:
            df = closed_df.copy()
            df[tc] = pd.to_datetime(df[tc], errors="coerce")
            df[pc] = pd.to_numeric(df[pc], errors="coerce").fillna(0)
            df = df.dropna(subset=[tc]).sort_values(tc)
            df["cum"] = df[pc].cumsum()
            df["dd"] = df["cum"] - df["cum"].cummax()
            df["day"] = df[tc].dt.date.astype(str)
            daily = df.groupby("day")[pc].sum().reset_index(name="pnl")
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Scatter(x=df[tc], y=df["cum"], fill="tozeroy", fillcolor="rgba(6,182,212,0.08)", line=dict(color="#06b6d4", width=2)))
                st.plotly_chart(sfig(fig, 280).update_layout(title="Cumulative Realized PnL"), use_container_width=True)
            with c2:
                fig = go.Figure(go.Scatter(x=df[tc], y=df["dd"], fill="tozeroy", fillcolor="rgba(239,68,68,0.08)", line=dict(color="#ef4444", width=2)))
                st.plotly_chart(sfig(fig, 280).update_layout(title="Drawdown Curve"), use_container_width=True)
            cols_ = ["#22c55e" if v >= 0 else "#ef4444" for v in daily["pnl"]]
            fig = go.Figure(go.Bar(x=daily["day"], y=daily["pnl"], marker_color=cols_))
            st.plotly_chart(sfig(fig, 240).update_layout(title="Daily PnL"), use_container_width=True)


def render_signal_cards(signals_df):
    if signals_df.empty:
        st.warning("No ranked opportunities yet. Run supervisor.py first.")
        return
    v = signals_df.copy()
    if "confidence" in v.columns:
        v = v.sort_values("confidence", ascending=False)
    v = v.head(8).reset_index(drop=True)
    h = '<div class="sig-grid">'
    for _, r in v.iterrows():
        m = r.get("market_title", r.get("market", "Unknown"))
        l = r.get("signal_label", "UNKNOWN")
        s = r.get("outcome_side", r.get("side", "?"))
        cf = r.get("confidence")
        ed = r.get("edge_score")
        pr = r.get("market_last_trade_price", r.get("current_price"))
        pt = r.get("p_tp_before_sl")
        er = r.get("expected_return")
        w = str(r.get("wallet_copied", r.get("trader_wallet", "")))[:12]
        rs = r.get("reason", r.get("reason_summary", "N/A"))
        ac = r.get("recommended_action", r.get("action", r.get("entry_intent", "ignore")))
        ts_ = r.get("timestamp", r.get("updated_at", "N/A"))
        cp = 0
        try:
            cp = max(0, min(100, int(float(cf) * 100)))
        except Exception:
            pass
        a_ = str(ac).strip().lower()
        al = "ENTER" if a_ in ["enter", "open_long", "buy"] else "HOLD/WATCH" if a_ in ["hold", "watch"] else "LEAVE/EXIT" if a_ in ["leave", "exit", "close", "sell"] else "IGNORE"
        h += f'<div class="sig-card"><div class="sig-market">{m}</div><div class="sig-badges"><span class="badge {badge_cls(l)}">{l}</span><span class="badge badge-ignore">{s}</span><span class="badge badge-watch">{al}</span></div><div class="conf-track"><div class="conf-fill" style="width:{cp}%"></div></div><div class="sig-row"><span>Confidence</span><span>{fmt_pct(cf)}</span></div><div class="sig-row"><span>P(TP before SL)</span><span>{fmt_num(pt, 3)}</span></div><div class="sig-row"><span>Edge score</span><span>{fmt_num(ed, 4)}</span></div><div class="sig-row"><span>Expected return</span><span>{fmt_num(er, 4)}</span></div><div class="sig-row"><span>Current price</span><span>{fmt_num(pr, 4)}</span></div><div class="sig-row"><span>Wallet</span><span>{w}</span></div><div class="sig-row"><span>Action</span><span>{al}</span></div><div class="sig-row"><span>Freshness</span><span>{ts_ if pd.notna(ts_) else "N/A"}</span></div><div class="reason-box">{rs}</div></div>'
    h += '</div>'
    st.markdown(h, unsafe_allow_html=True)


def render_factor_matrix(signals_df):
    st.markdown('<div class="sec-title">Signal Explanation Panel</div>', unsafe_allow_html=True)
    if signals_df.empty:
        st.info("No signals yet.")
        return
    v = signals_df.copy()
    if "confidence" in v.columns:
        v = v.sort_values("confidence", ascending=False)
    v = v.head(50).reset_index(drop=True)
    opts = [f"{i + 1}. {r.get('market_title', r.get('market', '?'))} | {r.get('outcome_side', r.get('side', '?'))} | {r.get('signal_label', '?')}" for i, r in v.iterrows()]
    sel = st.selectbox("Select signal", opts, key="fsel")
    sr = v.iloc[opts.index(sel)].to_dict()
    specs = [("Whale Pressure", "whale_pressure"), ("Market Structure", "market_structure_score"), ("Volatility Risk", "volatility_risk"), ("Time Decay", "time_decay_score"), ("Liquidity", "liquidity_score"), ("Volume", "volume_score"), ("Confidence", "confidence"), ("Edge Score", "edge_score")]
    rows = [{"factor": n, "score": optional_number(sr.get(c)), "status": "missing" if pd.isna(sr.get(c)) else ""} for n, c in specs]
    fdf = pd.DataFrame(rows)
    fdf["numeric"] = pd.to_numeric(fdf["score"], errors="coerce")
    fig = px.bar(fdf, x="numeric", y="factor", orientation="h", color_discrete_sequence=["#06b6d4"])
    st.plotly_chart(sfig(fig, 380).update_layout(title="Factor Breakdown", yaxis={"categoryorder": "total ascending"}), use_container_width=True)
    st.dataframe(ensure_safe(fdf[["factor", "score", "status"]]), use_container_width=True, hide_index=True)


def render_opp_table(signals_df):
    st.markdown('<div class="sec-title">Opportunity Ranking Table</div>', unsafe_allow_html=True)
    if signals_df.empty:
        st.info("No candidates yet.")
        return
    v = signals_df.copy()
    if "confidence" in v.columns:
        v = v.sort_values("confidence", ascending=False)
    rm = {"market_title": "Market", "outcome_side": "Side", "signal_label": "Label", "confidence": "Conf", "p_tp_before_sl": "P(TP)", "edge_score": "Edge", "expected_return": "E[R]", "wallet_copied": "Wallet", "market_last_trade_price": "Price", "recommended_action": "Action", "timestamp": "Time"}
    cs = [c for c in rm if c in v.columns]
    d = v[cs].rename(columns=rm).head(50)
    st.dataframe(ensure_safe(d), use_container_width=True, hide_index=True)
    st.download_button("Export CSV", d.to_csv(index=False).encode(), "opportunities.csv", "text/csv")


def render_action_board(signals_df, positions_df):
    st.markdown('<div class="sec-title">Recommended Paper Actions</div>', unsafe_allow_html=True)
    st.caption("Paper-trading decision board only — not live execution advice.")
    if signals_df.empty:
        st.info("No signals yet.")
        return
    rk = signals_df.copy()
    sc = [c for c in ["edge_score", "p_tp_before_sl", "confidence"] if c in rk.columns]
    if sc:
        rk = rk.sort_values(by=sc, ascending=[False] * len(sc))
    rk = rk.head(20)
    om = set(positions_df["market"].dropna().astype(str)) if not positions_df.empty and "market" in positions_df.columns else set()
    gs = {"Enter": [], "Hold": [], "Exit / Leave": [], "Watch": []}
    for _, r in rk.iterrows():
        m = r.get("market_title", r.get("market", "Unknown"))
        cf = pd.to_numeric(pd.Series([r.get("confidence")]), errors="coerce").iloc[0]
        pt = pd.to_numeric(pd.Series([r.get("p_tp_before_sl")]), errors="coerce").iloc[0]
        ed = pd.to_numeric(pd.Series([r.get("edge_score")]), errors="coerce").iloc[0]
        ao = m in om
        if ao and pd.notna(cf) and cf < 0.50:
            g = "Exit / Leave"
        elif ao:
            g = "Hold"
        elif pd.notna(pt) and pd.notna(ed) and pt >= 0.62 and ed > 0:
            g = "Enter"
        else:
            g = "Watch"
        gs[g].append({"market": m, "side": r.get("outcome_side", "N/A"), "confidence": optional_number(r.get("confidence"), 3), "edge": optional_number(r.get("edge_score"), 4), "E[R]": optional_number(r.get("expected_return"), 4), "price": optional_number(r.get("market_last_trade_price", r.get("current_price")), 4), "reason": r.get("reason_summary", r.get("reason", r.get("signal_label", "N/A")))})
    for gn in ["Enter", "Hold", "Exit / Leave", "Watch"]:
        st.markdown(f'<div class="action-group-title">{gn}</div>', unsafe_allow_html=True)
        gd = pd.DataFrame(gs[gn])
        if gd.empty:
            st.caption("No rows.")
        else:
            st.dataframe(ensure_safe(gd), use_container_width=True, hide_index=True)


def render_pnl_summary(positions_df, closed_df):
    st.markdown('<div class="sec-title">PnL Summary</div>', unsafe_allow_html=True)
    no = len(positions_df) if not positions_df.empty else 0
    nc = len(closed_df) if not closed_df.empty else 0
    rc = next((c for c in ["net_realized_pnl", "realized_pnl"] if c in closed_df.columns), None) if not closed_df.empty else None
    uc = "unrealized_pnl" if not positions_df.empty and "unrealized_pnl" in positions_df.columns else None
    rp = float(pd.to_numeric(closed_df[rc], errors="coerce").fillna(0).sum()) if rc else 0.0
    up = float(pd.to_numeric(positions_df[uc], errors="coerce").fillna(0).sum()) if uc else 0.0
    wr = "N/A"
    ar = "N/A"
    if rc and not closed_df.empty:
        ps = pd.to_numeric(closed_df[rc], errors="coerce").fillna(0)
        if len(ps):
            wr = f"{(ps > 0).mean() * 100:.1f}%"
            ar = f"{ps.mean():.2f}"
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Open", no)
    c2.metric("Closed", nc)
    c3.metric("Realized PnL", f"{rp:.2f}")
    c4.metric("Unrealized PnL", f"{up:.2f}")
    c5.metric("Win Rate", wr)
    c6.metric("Avg Return", ar)


def render_positions(positions_df, closed_df):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-title">Open Positions</div>', unsafe_allow_html=True)
        if positions_df.empty:
            st.info("No open positions.")
        else:
            cs = [c for c in ["market", "outcome_side", "entry_price", "current_price", "shares", "market_value", "unrealized_pnl", "confidence", "status"] if c in positions_df.columns]
            st.dataframe(ensure_safe(positions_df[cs].tail(20) if cs else positions_df.tail(20)), use_container_width=True, hide_index=True)
    with c2:
        st.markdown('<div class="sec-title">Closed Positions</div>', unsafe_allow_html=True)
        if closed_df.empty:
            st.info("No closed positions.")
        else:
            pc = next((c for c in ["net_realized_pnl", "realized_pnl", "net_pnl"] if c in closed_df.columns), None)
            cs = [c for c in ["market", "outcome_side", "entry_price", "exit_price", pc, "close_reason", "closed_at"] if c and c in closed_df.columns]
            st.dataframe(ensure_safe(closed_df[cs].tail(20) if cs else closed_df.tail(20)), use_container_width=True, hide_index=True)


def render_best_trades(closed_df, replay_df):
    st.markdown('<div class="sec-title">Best / Worst Trades</div>', unsafe_allow_html=True)
    src = closed_df if not closed_df.empty else replay_df
    if src.empty:
        st.info("No trade history yet.")
        return
    pc = next((c for c in ["net_realized_pnl", "realized_pnl", "net_pnl"] if c in src.columns), None)
    if not pc:
        st.dataframe(ensure_safe(src.head(10)), use_container_width=True)
        return
    cs = [c for c in ["market", "entry_price", "exit_price", pc, "wallet_copied", "close_reason"] if c in src.columns]
    l, r = st.columns(2)
    with l:
        st.markdown("**Top Winners**")
        st.dataframe(ensure_safe(src.sort_values(pc, ascending=False).head(10)[cs]), use_container_width=True, hide_index=True)
    with r:
        st.markdown("**Top Losers**")
        st.dataframe(ensure_safe(src.sort_values(pc, ascending=True).head(10)[cs]), use_container_width=True, hide_index=True)


def render_markets(markets_df):
    st.markdown('<div class="sec-title">BTC Market Tracker</div>', unsafe_allow_html=True)
    if markets_df.empty:
        st.info("No market data yet.")
        return
    search = st.text_input("Search markets", "", key="ms")
    mc = next((c for c in ["market_title", "question", "market"] if c in markets_df.columns), None)
    pc = next((c for c in ["last_trade_price", "current_price"] if c in markets_df.columns), None)
    v = markets_df.copy()
    if search and mc:
        v = v[v[mc].astype(str).str.contains(search, case=False, na=False)]
    if "liquidity" in v.columns:
        v = v.sort_values("liquidity", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Tracked", len(v))
    c2.metric("Avg Liquidity", f"{float(pd.to_numeric(v.get('liquidity', pd.Series()), errors='coerce').fillna(0).mean()):.2f}" if "liquidity" in v.columns else "N/A")
    hv = "-"
    if "volume" in v.columns and mc and not v.empty:
        i = pd.to_numeric(v["volume"], errors="coerce").fillna(0).idxmax()
        hv = str(v.loc[i, mc]) if i in v.index else "-"
    c3.metric("Highest Volume", hv)
    cs = [c for c in [mc, pc, "liquidity", "volume", "spread", "market_id", "url", "updated_at"] if c and c in v.columns]
    st.dataframe(ensure_safe(v[cs].head(50)), use_container_width=True, hide_index=True)


def render_whales(whales_df):
    st.markdown('<div class="sec-title">Whale Activity Tracker</div>', unsafe_allow_html=True)
    if whales_df.empty:
        st.info("No whale data.")
        return
    wc = next((c for c in ["wallet_copied", "wallet", "trader_wallet"] if c in whales_df.columns), None)
    mc = next((c for c in ["market", "market_title", "top_market"] if c in whales_df.columns), None)
    c1, c2, c3 = st.columns(3)
    c1.metric("Wallets", int(whales_df[wc].nunique()) if wc else len(whales_df))
    c2.metric("Most Active", str(whales_df[wc].astype(str).value_counts().idxmax()) if wc and not whales_df.empty else "-")
    c3.metric("Top Market", str(whales_df[mc].astype(str).value_counts().idxmax()) if mc and not whales_df.empty else "-")
    cs = [c for c in [wc, "trade_count", "avg_size", "unique_markets", "alpha_score", "profit", "timestamp"] if c and c in whales_df.columns]
    st.dataframe(ensure_safe(whales_df[cs].head(25) if cs else whales_df.head(25)), use_container_width=True, hide_index=True)


def render_alerts(alerts_df):
    st.markdown('<div class="sec-title">Alerts</div>', unsafe_allow_html=True)
    if alerts_df.empty:
        st.info("No alerts.")
        return
    ac = next((c for c in ["alert_type", "type"] if c in alerts_df.columns), None)
    sc = next((c for c in ["severity", "level"] if c in alerts_df.columns), None)
    af = st.selectbox("Type", ["All"] + (sorted(alerts_df[ac].dropna().astype(str).unique().tolist()) if ac else []), key="atf")
    sf = st.selectbox("Severity", ["All"] + (sorted(alerts_df[sc].dropna().astype(str).unique().tolist()) if sc else []), key="asf")
    v = alerts_df.copy()
    if ac and af != "All":
        v = v[v[ac].astype(str) == af]
    if sc and sf != "All":
        v = v[v[sc].astype(str) == sf]
    nc = int(v[sc].astype(str).str.contains("critical", case=False, na=False).sum()) if sc and not v.empty else 0
    nw = int(v[sc].astype(str).str.contains("warning", case=False, na=False).sum()) if sc and not v.empty else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", len(v))
    c2.metric("Critical", nc)
    c3.metric("Warning", nw)
    cs = [c for c in ["timestamp", sc, ac, "market_title", "message", "source_module", "status"] if c and c in v.columns]
    st.dataframe(ensure_safe(v[cs].tail(50) if cs else v.tail(50)), use_container_width=True, hide_index=True)


def render_models(msd, rpd, rgd, sdf, sup, tsd, wbt):
    st.markdown('<div class="sec-title">Model / Learning Status</div>', unsafe_allow_html=True)
    missing = [f"{l}: {Path(p).name}" for l, p in [("contract targets", LOGS / "contract_targets.csv"), ("CLOB history", LOGS / "clob_price_history.csv"), ("replay", FILES["replay"]), ("supervised eval", FILES["supervised_eval"]), ("time-split eval", FILES["time_split"])] if not Path(p).exists()]
    if missing:
        st.warning("Missing: " + "; ".join(missing))
    st.write(f"**Weights:** {'current' if (WEIGHTS / 'ppo_polytrader.zip').exists() else 'missing'}")
    isa = fmt_num(sup.iloc[-1]["accuracy"], 3) if not sup.empty and "accuracy" in sup.columns else "N/A"
    ta = fmt_num(tsd.iloc[-1]["test_accuracy"], 3) if not tsd.empty and "test_accuracy" in tsd.columns else "N/A"
    sh = fmt_num(sup.iloc[-1]["sharpe"], 3) if not sup.empty and "sharpe" in sup.columns else "N/A"
    ch = "N/A"
    lt = "N/A"
    if not rgd.empty:
        nc = next((c for c in ["model_name", "name", "model_version"] if c in rgd.columns), None)
        dc = next((c for c in ["promoted_at", "trained_at"] if c in rgd.columns), None)
        if nc:
            ch = str(rgd.iloc[-1][nc])
        if dc:
            lt = str(rgd.iloc[-1][dc])
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Test Acc", ta)
    c2.metric("Replay Trades", len(rpd))
    c3.metric("Sharpe", sh)
    c4.metric("In-Sample", isa)
    c5.metric("Champion", ch)
    c6.metric("Last Train", lt)
    if not msd.empty:
        la = msd.iloc[-1].to_dict()
        rows = int(la.get("closed_trade_rows", la.get("dataset_rows", 0)) or 0)
        th = int(la.get("closed_trade_threshold", la.get("retrain_threshold", 0)) or 0)
        pr = float(la.get("progress_ratio", 0) or 0)
        act = la.get("last_action", "Unknown")
        st.progress(max(0.0, min(1.0, pr)))
        st.caption(f"Closed: {rows}/{th} | Progress: {pr:.0%}")
        st.code(act, language="text")


def render_shadow(shadow_df):
    st.markdown('<div class="sec-title">Shadow Execution</div>', unsafe_allow_html=True)
    st.caption("Shadow intents, slippage tax, DOA vetoes, and realized post-signal outcomes.")
    if shadow_df.empty:
        st.info("No shadow data yet.")
        return
    res = shadow_df[shadow_df.get("outcome", pd.Series()) != "PENDING"] if "outcome" in shadow_df.columns else pd.DataFrame()
    doa = shadow_df[shadow_df.get("outcome", pd.Series()) == "DOA"] if "outcome" in shadow_df.columns else pd.DataFrame()
    tp = float((res["outcome"] == "TP").mean()) if not res.empty and "outcome" in res.columns else 0.0
    asl = float(pd.to_numeric(shadow_df.get("entry_slippage_bps"), errors="coerce").dropna().mean()) if "entry_slippage_bps" in shadow_df.columns else 0.0
    aev = float(pd.to_numeric(shadow_df.get("ev_adj"), errors="coerce").dropna().mean()) if "ev_adj" in shadow_df.columns else 0.0
    amp = float(pd.to_numeric(shadow_df.get("meta_prob"), errors="coerce").dropna().mean()) if "meta_prob" in shadow_df.columns else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shadow Intents", len(shadow_df))
    c2.metric("Resolved", len(res))
    c3.metric("DOA / Vetoed", len(doa))
    c4.metric("TP Rate", fmt_pct(tp))
    c5, c6, c7 = st.columns(3)
    c5.metric("Avg Slippage (bps)", f"{asl:.1f}")
    c6.metric("Avg EV_adj", f"{aev:+.2%}")
    c7.metric("Avg Meta Prob", f"{amp:.2%}")
    if "outcome" in shadow_df.columns:
        oc = shadow_df["outcome"].value_counts().reset_index()
        oc.columns = ["Outcome", "Count"]
        fig = px.bar(oc, x="Outcome", y="Count", color_discrete_sequence=COLORS)
        st.plotly_chart(sfig(fig, 260).update_layout(title="Shadow Outcome Mix"), use_container_width=True)
    cs = [c for c in ["timestamp", "market_title", "meta_prob", "entry_slippage_bps", "expected_slip_bps", "ev_adj", "outcome", "realized_return", "trades_in_window"] if c in shadow_df.columns]
    st.dataframe(shadow_df[cs].tail(100), use_container_width=True, hide_index=True)


def render_quality(sdf, tdf, mdf, wdf, adf, msd, pdf, cdf, rpd, hdf):
    st.markdown('<div class="sec-title">Data Quality and Pipeline Readiness</div>', unsafe_allow_html=True)
    frames = {"signals": sdf, "execution": tdf, "markets": mdf, "whales": wdf, "alerts": adf, "positions": pdf, "closed": cdf, "model_status": msd, "health": hdf, "replay": rpd}
    rows = []
    for n, df in frames.items():
        p = FILES.get(n)
        lt = latest_ts(df)
        rows.append({"file": n, "present": "Yes" if p and p.exists() else "No", "rows": len(df) if df is not None and not df.empty else 0, "latest": lt.strftime('%H:%M:%S') if lt else "N/A", "schema": "ok" if df is not None and not df.empty else "empty"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Training Ready", "Yes" if not cdf.empty and not msd.empty else "No")
    c2.metric("Replay Ready", "Yes" if not rpd.empty else "No")
    c3.metric("Targets", "Yes" if (LOGS / "contract_targets.csv").exists() else "No")


def render_raw(sdf, tdf, edf, mdf, wdf, adf, msd, pdf, cdf):
    tabs = st.tabs(["Signals", "Execution", "Episodes", "Markets", "Whales", "Alerts", "Learning", "Positions"])
    with tabs[0]:
        st.dataframe(ensure_safe(sdf), use_container_width=True)
    with tabs[1]:
        st.dataframe(ensure_safe(tdf), use_container_width=True)
    with tabs[2]:
        st.dataframe(edf, use_container_width=True)
    with tabs[3]:
        st.dataframe(mdf, use_container_width=True)
    with tabs[4]:
        st.dataframe(wdf, use_container_width=True)
    with tabs[5]:
        st.dataframe(adf, use_container_width=True)
    with tabs[6]:
        st.dataframe(msd, use_container_width=True)
    with tabs[7]:
        st.markdown("**Open**")
        st.dataframe(ensure_safe(pdf), use_container_width=True)
        st.markdown("**Closed**")
        st.dataframe(ensure_safe(cdf), use_container_width=True)


# ── FIX 2: Completely rewritten live sidebar using cached client ──
def render_live_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Polymarket Live Truth**")

    mode = get_trading_mode()
    if mode != "live":
        st.sidebar.caption(f"Mode: {mode}. Live sidebar requires TRADING_MODE=live.")
        # ── FIX: Still show available info even in paper mode ──
        addr = get_wallet_address()
        if addr:
            st.sidebar.markdown(f"**Address:** `{addr}`")
        return

    # ── FIX 2: Use cached client instead of creating new one each rerun ──
    client = get_execution_client_cached()
    if client is None:
        st.sidebar.error("ExecutionClient not available. Check credentials.")
        if is_interactive_mode():
            st.sidebar.caption("Running in interactive mode — credentials should be in memory from start.py.")
        else:
            st.sidebar.caption("Check .env file or run python start.py for interactive auth.")
        return

    try:
        balance_info = get_balance_info()
        if balance_info.get("error"):
            st.sidebar.warning(f"Balance fetch: {balance_info['error']}")
        else:
            st.sidebar.success(f"Live client connected (source: {balance_info['source']})")

        addr = get_wallet_address()
        st.sidebar.metric("On-chain USDC", f"${balance_info['onchain_balance']:.2f}")
        st.sidebar.metric("Available to Trade (CLOB/API)", f"${balance_info['clob_balance']:.2f}")
        st.sidebar.markdown(f"**Address:** `{addr}`")

        if balance_info["onchain_balance"] > 0 and balance_info["clob_balance"] <= 0:
            st.sidebar.warning("On-chain USDC is present, but CLOB/API balance is zero.")

    except Exception as e:
        st.sidebar.error("Live client failed")
        st.sidebar.caption(str(e))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    inject_theme()
    sdf = load("signals")
    tdf = load_trades()
    edf = load("episode_log")
    mdf = load("markets")
    wdf = load("whales")
    ddf = load("distribution")
    adf = load("alerts")
    msd = load("model_status")
    hdf = load("health")
    pdf = load("positions")
    cdf = load("closed")
    sup = load("supervised_eval")
    tsd = load("time_split")
    rpd = load("replay")
    wbt = load("wallet_backtest")
    rgd = load("registry")
    shd = load("shadow")

    st.sidebar.markdown("**Dashboard Controls**")
    ar = st.sidebar.checkbox("Auto-refresh", False)
    rs = st.sidebar.selectbox("Interval (s)", [5, 10, 15, 30, 60, 120], index=2)
    dbg = st.sidebar.checkbox("Show debug sections", True)

    # ── FIX 10: Show warning when auto-refresh package is missing ──
    if ar:
        if st_autorefresh:
            st_autorefresh(interval=rs * 1000, key="auto")
        else:
            st.sidebar.warning("Install streamlit-autorefresh for auto-refresh: pip install streamlit-autorefresh")

    # ── FIX: Show auth mode indicator ──
    if is_interactive_mode():
        st.sidebar.caption("Auth: interactive (from start.py)")
    elif os.getenv("PRIVATE_KEY"):
        st.sidebar.caption("Auth: .env file")
    else:
        st.sidebar.caption("Auth: paper mode (no credentials)")

    st.sidebar.markdown("**Global Filters**")
    dr = st.sidebar.selectbox("Date range (days)", [1, 3, 7, 14, 30], index=2)
    ms = st.sidebar.text_input("Market search", "")
    ws = st.sidebar.text_input("Wallet search", "")
    sf = st.sidebar.selectbox("Side", ["All", "YES", "NO"])
    lf = st.sidebar.selectbox("Label", ["All", "IGNORE", "LOW-CONFIDENCE WATCH", "STRONG PAPER OPPORTUNITY", "HIGHEST-RANKED PAPER SIGNAL"])
    mc = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.0, 0.01)
    me = st.sidebar.slider("Min edge", -1.0, 1.0, -1.0, 0.01)
    oa = st.sidebar.checkbox("Actionable only", False)
    th = dr * 24
    fk = dict(market_search=ms, wallet_search=ws, min_confidence=mc, signal_label=lf, side_filter=sf, min_edge=me, only_actionable=oa, time_hours=th)
    sdf = apply_filters(sdf, **fk)
    tdf = apply_filters(tdf, **fk)
    pdf = apply_filters(pdf, **fk)
    cdf = apply_filters(cdf, **fk)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Status**")
    sts = latest_ts(sdf)
    nm = sum(1 for p in FILES.values() if not p.exists())
    st.sidebar.write(f"Missing files: {nm}")
    st.sidebar.write(f"Alerts: {len(adf)}")
    st.sidebar.write(f"Last signal: {sts.strftime('%H:%M:%S') if sts else 'N/A'}")
    render_live_sidebar()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Export**")
    if not sdf.empty:
        st.sidebar.download_button("Signals", sdf.to_csv(index=False).encode(), "signals.csv", "text/csv")
    if not pdf.empty:
        st.sidebar.download_button("Positions", pdf.to_csv(index=False).encode(), "positions.csv", "text/csv")
    if not adf.empty:
        st.sidebar.download_button("Alerts", adf.to_csv(index=False).encode(), "alerts.csv", "text/csv")

    render_header(sdf, mdf)
    render_metrics(sdf, pdf, cdf, adf, mdf)
    st.caption("System Status = health + performance | Signals = ranked opportunities | Positions = paper trade state + PnL | Markets = market, whale, alert context | Models = learning outputs + data quality | Shadow Audit = slippage / DOA monitoring")

    t1, t2, t3, t4, t5, t6 = st.tabs(["System Status", "Signals and Opportunities", "Positions and PnL", "Markets Whales Alerts", "Models and Data Quality", "Shadow Audit"])

    with t1:
        render_freshness([("signals.csv", FILES["signals"], sdf), ("execution_log.csv", FILES["execution"], tdf), ("markets.csv", FILES["markets"], mdf), ("whales.csv", FILES["whales"], wdf), ("alerts.csv", FILES["alerts"], adf), ("positions.csv", FILES["positions"], pdf), ("model_status.csv", FILES["model_status"], msd), ("system_health.csv", FILES["health"], hdf)])
        render_health(sdf, mdf, pdf, msd, rpd, hdf)
        render_attention(sdf, tdf, adf, pdf, msd, rpd, hdf)
        render_perf_charts(tdf, cdf, adf, wbt, rgd, pdf)

    with t2:
        l, r = st.columns([1.2, 1])
        with l:
            render_signal_cards(sdf)
        with r:
            render_factor_matrix(sdf)
        render_opp_table(sdf)
        render_action_board(sdf, pdf)

    with t3:
        render_pnl_summary(pdf, cdf)
        render_positions(pdf, cdf)
        render_best_trades(cdf, rpd)
        if not edf.empty:
            st.markdown('<div class="sec-title">Episode Log</div>', unsafe_allow_html=True)
            st.dataframe(edf.tail(20), use_container_width=True)

    with t4:
        s1, s2, s3 = st.tabs(["Markets", "Whale Activity", "Alerts"])
        with s1:
            render_markets(mdf)
            if not ddf.empty:
                st.markdown('<div class="sec-title">Whale Market Distribution</div>', unsafe_allow_html=True)
                st.dataframe(ddf.head(15), use_container_width=True)
                if "unique_wallets" in ddf.columns and "market_title" in ddf.columns:
                    fig = px.bar(ddf.head(10), x="unique_wallets", y="market_title", orientation="h", color_discrete_sequence=["#a855f7"])
                    st.plotly_chart(sfig(fig, 320).update_layout(title="Wallet Clustering"), use_container_width=True)
        with s2:
            render_whales(wdf)
        with s3:
            render_alerts(adf)

    with t5:
        sm, sq = st.tabs(["Model Performance", "Data Quality and Readiness"])
        with sm:
            render_models(msd, rpd, rgd, sdf, sup, tsd, wbt)
        with sq:
            render_quality(sdf, tdf, mdf, wdf, adf, msd, pdf, cdf, rpd, hdf)
            if dbg:
                with st.expander("Debug / Raw Logs"):
                    render_raw(sdf, tdf, edf, mdf, wdf, adf, msd, pdf, cdf)

    with t6:
        render_shadow(shd)


if __name__ == "__main__":
    main()
