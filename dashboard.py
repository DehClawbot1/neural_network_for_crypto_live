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
import sqlite3
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from strategy_layers import EntryRuleLayer

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
    "stage2_temporal": LOGS / "stage2_temporal_eval.csv",
    "walk_forward": LOGS / "walk_forward_eval.csv",
    "replay": LOGS / "path_replay_backtest.csv",
    "backtest_summary": LOGS / "backtest_summary.csv",
    "wallet_backtest": LOGS / "backtest_by_wallet.csv",
    "registry": WEIGHTS / "model_registry.csv",
    "shadow": LOGS / "shadow_results.csv",
    "distribution": LOGS / "market_distribution.csv",
    "candidate_decisions": LOGS / "candidate_decisions.csv",
    "candidate_cycle_stats": LOGS / "candidate_cycle_stats.csv",
    "position_snapshots": LOGS / "position_snapshots.csv",
    "portfolio_curve": LOGS / "portfolio_equity_curve.csv",
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
@st.cache_data(show_spinner=False, ttl=30)
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


@st.cache_data(show_spinner=False, ttl=15)
def load_live_positions_from_db():
    db_path = LOGS / "trading.db"
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(live_positions)").fetchall()]
            if not cols:
                return pd.DataFrame()
            mapping = {
                "position_key": "position_id",
                "token_id": "token_id",
                "condition_id": "condition_id",
                "outcome_side": "outcome_side",
                "shares": "shares",
                "avg_entry_price": "entry_price",
                "avg_entry_price": "avg_entry_price",
                "mark_price": "current_price",
                "current_price": "current_price",
                "realized_pnl": "realized_pnl",
                "unrealized_pnl": "unrealized_pnl",
                "last_fill_at": "opened_at",
                "status": "status",
                "market": "market",
                "market_title": "market_title",
            }
            select_parts = []
            seen_aliases = set()
            for src, alias in mapping.items():
                if src in cols and alias not in seen_aliases:
                    select_parts.append(f"{src} AS {alias}" if src != alias else src)
                    seen_aliases.add(alias)
            query = f"SELECT {', '.join(select_parts)} FROM live_positions WHERE status = 'OPEN' AND shares > 0 ORDER BY COALESCE(last_fill_at, '') DESC"
            df = pd.read_sql_query(query, conn)
        df = normalize_dataframe_columns(df)
        if not df.empty:
            if "entry_price" not in df.columns and "avg_entry_price" in df.columns:
                df["entry_price"] = df["avg_entry_price"]
            if "current_price" not in df.columns and "entry_price" in df.columns:
                df["current_price"] = df["entry_price"]
            if {"shares", "current_price"}.issubset(df.columns) and "market_value" not in df.columns:
                df["market_value"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0) * pd.to_numeric(df["current_price"], errors="coerce").fillna(0)
            if {"shares", "entry_price", "current_price"}.issubset(df.columns) and "unrealized_pnl" not in df.columns:
                df["unrealized_pnl"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0) * (
                    pd.to_numeric(df["current_price"], errors="coerce").fillna(0) - pd.to_numeric(df["entry_price"], errors="coerce").fillna(0)
                )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=20)
def load_candidate_decisions_from_db(hours=6):
    db_path = LOGS / "trading.db"
    if not db_path.exists():
        return pd.DataFrame()
    try:
        cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(hours=max(1, int(hours)))).isoformat()
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    created_at,
                    cycle_id,
                    token_id,
                    market,
                    final_decision,
                    reject_reason,
                    reject_category,
                    gate,
                    order_id
                FROM candidate_decisions
                WHERE created_at >= ?
                ORDER BY decision_id DESC
                """,
                conn,
                params=(cutoff,),
            )
        return normalize_dataframe_columns(df)
    except Exception:
        return pd.DataFrame()


def latest_ts(df, cols=None):
    if df is None or df.empty:
        return None
    default_cols = [
        "timestamp",
        "updated_at",
        "created_at",
        "closed_at",
        "opened_at",
        "last_retrained_at",
        "entry_time",
        "exit_time",
        "last_signal_timestamp",
        "last_trade_timestamp",
        "last_alert_timestamp",
        "last_position_timestamp",
    ]
    for c in (cols or default_cols):
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=True).dropna()
            if not ts.empty:
                return ts.max()
    return None


def path_mtime(path):
    try:
        p = Path(path) if path is not None else None
        if p and p.exists():
            return pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC")
    except Exception:
        pass
    return None


def age_seconds(df, cols=None, fallback_path=None):
    ts = latest_ts(df, cols)
    fm = path_mtime(fallback_path) if fallback_path is not None else None
    if ts is None:
        ts = fm
    elif fm is not None and fm > ts:
        ts = fm
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


def to_float(v, default=None):
    try:
        if v is None or pd.isna(v):
            return default
        out = float(v)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def fmt_ts(v):
    try:
        ts = pd.to_datetime(v, errors="coerce", utc=True)
        if pd.isna(ts):
            return "N/A"
        return ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "N/A"


def latest_non_empty(df, columns):
    if df is None or df.empty:
        return None
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        for value in reversed(series.tolist()):
            if str(value).strip() not in {"", "None", "nan", "NaT"}:
                return value
    return None


def resolve_numeric_metric(candidates):
    for label, df, column in candidates:
        if df is None or df.empty or column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[-1]), label, column
    return None, None, None


def resolve_text_metric(candidates):
    for label, df, columns in candidates:
        value = latest_non_empty(df, columns)
        if value is not None:
            return str(value), label
    return None, None


def resolve_registry_snapshot(registry_df):
    if registry_df is None or registry_df.empty or "model_version" not in registry_df.columns:
        return {}, 0
    work = registry_df.copy()
    work["model_version"] = work["model_version"].astype(str).str.strip()
    work = work[work["model_version"] != ""].copy()
    if work.empty:
        return {}, 0
    malformed = 0
    if "promoted_at" in work.columns:
        work["_promoted_at_ts"] = pd.to_datetime(work["promoted_at"], errors="coerce", utc=True)
        malformed = int(work["_promoted_at_ts"].isna().sum())
        valid = work[work["_promoted_at_ts"].notna()].copy()
        if not valid.empty:
            valid = valid.sort_values("_promoted_at_ts")
            return valid.iloc[-1].to_dict(), malformed
    return work.iloc[-1].to_dict(), malformed


def resolve_retrainer_status(status_df):
    if status_df is None or status_df.empty:
        return {}
    row = status_df.iloc[-1].to_dict()
    closed_total = int(to_float(row.get("closed_trade_rows", row.get("dataset_rows", 0)), 0) or 0)
    replay_total = int(to_float(row.get("replay_rows", 0), 0) or 0)
    closed_threshold = int(
        to_float(
            row.get("closed_trade_threshold", row.get("retrain_threshold", row.get("min_new_closed_rows", 0))),
            0,
        )
        or 0
    )
    replay_threshold = int(to_float(row.get("replay_threshold", row.get("min_new_replay_rows", 0)), 0) or 0)
    last_trained_closed = to_float(row.get("last_trained_closed_rows"), None)
    last_trained_replay = to_float(row.get("last_trained_replay_rows"), None)
    new_closed = max(0, closed_total - int(last_trained_closed or 0)) if last_trained_closed is not None else None
    new_replay = max(0, replay_total - int(last_trained_replay or 0)) if last_trained_replay is not None else None
    progress = to_float(row.get("progress_ratio"), None)
    if progress is None:
        if new_closed is not None and closed_threshold > 0:
            progress = new_closed / max(1, closed_threshold)
        elif closed_total > 0 and closed_threshold > 0:
            progress = closed_total / max(1, closed_threshold)
        else:
            progress = 0.0
    return {
        "closed_total": closed_total,
        "replay_total": replay_total,
        "closed_threshold": closed_threshold,
        "replay_threshold": replay_threshold,
        "new_closed": new_closed,
        "new_replay": new_replay,
        "cooldown_minutes": int(to_float(row.get("cooldown_minutes"), 0) or 0),
        "progress_ratio": max(0.0, min(1.0, float(progress))),
        "last_action": str(row.get("last_action", "Unknown") or "Unknown"),
        "last_retrained_at": row.get("last_retrained_at"),
        "status_schema": row.get("status_schema", "unknown"),
    }


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


def _safe_cell(v):
    if v is None:
        return "N/A"
    try:
        if pd.isna(v):
            return "N/A"
    except Exception:
        pass
    if isinstance(v, (bytes, bytearray, memoryview)):
        raw = bytes(v)
        try:
            return raw.decode("utf-8")
        except Exception:
            return raw.hex()
    if isinstance(v, (dict, list, tuple, set, Path)):
        return str(v)
    return v


def ensure_safe(df):
    if df is None or getattr(df, "empty", False):
        return df
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].apply(_safe_cell).astype(str)
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
    cand_df = load_candidate_decisions_from_db(hours=1)
    cand_seen = int(len(cand_df)) if not cand_df.empty else 0
    cand_tradable = 0
    cand_rejected = 0
    cand_entries_sent = 0
    cand_fills = 0
    reject_by_gate = {}
    if not cand_df.empty:
        final_col = cand_df["final_decision"].astype(str) if "final_decision" in cand_df.columns else pd.Series(dtype=str)
        gate_col = cand_df["gate"].astype(str) if "gate" in cand_df.columns else pd.Series(dtype=str)
        if "gate" in cand_df.columns:
            cand_tradable = int(cand_df[gate_col.isin(["execution", "paper_execution"])].shape[0])
        if "final_decision" in cand_df.columns:
            cand_rejected = int(cand_df[final_col.isin(["SKIPPED", "REJECTED"])].shape[0])
            cand_fills = int(cand_df[final_col == "ENTRY_FILLED"].shape[0])
        if "order_id" in cand_df.columns:
            order_col = cand_df["order_id"].astype(str)
            cand_entries_sent = int(((~cand_df["order_id"].isna()) & (~order_col.isin(["", "None", "nan"]))).sum())
        if "gate" in cand_df.columns and "final_decision" in cand_df.columns:
            rej_df = cand_df[final_col.isin(["SKIPPED", "REJECTED"])]
            reject_by_gate = rej_df["gate"].fillna("unknown").astype(str).value_counts().head(4).to_dict()
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
        ("CANDIDATES 1H", str(cand_seen), "clr-blue", "coverage"),
        ("TRADABLE 1H", str(cand_tradable), "clr-green" if cand_tradable else "clr-amber", "post-gate"),
        ("REJECTED 1H", str(cand_rejected), "clr-red" if cand_rejected else "clr-green", "gated out"),
        ("ENTRIES/FILLS 1H", f"{cand_entries_sent}/{cand_fills}", "clr-cyan", "execution quality"),
    ]
    h = '<div class="metric-strip">'
    for l, v, c, s in cards:
        h += f'<div class="m-card"><div class="m-label">{l}</div><div class="m-value {c}">{v}</div>'
        if s:
            h += f'<div class="m-sub">{s}</div>'
        h += '</div>'
    h += '</div>'
    st.markdown(h, unsafe_allow_html=True)
    if reject_by_gate:
        st.caption("Rejected by gate (1h): " + " | ".join(f"{k}={v}" for k, v in reject_by_gate.items()))
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

    def chk(df, ma=600, path=None, cols=None):
        a = age_seconds(df, cols=cols, fallback_path=path)
        if a is None:
            return "dot-off"
        return "dot-ok" if a <= ma else "dot-warn" if a <= 1800 else "dot-err"

    whales_df = load("whales")
    signals_growing_dot = "dot-off"
    if health_df is not None and not health_df.empty and "signals_growing" in health_df.columns:
        sg = str(health_df.iloc[-1].get("signals_growing", "")).strip().lower()
        if sg in {"yes", "true", "1", "y"}:
            signals_growing_dot = "dot-ok"
        elif sg in {"no", "false", "0", "n"}:
            signals_growing_dot = "dot-warn"
    elif signals_df is not None and len(signals_df) > 0:
        signals_growing_dot = "dot-ok"

    items = [
        ("Market monitor", chk(markets_df, path=FILES["markets"])),
        ("Whale tracker", chk(whales_df, path=FILES["whales"])),
        ("Signal engine", chk(signals_df, path=FILES["signals"])),
        ("Order simulation", chk(positions_df, path=FILES["positions"])),
        ("Model status", chk(model_status_df, 86400, path=FILES["model_status"], cols=["last_retrained_at", "timestamp", "updated_at", "created_at"])),
        ("System health", chk(health_df, path=FILES["health"])),
        ("Replay available", "dot-ok" if replay_df is not None and not replay_df.empty else "dot-off"),
        ("Signals growing", signals_growing_dot),
    ]
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
    entry_rule = EntryRuleLayer()
    sc = [c for c in ["edge_score", "p_tp_before_sl", "confidence"] if c in rk.columns]
    if sc:
        rk = rk.sort_values(by=sc, ascending=[False] * len(sc))
    rk = rk.head(20)
    om = set(positions_df["market"].dropna().astype(str)) if not positions_df.empty and "market" in positions_df.columns else set()
    gs = {"Enter": [], "Hold": [], "Exit / Leave": [], "Watch": []}
    for _, r in rk.iterrows():
        m = r.get("market_title", r.get("market", "Unknown"))
        cf = pd.to_numeric(pd.Series([r.get("confidence")]), errors="coerce").iloc[0]
        entry_intent = str(r.get("entry_intent", "OPEN_LONG") or "OPEN_LONG").upper()
        rec_action = str(r.get("recommended_action", r.get("action", "")) or "").upper()
        ao = m in om
        if entry_intent == "CLOSE_LONG":
            g = "Exit / Leave"
        elif ao and pd.notna(cf) and cf < 0.50:
            g = "Exit / Leave"
        elif ao:
            g = "Hold"
        elif rec_action in {"SMALL_BUY", "LARGE_BUY", "BUY", "ENTER"}:
            g = "Enter"
        elif rec_action in {"IGNORE", "HOLD", "WATCH"}:
            g = "Watch"
        elif entry_rule.should_enter(r.to_dict()):
            # Fallback path for historical rows that don't carry explicit runtime action fields.
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


def _position_key_from_row(row):
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    position_key = str(getter("position_key", "") or "").strip()
    if position_key:
        return position_key
    token_id = str(getter("token_id", "") or "").strip()
    condition_id = str(getter("condition_id", "") or "").strip()
    outcome_side = str(getter("outcome_side", "") or "").strip()
    if token_id or condition_id:
        return f"{token_id}|{condition_id}|{outcome_side}"
    position_id = str(getter("position_id", "") or "").strip()
    if position_id:
        return position_id
    market = str(getter("market", "") or getter("market_title", "") or "").strip()
    return f"{market}|{outcome_side}"


def _prepare_position_history(position_history_df):
    if position_history_df is None or position_history_df.empty:
        return pd.DataFrame()
    df = normalize_dataframe_columns(position_history_df.copy())
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df[df["timestamp"].notna()].copy()
    if "position_key" not in df.columns:
        df["position_key"] = df.apply(_position_key_from_row, axis=1)
    if "mark_price" not in df.columns and "current_price" in df.columns:
        df["mark_price"] = df["current_price"]
    for col in ["mark_price", "current_price", "entry_price", "unrealized_pnl", "market_value", "shares", "drawdown_from_peak", "recent_return_3", "runup_from_entry"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["position_key", "timestamp"])


def _prepare_portfolio_curve(portfolio_curve_df, position_history_df):
    if portfolio_curve_df is not None and not portfolio_curve_df.empty:
        df = normalize_dataframe_columns(portfolio_curve_df.copy())
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df[df["timestamp"].notna()].copy()
        for col in ["unrealized_pnl", "realized_pnl", "total_pnl", "gross_market_value", "entry_notional", "open_positions"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("timestamp")

    if position_history_df is None or position_history_df.empty or "timestamp" not in position_history_df.columns:
        return pd.DataFrame()
    agg = (
        position_history_df.groupby("timestamp", as_index=False)
        .agg(
            open_positions=("position_key", "nunique"),
            gross_market_value=("market_value", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        .sort_values("timestamp")
    )
    agg["realized_pnl"] = 0.0
    agg["entry_notional"] = np.nan
    agg["total_pnl"] = agg["unrealized_pnl"]
    return agg


def render_positions(positions_df, closed_df, position_history_df=None, portfolio_curve_df=None):
    st.markdown('<div class="sec-title">Open Position Workstation</div>', unsafe_allow_html=True)
    if positions_df is None or positions_df.empty:
        st.info("No open positions.")
    else:
        pos = normalize_dataframe_columns(positions_df.copy())
        pos["position_key"] = pos.apply(_position_key_from_row, axis=1)
        for col in ["entry_price", "current_price", "mark_price", "shares", "market_value", "unrealized_pnl", "confidence", "drawdown_from_peak", "recent_return_3", "runup_from_entry"]:
            if col in pos.columns:
                pos[col] = pd.to_numeric(pos[col], errors="coerce")

        hist = _prepare_position_history(position_history_df)
        curve = _prepare_portfolio_curve(portfolio_curve_df, hist)

        if not curve.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=curve["timestamp"],
                    y=curve.get("unrealized_pnl", pd.Series(index=curve.index, dtype=float)),
                    name="Unrealized PnL",
                    mode="lines",
                    line=dict(color="#22c55e", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(34,197,94,0.12)",
                ),
                secondary_y=False,
            )
            if "gross_market_value" in curve.columns:
                fig.add_trace(
                    go.Scatter(
                        x=curve["timestamp"],
                        y=curve["gross_market_value"],
                        name="Gross Market Value",
                        mode="lines",
                        line=dict(color="#06b6d4", width=1.8),
                    ),
                    secondary_y=True,
                )
            fig.update_layout(
                title="Live Equity Curve",
                margin=dict(l=20, r=20, t=48, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            fig.update_yaxes(title_text="Unrealized PnL", secondary_y=False)
            fig.update_yaxes(title_text="Market Value", secondary_y=True)
            st.plotly_chart(sfig(fig, 320), use_container_width=True)
        else:
            st.caption("Live equity curve will appear after position telemetry snapshots accumulate.")

        card_cols = st.columns(2 if len(pos) > 1 else 1)
        for idx, (_, row) in enumerate(pos.sort_values("unrealized_pnl", ascending=False).iterrows()):
            market_name = str(row.get("market", row.get("market_title", row.get("token_id", "Unknown"))) or "Unknown")
            side = str(row.get("outcome_side", "N/A") or "N/A")
            position_key = str(row.get("position_key", "") or "")
            row_hist = hist[hist["position_key"].astype(str) == position_key].copy() if not hist.empty else pd.DataFrame()
            with card_cols[idx % len(card_cols)]:
                st.markdown(f"**{market_name}**")
                st.caption(f"{side} | key `{position_key[:24]}`")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Entry", fmt_num(row.get("entry_price"), 4))
                m2.metric("Mark", fmt_num(row.get("mark_price", row.get("current_price")), 4))
                m3.metric("Unrealized", fmt_money(row.get("unrealized_pnl")))
                m4.metric("Shares", fmt_num(row.get("shares"), 4))
                s1, s2, s3 = st.columns(3)
                s1.metric("Run-Up", fmt_pct(row.get("runup_from_entry")))
                s2.metric("Drawdown", fmt_pct(row.get("drawdown_from_peak")))
                s3.metric("3-Tick Return", fmt_pct(row.get("recent_return_3")))
                st.caption(f"Trajectory: {row.get('trajectory_state', 'N/A')} | Mark source: {row.get('mark_source', 'N/A')}")

                if not row_hist.empty and "timestamp" in row_hist.columns:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                        go.Scatter(
                            x=row_hist["timestamp"],
                            y=row_hist["mark_price"],
                            mode="lines",
                            name="Price",
                            line=dict(color="#3b82f6", width=2.3),
                        ),
                        secondary_y=False,
                    )
                    entry_price = to_float(row.get("entry_price"), default=None)
                    if entry_price is not None:
                        fig.add_hline(y=entry_price, line_dash="dash", line_color="#f59e0b", opacity=0.9)
                    fig.add_trace(
                        go.Scatter(
                            x=row_hist["timestamp"],
                            y=row_hist["unrealized_pnl"],
                            mode="lines",
                            name="Unrealized PnL",
                            line=dict(color="#22c55e", width=1.8),
                            fill="tozeroy",
                            fillcolor="rgba(34,197,94,0.10)",
                        ),
                        secondary_y=True,
                    )
                    fig.update_layout(
                        title=f"{market_name} Live Path",
                        margin=dict(l=18, r=18, t=44, b=18),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    )
                    fig.update_yaxes(title_text="Price", secondary_y=False)
                    fig.update_yaxes(title_text="Unrealized PnL", secondary_y=True)
                    st.plotly_chart(sfig(fig, 300), use_container_width=True)
                else:
                    st.info("Waiting for live trajectory snapshots for this position.")

        with st.expander("Position Blotter"):
            cs = [
                c for c in [
                    "market", "outcome_side", "entry_price", "mark_price", "current_price", "shares",
                    "market_value", "unrealized_pnl", "trajectory_state", "drawdown_from_peak",
                    "recent_return_3", "confidence", "status"
                ] if c in pos.columns
            ]
            st.dataframe(ensure_safe(pos[cs] if cs else pos), use_container_width=True, hide_index=True)

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


def render_models(msd, sup, tsd, s2d, wfd, rpd, bsd, rgd):
    st.markdown('<div class="sec-title">Model / Learning Status</div>', unsafe_allow_html=True)
    test_acc, test_src, test_col = resolve_numeric_metric([
        ("time_split_eval.csv", tsd, "test_accuracy"),
        ("walk_forward_eval.csv", wfd, "accuracy"),
        ("supervised_eval.csv", sup, "accuracy"),
        ("stage2_temporal_eval.csv", s2d, "temporal_walk_forward_accuracy"),
    ])
    val_acc, val_src, val_col = resolve_numeric_metric([
        ("time_split_eval.csv", tsd, "val_accuracy"),
        ("supervised_eval.csv", sup, "accuracy"),
    ])
    sharpe, sharpe_src, sharpe_col = resolve_numeric_metric([
        ("supervised_eval.csv", sup, "sharpe"),
        ("backtest_summary.csv", bsd, "sharpe_like"),
    ])
    profit_factor, pf_src, pf_col = resolve_numeric_metric([
        ("backtest_summary.csv", bsd, "profit_factor"),
    ])
    replay_trades, replay_src, replay_col = resolve_numeric_metric([
        ("backtest_summary.csv", bsd, "trades"),
    ])
    if replay_trades is None and rpd is not None and not rpd.empty:
        replay_trades = float(len(rpd))
        replay_src = "path_replay_backtest.csv"
        replay_col = "rows"

    registry_row, malformed_registry_rows = resolve_registry_snapshot(rgd)
    retrain_status = resolve_retrainer_status(msd)
    champion = str(registry_row.get("model_version") or "N/A")
    champion_src = "model_registry.csv" if registry_row else None
    last_train_raw = retrain_status.get("last_retrained_at") or registry_row.get("promoted_at")
    if not last_train_raw:
        weight_times = [t for t in [path_mtime(WEIGHTS / "ppo_polytrader.zip"), path_mtime(WEIGHTS / "ppo_polytrader_vecnormalize.pkl")] if t is not None]
        last_train_raw = max(weight_times) if weight_times else None
    last_train = fmt_ts(last_train_raw)
    last_train_src = "model_status.csv" if retrain_status.get("last_retrained_at") else ("model_registry.csv" if registry_row.get("promoted_at") else "weights")

    missing = []
    if not (LOGS / "contract_targets.csv").exists():
        missing.append("contract targets")
    if not (LOGS / "clob_price_history.csv").exists():
        missing.append("CLOB history")
    if replay_trades is None:
        missing.append("replay performance")
    if test_acc is None and val_acc is None:
        missing.append("accuracy artifacts")
    if sharpe is None:
        missing.append("risk-adjusted performance")
    if not registry_row:
        missing.append("champion registry")
    if missing:
        st.warning("Missing or unresolved: " + "; ".join(missing))
    elif malformed_registry_rows:
        st.info(f"Registry audit: ignored {malformed_registry_rows} malformed legacy registry row(s) and used the last valid champion snapshot.")

    st.write(f"**Weights:** {'current' if (WEIGHTS / 'ppo_polytrader.zip').exists() else 'missing'}")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Test Acc", fmt_num(test_acc, 3) if test_acc is not None else "N/A")
    c2.metric("Val Acc", fmt_num(val_acc, 3) if val_acc is not None else "N/A")
    c3.metric("Replay Trades", str(int(replay_trades)) if replay_trades is not None else "N/A")
    c4.metric("Sharpe", fmt_num(sharpe, 3) if sharpe is not None else "N/A")
    c5.metric("Profit Factor", fmt_num(profit_factor, 3) if profit_factor is not None else "N/A")
    c6.metric("Champion", champion)
    c7.metric("Last Train", last_train)

    resolved_bits = []
    if test_src:
        resolved_bits.append(f"test <- {test_src}:{test_col}")
    if val_src:
        resolved_bits.append(f"validation <- {val_src}:{val_col}")
    if sharpe_src:
        resolved_bits.append(f"sharpe <- {sharpe_src}:{sharpe_col}")
    if pf_src:
        resolved_bits.append(f"profit_factor <- {pf_src}:{pf_col}")
    if last_train_src:
        resolved_bits.append(f"last_train <- {last_train_src}")
    if resolved_bits:
        st.caption("Resolved sources: " + " | ".join(resolved_bits))

    if retrain_status:
        pr = retrain_status["progress_ratio"]
        st.progress(pr)
        if retrain_status.get("new_closed") is not None:
            parts = [f"New closed: {retrain_status['new_closed']}/{retrain_status['closed_threshold'] or 0}"]
            if retrain_status.get("replay_threshold"):
                parts.append(f"New replay: {retrain_status.get('new_replay', 0)}/{retrain_status['replay_threshold']}")
        else:
            parts = [f"Closed total: {retrain_status['closed_total']}"]
            if retrain_status.get("closed_threshold"):
                parts[0] += f"/{retrain_status['closed_threshold']}"
        parts.append(f"Progress: {pr:.0%}")
        if retrain_status.get("cooldown_minutes"):
            parts.append(f"Cooldown: {retrain_status['cooldown_minutes']}m")
        if retrain_status.get("last_retrained_at"):
            parts.append(f"Last retrain: {fmt_ts(retrain_status['last_retrained_at'])}")
        st.caption(" | ".join(parts))
        st.code(retrain_status.get("last_action", "Unknown"), language="text")

    audit_rows = [
        {
            "field": "Test accuracy",
            "value": fmt_num(test_acc, 3) if test_acc is not None else "N/A",
            "resolved_source": test_src or "missing",
            "status": "ok" if test_src else "missing",
        },
        {
            "field": "Validation accuracy",
            "value": fmt_num(val_acc, 3) if val_acc is not None else "N/A",
            "resolved_source": val_src or "missing",
            "status": "ok" if val_src else "missing",
        },
        {
            "field": "Sharpe / risk-adjusted score",
            "value": fmt_num(sharpe, 3) if sharpe is not None else "N/A",
            "resolved_source": sharpe_src or "missing",
            "status": "ok" if sharpe_src else "missing",
        },
        {
            "field": "Profit factor",
            "value": fmt_num(profit_factor, 3) if profit_factor is not None else "N/A",
            "resolved_source": pf_src or "missing",
            "status": "ok" if pf_src else "missing",
        },
        {
            "field": "Replay trades",
            "value": str(int(replay_trades)) if replay_trades is not None else "N/A",
            "resolved_source": replay_src or "missing",
            "status": "ok" if replay_src else "missing",
        },
        {
            "field": "Champion version",
            "value": champion,
            "resolved_source": champion_src or "missing",
            "status": "ok" if champion_src else "missing",
        },
        {
            "field": "Last retrain timestamp",
            "value": last_train,
            "resolved_source": last_train_src or "missing",
            "status": "ok" if last_train != "N/A" else "missing",
        },
    ]
    st.dataframe(ensure_safe(pd.DataFrame(audit_rows)), use_container_width=True, hide_index=True)

    artifact_specs = [
        ("supervised_eval.csv", FILES["supervised_eval"], sup, "Evaluator / fallback aggregator", "dashboard accuracy + sharpe"),
        ("time_split_eval.csv", FILES["time_split"], tsd, "TimeSplitTrainer", "dashboard test/val accuracy"),
        ("stage2_temporal_eval.csv", FILES["stage2_temporal"], s2d, "Stage2TemporalModels", "dashboard fallback temporal metrics"),
        ("walk_forward_eval.csv", FILES["walk_forward"], wfd, "WalkForwardEvaluator", "dashboard fallback walk-forward accuracy"),
        ("backtest_summary.csv", FILES["backtest_summary"], bsd, "Backtester", "dashboard replay trades + sharpe + profit factor"),
        ("model_status.csv", FILES["model_status"], msd, "Retrainer / runtime patch", "dashboard retrain progress + last train"),
        ("model_registry.csv", FILES["registry"], rgd, "Retrainer promotion registry", "dashboard champion version"),
    ]
    artifact_rows = []
    for label, path, df, producer, consumer in artifact_specs:
        file_ts = path_mtime(path)
        latest_row_ts = latest_ts(df)
        best_ts = max([t for t in [file_ts, latest_row_ts] if t is not None], default=None)
        artifact_rows.append(
            {
                "artifact": label,
                "present": "Yes" if Path(path).exists() else "No",
                "rows": len(df) if df is not None and not df.empty else 0,
                "latest_seen": best_ts.strftime("%Y-%m-%d %H:%M:%S UTC") if best_ts is not None else "N/A",
                "producer": producer,
                "consumed_by": consumer,
            }
        )
    with st.expander("Model Artifact Audit"):
        st.dataframe(ensure_safe(pd.DataFrame(artifact_rows)), use_container_width=True, hide_index=True)


def render_shadow(shadow_df):
    st.markdown('<div class="sec-title">Shadow Execution</div>', unsafe_allow_html=True)
    st.caption("Shadow intents, slippage tax, DOA vetoes, and realized post-signal outcomes.")
    if shadow_df.empty:
        bundle_candidates = sorted((WEIGHTS).glob("meta_model_bundle_*.pkl"))
        raw_candidates_path = LOGS / "raw_candidates.csv"
        signals_path = FILES["signals"]
        diagnostic_rows = [
            {
                "artifact": "shadow_results.csv",
                "present": "Yes" if FILES["shadow"].exists() else "No",
                "latest_seen": fmt_ts(path_mtime(FILES["shadow"])),
                "note": "Shadow audit log consumed by this tab",
            },
            {
                "artifact": "meta_model_bundle_*.pkl",
                "present": "Yes" if bundle_candidates else "No",
                "latest_seen": fmt_ts(max([path_mtime(p) for p in bundle_candidates], default=None)),
                "note": "Shadow meta-model used for intent scoring",
            },
            {
                "artifact": "raw_candidates.csv",
                "present": "Yes" if raw_candidates_path.exists() else "No",
                "latest_seen": fmt_ts(path_mtime(raw_candidates_path)),
                "note": "Upstream feature source for shadow intent rows",
            },
            {
                "artifact": "signals.csv",
                "present": "Yes" if signals_path.exists() else "No",
                "latest_seen": fmt_ts(path_mtime(signals_path)),
                "note": "Signal ranking feed",
            },
        ]
        st.warning("Shadow audit log is empty. The producer was not writing rows yet, or it is still warming up.")
        st.dataframe(ensure_safe(pd.DataFrame(diagnostic_rows)), use_container_width=True, hide_index=True)
        st.caption(
            "Shadow logging now imputes missing meta-model features, skips duplicate intents, and resolves pending rows each cycle. "
            "Refresh after the supervisor runs another cycle."
        )
        return
    res = shadow_df[shadow_df.get("outcome", pd.Series()) != "PENDING"] if "outcome" in shadow_df.columns else pd.DataFrame()
    doa = shadow_df[shadow_df.get("outcome", pd.Series()) == "DOA"] if "outcome" in shadow_df.columns else pd.DataFrame()
    blocked = shadow_df[shadow_df.get("outcome", pd.Series()) == "FEATURE_BLOCKED"] if "outcome" in shadow_df.columns else pd.DataFrame()
    tp = float((res["outcome"] == "TP").mean()) if not res.empty and "outcome" in res.columns else 0.0
    asl = float(pd.to_numeric(shadow_df.get("entry_slippage_bps"), errors="coerce").dropna().mean()) if "entry_slippage_bps" in shadow_df.columns else 0.0
    aev = float(pd.to_numeric(shadow_df.get("ev_adj"), errors="coerce").dropna().mean()) if "ev_adj" in shadow_df.columns else 0.0
    amp = float(pd.to_numeric(shadow_df.get("meta_prob"), errors="coerce").dropna().mean()) if "meta_prob" in shadow_df.columns else 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Shadow Intents", len(shadow_df))
    c2.metric("Resolved", len(res))
    c3.metric("DOA / Vetoed", len(doa))
    c4.metric("TP Rate", fmt_pct(tp))
    c5.metric("Feature Blocked", len(blocked))
    c6, c7, c8 = st.columns(3)
    c6.metric("Avg Slippage (bps)", f"{0.0 if pd.isna(asl) else asl:.1f}") # BUG FIX 10
    c7.metric("Avg EV_adj", f"{0.0 if pd.isna(aev) else aev:+.2%}")
    c8.metric("Avg Meta Prob", f"{0.0 if pd.isna(amp) else amp:.2%}")
    if "outcome" in shadow_df.columns:
        oc = shadow_df["outcome"].value_counts().reset_index()
        oc.columns = ["Outcome", "Count"]
        fig = px.bar(oc, x="Outcome", y="Count", color_discrete_sequence=COLORS)
        st.plotly_chart(sfig(fig, 260).update_layout(title="Shadow Outcome Mix"), use_container_width=True)
    cs = [c for c in ["timestamp", "market_title", "meta_prob", "entry_slippage_bps", "expected_slip_bps", "ev_adj", "outcome", "feature_status", "imputed_feature_count", "missing_feature_count", "realized_return", "trades_in_window"] if c in shadow_df.columns]
    st.dataframe(ensure_safe(shadow_df[cs].tail(100)), use_container_width=True, hide_index=True)


def render_quality(sdf, tdf, mdf, wdf, adf, msd, pdf, cdf, rpd, hdf):
    st.markdown('<div class="sec-title">Data Quality and Pipeline Readiness</div>', unsafe_allow_html=True)
    frames = {
        "signals": sdf,
        "execution": tdf,
        "markets": mdf,
        "whales": wdf,
        "alerts": adf,
        "positions": pdf,
        "closed": cdf,
        "model_status": msd,
        "health": hdf,
        "replay": rpd,
        "supervised_eval": load("supervised_eval"),
        "time_split": load("time_split"),
        "stage2_temporal": load("stage2_temporal"),
        "walk_forward": load("walk_forward"),
        "backtest_summary": load("backtest_summary"),
    }
    rows = []
    for n, df in frames.items():
        p = FILES.get(n)
        lt = latest_ts(df)
        rows.append({"file": n, "present": "Yes" if p and p.exists() else "No", "rows": len(df) if df is not None and not df.empty else 0, "latest": lt.strftime('%H:%M:%S') if lt else "N/A", "schema": "ok" if df is not None and not df.empty else "empty"})
    st.dataframe(ensure_safe(pd.DataFrame(rows)), use_container_width=True, hide_index=True)
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
        st.dataframe(ensure_safe(edf), use_container_width=True)
    with tabs[3]:
        st.dataframe(ensure_safe(mdf), use_container_width=True)
    with tabs[4]:
        st.dataframe(ensure_safe(wdf), use_container_width=True)
    with tabs[5]:
        st.dataframe(ensure_safe(adf), use_container_width=True)
    with tabs[6]:
        st.dataframe(ensure_safe(msd), use_container_width=True)
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
    if is_live_mode():
        live_pdf = load_live_positions_from_db()
        if not live_pdf.empty:
            if pdf.empty:
                pdf = live_pdf
            else:
                merge_keys = [c for c in ["position_id", "token_id", "condition_id", "outcome_side"] if c in live_pdf.columns and c in pdf.columns]
                if merge_keys:
                    merged = live_pdf.merge(pdf, on=merge_keys, how="left", suffixes=("", "_csv"))
                    for col in ["market", "market_title", "current_price", "unrealized_pnl", "confidence", "signal_label", "status"]:
                        csv_col = f"{col}_csv"
                        if col not in merged.columns and csv_col in merged.columns:
                            merged[col] = merged[csv_col]
                        elif col in merged.columns and csv_col in merged.columns:
                            merged[col] = merged[col].where(merged[col].notna(), merged[csv_col])
                    pdf = normalize_dataframe_columns(merged)
                else:
                    pdf = pd.concat([live_pdf, pdf], ignore_index=True) # BUG FIX 4: Append instead of wiping paper book
    cdf = load("closed")
    sup = load("supervised_eval")
    tsd = load("time_split")
    s2d = load("stage2_temporal")
    wfd = load("walk_forward")
    rpd = load("replay")
    bsd = load("backtest_summary")
    wbt = load("wallet_backtest")
    rgd = load("registry")
    shd = load("shadow")
    phd = load("position_snapshots")
    pcd = load("portfolio_curve")

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
        render_positions(pdf, cdf, phd, pcd)
        render_best_trades(cdf, rpd)
        if not edf.empty:
            st.markdown('<div class="sec-title">Episode Log</div>', unsafe_allow_html=True)
            st.dataframe(ensure_safe(edf.tail(20)), use_container_width=True)

    with t4:
        s1, s2, s3 = st.tabs(["Markets", "Whale Activity", "Alerts"])
        with s1:
            render_markets(mdf)
            if not ddf.empty:
                st.markdown('<div class="sec-title">Whale Market Distribution</div>', unsafe_allow_html=True)
                st.dataframe(ensure_safe(ddf.head(15)), use_container_width=True)
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
            render_models(msd, sup, tsd, s2d, wfd, rpd, bsd, rgd)
        with sq:
            render_quality(sdf, tdf, mdf, wdf, adf, msd, pdf, cdf, rpd, hdf)
            if dbg:
                with st.expander("Debug / Raw Logs"):
                    render_raw(sdf, tdf, edf, mdf, wdf, adf, msd, pdf, cdf)

    with t6:
        render_shadow(shd)


if __name__ == "__main__":
    main()
