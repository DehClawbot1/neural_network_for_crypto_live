import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from log_loader import load_execution_history as shared_load_execution_history

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
SIGNALS_FILE = LOGS_DIR / "signals.csv"
EXECUTION_FILE = LOGS_DIR / "execution_log.csv"
LEGACY_EXECUTION_FILE = LOGS_DIR / "daily_summary.txt"
EPISODE_LOG_FILE = LOGS_DIR / "episode_log.csv"
MARKETS_FILE = LOGS_DIR / "markets.csv"
WHALES_FILE = LOGS_DIR / "whales.csv"
ALERTS_FILE = LOGS_DIR / "alerts.csv"
MODEL_STATUS_FILE = LOGS_DIR / "model_status.csv"
WEIGHTS_FILE = BASE_DIR / "weights" / "ppo_polytrader.zip"
TRADER_ANALYTICS_FILE = LOGS_DIR / "trader_analytics.csv"
BACKTEST_FILE = LOGS_DIR / "backtest_summary.csv"
DATASET_FILE = LOGS_DIR / "historical_dataset.csv"
POSITIONS_FILE = LOGS_DIR / "positions.csv"
CLOSED_POSITIONS_FILE = LOGS_DIR / "closed_positions.csv"
MARKET_DISTRIBUTION_FILE = LOGS_DIR / "market_distribution.csv"
SUPERVISED_EVAL_FILE = LOGS_DIR / "supervised_eval.csv"
TIME_SPLIT_EVAL_FILE = LOGS_DIR / "time_split_eval.csv"
PATH_REPLAY_FILE = LOGS_DIR / "path_replay_backtest.csv"
BACKTEST_BY_WALLET_FILE = LOGS_DIR / "backtest_by_wallet.csv"
MODEL_REGISTRY_FILE = BASE_DIR / "weights" / "model_registry.csv"

st.set_page_config(page_title="Neural Network for Crypto", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def _cached_csv_read(path_str, mtime):
    try:
        return pd.read_csv(path_str, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def load_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return _cached_csv_read(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _cached_execution_history(logs_dir_str, execution_mtime, legacy_mtime):
    return shared_load_execution_history(logs_dir_str)


def load_execution_history():
    execution_path = LOGS_DIR / "execution_log.csv"
    legacy_path = LEGACY_EXECUTION_FILE
    execution_mtime = execution_path.stat().st_mtime if execution_path.exists() else 0
    legacy_mtime = legacy_path.stat().st_mtime if legacy_path.exists() else 0
    return _cached_execution_history(str(LOGS_DIR), execution_mtime, legacy_mtime)


def apply_dashboard_filters(df, market_search="", wallet_search="", min_confidence=0.0, signal_label="All", position_status="All", side_filter="All", min_edge_score=None, only_open_positions=False, only_actionable=False, time_range_hours=None):
    if df is None or df.empty:
        return df
    out = df.copy()
    if market_search:
        market_col = "market_title" if "market_title" in out.columns else "market" if "market" in out.columns else None
        if market_col:
            out = out[out[market_col].astype(str).str.contains(market_search, case=False, na=False)]
    if wallet_search:
        wallet_col = "trader_wallet" if "trader_wallet" in out.columns else "wallet_copied" if "wallet_copied" in out.columns else None
        if wallet_col:
            out = out[out[wallet_col].astype(str).str.contains(wallet_search, case=False, na=False)]
    if "confidence" in out.columns:
        out = out[pd.to_numeric(out["confidence"], errors="coerce").fillna(0) >= float(min_confidence)]
    if min_edge_score is not None and "edge_score" in out.columns:
        out = out[pd.to_numeric(out["edge_score"], errors="coerce").fillna(0) >= float(min_edge_score)]
    if signal_label != "All" and "signal_label" in out.columns:
        out = out[out["signal_label"].astype(str) == signal_label]
    if position_status != "All" and "status" in out.columns:
        out = out[out["status"].astype(str) == position_status]
    if side_filter != "All":
        side_col = "outcome_side" if "outcome_side" in out.columns else "side" if "side" in out.columns else None
        if side_col:
            if side_filter == "unknown":
                out = out[out[side_col].isna() | ~out[side_col].astype(str).str.upper().isin(["YES", "NO"])]
            else:
                out = out[out[side_col].astype(str).str.upper() == side_filter]
    if only_open_positions and "status" in out.columns:
        out = out[out["status"].astype(str).str.upper() == "OPEN"]
    if only_actionable and "signal_label" in out.columns:
        out = out[~out["signal_label"].astype(str).str.upper().isin(["IGNORE", "NO_ACTION"])]
    if time_range_hours is not None:
        for col in ["timestamp", "created_at", "updated_at", "opened_at"]:
            if col in out.columns:
                ts = pd.to_datetime(out[col], errors="coerce", utc=True)
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=float(time_range_hours))
                out = out[ts >= cutoff]
                break
    return out


def inject_styles():
    st.markdown(
        """
        <style>
            .main { background-color: #0e1117; }
            .hero-box {
                padding: 1.2rem 1.4rem;
                border-radius: 18px;
                background: linear-gradient(135deg, #121826 0%, #0f172a 100%);
                border: 1px solid rgba(255,255,255,0.08);
                margin-bottom: 1rem;
            }
            .market-card {
                background: linear-gradient(180deg, #121826 0%, #111827 100%);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 20px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 8px 20px rgba(0,0,0,0.18);
            }
            .market-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 0.65rem;
                line-height: 1.35;
            }
            .signal-badge {
                display: inline-block;
                padding: 0.35rem 0.65rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
                margin-right: 0.35rem;
                margin-bottom: 0.45rem;
            }
            .badge-watch { background: rgba(245, 158, 11, 0.16); color: #fbbf24; }
            .badge-strong { background: rgba(16, 185, 129, 0.16); color: #34d399; }
            .badge-top { background: rgba(59, 130, 246, 0.16); color: #60a5fa; }
            .badge-ignore { background: rgba(148, 163, 184, 0.16); color: #cbd5e1; }
            .meta-line {
                color: #cbd5e1;
                font-size: 0.92rem;
                margin-bottom: 0.3rem;
            }
            .reason-box {
                margin-top: 0.7rem;
                color: #94a3b8;
                font-size: 0.88rem;
                padding: 0.7rem 0.8rem;
                border-radius: 14px;
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.05);
            }
            .section-title {
                font-size: 1.1rem;
                font-weight: 700;
                margin-top: 0.5rem;
                margin-bottom: 0.9rem;
                color: #f8fafc;
            }
            .overview-card {
                background: linear-gradient(180deg, #121826 0%, #111827 100%);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                padding: 0.9rem 1rem;
                margin-bottom: 1rem;
            }
            .live-dot {
                display:inline-block;
                width:10px;
                height:10px;
                background:#22c55e;
                border-radius:50%;
                margin-right:8px;
                box-shadow:0 0 10px rgba(34,197,94,0.8);
            }
            .confidence-bar-wrap {
                width:100%;
                height:10px;
                background:rgba(255,255,255,0.08);
                border-radius:999px;
                overflow:hidden;
                margin-top:0.45rem;
                margin-bottom:0.2rem;
            }
            .confidence-bar-fill {
                height:100%;
                border-radius:999px;
                background:linear-gradient(90deg, #3b82f6 0%, #10b981 100%);
            }
            .small-muted {
                color:#94a3b8;
                font-size:0.85rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_data_freshness(*paths):
    timestamps = []
    for path in paths:
        p = Path(path)
        if p.exists():
            timestamps.append(datetime.fromtimestamp(p.stat().st_mtime))
    if not timestamps:
        return "unknown"
    return max(timestamps).strftime('%Y-%m-%d %H:%M:%S')


def _latest_timestamp_from_df(df):
    if df is None or df.empty:
        return None
    for col in ["timestamp", "created_at", "updated_at", "logged_at", "closed_at", "opened_at"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce", utc=True).dropna()
            if not ts.empty:
                return ts.max()
    return None


def _freshness_status(age_seconds):
    if age_seconds is None:
        return "missing", "⚪"
    if age_seconds < 120:
        return "fresh", "🟢"
    if age_seconds <= 600:
        return "delayed", "🟡"
    return "stale", "🔴"


def render_data_freshness_panel(source_frames):
    st.markdown('<div class="section-title">Data Freshness</div>', unsafe_allow_html=True)
    now = pd.Timestamp.utcnow()
    rows = []
    for label, path, df in source_frames:
        path = Path(path)
        file_modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC") if path.exists() else None
        latest_row_ts = _latest_timestamp_from_df(df)
        latest_ts = max([ts for ts in [file_modified, latest_row_ts] if ts is not None], default=None)
        age_seconds = int((now - latest_ts).total_seconds()) if latest_ts is not None else None
        status, badge = _freshness_status(age_seconds)
        rows.append(
            {
                "source": label,
                "latest_row_timestamp": latest_row_ts.strftime('%Y-%m-%d %H:%M:%S') if latest_row_ts is not None else "N/A",
                "file_modified_timestamp": file_modified.strftime('%Y-%m-%d %H:%M:%S') if file_modified is not None else "N/A",
                "age": f"{age_seconds}s" if age_seconds is not None and age_seconds < 120 else (f"{round(age_seconds / 60, 1)}m" if age_seconds is not None else "N/A"),
                "status": f"{badge} {status}",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_pipeline_health_strip(signals_df, markets_df, positions_df, model_status_df, path_replay_df):
    st.markdown('<div class="section-title">Pipeline Health</div>', unsafe_allow_html=True)

    def yes_no(df, max_age_seconds=600):
        ts = _latest_timestamp_from_df(df)
        if ts is None:
            return "No"
        age = (pd.Timestamp.utcnow() - ts).total_seconds()
        return "Yes" if age <= max_age_seconds else "No"

    cards = [
        ("Market monitor", yes_no(markets_df)),
        ("Whale tracker", yes_no(load_csv(WHALES_FILE))),
        ("Signal engine", yes_no(signals_df)),
        ("Order simulation", yes_no(positions_df)),
        ("Model status", yes_no(model_status_df, max_age_seconds=1800)),
        ("Replay/backtest available", "Yes" if path_replay_df is not None and not path_replay_df.empty else "No"),
        ("Signals growing", "Yes" if signals_df is not None and len(signals_df) > 0 else "No"),
        ("Markets updated", "Yes" if yes_no(markets_df) == "Yes" else "No"),
        ("Positions changing", "Yes" if yes_no(positions_df) == "Yes" else "No"),
        ("Replay file exists", "Yes" if PATH_REPLAY_FILE.exists() else "No"),
    ]

    cols = st.columns(5)
    for idx, (label, value) in enumerate(cards):
        with cols[idx % 5]:
            st.markdown('<div class="overview-card">', unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown('</div>', unsafe_allow_html=True)


def render_attention_needed(signals_df, trades_df, alerts_df, positions_df, model_status_df, path_replay_df):
    st.markdown('<div class="section-title">Attention Needed</div>', unsafe_allow_html=True)
    warnings = []
    now = pd.Timestamp.utcnow()

    signal_ts = _latest_timestamp_from_df(signals_df)
    if signal_ts is None or (now - signal_ts).total_seconds() > 1800:
        warnings.append("No signals in last 30 min")
    positions_ts = _latest_timestamp_from_df(positions_df)
    if positions_ts is None or (now - positions_ts).total_seconds() > 600:
        warnings.append("Positions file stale")
    if not ALERTS_FILE.exists() or alerts_df is None or alerts_df.empty:
        warnings.append("Alerts file missing")
    if signals_df is None or signals_df.empty or "confidence" not in signals_df.columns:
        warnings.append("Confidence column missing")
    if EXECUTION_FILE.exists() and LEGACY_EXECUTION_FILE.exists():
        exec_rows = len(load_csv(EXECUTION_FILE))
        legacy_rows = len(load_csv(LEGACY_EXECUTION_FILE))
        if abs(exec_rows - legacy_rows) > 0:
            warnings.append("Trade source mismatch")
    if model_status_df is None or model_status_df.empty:
        warnings.append("Model outputs missing")
    if path_replay_df is None or path_replay_df.empty:
        warnings.append("No replay outputs available")

    if warnings:
        for item in warnings:
            st.warning(item)
    else:
        st.success("No immediate incidents detected.")


def render_header():
    freshness = get_data_freshness(SIGNALS_FILE, MARKETS_FILE, ALERTS_FILE)
    st.markdown(
        f"""
        <div class="hero-box">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
                <div>
                    <h1 style="margin-bottom:0.35rem;">📈 Neural Network for Crypto</h1>
                    <div style="color:#94a3b8; font-size:1rem;">
                        Real-time public-data market tracker + whale tracker + paper-trading dashboard
                    </div>
                </div>
                <div class="overview-card" style="min-width:240px; margin-bottom:0;">
                    <div><span class="live-dot"></span><strong>Data freshness</strong></div>
                    <div class="small-muted">Latest file update: {freshness}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "This interface shows ranked paper-trading opportunities, public market tracking, whale activity, and alerts only. "
        "It does not place real bets or connect to a live account."
    )


def render_overview(signals_df, trades_df, markets_df, alerts_df, positions_df, closed_positions_df):
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    now = pd.Timestamp.utcnow()
    stale_threshold_seconds = 120

    def _find_timestamp_col(df):
        for col in ["timestamp", "created_at", "updated_at", "logged_at", "closed_at", "opened_at"]:
            if col in df.columns:
                return col
        return None

    def _series_datetime(df, preferred=None):
        if df is None or df.empty:
            return pd.Series(dtype="datetime64[ns, UTC]")
        candidates = []
        if preferred:
            candidates.extend(preferred)
        candidates.extend(["timestamp", "created_at", "updated_at", "logged_at", "closed_at", "opened_at"])
        seen = set()
        for col in candidates:
            if col in seen:
                continue
            seen.add(col)
            if col in df.columns:
                series = pd.to_datetime(df[col], errors="coerce", utc=True)
                if series.notna().any():
                    return series
        return pd.Series(dtype="datetime64[ns, UTC]")

    def _last_age_seconds(df, preferred=None):
        ts = _series_datetime(df, preferred)
        if ts.empty or not ts.notna().any():
            return None
        latest = ts.dropna().max()
        age = (now - latest).total_seconds()
        return max(0, int(age))

    signal_ts = _series_datetime(signals_df)
    signals_last_60m = 0
    latest_conf_max = "-"
    if not signals_df.empty and signal_ts.notna().any():
        recent_mask = signal_ts >= (now - pd.Timedelta(minutes=60))
        signals_last_60m = int(recent_mask.fillna(False).sum())
        if "confidence" in signals_df.columns:
            try:
                conf_series = pd.to_numeric(signals_df.loc[recent_mask, "confidence"], errors="coerce").dropna()
                if conf_series.empty:
                    conf_series = pd.to_numeric(signals_df["confidence"], errors="coerce").dropna()
                if not conf_series.empty:
                    latest_conf_max = f"{float(conf_series.max()):.2f}"
            except Exception:
                pass
    elif not signals_df.empty and "confidence" in signals_df.columns:
        try:
            conf_series = pd.to_numeric(signals_df["confidence"], errors="coerce").dropna()
            if not conf_series.empty:
                latest_conf_max = f"{float(conf_series.max()):.2f}"
        except Exception:
            pass

    open_positions = len(positions_df) if positions_df is not None else 0
    unrealized_pnl = 0.0
    if not positions_df.empty and "unrealized_pnl" in positions_df.columns:
        unrealized_pnl = float(pd.to_numeric(positions_df["unrealized_pnl"], errors="coerce").fillna(0).sum())

    realized_pnl_today = 0.0
    win_rate_today = "-"
    if not closed_positions_df.empty:
        pnl_col = "net_realized_pnl" if "net_realized_pnl" in closed_positions_df.columns else "realized_pnl" if "realized_pnl" in closed_positions_df.columns else None
        closed_ts = _series_datetime(closed_positions_df, preferred=["closed_at", "timestamp", "updated_at"])
        closed_today = closed_positions_df.copy()
        if not closed_ts.empty and closed_ts.notna().any():
            today_mask = closed_ts.dt.date == now.date()
            closed_today = closed_positions_df.loc[today_mask.fillna(False)].copy()
        if pnl_col and not closed_today.empty:
            pnl_series = pd.to_numeric(closed_today[pnl_col], errors="coerce").fillna(0)
            realized_pnl_today = float(pnl_series.sum())
            if len(pnl_series) > 0:
                win_rate_today = f"{float((pnl_series > 0).mean() * 100):.1f}%"

    critical_alerts_open = 0
    if not alerts_df.empty:
        severity_col = next((c for c in ["severity", "level", "priority", "alert_level"] if c in alerts_df.columns), None)
        status_col = next((c for c in ["status", "state", "resolved"] if c in alerts_df.columns), None)
        alert_view = alerts_df.copy()
        if severity_col:
            sev = alert_view[severity_col].astype(str).str.lower()
            alert_view = alert_view[sev.str.contains("critical", na=False)]
        if status_col and not alert_view.empty:
            if status_col == "resolved":
                resolved = alert_view[status_col].astype(str).str.lower().isin(["true", "1", "yes"])
                alert_view = alert_view[~resolved]
            else:
                status = alert_view[status_col].astype(str).str.lower()
                alert_view = alert_view[~status.isin(["closed", "resolved", "done"])]
        critical_alerts_open = len(alert_view)

    markets_actively_watched = len(markets_df)
    if not markets_df.empty and "market_id" in markets_df.columns:
        markets_actively_watched = int(markets_df["market_id"].nunique())

    feed_ages = {
        "signals": _last_age_seconds(signals_df),
        "markets": _last_age_seconds(markets_df),
        "alerts": _last_age_seconds(alerts_df),
        "positions": _last_age_seconds(positions_df, preferred=["updated_at", "timestamp", "opened_at"]),
    }
    available_feed_ages = [age for age in feed_ages.values() if age is not None]
    freshness_age_max = max(available_feed_ages) if available_feed_ages else None
    stale_feeds = sum(1 for age in available_feed_ages if age > stale_threshold_seconds)

    rows = [
        ("Signals generated (last 60 min)", signals_last_60m),
        ("Open paper positions", open_positions),
        ("Realized PnL today", f"{realized_pnl_today:.2f}"),
        ("Unrealized PnL now", f"{unrealized_pnl:.2f}"),
        ("Critical alerts open", critical_alerts_open),
        ("Freshness age (max seconds)", freshness_age_max if freshness_age_max is not None else "-"),
        ("Win rate today", win_rate_today),
        ("Number of stale feeds", stale_feeds),
        ("Latest confidence max", latest_conf_max),
        ("Markets actively watched", markets_actively_watched),
    ]

    cols = st.columns(5)
    for idx, (label, value) in enumerate(rows):
        with cols[idx % 5]:
            st.markdown('<div class="overview-card">', unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown('</div>', unsafe_allow_html=True)

    freshness_bits = " | ".join(
        f"{name}: {age}s" if age is not None else f"{name}: -"
        for name, age in feed_ages.items()
    )
    st.caption(f"Feed freshness — {freshness_bits} | stale threshold: {stale_threshold_seconds}s")


def badge_class(label: str) -> str:
    label = str(label).upper()
    if "HIGHEST" in label:
        return "badge-top"
    if "STRONG" in label:
        return "badge-strong"
    if "WATCH" in label:
        return "badge-watch"
    return "badge-ignore"


def render_factor_matrix(signals_df):
    st.markdown('<div class="section-title">Signal Explanation Panel</div>', unsafe_allow_html=True)
    if signals_df.empty:
        st.info("No ranked signals available yet.")
        return

    view = signals_df.copy()
    if "confidence" in view.columns:
        view = view.sort_values(by="confidence", ascending=False)
    view = view.head(50).reset_index(drop=True)

    options = []
    for idx, row in view.iterrows():
        market = row.get("market_title", row.get("market", f"Signal {idx + 1}"))
        side = row.get("outcome_side", row.get("side", "?"))
        label = row.get("signal_label", "UNKNOWN")
        options.append(f"{idx + 1}. {market} | {side} | {label}")

    selected = st.selectbox("Select signal to explain", options)
    selected_idx = options.index(selected)
    selected_row = view.iloc[selected_idx].to_dict()

    factor_specs = [
        ("Whale Pressure", "whale_pressure", "Higher usually means stronger wallet-following pressure."),
        ("Market Structure", "market_structure_score", "Higher suggests cleaner structure / setup quality."),
        ("Volatility Risk", "volatility_risk", "Higher means more risk / instability, usually worse."),
        ("Time Decay", "time_decay_score", "Higher time decay penalty is usually worse near expiry/event."),
        ("Liquidity", "liquidity_score", "Higher usually means better execution quality."),
        ("Volume", "volume_score", "Higher usually means healthier participation / tradability."),
        ("Confidence", "confidence", "Higher means stronger model conviction."),
        ("Edge Score", "edge_score", "Higher means stronger expected advantage."),
    ]

    rows = []
    for label, col, desc in factor_specs:
        value = selected_row.get(col)
        missing = pd.isna(value)
        rows.append({
            "factor": label,
            "score": value if not missing else "N/A",
            "direction_meaning": desc,
            "missing": "⚠ missing" if missing else "",
        })

    factor_df = pd.DataFrame(rows)
    numeric_plot = factor_df.copy()
    numeric_plot["numeric_score"] = pd.to_numeric(numeric_plot["score"], errors="coerce")

    st.caption(f"Explaining: {selected_row.get('market_title', selected_row.get('market', 'Unknown Market'))}")
    fig = px.bar(numeric_plot, x="numeric_score", y="factor", orientation="h", title="Signal Factor Breakdown")
    fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(factor_df, width="stretch", hide_index=True)


def render_top_opportunities(signals_df):
    st.markdown('<div class="section-title">Top Paper-Trading Opportunities</div>', unsafe_allow_html=True)
    if signals_df.empty:
        st.warning("No ranked opportunities yet. Run supervisor.py first.")
        return

    def fmt_pct(value):
        try:
            if pd.isna(value):
                return "N/A"
            return f"{float(value) * 100:.1f}%"
        except Exception:
            return "N/A"

    def fmt_num(value, digits=4):
        try:
            if pd.isna(value):
                return "N/A"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "N/A"

    def action_badge(action):
        action = str(action).strip().lower()
        if action in ["enter", "open_long", "buy"]:
            return "🟢 ENTER"
        if action in ["hold", "watch", "monitor"]:
            return "🟡 HOLD/WATCH"
        if action in ["leave", "exit", "close", "sell"]:
            return "🔴 LEAVE/EXIT"
        return "⚪ IGNORE"

    sort_df = signals_df.copy()
    if "confidence" in sort_df.columns:
        sort_df = sort_df.sort_values(by="confidence", ascending=False)

    top_df = sort_df.head(8).reset_index(drop=True)
    cols = st.columns(2)

    for idx, (_, row) in enumerate(top_df.iterrows()):
        with cols[idx % 2]:
            label = row.get("signal_label", "UNKNOWN")
            side = row.get("outcome_side", row.get("side", "UNKNOWN"))
            market = row.get("market_title", row.get("market", "Unknown Market"))
            wallet = row.get("wallet_copied", row.get("trader_wallet", "N/A"))
            confidence = row.get("confidence")
            edge_score = row.get("edge_score")
            expected_return = row.get("expected_return")
            current_price = row.get("market_last_trade_price", row.get("current_price"))
            action = row.get("recommended_action", row.get("action", row.get("entry_intent", "ignore")))
            reason = row.get("reason_summary", row.get("reason", "N/A"))
            freshness_ts = row.get("timestamp", row.get("updated_at", "N/A"))
            market_url = row.get("market_url", row.get("url"))
            missing_scores = []
            if pd.isna(confidence):
                missing_scores.append("confidence")
            if pd.isna(edge_score):
                missing_scores.append("edge")
            if pd.isna(expected_return):
                missing_scores.append("expected return")

            conf_for_bar = 0
            try:
                conf_for_bar = max(0, min(100, int(float(confidence) * 100))) if not pd.isna(confidence) else 0
            except Exception:
                conf_for_bar = 0

            st.markdown(
                f"""
                <div class="market-card">
                    <div class="market-title">{market}</div>
                    <div>
                        <span class="signal-badge {badge_class(label)}">{label}</span>
                        <span class="signal-badge badge-ignore">Side: {side}</span>
                        <span class="signal-badge badge-watch">{action_badge(action)}</span>
                    </div>
                    <div class="small-muted">Confidence: {fmt_pct(confidence)}</div>
                    <div class="confidence-bar-wrap">
                        <div class="confidence-bar-fill" style="width:{conf_for_bar}%;"></div>
                    </div>
                    <div class="meta-line"><b>Edge score:</b> {fmt_num(edge_score)}</div>
                    <div class="meta-line"><b>Expected return:</b> {fmt_num(expected_return)}</div>
                    <div class="meta-line"><b>Wallet source:</b> {wallet}</div>
                    <div class="meta-line"><b>Current market price:</b> {fmt_num(current_price)}</div>
                    <div class="meta-line"><b>Recommended action:</b> {action_badge(action)}</div>
                    <div class="meta-line"><b>Freshness:</b> {freshness_ts if pd.notna(freshness_ts) else 'N/A'}</div>
                    <div class="reason-box">{reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if missing_scores:
                st.caption(f"⚠ Missing score data: {', '.join(missing_scores)}")
            if pd.notna(market_url) and market_url:
                st.link_button("Open market on Polymarket", market_url, width="stretch")


def render_opportunity_table(signals_df):
    st.markdown('<div class="section-title">Opportunity Ranking Table</div>', unsafe_allow_html=True)
    if signals_df is None or signals_df.empty:
        st.info("No signal candidates available yet.")
        return

    view = signals_df.copy()
    if "timestamp" in view.columns:
        ts = pd.to_datetime(view["timestamp"], errors="coerce", utc=True)
        now = pd.Timestamp.utcnow()
        view["freshness_age"] = ((now - ts).dt.total_seconds() / 60).round(1)

    rename_map = {
        "market_title": "market",
        "outcome_side": "side",
        "signal_label": "signal label",
        "confidence": "confidence",
        "p_tp_before_sl": "p_tp_before_sl",
        "edge_score": "edge score",
        "expected_return": "expected return",
        "wallet_copied": "wallet",
        "market_last_trade_price": "current price",
        "recommended_action": "action",
        "timestamp": "timestamp",
        "freshness_age": "freshness age",
    }
    display = view.rename(columns=rename_map)
    preferred = ["market", "side", "signal label", "confidence", "p_tp_before_sl", "edge score", "expected return", "wallet", "current price", "action", "timestamp", "freshness age"]
    cols = [c for c in preferred if c in display.columns]
    display = display[cols].copy()
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        "Export opportunity table CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="opportunity_ranking_table.csv",
        mime="text/csv",
    )


def render_market_tracker(markets_df):
    st.markdown('<div class="section-title">BTC Market Tracker</div>', unsafe_allow_html=True)
    st.caption("Sortable market coverage view for BTC-related markets being tracked by the system.")
    if markets_df.empty:
        st.info("No BTC market snapshots yet.")
        return

    view = markets_df.copy().tail(20)
    st.dataframe(view[[c for c in ["question", "last_trade_price", "liquidity", "volume", "url"] if c in view.columns]], width="stretch")

    if "last_trade_price" in view.columns and "question" in view.columns:
        chart_df = view.dropna(subset=["last_trade_price"]).tail(12)
        if not chart_df.empty:
            fig = px.bar(chart_df, x="last_trade_price", y="question", orientation="h", title="Current BTC Market Prices")
            fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")


def render_whale_tracker(whales_df):
    st.markdown('<div class="section-title">Whale Activity Tracker</div>', unsafe_allow_html=True)
    st.caption("Public wallet summaries showing who is most active and where concentration is forming.")
    if whales_df.empty:
        st.info("No whale summary yet.")
        return

    st.dataframe(whales_df.head(15), width="stretch")


def render_market_distribution(distribution_df):
    st.markdown('<div class="section-title">Whale Market Distribution</div>', unsafe_allow_html=True)
    if distribution_df.empty:
        st.info("No market distribution data yet.")
        return

    st.dataframe(distribution_df.head(15), width="stretch")
    if "unique_wallets" in distribution_df.columns and "market_title" in distribution_df.columns:
        chart_df = distribution_df.head(10)
        fig = px.bar(chart_df, x="unique_wallets", y="market_title", orientation="h", title="Where the watched wallets are clustering")
        fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")


def render_alerts(alerts_df):
    st.markdown('<div class="section-title">Alerts</div>', unsafe_allow_html=True)
    st.caption("Recent notable changes detected by the monitoring logic.")
    if alerts_df.empty:
        st.info("No alerts generated yet.")
        return

    st.dataframe(alerts_df.tail(20), width="stretch")


def render_simulated_decisions(positions_df, closed_positions_df):
    st.markdown('<div class="section-title">Trade Decisions</div>', unsafe_allow_html=True)
    rows = []

    if not positions_df.empty:
        for _, row in positions_df.tail(10).iterrows():
            rows.append(
                {
                    "market": row.get("market"),
                    "token_id": row.get("token_id"),
                    "status": "HOLDING",
                    "outcome_side": row.get("outcome_side", row.get("side")),
                    "entry_price": row.get("entry_price"),
                    "live_price": row.get("current_price"),
                    "shares": row.get("shares"),
                    "cost_basis_usdc": (float(row.get("shares", 0.0) or 0.0) * float(row.get("entry_price", 0.0) or 0.0)),
                    "market_value": row.get("market_value"),
                    "realized_pnl": row.get("realized_pnl", 0.0),
                    "profit_usdc": row.get("unrealized_pnl", 0.0),
                    "reason": row.get("signal_label", "paper_hold"),
                }
            )

    if not closed_positions_df.empty:
        for _, row in closed_positions_df.tail(10).iterrows():
            rows.append(
                {
                    "market": row.get("market"),
                    "token_id": row.get("token_id"),
                    "status": "CLOSED",
                    "outcome_side": row.get("outcome_side", row.get("side")),
                    "entry_price": row.get("entry_price"),
                    "live_price": row.get("current_price", row.get("exit_price")),
                    "shares": row.get("shares"),
                    "market_value": row.get("market_value"),
                    "realized_pnl": row.get("realized_pnl", row.get("net_pnl", 0.0)),
                    "profit_usdc": row.get("unrealized_pnl", row.get("net_pnl", 0.0)),
                    "reason": row.get("close_reason", row.get("exit_reason", "paper_exit")),
                }
            )

    if not rows:
        st.info("No simulated trade decisions yet.")
        return

    decisions_df = pd.DataFrame(rows)
    st.dataframe(decisions_df.sort_values(by="profit_usdc", ascending=False), width="stretch")


def render_positions_pnl_summary(positions_df, closed_positions_df):
    st.markdown('<div class="section-title">PnL Summary</div>', unsafe_allow_html=True)

    open_positions = len(positions_df) if positions_df is not None else 0
    closed_positions = len(closed_positions_df) if closed_positions_df is not None else 0
    realized_col = "net_realized_pnl" if "net_realized_pnl" in closed_positions_df.columns else "realized_pnl" if "realized_pnl" in closed_positions_df.columns else None
    unrealized_col = "unrealized_pnl" if "unrealized_pnl" in positions_df.columns else None

    realized_pnl = float(pd.to_numeric(closed_positions_df[realized_col], errors="coerce").fillna(0).sum()) if realized_col and not closed_positions_df.empty else 0.0
    unrealized_pnl = float(pd.to_numeric(positions_df[unrealized_col], errors="coerce").fillna(0).sum()) if unrealized_col and not positions_df.empty else 0.0
    win_rate = "-"
    avg_trade_return = "-"
    if realized_col and not closed_positions_df.empty:
        pnl = pd.to_numeric(closed_positions_df[realized_col], errors="coerce").fillna(0)
        if len(pnl) > 0:
            win_rate = f"{float((pnl > 0).mean() * 100):.1f}%"
            avg_trade_return = f"{float(pnl.mean()):.2f}"

    cols = st.columns(6)
    metrics = [
        ("Open Positions", open_positions),
        ("Closed Positions", closed_positions),
        ("Realized PnL", f"{realized_pnl:.2f}"),
        ("Unrealized PnL", f"{unrealized_pnl:.2f}"),
        ("Win Rate", win_rate),
        ("Average Trade Return", avg_trade_return),
    ]
    for idx, (label, value) in enumerate(metrics):
        with cols[idx]:
            st.metric(label, value)


def render_positions(positions_df, closed_positions_df):
    st.markdown('<div class="section-title">Paper Positions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Open Positions**")
        if positions_df.empty:
            st.info("No open paper positions.")
        else:
            open_view = positions_df.copy().tail(20)
            if "opened_at" in open_view.columns:
                opened_ts = pd.to_datetime(open_view["opened_at"], errors="coerce", utc=True)
                open_view["position_age"] = (((pd.Timestamp.utcnow() - opened_ts).dt.total_seconds()) / 60).round(1)
            if "confidence_at_entry" not in open_view.columns and "confidence" in open_view.columns:
                open_view["confidence_at_entry"] = open_view["confidence"]
            if "max_favorable_excursion" not in open_view.columns and "mfe" in open_view.columns:
                open_view["max_favorable_excursion"] = open_view["mfe"]
            if "max_adverse_excursion" not in open_view.columns and "mae" in open_view.columns:
                open_view["max_adverse_excursion"] = open_view["mae"]
            if "status" not in open_view.columns:
                open_view["status"] = "OPEN"
            display_cols = [c for c in ["market", "outcome_side", "entry_price", "current_price", "shares", "market_value", "unrealized_pnl", "realized_pnl", "confidence_at_entry", "position_age", "max_favorable_excursion", "max_adverse_excursion", "status"] if c in open_view.columns]
            open_view = open_view[display_cols].rename(columns={"outcome_side": "side"}).fillna("N/A")

            def _row_style(row):
                pnl = row.get("unrealized_pnl", "N/A")
                current_price = row.get("current_price", "N/A")
                if pnl == "N/A" or current_price == "N/A":
                    return ["background-color: rgba(245, 158, 11, 0.18)"] * len(row)
                try:
                    pnl_value = float(pnl)
                    if pnl_value > 0:
                        return ["background-color: rgba(34, 197, 94, 0.14)"] * len(row)
                    if pnl_value < 0:
                        return ["background-color: rgba(239, 68, 68, 0.14)"] * len(row)
                except Exception:
                    return ["background-color: rgba(245, 158, 11, 0.18)"] * len(row)
                return [""] * len(row)

            st.dataframe(open_view.style.apply(_row_style, axis=1), width="stretch")
    with c2:
        st.markdown("**Closed Positions**")
        if closed_positions_df.empty:
            st.info("No closed paper positions yet.")
        else:
            closed_view = closed_positions_df.copy().tail(20)
            if "opened_at" in closed_view.columns and "closed_at" in closed_view.columns:
                opened_ts = pd.to_datetime(closed_view["opened_at"], errors="coerce", utc=True)
                closed_ts = pd.to_datetime(closed_view["closed_at"], errors="coerce", utc=True)
                closed_view["hold_duration"] = (((closed_ts - opened_ts).dt.total_seconds()) / 60).round(1)
            if "net_pnl" not in closed_view.columns:
                if "net_realized_pnl" in closed_view.columns:
                    closed_view["net_pnl"] = closed_view["net_realized_pnl"]
                elif "realized_pnl" in closed_view.columns:
                    closed_view["net_pnl"] = closed_view["realized_pnl"]
            if "fees" not in closed_view.columns and "fees_paid" in closed_view.columns:
                closed_view["fees"] = closed_view["fees_paid"]
            if "signal_label_at_entry" not in closed_view.columns and "signal_label" in closed_view.columns:
                closed_view["signal_label_at_entry"] = closed_view["signal_label"]
            display_cols = [c for c in ["market", "outcome_side", "entry_price", "exit_price", "opened_at", "closed_at", "hold_duration", "close_reason", "fees", "net_pnl", "wallet_copied", "signal_label_at_entry"] if c in closed_view.columns]
            closed_view = closed_view[display_cols].rename(columns={"outcome_side": "side", "opened_at": "entry time", "closed_at": "exit time", "close_reason": "exit reason", "wallet_copied": "copied wallet", "net_pnl": "net PnL", "signal_label_at_entry": "signal label at entry"}).fillna("N/A")
            st.dataframe(closed_view, width="stretch", hide_index=True)


def render_paper_trades(trades_df):
    st.markdown('<div class="section-title">Paper Trade Ledger</div>', unsafe_allow_html=True)
    if trades_df.empty:
        st.warning("No paper trades yet. Run supervisor.py first.")
        return

    st.dataframe(trades_df.sort_index(ascending=False).tail(30), width="stretch", height=420)


def render_trade_chart(trades_df, positions_df=None, closed_positions_df=None):
    st.markdown('<div class="section-title">Paper Equity Curves</div>', unsafe_allow_html=True)
    if closed_positions_df is None or closed_positions_df.empty:
        st.info("No closed-trade equity data yet.")
        return

    pnl_col = "net_realized_pnl" if "net_realized_pnl" in closed_positions_df.columns else "realized_pnl" if "realized_pnl" in closed_positions_df.columns else None
    time_col = "closed_at" if "closed_at" in closed_positions_df.columns else "timestamp" if "timestamp" in closed_positions_df.columns else None
    if pnl_col is None or time_col is None:
        st.info("No realized PnL timeline available yet.")
        return

    curve_df = closed_positions_df.copy()
    curve_df[time_col] = pd.to_datetime(curve_df[time_col], errors="coerce")
    curve_df[pnl_col] = pd.to_numeric(curve_df[pnl_col], errors="coerce").fillna(0)
    curve_df = curve_df.dropna(subset=[time_col]).sort_values(time_col)
    curve_df["cumulative_realized_pnl"] = curve_df[pnl_col].cumsum()
    curve_df["rolling_peak"] = curve_df["cumulative_realized_pnl"].cummax()
    curve_df["drawdown"] = curve_df["cumulative_realized_pnl"] - curve_df["rolling_peak"]
    curve_df["day"] = curve_df[time_col].dt.date.astype(str)
    daily = curve_df.groupby("day")[pnl_col].sum().reset_index(name="daily_pnl")

    unrealized_total = 0.0
    if positions_df is not None and not positions_df.empty and "unrealized_pnl" in positions_df.columns:
        unrealized_total = float(pd.to_numeric(positions_df["unrealized_pnl"], errors="coerce").fillna(0).sum())
    curve_df["realized_plus_unrealized"] = curve_df["cumulative_realized_pnl"] + unrealized_total

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(curve_df, x=time_col, y="cumulative_realized_pnl", title="Cumulative Realized PnL Over Time"), width="stretch")
    with c2:
        st.plotly_chart(px.line(curve_df, x=time_col, y="realized_plus_unrealized", title="Realized + Unrealized Equity Curve"), width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.bar(daily, x="day", y="daily_pnl", title="Daily PnL"), width="stretch")
    with c4:
        st.plotly_chart(px.line(curve_df, x=time_col, y="drawdown", title="Drawdown Curve"), width="stretch")


def render_performance_charts(trades_df, closed_positions_df, alerts_df, backtest_wallet_df, model_registry_df, positions_df=None):
    st.markdown('<div class="section-title">Performance Charts</div>', unsafe_allow_html=True)
    chart_cols = st.columns(2)

    pnl_source = closed_positions_df.copy()
    if not pnl_source.empty:
        pnl_col = "net_realized_pnl" if "net_realized_pnl" in pnl_source.columns else "realized_pnl" if "realized_pnl" in pnl_source.columns else None
        time_col = "closed_at" if "closed_at" in pnl_source.columns else "timestamp" if "timestamp" in pnl_source.columns else None
        if pnl_col and time_col:
            pnl_source[time_col] = pd.to_datetime(pnl_source[time_col], errors="coerce")
            pnl_source = pnl_source.dropna(subset=[time_col]).sort_values(time_col)
            pnl_source["cumulative_pnl"] = pd.to_numeric(pnl_source[pnl_col], errors="coerce").fillna(0).cumsum()
            pnl_source["day"] = pnl_source[time_col].dt.date.astype(str)
            daily = pnl_source.groupby("day")[pnl_col].agg(win_rate=lambda s: float((s > 0).mean() * 100), pnl="sum").reset_index()
            with chart_cols[0]:
                st.plotly_chart(px.line(pnl_source, x=time_col, y="cumulative_pnl", title="Cumulative Paper PnL"), width="stretch")
            with chart_cols[1]:
                st.plotly_chart(px.bar(daily, x="day", y="pnl", title="Win/Loss by Day"), width="stretch")

    if not trades_df.empty and "confidence" in trades_df.columns:
        outcome_col = "net_realized_pnl" if "net_realized_pnl" in trades_df.columns else "realized_pnl" if "realized_pnl" in trades_df.columns else None
        if outcome_col:
            plot_df = trades_df.copy()
            plot_df[outcome_col] = pd.to_numeric(plot_df[outcome_col], errors="coerce")
            st.plotly_chart(px.scatter(plot_df, x="confidence", y=outcome_col, title="Signal Confidence vs Realized Outcome"), width="stretch")

    if not alerts_df.empty:
        alert_col = "close_reason" if "close_reason" in alerts_df.columns else "alert_type" if "alert_type" in alerts_df.columns else None
        if alert_col:
            counts = alerts_df[alert_col].astype(str).value_counts().reset_index()
            counts.columns = [alert_col, "count"]
            st.plotly_chart(px.bar(counts, x=alert_col, y="count", title="Alert Counts by Type"), width="stretch")

    if not backtest_wallet_df.empty and "total_pnl" in backtest_wallet_df.columns:
        st.plotly_chart(px.bar(backtest_wallet_df.head(10), x="wallet_copied", y="total_pnl", title="Top Wallets by Profitability"), width="stretch")

    if positions_df is not None and not positions_df.empty and "market" in positions_df.columns:
        active = positions_df["market"].astype(str).value_counts().reset_index()
        active.columns = ["market", "count"]
        st.plotly_chart(px.bar(active, x="market", y="count", title="Active Positions by Market"), width="stretch")

    if not model_registry_df.empty:
        reg = model_registry_df.copy()
        if "promoted_at" in reg.columns:
            reg["promoted_at"] = pd.to_datetime(reg["promoted_at"], errors="coerce")
        metric_col = "average_pnl" if "average_pnl" in reg.columns else None
        if metric_col and "promoted_at" in reg.columns:
            st.plotly_chart(px.line(reg, x="promoted_at", y=metric_col, title="Model Performance Over Retrains"), width="stretch")


def render_best_trades(closed_positions_df, path_replay_df):
    st.markdown('<div class="section-title">Most Successful Trades</div>', unsafe_allow_html=True)
    source_df = closed_positions_df if not closed_positions_df.empty else path_replay_df
    if source_df.empty:
        st.info("No successful trade history yet.")
        return

    pnl_col = "unrealized_pnl" if "unrealized_pnl" in source_df.columns else "net_pnl" if "net_pnl" in source_df.columns else None
    if pnl_col is None:
        st.dataframe(source_df.head(10), width="stretch")
        return

    best_df = source_df.sort_values(by=pnl_col, ascending=False).head(10)
    cols = [c for c in ["market", "entry_price", "current_price", "exit_price", pnl_col, "close_reason", "exit_reason", "wallet_copied"] if c in best_df.columns]
    st.dataframe(best_df[cols], width="stretch")


def render_action_board(signals_df, positions_df):
    st.markdown('<div class="section-title">Recommended Paper Actions</div>', unsafe_allow_html=True)
    st.caption("Paper-trading decision board only — not live execution advice.")
    if signals_df.empty:
        st.info("No ranked signals available yet.")
        return

    ranked = signals_df.copy()
    sort_cols = [c for c in ["edge_score", "p_tp_before_sl", "confidence"] if c in ranked.columns]
    if sort_cols:
        ranked = ranked.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    ranked = ranked.head(20).copy()

    open_markets = set()
    positions_lookup = {}
    if not positions_df.empty and "market" in positions_df.columns:
        open_markets = set(positions_df["market"].dropna().astype(str).tolist())
        positions_lookup = positions_df.drop_duplicates(subset=["market"], keep="last").set_index("market").to_dict("index")

    grouped_rows = {"Enter": [], "Hold": [], "Exit / Leave": [], "Watch": []}

    for _, row in ranked.iterrows():
        market = row.get("market_title", row.get("market", "Unknown Market"))
        confidence = pd.to_numeric(pd.Series([row.get("confidence")]), errors="coerce").iloc[0]
        p_tp = pd.to_numeric(pd.Series([row.get("p_tp_before_sl")]), errors="coerce").iloc[0]
        edge = pd.to_numeric(pd.Series([row.get("edge_score")]), errors="coerce").iloc[0]
        current_price = row.get("market_last_trade_price", row.get("current_price"))
        position_row = positions_lookup.get(market, {})
        open_pnl = position_row.get("unrealized_pnl") if position_row else None
        already_open = market in open_markets

        if already_open and pd.notna(confidence) and confidence < 0.50:
            group = "Exit / Leave"
        elif already_open:
            group = "Hold"
        elif pd.notna(p_tp) and pd.notna(edge) and p_tp >= 0.62 and edge > 0:
            group = "Enter"
        else:
            group = "Watch"

        grouped_rows[group].append(
            {
                "market": market,
                "side": row.get("outcome_side", row.get("side", "N/A")),
                "confidence": optional_number(row.get("confidence"), 3),
                "edge score": optional_number(row.get("edge_score"), 4),
                "expected return": optional_number(row.get("expected_return"), 4),
                "current price": optional_number(current_price, 4),
                "open PnL if held": optional_number(open_pnl, 4),
                "reason": row.get("reason_summary", row.get("reason", row.get("signal_label", "N/A"))),
                "link": row.get("market_url", row.get("url", "")),
            }
        )

    for group_name in ["Enter", "Hold", "Exit / Leave", "Watch"]:
        st.markdown(f"**{group_name}**")
        group_df = pd.DataFrame(grouped_rows[group_name])
        if group_df.empty:
            st.caption("No rows in this group.")
        else:
            st.dataframe(group_df, width="stretch", hide_index=True)


def render_model_status(model_status_df, supervised_eval_df, time_split_eval_df, path_replay_df, backtest_wallet_df, model_registry_df):
    st.markdown('<div class="section-title">Model / Learning Status</div>', unsafe_allow_html=True)
    st.caption("This tab shows whether the paper-trading system has enough historical rows to train/evaluate the newer supervised models.")
    missing_outputs = []
    for label, path in [
        ("contract targets", LOGS_DIR / "contract_targets.csv"),
        ("CLOB price history", LOGS_DIR / "clob_price_history.csv"),
        ("replay backtest", PATH_REPLAY_FILE),
        ("supervised eval", SUPERVISED_EVAL_FILE),
        ("time-split eval", TIME_SPLIT_EVAL_FILE),
    ]:
        if not Path(path).exists():
            missing_outputs.append(f"{label}: {path.name}")

    if missing_outputs:
        st.warning("Learning outputs still missing: " + "; ".join(missing_outputs))

    weights_status = "🟢 current" if WEIGHTS_FILE.exists() else "🔴 missing"
    st.write(f"**Weights file:** {weights_status}")

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Replay Trades", len(path_replay_df))
    with top2:
        acc = "-"
        if not supervised_eval_df.empty and "accuracy" in supervised_eval_df.columns:
            acc = f"{float(supervised_eval_df.iloc[-1]['accuracy']):.3f}"
        st.metric("Supervised Accuracy", acc)
    with top3:
        test_acc = "-"
        if not time_split_eval_df.empty and "test_accuracy" in time_split_eval_df.columns:
            test_acc = f"{float(time_split_eval_df.iloc[-1]['test_accuracy']):.3f}"
        st.metric("Time-Split Test Acc", test_acc)
    with top4:
        sharpe = "-"
        if not supervised_eval_df.empty and "sharpe" in supervised_eval_df.columns:
            sharpe = f"{float(supervised_eval_df.iloc[-1]['sharpe']):.3f}"
        st.metric("Sharpe-like", sharpe)

    if not model_status_df.empty:
        latest = model_status_df.iloc[-1].to_dict()
        dataset_rows = int(latest.get('dataset_rows', latest.get('closed_trade_rows', 0)) or 0)
        retrain_threshold = int(latest.get('retrain_threshold', latest.get('closed_trade_threshold', 0)) or 0)
        replay_rows = int(latest.get('replay_rows', 0) or 0)
        replay_threshold = int(latest.get('replay_threshold', 0) or 0)
        progress_ratio = float(latest.get('progress_ratio', 0) or 0)
        last_action = latest.get('last_action', 'Unknown')

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Closed Trade Rows", dataset_rows)
            st.metric("Closed Trade Threshold", retrain_threshold)
            st.metric("Replay Rows", replay_rows)
        with c2:
            st.metric("Replay Threshold", replay_threshold)
            st.metric("Progress Ratio", f"{progress_ratio:.2f}")

        st.progress(max(0.0, min(1.0, progress_ratio)))
        if retrain_threshold and dataset_rows < retrain_threshold and replay_threshold and replay_rows < replay_threshold:
            st.warning(
                f"Not enough real outcomes yet for stronger retraining: closed={dataset_rows}/{retrain_threshold}, replay={replay_rows}/{replay_threshold}."
            )
        else:
            st.success("Enough outcome history is available for stronger supervised/replay training passes.")
        st.code(last_action, language="text")
    else:
        st.info("No model status rows yet. Run `python run_bot.py` for a while so the system can collect signals, markets, and replay data.")

    if supervised_eval_df.empty and time_split_eval_df.empty and path_replay_df.empty:
        st.info("Learning outputs are still empty because the newer supervised / replay pipeline does not have enough built history yet. This is expected on early runs.")

    if not model_registry_df.empty:
        latest_model = model_registry_df.iloc[-1].to_dict()
        st.markdown("**Current Champion Model**")
        st.code(str(latest_model), language="text")

    if not path_replay_df.empty:
        pnl_col = "net_pnl" if "net_pnl" in path_replay_df.columns else "gross_pnl" if "gross_pnl" in path_replay_df.columns else None
        if pnl_col:
            fig = px.histogram(path_replay_df, x=pnl_col, title="Replay PnL Distribution")
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")

            equity_df = path_replay_df.copy()
            equity_df["equity_curve"] = equity_df[pnl_col].astype(float).cumsum()
            equity_df["rolling_win_rate"] = (equity_df[pnl_col].astype(float) > 0).rolling(20, min_periods=1).mean()
            equity_df["rolling_drawdown"] = equity_df["equity_curve"] - equity_df["equity_curve"].cummax()
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.line(equity_df, y="equity_curve", title="Equity Curve"), width="stretch")
            with c2:
                st.plotly_chart(px.line(equity_df, y="rolling_win_rate", title="Rolling Win Rate"), width="stretch")
            st.plotly_chart(px.line(equity_df, y="rolling_drawdown", title="Rolling Drawdown"), width="stretch")

    if not backtest_wallet_df.empty:
        st.markdown("**Wallet Alpha Evolution / Leaders**")
        st.dataframe(backtest_wallet_df.head(15), width="stretch")


def render_raw_data(signals_df, trades_df, episode_log_df, markets_df, whales_df, alerts_df, model_status_df, positions_df, closed_positions_df):
    st.caption("Raw data is split into sub-tabs for faster inspection and export-oriented review.")
    raw_tabs = st.tabs(["Signals", "Execution", "Episodes", "Markets", "Whales", "Alerts", "Learning", "Positions"])

    with raw_tabs[0]:
        st.dataframe(signals_df, width="stretch")
    with raw_tabs[1]:
        st.dataframe(trades_df, width="stretch")
    with raw_tabs[2]:
        st.dataframe(episode_log_df, width="stretch")
    with raw_tabs[3]:
        st.dataframe(markets_df, width="stretch")
    with raw_tabs[4]:
        st.dataframe(whales_df, width="stretch")
    with raw_tabs[5]:
        st.dataframe(alerts_df, width="stretch")
    with raw_tabs[6]:
        st.dataframe(model_status_df, width="stretch")
    with raw_tabs[7]:
        st.markdown("**Open Positions**")
        st.dataframe(positions_df, width="stretch")
        st.markdown("**Closed Positions**")
        st.dataframe(closed_positions_df, width="stretch")


def main():
    inject_styles()
    render_header()

    refresh_seconds = st.sidebar.slider("Auto-refresh hint (seconds)", min_value=5, max_value=120, value=15)
    st.sidebar.caption("Tip: rerun/refresh after a supervisor cycle completes.")
    st.sidebar.write(f"Suggested refresh interval: {refresh_seconds}s")
    market_search = st.sidebar.text_input("Market search", "")
    wallet_search = st.sidebar.text_input("Wallet search", "")
    side_filter = st.sidebar.selectbox("Side filter", ["All", "YES", "NO", "unknown"])
    signal_label_filter = st.sidebar.selectbox("Signal label filter", ["All", "IGNORE", "LOW-CONFIDENCE WATCH", "STRONG PAPER OPPORTUNITY", "HIGHEST-RANKED PAPER SIGNAL"])
    min_confidence = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.01)
    min_edge_score = st.sidebar.slider("Minimum edge score", -1.0, 1.0, -1.0, 0.01)
    only_open_positions = st.sidebar.checkbox("Only open positions", value=False)
    only_actionable = st.sidebar.checkbox("Only actionable signals", value=False)
    time_range_hours = st.sidebar.selectbox("Time range", [1, 6, 12, 24, 72, 168], index=3)
    st.sidebar.caption(f"Signals file: {SIGNALS_FILE}")
    st.sidebar.caption(f"Execution file: {EXECUTION_FILE}")
    st.sidebar.caption(f"Markets file: {MARKETS_FILE}")
    st.sidebar.caption(f"Whales file: {WHALES_FILE}")
    st.sidebar.caption(f"Alerts file: {ALERTS_FILE}")

    signals_df = load_csv(SIGNALS_FILE)
    trades_df = load_execution_history()
    episode_log_df = load_csv(EPISODE_LOG_FILE)
    markets_df = load_csv(MARKETS_FILE)
    whales_df = load_csv(WHALES_FILE)
    distribution_df = load_csv(MARKET_DISTRIBUTION_FILE)
    alerts_df = load_csv(ALERTS_FILE)
    model_status_df = load_csv(MODEL_STATUS_FILE)
    positions_df = load_csv(POSITIONS_FILE)
    closed_positions_df = load_csv(CLOSED_POSITIONS_FILE)
    supervised_eval_df = load_csv(SUPERVISED_EVAL_FILE)
    time_split_eval_df = load_csv(TIME_SPLIT_EVAL_FILE)
    path_replay_df = load_csv(PATH_REPLAY_FILE)
    backtest_wallet_df = load_csv(BACKTEST_BY_WALLET_FILE)
    model_registry_df = load_csv(MODEL_REGISTRY_FILE)

    signals_df = apply_dashboard_filters(signals_df, market_search=market_search, wallet_search=wallet_search, min_confidence=min_confidence, signal_label=signal_label_filter, side_filter=side_filter, min_edge_score=min_edge_score, only_actionable=only_actionable, time_range_hours=time_range_hours)
    trades_df = apply_dashboard_filters(trades_df, market_search=market_search, wallet_search=wallet_search, min_confidence=min_confidence, signal_label=signal_label_filter, side_filter=side_filter, min_edge_score=min_edge_score, only_actionable=only_actionable, time_range_hours=time_range_hours)
    positions_df = apply_dashboard_filters(positions_df, market_search=market_search, wallet_search=wallet_search, min_confidence=min_confidence, signal_label=signal_label_filter, position_status="OPEN" if only_open_positions else "All", side_filter=side_filter, min_edge_score=min_edge_score, only_open_positions=only_open_positions, only_actionable=only_actionable, time_range_hours=time_range_hours)
    closed_positions_df = apply_dashboard_filters(closed_positions_df, market_search=market_search, wallet_search=wallet_search, min_confidence=min_confidence, signal_label=signal_label_filter, side_filter=side_filter, min_edge_score=min_edge_score, only_actionable=only_actionable, time_range_hours=time_range_hours)

    st.caption("Quick guide: System Status = health and performance, Signals = ranked opportunities, Positions = paper trade state and PnL, Markets = market, whale, and alert context, Models = learning outputs and raw data.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["System Status", "Signals & Opportunities", "Positions & PnL", "Markets, Whales & Alerts", "Models & Data Quality"])

    with tab1:
        render_overview(signals_df, trades_df, markets_df, alerts_df, positions_df, closed_positions_df)
        render_data_freshness_panel([
            ("signals.csv", SIGNALS_FILE, signals_df),
            ("execution_log.csv", EXECUTION_FILE, trades_df),
            ("markets.csv", MARKETS_FILE, markets_df),
            ("whales.csv", WHALES_FILE, whales_df),
            ("alerts.csv", ALERTS_FILE, alerts_df),
            ("positions.csv", POSITIONS_FILE, positions_df),
            ("model_status.csv", MODEL_STATUS_FILE, model_status_df),
        ])
        render_pipeline_health_strip(signals_df, markets_df, positions_df, model_status_df, path_replay_df)
        render_attention_needed(signals_df, trades_df, alerts_df, positions_df, model_status_df, path_replay_df)
        render_performance_charts(trades_df, closed_positions_df, alerts_df, backtest_wallet_df, model_registry_df, positions_df=positions_df)

    with tab2:
        top_left, top_right = st.columns([1.2, 1])
        with top_left:
            render_top_opportunities(signals_df)
        with top_right:
            render_factor_matrix(signals_df)
        render_opportunity_table(signals_df)
        render_action_board(signals_df, positions_df)

    with tab3:
        render_positions_pnl_summary(positions_df, closed_positions_df)
        render_simulated_decisions(positions_df, closed_positions_df)
        render_positions(positions_df, closed_positions_df)
        render_best_trades(closed_positions_df, path_replay_df)
        if not episode_log_df.empty:
            st.markdown('<div class="section-title">Episode Log</div>', unsafe_allow_html=True)
            st.dataframe(episode_log_df.tail(20), width="stretch")
        bottom_left, bottom_right = st.columns([1, 1])
        with bottom_left:
            render_paper_trades(trades_df)
        with bottom_right:
            render_trade_chart(trades_df, positions_df=positions_df, closed_positions_df=closed_positions_df)

    with tab4:
        top_left, top_right = st.columns([1.1, 0.9])
        with top_left:
            render_market_tracker(markets_df)
        with top_right:
            render_alerts(alerts_df)
        lower_left, lower_right = st.columns([1, 1])
        with lower_left:
            render_whale_tracker(whales_df)
        with lower_right:
            render_market_distribution(distribution_df)

    with tab5:
        render_model_status(model_status_df, supervised_eval_df, time_split_eval_df, path_replay_df, backtest_wallet_df, model_registry_df)
        render_raw_data(signals_df, trades_df, episode_log_df, markets_df, whales_df, alerts_df, model_status_df, positions_df, closed_positions_df)


if __name__ == "__main__":
    main()
