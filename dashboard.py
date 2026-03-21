import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
SIGNALS_FILE = LOGS_DIR / "signals.csv"
SUMMARY_FILE = LOGS_DIR / "daily_summary.txt"
MARKETS_FILE = LOGS_DIR / "markets.csv"
WHALES_FILE = LOGS_DIR / "whales.csv"
ALERTS_FILE = LOGS_DIR / "alerts.csv"
MODEL_STATUS_FILE = LOGS_DIR / "model_status.csv"
WEIGHTS_FILE = BASE_DIR / "weights" / "ppo_polytrader.zip"
TRADER_ANALYTICS_FILE = LOGS_DIR / "trader_analytics.csv"
BACKTEST_FILE = LOGS_DIR / "backtest_summary.csv"
DATASET_FILE = LOGS_DIR / "historical_dataset.csv"

st.set_page_config(page_title="Neural Network for Crypto", page_icon="📈", layout="wide")


def load_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="hero-box">
            <h1 style="margin-bottom:0.35rem;">📈 Neural Network for Crypto</h1>
            <div style="color:#94a3b8; font-size:1rem;">
                Real-time public-data market tracker + whale tracker + paper-trading dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "This interface shows ranked paper-trading opportunities, public market tracking, whale activity, and alerts only. "
        "It does not place real bets or connect to a live account."
    )


def render_overview(signals_df, trades_df, markets_df, alerts_df):
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    top_conf = "-"
    if not signals_df.empty and "confidence" in signals_df.columns:
        try:
            top_conf = f"{float(signals_df['confidence'].max()):.2f}"
        except Exception:
            top_conf = "-"

    with c1:
        st.metric("Ranked Signals", len(signals_df))
    with c2:
        st.metric("Paper Trades", len(trades_df))
    with c3:
        st.metric("Tracked BTC Markets", len(markets_df))
    with c4:
        st.metric("Active Alerts", len(alerts_df))

    st.caption(f"Highest confidence: {top_conf} | Last refresh: {datetime.now().strftime('%H:%M:%S')}")


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
    st.markdown('<div class="section-title">Confidence Matrix</div>', unsafe_allow_html=True)
    if signals_df.empty:
        st.info("No ranked signals available yet.")
        return

    top_row = signals_df.sort_values(by="confidence", ascending=False).iloc[0].to_dict() if "confidence" in signals_df.columns else signals_df.iloc[0].to_dict()
    factor_df = pd.DataFrame(
        [
            {"factor": "Whale Pressure", "score": float(top_row.get("whale_pressure", 0.0))},
            {"factor": "Market Structure", "score": float(top_row.get("market_structure_score", 0.0))},
            {"factor": "Volatility Risk", "score": float(top_row.get("volatility_risk", 0.0))},
            {"factor": "Time Decay", "score": float(top_row.get("time_decay_score", 0.0))},
            {"factor": "Liquidity", "score": float(top_row.get("liquidity_score", 0.0))},
            {"factor": "Volume", "score": float(top_row.get("volume_score", 0.0))},
        ]
    )
    st.caption(f"Top signal: {top_row.get('market', top_row.get('market_title', 'Unknown Market'))}")
    fig = px.bar(factor_df, x="score", y="factor", orientation="h", title="Top Signal Factor Breakdown")
    fig.update_layout(height=360, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")


def render_top_opportunities(signals_df):
    st.markdown('<div class="section-title">Top Paper-Trading Opportunities</div>', unsafe_allow_html=True)
    if signals_df.empty:
        st.warning("No ranked opportunities yet. Run supervisor.py first.")
        return

    sort_df = signals_df.copy()
    if "confidence" in sort_df.columns:
        sort_df = sort_df.sort_values(by="confidence", ascending=False)

    top_df = sort_df.head(8).reset_index(drop=True)
    cols = st.columns(2)

    for idx, (_, row) in enumerate(top_df.iterrows()):
        with cols[idx % 2]:
            label = row.get("signal_label", "UNKNOWN")
            side = row.get("side", "UNKNOWN")
            market = row.get("market", row.get("market_title", "Unknown Market"))
            wallet = row.get("wallet_copied", row.get("trader_wallet", "Unknown"))
            confidence = row.get("confidence", "-")
            reason = row.get("reason", "No reason available")
            market_url = row.get("market_url")

            st.markdown(
                f"""
                <div class="market-card">
                    <div class="market-title">{market}</div>
                    <div>
                        <span class="signal-badge {badge_class(label)}">{label}</span>
                        <span class="signal-badge badge-ignore">Observed side: {side}</span>
                        <span class="signal-badge badge-ignore">Confidence: {confidence}</span>
                    </div>
                    <div class="meta-line"><b>Source wallet:</b> {wallet}</div>
                    <div class="reason-box">{reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if pd.notna(market_url) and market_url:
                st.link_button("Open market on Polymarket", market_url, width="stretch")


def render_market_tracker(markets_df):
    st.markdown('<div class="section-title">BTC Market Tracker</div>', unsafe_allow_html=True)
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
    if whales_df.empty:
        st.info("No whale summary yet.")
        return

    st.dataframe(whales_df.head(15), width="stretch")


def render_alerts(alerts_df):
    st.markdown('<div class="section-title">Alerts</div>', unsafe_allow_html=True)
    if alerts_df.empty:
        st.info("No alerts generated yet.")
        return

    st.dataframe(alerts_df.tail(20), width="stretch")


def render_paper_trades(trades_df):
    st.markdown('<div class="section-title">Paper Trade Ledger</div>', unsafe_allow_html=True)
    if trades_df.empty:
        st.warning("No paper trades yet. Run supervisor.py first.")
        return

    st.dataframe(trades_df.sort_index(ascending=False).tail(30), width="stretch", height=420)


def render_trade_chart(trades_df):
    st.markdown('<div class="section-title">Simulated Fill Prices</div>', unsafe_allow_html=True)
    if trades_df.empty or "fill_price" not in trades_df.columns:
        st.info("No fill-price data yet.")
        return

    chart_df = trades_df.copy().tail(30)
    chart_df["trade_index"] = range(1, len(chart_df) + 1)
    fig = px.line(
        chart_df,
        x="trade_index",
        y="fill_price",
        color="side" if "side" in chart_df.columns else None,
        hover_data=[col for col in ["market", "signal_label", "confidence"] if col in chart_df.columns],
        title="Recent Simulated Fill Prices",
        markers=True,
    )
    st.plotly_chart(fig, width="stretch")


def render_model_status(model_status_df):
    st.markdown('<div class="section-title">Model / Learning Status</div>', unsafe_allow_html=True)
    weights_status = "present" if WEIGHTS_FILE.exists() else "missing"
    st.write(f"**Weights file:** {weights_status}")

    if model_status_df.empty:
        st.info("No retraining status yet.")
        return

    latest = model_status_df.iloc[-1].to_dict()
    st.write(f"**Dataset rows:** {latest.get('dataset_rows', 0)}")
    st.write(f"**Retrain threshold:** {latest.get('retrain_threshold', 0)}")
    st.write(f"**Progress ratio:** {latest.get('progress_ratio', 0)}")
    st.write(f"**Last action:** {latest.get('last_action', 'Unknown')}")


def render_raw_data(signals_df, trades_df, markets_df, whales_df, alerts_df, model_status_df):
    with st.expander("Raw data"):
        st.markdown("**Signals CSV**")
        st.dataframe(signals_df, width="stretch")
        st.markdown("**Paper Trade Ledger**")
        st.dataframe(trades_df, width="stretch")
        st.markdown("**Markets CSV**")
        st.dataframe(markets_df, width="stretch")
        st.markdown("**Whales CSV**")
        st.dataframe(whales_df, width="stretch")
        st.markdown("**Alerts CSV**")
        st.dataframe(alerts_df, width="stretch")
        st.markdown("**Model Status CSV**")
        st.dataframe(model_status_df, width="stretch")


def main():
    inject_styles()
    render_header()

    refresh_seconds = st.sidebar.slider("Auto-refresh hint (seconds)", min_value=5, max_value=120, value=15)
    st.sidebar.caption("Tip: rerun/refresh after a supervisor cycle completes.")
    st.sidebar.write(f"Suggested refresh interval: {refresh_seconds}s")
    st.sidebar.caption(f"Signals file: {SIGNALS_FILE}")
    st.sidebar.caption(f"Trades file: {SUMMARY_FILE}")
    st.sidebar.caption(f"Markets file: {MARKETS_FILE}")
    st.sidebar.caption(f"Whales file: {WHALES_FILE}")
    st.sidebar.caption(f"Alerts file: {ALERTS_FILE}")

    signals_df = load_csv(SIGNALS_FILE)
    trades_df = load_csv(SUMMARY_FILE)
    markets_df = load_csv(MARKETS_FILE)
    whales_df = load_csv(WHALES_FILE)
    alerts_df = load_csv(ALERTS_FILE)
    model_status_df = load_csv(MODEL_STATUS_FILE)

    render_overview(signals_df, trades_df, markets_df, alerts_df)

    st.caption("Quick guide: Overview = status, Opportunities = strongest paper signals, Markets = BTC tracker, Whales = public wallet summaries, Alerts = notable changes, Learning = model/retraining state.")

    tab1, tab2, tab3, tab4 = st.tabs(["Opportunities", "Markets & Whales", "Learning", "Raw Data"])

    with tab1:
        top_left, top_right = st.columns([1.2, 1])
        with top_left:
            render_top_opportunities(signals_df)
        with top_right:
            render_factor_matrix(signals_df)

        bottom_left, bottom_right = st.columns([1, 1])
        with bottom_left:
            render_paper_trades(trades_df)
        with bottom_right:
            render_trade_chart(trades_df)

    with tab2:
        top_left, top_right = st.columns([1.1, 0.9])
        with top_left:
            render_market_tracker(markets_df)
        with top_right:
            render_alerts(alerts_df)
        render_whale_tracker(whales_df)

    with tab3:
        render_model_status(model_status_df)

    with tab4:
        render_raw_data(signals_df, trades_df, markets_df, whales_df, alerts_df, model_status_df)


if __name__ == "__main__":
    main()
