import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

LOGS_DIR = "logs"
SIGNALS_FILE = os.path.join(LOGS_DIR, "signals.csv")
SUMMARY_FILE = os.path.join(LOGS_DIR, "daily_summary.txt")

st.set_page_config(page_title="Neural Network for Crypto", page_icon="📈", layout="wide")


def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def metric_card(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)


def render_header():
    st.title("📈 Neural Network for Crypto")
    st.caption("Real-time public-data research + paper-trading dashboard")
    st.info(
        "This interface shows ranked paper-trading opportunities and simulated trades only. "
        "It does not place real bets or connect to a live account."
    )


def render_overview(signals_df, trades_df):
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)

    top_conf = "-"
    if not signals_df.empty and "confidence" in signals_df.columns:
        try:
            top_conf = f"{float(signals_df['confidence'].max()):.2f}"
        except Exception:
            top_conf = "-"

    with c1:
        metric_card("Ranked Signals", len(signals_df))
    with c2:
        metric_card("Paper Trades", len(trades_df))
    with c3:
        metric_card("Highest Confidence", top_conf)
    with c4:
        metric_card("Last Refresh", datetime.now().strftime("%H:%M:%S"))


def render_top_opportunities(signals_df):
    st.subheader("Top Paper-Trading Opportunities")
    if signals_df.empty:
        st.warning("No ranked opportunities yet. Run supervisor.py first.")
        return

    sort_df = signals_df.copy()
    if "confidence" in sort_df.columns:
        sort_df = sort_df.sort_values(by="confidence", ascending=False)

    top_df = sort_df.head(10)

    for _, row in top_df.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row.get('signal_label', 'UNKNOWN')}")
            st.write(f"**Market:** {row.get('market', row.get('market_title', 'Unknown'))}")
            st.write(f"**Side:** {row.get('side', 'UNKNOWN')}")
            st.write(f"**Wallet:** {row.get('wallet_copied', row.get('trader_wallet', 'Unknown'))}")
            st.write(f"**Confidence:** {row.get('confidence', '-')}")
            st.write(f"**Reason:** {row.get('reason', 'No reason available')}")


def render_signal_charts(signals_df):
    st.subheader("Signal Visualizations")
    if signals_df.empty or "confidence" not in signals_df.columns:
        st.info("No signal chart data yet.")
        return

    chart_df = signals_df.copy().head(25)
    fig = px.bar(
        chart_df,
        x="confidence",
        y="market",
        color="signal_label",
        orientation="h",
        title="Top Ranked Opportunities by Confidence",
    )
    fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def render_paper_trades(trades_df):
    st.subheader("Paper Trade Ledger")
    if trades_df.empty:
        st.warning("No paper trades yet. Run supervisor.py first.")
        return

    st.dataframe(trades_df.sort_index(ascending=False), use_container_width=True)


def render_trade_chart(trades_df):
    st.subheader("Simulated Fill Prices")
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
    st.plotly_chart(fig, use_container_width=True)


def render_raw_data(signals_df, trades_df):
    with st.expander("Raw data"):
        st.markdown("**Signals CSV**")
        st.dataframe(signals_df, use_container_width=True)
        st.markdown("**Paper Trade Ledger**")
        st.dataframe(trades_df, use_container_width=True)


def main():
    render_header()

    refresh_seconds = st.sidebar.slider("Auto-refresh hint (seconds)", min_value=5, max_value=120, value=15)
    st.sidebar.caption(
        "Tip: Streamlit does not auto-refresh by itself here. Re-run manually or use a refresh extension if desired."
    )
    st.sidebar.write(f"Suggested refresh interval: {refresh_seconds}s")

    signals_df = load_csv(SIGNALS_FILE)
    trades_df = load_csv(SUMMARY_FILE)

    render_overview(signals_df, trades_df)

    left, right = st.columns([1.2, 1])
    with left:
        render_top_opportunities(signals_df)
        render_signal_charts(signals_df)
    with right:
        render_paper_trades(trades_df)
        render_trade_chart(trades_df)

    render_raw_data(signals_df, trades_df)


if __name__ == "__main__":
    main()
