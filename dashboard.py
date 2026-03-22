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
POSITIONS_FILE = LOGS_DIR / "positions.csv"
CLOSED_POSITIONS_FILE = LOGS_DIR / "closed_positions.csv"
MARKET_DISTRIBUTION_FILE = LOGS_DIR / "market_distribution.csv"
SUPERVISED_EVAL_FILE = LOGS_DIR / "supervised_eval.csv"
TIME_SPLIT_EVAL_FILE = LOGS_DIR / "time_split_eval.csv"
PATH_REPLAY_FILE = LOGS_DIR / "path_replay_backtest.csv"

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


def render_header():
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
                    <div><span class="live-dot"></span><strong>Live</strong></div>
                    <div class="small-muted">Last refresh: {datetime.now().strftime('%H:%M:%S')}</div>
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


def render_overview(signals_df, trades_df, markets_df, alerts_df):
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    top_conf = "-"
    if not signals_df.empty and "confidence" in signals_df.columns:
        try:
            top_conf = f"{float(signals_df['confidence'].max()):.2f}"
        except Exception:
            top_conf = "-"

    tracked_market_count = len(markets_df)
    if not markets_df.empty and "market_id" in markets_df.columns:
        tracked_market_count = markets_df["market_id"].nunique()

    with c1:
        st.markdown('<div class="overview-card">', unsafe_allow_html=True)
        st.metric("Ranked Signals", len(signals_df))
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="overview-card">', unsafe_allow_html=True)
        st.metric("Paper Trades", len(trades_df))
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="overview-card">', unsafe_allow_html=True)
        st.metric("Tracked BTC Markets", tracked_market_count)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="overview-card">', unsafe_allow_html=True)
        st.metric("Recent Alerts", len(alerts_df))
        st.markdown('</div>', unsafe_allow_html=True)

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

            conf_pct = 0
            try:
                conf_pct = max(0, min(100, int(float(confidence) * 100)))
            except Exception:
                conf_pct = 0

            st.markdown(
                f"""
                <div class="market-card">
                    <div class="market-title">{market}</div>
                    <div>
                        <span class="signal-badge {badge_class(label)}">{label}</span>
                        <span class="signal-badge badge-ignore">Observed side: {side}</span>
                    </div>
                    <div class="small-muted">Confidence score: {conf_pct}%</div>
                    <div class="confidence-bar-wrap">
                        <div class="confidence-bar-fill" style="width:{conf_pct}%;"></div>
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
    st.markdown('<div class="section-title">Simulated Trade Decisions</div>', unsafe_allow_html=True)
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


def render_positions(positions_df, closed_positions_df):
    st.markdown('<div class="section-title">Paper Positions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Open Positions**")
        if positions_df.empty:
            st.info("No open paper positions.")
        else:
            cols = [c for c in ["position_id", "market", "token_id", "condition_id", "outcome_side", "entry_price", "current_price", "shares", "market_value", "unrealized_pnl", "realized_pnl", "opened_at"] if c in positions_df.columns]
            st.dataframe(positions_df.tail(20)[cols] if cols else positions_df.tail(20), width="stretch")
    with c2:
        st.markdown("**Closed Positions**")
        if closed_positions_df.empty:
            st.info("No closed paper positions yet.")
        else:
            cols = [c for c in ["position_id", "market", "token_id", "condition_id", "outcome_side", "entry_price", "current_price", "shares", "market_value", "unrealized_pnl", "realized_pnl", "close_reason", "closed_at"] if c in closed_positions_df.columns]
            st.dataframe(closed_positions_df.tail(20)[cols] if cols else closed_positions_df.tail(20), width="stretch")


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
    st.markdown('<div class="section-title">Top 10 Entry / Hold / Leave Board</div>', unsafe_allow_html=True)
    st.caption("Paper-trading action board only — not live execution advice.")
    if signals_df.empty:
        st.info("No ranked signals available yet.")
        return

    ranked = signals_df.copy()
    sort_cols = [c for c in ["edge_score", "p_tp_before_sl", "confidence"] if c in ranked.columns]
    if sort_cols:
        ranked = ranked.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    ranked = ranked.head(10).copy()

    open_markets = set()
    positions_lookup = {}
    if not positions_df.empty and "market" in positions_df.columns:
        open_markets = set(positions_df["market"].dropna().astype(str).tolist())
        positions_lookup = positions_df.drop_duplicates(subset=["market"], keep="last").set_index("market").to_dict("index")

    rows = []
    for _, row in ranked.iterrows():
        market = row.get("market_title", row.get("market", "Unknown Market"))
        confidence = float(row.get("confidence", 0.0) or 0.0)
        p_tp = float(row.get("p_tp_before_sl", 0.0) or 0.0)
        expected_return = float(row.get("expected_return", 0.0) or 0.0)
        edge = float(row.get("edge_score", 0.0) or 0.0)
        entry_price_now = float(row.get("current_price", row.get("entry_price", 0.0)) or 0.0)
        live_market_price = float(row.get("market_last_trade_price", row.get("current_price", 0.0)) or 0.0)
        price_delta = live_market_price - entry_price_now
        position_row = positions_lookup.get(market, {})
        open_pnl = float(position_row.get("unrealized_pnl", 0.0) or 0.0) if position_row else 0.0
        shares = float(position_row.get("shares", 0.0) or 0.0) if position_row else 0.0
        already_open = market in open_markets

        if already_open and confidence < 0.50:
            action = "LEAVE / EXIT PAPER POSITION"
            alert = "🔴 Exit watch"
        elif already_open:
            action = "HOLD PAPER POSITION"
            alert = "🟡 Hold / monitor"
        elif p_tp >= 0.62 and edge > 0:
            action = "ENTER PAPER POSITION"
            alert = "🟢 Entry alert"
        else:
            action = "WATCH ONLY"
            alert = "⚪ No entry yet"

        rows.append(
            {
                "market": market,
                "side": row.get("side"),
                "signal": row.get("signal_label"),
                "entry_price_now": round(entry_price_now, 4),
                "live_market_price": round(live_market_price, 4),
                "price_delta": round(price_delta, 4),
                "shares": round(shares, 4),
                "paper_profit_usdc": round(open_pnl, 4),
                "p_tp_before_sl": round(p_tp, 3),
                "expected_return": round(expected_return, 4),
                "edge_score": round(edge, 4),
                "confidence": round(confidence, 3),
                "action": action,
                "alert": alert,
                "link": row.get("market_url"),
            }
        )

    board_df = pd.DataFrame(rows)
    st.dataframe(board_df, width="stretch")


def render_model_status(model_status_df, supervised_eval_df, time_split_eval_df, path_replay_df):
    st.markdown('<div class="section-title">Model / Learning Status</div>', unsafe_allow_html=True)
    st.caption("This tab shows whether the paper-trading system has enough historical rows to train/evaluate the newer supervised models.")
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

    if not path_replay_df.empty:
        pnl_col = "net_pnl" if "net_pnl" in path_replay_df.columns else "gross_pnl" if "gross_pnl" in path_replay_df.columns else None
        if pnl_col:
            fig = px.histogram(path_replay_df, x=pnl_col, title="Replay PnL Distribution")
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")


def render_raw_data(signals_df, trades_df, markets_df, whales_df, alerts_df, model_status_df, positions_df, closed_positions_df):
    st.caption("Raw data is split into sub-tabs for faster inspection and export-oriented review.")
    raw_tabs = st.tabs(["Signals", "Trades", "Markets", "Whales", "Alerts", "Learning", "Positions"])

    with raw_tabs[0]:
        st.dataframe(signals_df, width="stretch")
    with raw_tabs[1]:
        st.dataframe(trades_df, width="stretch")
    with raw_tabs[2]:
        st.dataframe(markets_df, width="stretch")
    with raw_tabs[3]:
        st.dataframe(whales_df, width="stretch")
    with raw_tabs[4]:
        st.dataframe(alerts_df, width="stretch")
    with raw_tabs[5]:
        st.dataframe(model_status_df, width="stretch")
    with raw_tabs[6]:
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
    st.sidebar.caption(f"Signals file: {SIGNALS_FILE}")
    st.sidebar.caption(f"Trades file: {SUMMARY_FILE}")
    st.sidebar.caption(f"Markets file: {MARKETS_FILE}")
    st.sidebar.caption(f"Whales file: {WHALES_FILE}")
    st.sidebar.caption(f"Alerts file: {ALERTS_FILE}")

    signals_df = load_csv(SIGNALS_FILE)
    trades_df = load_csv(SUMMARY_FILE)
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

    render_overview(signals_df, trades_df, markets_df, alerts_df)

    st.caption("Quick guide: Overview = status, Opportunities = strongest paper signals, Markets = BTC tracker, Whales = public wallet summaries, Alerts = notable changes, Learning = model/retraining state.")

    tab1, tab2, tab3, tab4 = st.tabs(["Opportunities", "Markets & Whales", "Learning", "Raw Data"])

    with tab1:
        top_left, top_right = st.columns([1.2, 1])
        with top_left:
            render_top_opportunities(signals_df)
        with top_right:
            render_factor_matrix(signals_df)

        render_action_board(signals_df, positions_df)
        render_simulated_decisions(positions_df, closed_positions_df)
        render_positions(positions_df, closed_positions_df)
        render_best_trades(closed_positions_df, path_replay_df)
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
        lower_left, lower_right = st.columns([1, 1])
        with lower_left:
            render_whale_tracker(whales_df)
        with lower_right:
            render_market_distribution(distribution_df)

    with tab3:
        render_model_status(model_status_df, supervised_eval_df, time_split_eval_df, path_replay_df)

    with tab4:
        render_raw_data(signals_df, trades_df, markets_df, whales_df, alerts_df, model_status_df, positions_df, closed_positions_df)


if __name__ == "__main__":
    main()
