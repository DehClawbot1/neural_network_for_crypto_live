import pandas as pd
import plotly.express as px
import streamlit as st


def render_market_tracker(markets_df):
    st.markdown('<div class="section-title">BTC Market Tracker</div>', unsafe_allow_html=True)
    st.caption("Sortable market coverage view for BTC-related markets being tracked by the system.")
    if markets_df.empty:
        st.info("No BTC market snapshots yet.")
        return

    search_text = st.text_input("Search markets", "", key="markets_search")
    min_volume = st.number_input("Volume minimum", min_value=0.0, value=0.0, step=100.0)
    min_liquidity = st.number_input("Liquidity minimum", min_value=0.0, value=0.0, step=100.0)
    price_min, price_max = st.slider("Price range", 0.0, 1.0, (0.0, 1.0), 0.01)
    sort_by = st.selectbox("Sort markets by", ["recent movement", "liquidity", "volume"])

    view = markets_df.copy()
    market_col = "question" if "question" in view.columns else "market" if "market" in view.columns else None
    price_col = "last_trade_price" if "last_trade_price" in view.columns else "current_price" if "current_price" in view.columns else None
    if search_text and market_col:
        view = view[view[market_col].astype(str).str.contains(search_text, case=False, na=False)]
    if "volume" in view.columns:
        view = view[pd.to_numeric(view["volume"], errors="coerce").fillna(0) >= min_volume]
    if "liquidity" in view.columns:
        view = view[pd.to_numeric(view["liquidity"], errors="coerce").fillna(0) >= min_liquidity]
    if price_col:
        prices = pd.to_numeric(view[price_col], errors="coerce")
        view = view[(prices >= price_min) & (prices <= price_max)]

    if sort_by == "liquidity" and "liquidity" in view.columns:
        view = view.sort_values("liquidity", ascending=False)
    elif sort_by == "volume" and "volume" in view.columns:
        view = view.sort_values("volume", ascending=False)
    elif sort_by == "recent movement":
        if "price_change" in view.columns:
            view = view.sort_values("price_change", ascending=False)
        else:
            st.warning("Price change data unavailable for sorting.")

    tracked_markets = len(view)
    avg_liquidity = float(pd.to_numeric(view["liquidity"], errors="coerce").fillna(0).mean()) if "liquidity" in view.columns and not view.empty else 0.0
    highest_volume_market = "-"
    if "volume" in view.columns and market_col and not view.empty:
        top_idx = pd.to_numeric(view["volume"], errors="coerce").fillna(0).idxmax()
        highest_volume_market = str(view.loc[top_idx, market_col]) if top_idx in view.index else "-"
    recently_updated = 0
    freshness_col = "updated_at" if "updated_at" in view.columns else "timestamp" if "timestamp" in view.columns else None
    if freshness_col:
        ts = pd.to_datetime(view[freshness_col], errors="coerce", utc=True)
        recently_updated = int((ts >= (pd.Timestamp.utcnow() - pd.Timedelta(minutes=10))).fillna(False).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked Markets", tracked_markets)
    c2.metric("Average Liquidity", f"{avg_liquidity:.2f}")
    c3.metric("Highest Volume Market", highest_volume_market)
    c4.metric("Price-updated Recently", recently_updated)

    table_cols = [c for c in [market_col, price_col, "liquidity", "volume", "market_id", "url", "updated_at", "timestamp"] if c and c in view.columns]
    st.dataframe(view[table_cols], use_container_width=True, hide_index=True)

    if market_col and "liquidity" in view.columns:
        liq_df = view.dropna(subset=[market_col]).copy().head(12)
        liq_df["liquidity"] = pd.to_numeric(liq_df["liquidity"], errors="coerce")
        liq_df = liq_df.dropna(subset=["liquidity"]).sort_values("liquidity", ascending=False).head(12)
        if not liq_df.empty:
            st.plotly_chart(px.bar(liq_df, x="liquidity", y=market_col, orientation="h", title="Top Markets by Liquidity"), use_container_width=True)

    if market_col and "volume" in view.columns:
        vol_df = view.dropna(subset=[market_col]).copy()
        vol_df["volume"] = pd.to_numeric(vol_df["volume"], errors="coerce")
        vol_df = vol_df.dropna(subset=["volume"]).sort_values("volume", ascending=False).head(12)
        if not vol_df.empty:
            st.plotly_chart(px.bar(vol_df, x="volume", y=market_col, orientation="h", title="Top Markets by Volume"), use_container_width=True)

    if price_col:
        price_df = view.copy()
        price_df[price_col] = pd.to_numeric(price_df[price_col], errors="coerce")
        price_df = price_df.dropna(subset=[price_col])
        if not price_df.empty:
            st.plotly_chart(px.histogram(price_df, x=price_col, nbins=20, title="Last Trade Price Distribution"), use_container_width=True)

    time_col = "updated_at" if "updated_at" in view.columns else "timestamp" if "timestamp" in view.columns else None
    if time_col:
        timeline = view.copy()
        timeline[time_col] = pd.to_datetime(timeline[time_col], errors="coerce")
        timeline = timeline.dropna(subset=[time_col])
        if not timeline.empty:
            counts = timeline.groupby(timeline[time_col].dt.floor("h")).size().reset_index(name="tracked_market_count")
            st.plotly_chart(px.line(counts, x=time_col, y="tracked_market_count", title="Tracked Market Count Over Time"), use_container_width=True)


def render_whale_tracker(whales_df):
    st.markdown('<div class="section-title">Whale Activity Tracker</div>', unsafe_allow_html=True)
    st.caption("Public wallet summaries showing who is most active and where concentration is forming.")
    if whales_df.empty:
        st.info("No whale summary yet.")
        return

    wallet_col = "wallet" if "wallet" in whales_df.columns else "trader_wallet" if "trader_wallet" in whales_df.columns else None
    market_col = "market" if "market" in whales_df.columns else "market_title" if "market_title" in whales_df.columns else "top_market" if "top_market" in whales_df.columns else None
    watched_wallets_count = int(whales_df[wallet_col].nunique()) if wallet_col else len(whales_df)
    most_active_wallet = str(whales_df[wallet_col].astype(str).value_counts().idxmax()) if wallet_col else "-"
    highest_concentration_market = str(whales_df[market_col].astype(str).value_counts().idxmax()) if market_col else "-"
    total_unique_markets = int(whales_df[market_col].nunique()) if market_col else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Watched Wallets", watched_wallets_count)
    c2.metric("Most Active Wallet", most_active_wallet)
    c3.metric("Highest Concentration Market", highest_concentration_market)
    c4.metric("Unique Markets Touched", total_unique_markets)

    if wallet_col:
        summary = whales_df.copy()
        if market_col:
            grouped = summary.groupby(wallet_col).agg(
                action_count=(wallet_col, "size"),
                unique_markets=(market_col, pd.Series.nunique),
            )
        else:
            grouped = summary.groupby(wallet_col).agg(action_count=(wallet_col, "size"))
            grouped["unique_markets"] = 0
        if market_col:
            grouped["concentration_score"] = grouped["action_count"] / grouped["unique_markets"].replace(0, 1)
        else:
            grouped["concentration_score"] = grouped["action_count"]
        if "alpha_score" in summary.columns:
            grouped["alpha_score"] = summary.groupby(wallet_col)["alpha_score"].mean()
        time_col = "timestamp" if "timestamp" in summary.columns else "updated_at" if "updated_at" in summary.columns else None
        if time_col:
            grouped["latest_activity_time"] = pd.to_datetime(summary[time_col], errors="coerce").groupby(summary[wallet_col]).max()
        grouped = grouped.reset_index().sort_values("action_count", ascending=False)
        st.dataframe(grouped.head(20), use_container_width=True, hide_index=True)
    else:
        st.dataframe(whales_df.head(20), use_container_width=True, hide_index=True)

    if wallet_col:
        wallet_counts = whales_df[wallet_col].astype(str).value_counts().head(15).reset_index()
        wallet_counts.columns = [wallet_col, "actions"]
        st.plotly_chart(px.bar(wallet_counts, x=wallet_col, y="actions", title="Trades / Actions by Wallet"), use_container_width=True)

    if market_col:
        market_counts = whales_df[market_col].astype(str).value_counts().head(15).reset_index()
        market_counts.columns = [market_col, "wallet_count"]
        st.plotly_chart(px.bar(market_counts, x=market_col, y="wallet_count", title="Wallet Concentration by Market"), use_container_width=True)

    if wallet_col and "profit" in whales_df.columns:
        profit_df = whales_df.copy()
        profit_df["profit"] = pd.to_numeric(profit_df["profit"], errors="coerce")
        profit_df = profit_df.dropna(subset=["profit"]).sort_values("profit", ascending=False).head(15)
        if not profit_df.empty:
            st.plotly_chart(px.bar(profit_df, x=wallet_col, y="profit", title="Top Wallets by Profitability"), use_container_width=True)

    time_col = "timestamp" if "timestamp" in whales_df.columns else "updated_at" if "updated_at" in whales_df.columns else None
    if wallet_col and time_col:
        activity_df = whales_df.copy()
        activity_df[time_col] = pd.to_datetime(activity_df[time_col], errors="coerce")
        activity_df = activity_df.dropna(subset=[time_col])
        if not activity_df.empty:
            timeline = activity_df.groupby(activity_df[time_col].dt.floor("H")).size().reset_index(name="activity_count")
            st.plotly_chart(px.line(timeline, x=time_col, y="activity_count", title="Wallet Activity Over Time"), use_container_width=True)
