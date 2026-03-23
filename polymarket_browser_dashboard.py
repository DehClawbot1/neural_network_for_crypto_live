import pandas as pd
import streamlit as st

from polymarket_profile_client import PolymarketProfileClient


st.set_page_config(page_title="Polymarket Browser", page_icon="🧭", layout="wide")


@st.cache_resource
def get_client() -> PolymarketProfileClient:
    return PolymarketProfileClient()


def as_dataframe(payload):
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    return pd.DataFrame()


def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, str(exc)


def render_metric_row(positions_df, closed_df, value_payload, traded_payload, activity_df, trades_df):
    total_value = "-"
    if isinstance(value_payload, list) and value_payload:
        total_value = value_payload[0].get("value", "-")
    elif isinstance(value_payload, dict):
        total_value = value_payload.get("value", "-")

    traded_markets = "-"
    if isinstance(traded_payload, dict):
        traded_markets = traded_payload.get("traded", "-")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Open Positions", len(positions_df))
    c2.metric("Closed Positions", len(closed_df))
    c3.metric("Activity Rows", len(activity_df))
    c4.metric("Trade Rows", len(trades_df))
    c5.metric("Total Value", total_value)
    c6.metric("Markets Traded", traded_markets)


def main():
    st.title("🧭 Polymarket Browser")
    st.caption("Public Polymarket profile and data explorer for wallet-based browsing.")

    client = get_client()

    st.sidebar.header("Inputs")
    wallet = st.sidebar.text_input("Wallet address", "")
    market_id = st.sidebar.text_input("Condition ID for market positions", "")
    limit = st.sidebar.slider("Limit", min_value=10, max_value=500, value=100, step=10)
    offset = st.sidebar.number_input("Offset", min_value=0, max_value=10000, value=0, step=10)
    title_filter = st.sidebar.text_input("Position title filter", "")
    fetch_button = st.sidebar.button("Load Polymarket Data", type="primary")

    st.info(
        "This dashboard uses public Polymarket profile/data endpoints. "
        "Use a wallet address for profile, positions, activity, and trade browsing."
    )

    if not fetch_button:
        st.stop()

    if not wallet:
        st.error("Enter a wallet address first.")
        st.stop()

    profile_payload, profile_error = safe_call(client.get_public_profile, wallet)
    positions_payload, positions_error = safe_call(
        client.get_positions,
        user=wallet,
        limit=limit,
        offset=offset,
        title=title_filter or None,
    )
    closed_payload, closed_error = safe_call(client.get_closed_positions, user=wallet, limit=limit, offset=offset)
    activity_payload, activity_error = safe_call(client.get_activity, user=wallet, limit=limit, offset=offset)
    trades_payload, trades_error = safe_call(client.get_trades, user=wallet, limit=limit, offset=offset)
    value_payload, value_error = safe_call(client.get_value, wallet)
    traded_payload, traded_error = safe_call(client.get_traded, wallet)

    market_positions_payload = None
    market_positions_error = None
    if market_id:
        market_positions_payload, market_positions_error = safe_call(
            client.get_market_positions,
            market=market_id,
            limit=limit,
            offset=offset,
        )

    errors = [
        msg for msg in [
            profile_error,
            positions_error,
            closed_error,
            activity_error,
            trades_error,
            value_error,
            traded_error,
            market_positions_error,
        ] if msg
    ]
    if errors:
        for err in errors:
            st.warning(err)

    positions_df = as_dataframe(positions_payload)
    closed_df = as_dataframe(closed_payload)
    activity_df = as_dataframe(activity_payload)
    trades_df = as_dataframe(trades_payload)
    market_positions_df = as_dataframe(market_positions_payload)

    render_metric_row(positions_df, closed_df, value_payload, traded_payload, activity_df, trades_df)

    profile_tab, positions_tab, closed_tab, activity_tab, trades_tab, value_tab, market_tab = st.tabs(
        [
            "Profile",
            "Current Positions",
            "Closed Positions",
            "Activity",
            "Trades",
            "Value & Totals",
            "Market Positions",
        ]
    )

    with profile_tab:
        st.subheader("Public Profile")
        if isinstance(profile_payload, dict) and profile_payload:
            left, right = st.columns([1, 2])
            with left:
                image_url = profile_payload.get("profileImage")
                if image_url:
                    st.image(image_url, width=140)
            with right:
                st.write({
                    "name": profile_payload.get("name"),
                    "pseudonym": profile_payload.get("pseudonym"),
                    "proxyWallet": profile_payload.get("proxyWallet"),
                    "bio": profile_payload.get("bio"),
                    "xUsername": profile_payload.get("xUsername"),
                    "verifiedBadge": profile_payload.get("verifiedBadge"),
                    "createdAt": profile_payload.get("createdAt"),
                })
            st.json(profile_payload)
        else:
            st.info("No profile payload returned.")

    with positions_tab:
        st.subheader("Current Positions")
        if positions_df.empty:
            st.info("No open/current positions found.")
        else:
            st.dataframe(positions_df, width="stretch", hide_index=True)
            numeric_cols = [c for c in ["currentValue", "cashPnl", "percentPnl", "size", "avgPrice", "curPrice"] if c in positions_df.columns]
            for col in numeric_cols:
                positions_df[col] = pd.to_numeric(positions_df[col], errors="coerce")
            if "currentValue" in positions_df.columns and "title" in positions_df.columns:
                top_positions = positions_df.sort_values("currentValue", ascending=False).head(15)
                st.bar_chart(top_positions.set_index("title")["currentValue"])

    with closed_tab:
        st.subheader("Closed Positions")
        if closed_df.empty:
            st.info("No closed positions found.")
        else:
            st.dataframe(closed_df, width="stretch", hide_index=True)
            if "realizedPnl" in closed_df.columns and "title" in closed_df.columns:
                closed_df["realizedPnl"] = pd.to_numeric(closed_df["realizedPnl"], errors="coerce")
                winners = closed_df.sort_values("realizedPnl", ascending=False).head(15)
                st.bar_chart(winners.set_index("title")["realizedPnl"])

    with activity_tab:
        st.subheader("User Activity")
        if activity_df.empty:
            st.info("No activity found.")
        else:
            st.dataframe(activity_df, width="stretch", hide_index=True)
            if "type" in activity_df.columns:
                st.bar_chart(activity_df["type"].astype(str).value_counts())

    with trades_tab:
        st.subheader("Trades")
        if trades_df.empty:
            st.info("No trades found.")
        else:
            st.dataframe(trades_df, width="stretch", hide_index=True)
            if "side" in trades_df.columns:
                st.bar_chart(trades_df["side"].astype(str).value_counts())

    with value_tab:
        st.subheader("Value & Totals")
        st.write("Total Value")
        st.json(value_payload if value_payload is not None else {})
        st.write("Total Markets Traded")
        st.json(traded_payload if traded_payload is not None else {})

    with market_tab:
        st.subheader("Positions for a Market")
        if not market_id:
            st.info("Enter a condition ID in the sidebar to load market positions.")
        elif market_positions_df.empty:
            st.info("No market positions returned.")
        else:
            st.dataframe(market_positions_df, width="stretch", hide_index=True)
            if "positions" in market_positions_df.columns:
                st.caption("Nested market position arrays are shown in raw form below.")
                st.json(market_positions_payload)


if __name__ == "__main__":
    main()
