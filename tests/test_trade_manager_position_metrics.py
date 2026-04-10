import pandas as pd
import json

from trade_lifecycle import TradeLifecycle
from trade_manager import TradeManager


def test_trade_manager_persists_ui_style_position_metrics(tmp_path):
    manager = TradeManager(logs_dir=tmp_path)
    trade = TradeLifecycle(market="Will the price of Bitcoin be above $72,000 on April 15?", token_id="tok-1", condition_id="cond-1", outcome_side="YES")
    trade.on_signal(
        {
            "market": "Will the price of Bitcoin be above $72,000 on April 15?",
            "btc_live_index_price": 72123.0,
            "twitter_sentiment": 0.18,
            "open_positions_count": 2,
        }
    )
    trade.enter(size_usdc=2.87, entry_price=0.39)
    trade.current_price = 0.415
    trade.unrealized_pnl = trade.shares * (trade.current_price - trade.entry_price)
    manager.active_trades["tok-1|cond-1|YES"] = trade

    manager.persist_open_positions()

    df = pd.read_csv(tmp_path / "positions.csv")
    row = df.iloc[0]
    assert round(float(row["negotiated_value_usdc"]), 2) == 2.87
    assert round(float(row["max_payout_usdc"]), 2) == round(float(row["shares"]), 2)
    assert round(float(row["current_value_usdc"]), 2) == round(float(row["market_value"]), 2)
    assert round(float(row["current_value_usdc"]), 2) == 3.05
    assert round(float(row["avg_to_now_price_change"]), 3) == 0.025
    assert round(float(row["avg_to_now_price_change_pct"]), 6) == round((0.415 - 0.39) / 0.39, 6)
    assert round(float(row["unrealized_pnl"]), 2) == 0.18
    expected_unrealized_pnl_pct = (float(row["current_value_usdc"]) - float(row["negotiated_value_usdc"])) / float(row["negotiated_value_usdc"])
    assert round(float(row["unrealized_pnl_pct"]), 6) == round(expected_unrealized_pnl_pct, 6)
    snapshot = json.loads(row["entry_signal_snapshot_json"])
    assert snapshot["btc_live_index_price"] == 72123.0
    assert snapshot["twitter_sentiment"] == 0.18
    assert int(row["entry_signal_snapshot_feature_count"]) == len(snapshot)

    db_row = manager.db.query_all(
        "SELECT entry_signal_snapshot_json, entry_signal_snapshot_feature_count FROM positions WHERE token_id = ?",
        ("tok-1",),
    )[0]
    assert "btc_live_index_price" in db_row["entry_signal_snapshot_json"]
    assert int(db_row["entry_signal_snapshot_feature_count"]) == len(snapshot)
