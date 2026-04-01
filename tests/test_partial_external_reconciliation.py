import pandas as pd

from live_position_book import LivePositionBook
from trade_feedback_learner import TradeFeedbackLearner
from trade_lifecycle import TradeLifecycle, TradeState
from trade_manager import TradeManager


def test_rebuild_collapses_consecutive_external_sync_sells(tmp_path):
    book = LivePositionBook(logs_dir=tmp_path)
    book.db.execute(
        "INSERT INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("buy_1", "order_buy_1", "tok-1", "cond-1", "Yes", "BUY", 0.50, 10.0, "2026-04-01T10:00:00+00:00"),
    )
    book.db.execute(
        "INSERT INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ext_sync_1", "external_manual", "tok-1", "cond-1", "Yes", "SELL", 0.50, 3.0, "2026-04-01T10:01:00+00:00"),
    )
    book.db.execute(
        "INSERT INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ext_sync_2", "external_manual", "tok-1", "cond-1", "Yes", "SELL", 0.50, 7.0, "2026-04-01T10:02:00+00:00"),
    )

    rebuilt = book.rebuild_from_db()

    assert len(rebuilt.index) == 1
    row = rebuilt.iloc[0]
    assert float(row["shares"]) == 3.0
    assert float(row["realized_pnl"]) == 0.0


def test_trade_manager_reconcile_updates_existing_trade_shares(tmp_path):
    manager = TradeManager(logs_dir=tmp_path)
    trade = TradeLifecycle(market="BTC", token_id="tok-1", condition_id="cond-1", outcome_side="Yes")
    trade.enter(size_usdc=5.0, entry_price=0.50)
    trade.update_market(0.54)
    manager.active_trades["tok-1|cond-1|Yes"] = trade

    reconciled = pd.DataFrame(
        [
            {
                "token_id": "tok-1",
                "condition_id": "cond-1",
                "outcome_side": "Yes",
                "market": "BTC",
                "avg_entry_price": 0.50,
                "mark_price": 0.55,
                "current_price": 0.55,
                "shares": 4.0,
                "realized_pnl": 0.75,
                "unrealized_pnl": 0.20,
                "last_fill_at": "2026-04-01T10:03:00+00:00",
            }
        ]
    )

    manager.reconcile_live_positions(reconciled_positions_df=reconciled)
    synced = manager.active_trades["tok-1|cond-1|Yes"]

    assert float(synced.shares) == 4.0
    assert float(synced.size_usdc) == 2.0
    assert float(synced.realized_pnl) == 0.75
    assert float(synced.unrealized_pnl) == 0.20
    assert float(synced.current_price) == 0.55


def test_closed_trade_persistence_is_idempotent(tmp_path):
    manager = TradeManager(logs_dir=tmp_path)
    trade = TradeLifecycle(market="BTC", token_id="tok-1", condition_id="cond-1", outcome_side="Yes")
    trade.enter(size_usdc=5.0, entry_price=0.50)
    trade.close(exit_price=0.60, reason="rl_exit")

    manager.persist_closed_trades([trade])
    manager.persist_closed_trades([trade])

    df = pd.read_csv(tmp_path / "closed_positions.csv")
    assert len(df.index) == 1
    assert df.iloc[0]["close_reason"] == "rl_exit"


def test_external_manual_close_is_skipped_from_learning_reports(tmp_path):
    learner = TradeFeedbackLearner(logs_dir=tmp_path)
    trade = TradeLifecycle(market="BTC", token_id="tok-1", condition_id="cond-1", outcome_side="Yes", logs_dir=str(tmp_path))
    trade.enter(size_usdc=5.0, entry_price=0.50)
    trade.state = TradeState.CLOSED
    trade.close_reason = "external_manual_close"
    trade.current_price = 0.50
    trade.closed_at = "2026-04-01T10:05:00+00:00"

    processed = learner.record_closed_trades([trade])

    assert processed == 0
    assert not learner.report_csv.exists()
