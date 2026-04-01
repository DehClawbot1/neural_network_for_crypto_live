import pandas as pd

from trade_feedback_learner import TradeFeedbackLearner
from trade_lifecycle import TradeLifecycle
from trade_manager import TradeManager


def test_persist_closed_trade_upserts_db_position_row(tmp_path):
    manager = TradeManager(logs_dir=tmp_path)
    trade = TradeLifecycle(market="BTC", token_id="tok-1", condition_id="cond-1", outcome_side="Yes")
    trade.enter(size_usdc=5.0, entry_price=0.50)
    trade.close(exit_price=0.60, reason="take_profit_roi")

    manager.persist_closed_trades([trade])

    db_rows = manager.db.query_all(
        """
        SELECT position_id, token_id, condition_id, outcome_side, status, close_reason, close_fingerprint
        FROM positions
        WHERE UPPER(COALESCE(status, '')) = 'CLOSED'
        """
    )
    assert len(db_rows) == 1
    assert db_rows[0]["token_id"] == "tok-1"
    assert db_rows[0]["close_reason"] == "take_profit_roi"
    assert db_rows[0]["close_fingerprint"]


def test_backfill_closed_positions_db_from_csv(tmp_path):
    closed_df = pd.DataFrame(
        [
            {
                "position_id": "legacy-id",
                "market": "BTC",
                "market_title": "BTC",
                "token_id": "tok-1",
                "condition_id": "cond-1",
                "outcome_side": "Yes",
                "order_side": "BUY",
                "entry_price": 0.5,
                "current_price": 0.6,
                "exit_price": 0.6,
                "size_usdc": 5.0,
                "shares": 10.0,
                "market_value": 6.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 1.0,
                "net_realized_pnl": 1.0,
                "opened_at": "2026-04-01T10:00:00+00:00",
                "closed_at": "2026-04-01T10:05:00+00:00",
                "status": "CLOSED",
                "confidence": 0.7,
                "confidence_at_entry": 0.7,
                "signal_label": "LARGE_BUY",
                "close_reason": "take_profit_roi",
                "close_fingerprint": "fp-1",
                "is_reconciliation_close": False,
            }
        ]
    )
    closed_df.to_csv(tmp_path / "closed_positions.csv", index=False)

    manager = TradeManager(logs_dir=tmp_path)
    result = manager.backfill_closed_positions_db_from_csv()

    db_rows = manager.db.query_all("SELECT position_id, close_fingerprint, status FROM positions")
    assert result["db_rows_upserted"] == 1
    assert len(db_rows) == 1
    assert db_rows[0]["close_fingerprint"] == "fp-1"
    assert db_rows[0]["status"] == "CLOSED"


def test_feedback_backfill_from_closed_positions_csv_is_idempotent_and_skips_reconciliation(tmp_path):
    closed_df = pd.DataFrame(
        [
            {
                "position_id": "p1",
                "market": "BTC",
                "market_title": "BTC",
                "token_id": "tok-1",
                "condition_id": "cond-1",
                "outcome_side": "Yes",
                "entry_price": 0.5,
                "current_price": 0.6,
                "exit_price": 0.6,
                "size_usdc": 5.0,
                "shares": 10.0,
                "opened_at": "2026-04-01T10:00:00+00:00",
                "closed_at": "2026-04-01T10:05:00+00:00",
                "close_reason": "take_profit_roi",
                "realized_pnl": 1.0,
                "confidence": 0.6,
                "confidence_at_entry": 0.6,
                "signal_label": "WATCH",
                "close_fingerprint": "fp-1",
            },
            {
                "position_id": "p2",
                "market": "BTC",
                "market_title": "BTC",
                "token_id": "tok-2",
                "condition_id": "cond-2",
                "outcome_side": "No",
                "entry_price": 0.5,
                "current_price": 0.5,
                "exit_price": 0.5,
                "size_usdc": 5.0,
                "shares": 10.0,
                "opened_at": "2026-04-01T11:00:00+00:00",
                "closed_at": "2026-04-01T11:05:00+00:00",
                "close_reason": "external_manual_close",
                "realized_pnl": 0.0,
                "confidence": 0.4,
                "confidence_at_entry": 0.4,
                "signal_label": "IGNORE",
                "close_fingerprint": "fp-2",
            },
        ]
    )
    closed_df.to_csv(tmp_path / "closed_positions.csv", index=False)

    learner = TradeFeedbackLearner(logs_dir=tmp_path)
    first = learner.backfill_from_closed_positions_csv()
    second = learner.backfill_from_closed_positions_csv()

    reports_df = pd.read_csv(tmp_path / "trade_feedback_reports.csv")
    assert first["processed_reports"] == 1
    assert second["processed_reports"] == 0
    assert len(reports_df.index) == 1
    assert reports_df.iloc[0]["close_fingerprint"] == "fp-1"
