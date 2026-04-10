import pandas as pd
import warnings

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


def test_append_closed_rows_avoids_pandas_concat_futurewarning(tmp_path):
    manager = TradeManager(logs_dir=tmp_path)
    existing = pd.DataFrame(
        [
            {
                "token_id": "tok-0",
                "condition_id": "cond-0",
                "outcome_side": "Yes",
                "close_reason": "rl_exit",
                "close_fingerprint": "fp-0",
                "legacy_empty_col": None,
            }
        ]
    )
    existing.to_csv(tmp_path / "closed_positions.csv", index=False)

    row = {
        "token_id": "tok-1",
        "condition_id": "cond-1",
        "outcome_side": "No",
        "close_reason": "take_profit_roi",
        "close_fingerprint": "fp-1",
        "shares": 10.0,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        appended = manager._append_closed_rows([row])

    assert appended == 1
    df = pd.read_csv(tmp_path / "closed_positions.csv")
    assert len(df.index) == 2
    assert "legacy_empty_col" in df.columns
    assert "shares" in df.columns


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
    assert list(learner.reports_dir.glob("*")) == []


def test_archive_and_prune_redundant_external_sync_fills(tmp_path):
    book = LivePositionBook(logs_dir=tmp_path)
    fills = [
        ("buy_1", "order_buy_1", "tok-1", "cond-1", "Yes", "BUY", 0.50, 10.0, "2026-04-01T10:00:00+00:00"),
        ("ext_sync_1", "external_manual", "tok-1", "cond-1", "Yes", "SELL", 0.50, 3.0, "2026-04-01T10:01:00+00:00"),
        ("ext_sync_2", "external_manual", "tok-1", "cond-1", "Yes", "SELL", 0.50, 7.0, "2026-04-01T10:02:00+00:00"),
        ("buy_2", "order_buy_2", "tok-2", "cond-2", "No", "BUY", 0.40, 5.0, "2026-04-01T10:00:00+00:00"),
        ("ext_sync_3", "external_manual", "tok-2", "cond-2", "No", "SELL", 0.40, 1.0, "2026-04-01T10:01:00+00:00"),
    ]
    for row in fills:
        book.db.execute(
            "INSERT INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    result = book.archive_and_prune_redundant_external_sync_fills(vacuum=False)

    assert result["archived_rows"] == 1
    assert result["deleted_rows"] == 1
    assert result["archive_csv"] is not None
    remaining = pd.DataFrame(book.db.query_all("SELECT fill_id FROM fills ORDER BY filled_at, fill_id"))
    assert "ext_sync_1" not in remaining["fill_id"].tolist()
    assert "ext_sync_2" in remaining["fill_id"].tolist()
    assert "ext_sync_3" in remaining["fill_id"].tolist()


def test_get_open_positions_merges_profile_snapshot_for_missing_live_row(tmp_path, monkeypatch):
    book = LivePositionBook(logs_dir=tmp_path)
    book.db.execute(
        "INSERT INTO live_positions (position_key, token_id, condition_id, outcome_side, shares, avg_entry_price, realized_pnl, last_fill_at, source, status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "tok-local|cond-local|Yes",
            "tok-local",
            "cond-local",
            "Yes",
            3.0,
            0.45,
            0.0,
            "2026-04-01T10:00:00+00:00",
            "rebuild",
            "OPEN",
            "2026-04-01T10:00:00+00:00",
        ),
    )

    monkeypatch.setattr(book, "_verify_open_positions_against_exchange", lambda rows: rows)
    monkeypatch.setattr(
        book,
        "_load_profile_snapshot_rows",
        lambda: [
            {
                "position_key": "tok-local|cond-local|Yes",
                "token_id": "tok-local",
                "condition_id": "cond-local",
                "outcome_side": "Yes",
                "shares": 4.0,
                "avg_entry_price": 0.46,
                "realized_pnl": 0.1,
                "unrealized_pnl": 0.2,
                "current_price": 0.51,
                "market_title": "Local upgraded",
                "market": "local-upgraded",
                "last_fill_at": None,
                "source": "profile_snapshot",
                "status": "OPEN",
            },
            {
                "position_key": "tok-profile|cond-profile|No",
                "token_id": "tok-profile",
                "condition_id": "cond-profile",
                "outcome_side": "No",
                "shares": 2.5,
                "avg_entry_price": 0.61,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.15,
                "current_price": 0.67,
                "market_title": "Profile only",
                "market": "profile-only",
                "last_fill_at": None,
                "source": "profile_snapshot",
                "status": "OPEN",
            },
        ],
    )

    open_df = book.get_open_positions()

    assert len(open_df.index) == 2

    upgraded = open_df.loc[open_df["token_id"] == "tok-local"].iloc[0]
    assert float(upgraded["shares"]) == 4.0
    assert str(upgraded["source"]) == "rebuild+profile_snapshot"

    added = open_df.loc[open_df["token_id"] == "tok-profile"].iloc[0]
    assert float(added["shares"]) == 2.5
    assert str(added["source"]) == "profile_snapshot"
