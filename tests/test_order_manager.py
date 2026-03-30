import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from live_risk_manager import RiskDecision
from order_manager import OrderManager


class _FixedDateTime:
    @classmethod
    def utcnow(cls):
        return datetime(2026, 3, 23, 12, 0, 0)

    @classmethod
    def now(cls, tz=None):
        base = datetime(2026, 3, 23, 12, 0, 0)
        return base.replace(tzinfo=tz) if tz is not None else base


class TestOrderManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.exec_patcher = patch("order_manager.ExecutionClient")
        self.db_patcher = patch("order_manager.Database")
        self.risk_patcher = patch("order_manager.LiveRiskManager")
        self.mock_exec_cls = self.exec_patcher.start()
        self.mock_db_cls = self.db_patcher.start()
        self.mock_risk_cls = self.risk_patcher.start()

        self.client = MagicMock()
        self.db = MagicMock()
        self.risk = MagicMock()
        self.risk.pre_trade_check.return_value = RiskDecision(True, "ok")

        self.mock_exec_cls.return_value = self.client
        self.mock_db_cls.return_value = self.db
        self.mock_risk_cls.return_value = self.risk
        self.manager = OrderManager(logs_dir=self.tmp.name)

    def tearDown(self):
        self.exec_patcher.stop()
        self.db_patcher.stop()
        self.risk_patcher.stop()
        self.tmp.cleanup()

    def test_submit_entry_rejects_duplicate_idempotency_key(self):
        duplicate_key = "2026-03-23T12:00|tok-1|cond-1|BUY|10|0.5"
        pd.DataFrame([
            {"idempotency_key": duplicate_key, "status": "SUBMITTED"}
        ]).to_csv(self.manager.orders_file, index=False)

        with patch("order_manager.datetime", _FixedDateTime):
            row, response = self.manager.submit_entry(
                token_id="tok-1",
                price=0.5,
                size=10,
                side="BUY",
                condition_id="cond-1",
            )

        self.assertEqual(row["status"], "REJECTED")
        self.assertEqual(row["reason"], "duplicate_idempotency_key")
        self.assertIsNone(response)
        self.client.create_and_post_order.assert_not_called()

    def test_submit_entry_rejects_when_risk_manager_blocks(self):
        self.risk.pre_trade_check.return_value = RiskDecision(False, "spread_too_wide")

        with patch("order_manager.datetime", _FixedDateTime):
            row, response = self.manager.submit_entry(
                token_id="tok-2",
                price=0.45,
                size=12,
                side="BUY",
                condition_id="cond-2",
            )

        self.assertEqual(row["status"], "REJECTED")
        self.assertEqual(row["reason"], "spread_too_wide")
        self.assertIsNone(response)
        self.client.create_and_post_order.assert_not_called()
        saved = pd.read_csv(self.manager.orders_file)
        self.assertEqual(saved.iloc[-1]["reason"], "spread_too_wide")

    def test_submit_entry_rejects_when_readiness_missing(self):
        self.client.get_balance_allowance.return_value = None

        with patch("order_manager.datetime", _FixedDateTime):
            row, response = self.manager.submit_entry(
                token_id="tok-3",
                price=0.41,
                size=8,
                side="BUY",
                condition_id="cond-3",
            )

        self.assertEqual(row["status"], "REJECTED")
        self.assertEqual(row["reason"], "missing_readiness")
        self.assertIsNone(response)
        self.client.create_and_post_order.assert_not_called()

    def test_submit_entry_records_failed_order_on_client_exception(self):
        self.client.get_balance_allowance.return_value = {"balance": 100.0}
        self.client.create_and_post_order.side_effect = RuntimeError("boom")

        with patch("order_manager.datetime", _FixedDateTime):
            row, response = self.manager.submit_entry(
                token_id="tok-4",
                price=0.51,
                size=11,
                side="BUY",
                condition_id="cond-4",
            )

        self.assertEqual(row["status"], "FAILED")
        self.assertIn("boom", row["reason"])
        self.assertIsNone(response)
        self.risk.record_failed_order.assert_called_once()

    def test_submit_entry_persists_successful_order_and_db_row(self):
        self.client.get_balance_allowance.return_value = {"balance": 100.0, "allowance": 100.0}
        self.client.create_and_post_order.return_value = {"orderID": "ord-1", "status": "SUBMITTED"}

        with patch("order_manager.datetime", _FixedDateTime):
            row, response = self.manager.submit_entry(
                token_id="tok-5",
                price=0.61,
                size=13,
                side="BUY",
                condition_id="cond-5",
                outcome_side="YES",
            )

        self.assertEqual(row["order_id"], "ord-1")
        self.assertEqual(row["status"], "SUBMITTED")
        self.assertEqual(response["orderID"], "ord-1")
        self.db.execute.assert_called_once()
        sql, params = self.db.execute.call_args[0]
        self.assertIn("INSERT OR REPLACE INTO orders", sql)
        self.assertEqual(params[0], "ord-1")
        self.assertEqual(params[1], "tok-5")
        self.assertEqual(params[2], "cond-5")
        self.assertEqual(params[3], "YES")
        self.assertEqual(params[4], "BUY")
        self.assertEqual(params[5], 0.61)
        self.assertEqual(params[6], 13)
        self.assertEqual(params[7], "SUBMITTED")
        saved = pd.read_csv(self.manager.orders_file)
        self.assertIn("ord-1", saved["order_id"].astype(str).tolist())

    def test_wait_for_fill_returns_filled_on_filled_status(self):
        self.manager.get_order_status = MagicMock(side_effect=[{"status": "OPEN"}, {"status": "FILLED"}])
        with patch("order_manager.time.sleep", return_value=None):
            result = self.manager.wait_for_fill("ord-2", timeout_seconds=5, poll_seconds=0)
        self.assertTrue(result["filled"])
        self.assertEqual(result["response"]["status"], "FILLED")

    def test_wait_for_fill_returns_false_on_rejected_status(self):
        self.manager.get_order_status = MagicMock(return_value={"status": "REJECTED"})
        with patch("order_manager.time.sleep", return_value=None):
            result = self.manager.wait_for_fill("ord-3", timeout_seconds=5, poll_seconds=0)
        self.assertFalse(result["filled"])
        self.assertEqual(result["response"]["status"], "REJECTED")

    def test_monitor_and_trigger_exit_submits_sell_when_target_hit(self):
        with patch("market_price_service.MarketPriceService") as mock_price_cls:
            mock_price_cls.return_value.get_quote.return_value = {"best_bid": 0.72}
            self.manager.submit_entry = MagicMock(return_value=({"status": "SUBMITTED"}, {"ok": True}))
            row, response = self.manager.monitor_and_trigger_exit(
                token_id="tok-6",
                target_price=0.70,
                size=5,
                condition_id="cond-6",
                outcome_side="YES",
            )

        self.manager.submit_entry.assert_called_once_with(
            token_id="tok-6",
            price=0.72,
            size=5,
            side="SELL",
            condition_id="cond-6",
            outcome_side="YES",
        )
        self.assertEqual(row["status"], "SUBMITTED")
        self.assertEqual(response, {"ok": True})
