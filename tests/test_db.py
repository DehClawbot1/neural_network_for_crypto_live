import json
import tempfile
import unittest
from pathlib import Path

from db import Database, _instances, _instance_lock


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_trading.db"
        self.db = Database(self.db_path)

    def tearDown(self):
        self.db.conn.close()
        # Clear the singleton so each test gets a fresh DB
        resolved = str(self.db.db_path.resolve())
        with _instance_lock:
            _instances.pop(resolved, None)
        self.db._initialized = False
        self.tmp_dir.cleanup()

    def test_schema_initialization(self):
        tables_to_check = [
            "positions",
            "orders",
            "fills",
            "trade_events",
            "model_decisions",
            "risk_events",
        ]

        for table in tables_to_check:
            query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';"
            result = self.db.query_all(query)
            self.assertEqual(len(result), 1, f"Table {table} was not created.")

    def test_insert_and_query_all(self):
        insert_query = """
        INSERT INTO orders (order_id, token_id, status, price, size)
        VALUES (?, ?, ?, ?, ?)
        """
        params = ("ord-123", "tok-abc", "SUBMITTED", 0.55, 100.0)
        self.db.execute(insert_query, params)

        select_query = "SELECT * FROM orders WHERE order_id = ?"
        results = self.db.query_all(select_query, ("ord-123",))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["token_id"], "tok-abc")
        self.assertEqual(results[0]["status"], "SUBMITTED")
        self.assertEqual(results[0]["price"], 0.55)

    def test_update_record(self):
        self.db.execute("INSERT INTO orders (order_id, status) VALUES (?, ?)", ("ord-999", "OPEN"))
        self.db.execute("UPDATE orders SET status = ? WHERE order_id = ?", ("FILLED", "ord-999"))

        result = self.db.query_all("SELECT status FROM orders WHERE order_id = ?", ("ord-999",))
        self.assertEqual(result[0]["status"], "FILLED")

    def test_query_empty_results(self):
        results = self.db.query_all("SELECT * FROM orders WHERE order_id = ?", ("non-existent",))
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_persistence_between_connections(self):
        self.db.execute("INSERT INTO fills (fill_id, price) VALUES (?, ?)", ("fill-1", 0.42))
        self.db.conn.close()

        # Clear singleton so a new Database() creates a fresh connection
        resolved = str(self.db.db_path.resolve())
        with _instance_lock:
            _instances.pop(resolved, None)
        self.db._initialized = False

        new_db = Database(self.db_path)
        results = new_db.query_all("SELECT price FROM fills WHERE fill_id = ?", ("fill-1",))
        self.assertEqual(results[0]["price"], 0.42)
        # Reassign so tearDown closes the right connection
        self.db = new_db


class TestDatabaseLogging(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_events.db"
        self.db = Database(self.db_path)

    def tearDown(self):
        self.db.conn.close()
        resolved = str(self.db.db_path.resolve())
        with _instance_lock:
            _instances.pop(resolved, None)
        self.db._initialized = False
        self.tmp_dir.cleanup()

    def test_log_model_decision(self):
        query = """
        INSERT INTO model_decisions (token_id, model_name, score, action)
        VALUES (?, ?, ?, ?)
        """
        params = ("0xtoken_abc", "stage3_hybrid_v1", 0.88, "BUY")
        self.db.execute(query, params)

        results = self.db.query_all("SELECT * FROM model_decisions")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model_name"], "stage3_hybrid_v1")
        self.assertEqual(results[0]["score"], 0.88)
        self.assertIsNotNone(results[0]["created_at"])

    def test_log_risk_event(self):
        query = "INSERT INTO risk_events (token_id, event_type, detail) VALUES (?, ?, ?)"
        detail = "Spread 0.06 exceeds max_spread 0.05"
        self.db.execute(query, ("0xtoken_xyz", "spread_violation", detail))

        results = self.db.query_all("SELECT * FROM risk_events WHERE event_type = ?", ("spread_violation",))
        self.assertEqual(results[0]["detail"], detail)

    def test_log_trade_event_with_json_payload(self):
        payload = {
            "side": "BUY",
            "size_usdc": 100.0,
            "slippage_bps": 15,
            "whale_wallet": "0xwhale_123",
        }
        json_payload = json.dumps(payload)

        query = "INSERT INTO trade_events (event_type, token_id, payload) VALUES (?, ?, ?)"
        self.db.execute(query, ("whale_order_detected", "0xtoken_123", json_payload))

        results = self.db.query_all("SELECT payload FROM trade_events WHERE token_id = ?", ("0xtoken_123",))
        recovered_payload = json.loads(results[0]["payload"])
        self.assertEqual(recovered_payload["whale_wallet"], "0xwhale_123")
        self.assertEqual(recovered_payload["size_usdc"], 100.0)


if __name__ == "__main__":
    unittest.main()
