import sqlite3
from pathlib import Path


class Database:
    def __init__(self, db_path="logs/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                token_id TEXT,
                condition_id TEXT,
                outcome_side TEXT,
                status TEXT,
                entry_price REAL,
                current_price REAL,
                shares REAL,
                market_value REAL,
                realized_pnl REAL,
                unrealized_pnl REAL,
                opened_at TEXT,
                closed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                token_id TEXT,
                condition_id TEXT,
                outcome_side TEXT,
                order_side TEXT,
                price REAL,
                size REAL,
                status TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT,
                token_id TEXT,
                price REAL,
                size REAL,
                filled_at TEXT
            );
            CREATE TABLE IF NOT EXISTS trade_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                token_id TEXT,
                payload TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS model_decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT,
                model_name TEXT,
                score REAL,
                action TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS risk_events (
                risk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT,
                event_type TEXT,
                detail TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def execute(self, query, params=()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        self.conn.commit()
        return cur

    def query_all(self, query, params=()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

