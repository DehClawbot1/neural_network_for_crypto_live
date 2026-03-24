import sqlite3
from pathlib import Path


class Database:
    def __init__(self, db_path="logs/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA busy_timeout=30000;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        self._ensure_column("model_decisions", "condition_id", "TEXT")
        self._ensure_column("model_decisions", "outcome_side", "TEXT")
        self._ensure_column("model_decisions", "feature_snapshot", "TEXT")
        self._ensure_column("model_decisions", "model_artifact", "TEXT")
        self._ensure_column("model_decisions", "normalization_artifact", "TEXT")

    def _ensure_column(self, table_name, column_name, column_type):
        existing_columns = {
            row["name"] if isinstance(row, sqlite3.Row) else row[1]
            for row in self.conn.execute(f"PRAGMA table_info({table_name})")
        }
        if column_name not in existing_columns:
            self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            self.conn.commit()

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
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                category TEXT,
                severity TEXT,
                summary TEXT,
                detail TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS service_heartbeats (
                heartbeat_id TEXT PRIMARY KEY,
                service TEXT,
                status TEXT,
                detail TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS system_health (
                health_id TEXT PRIMARY KEY,
                component TEXT,
                status TEXT,
                detail TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_positions_token_id ON positions(token_id);
            CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
            CREATE INDEX IF NOT EXISTS idx_orders_token_id ON orders(token_id);
            CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
            CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);
            CREATE INDEX IF NOT EXISTS idx_fills_token_id ON fills(token_id);
            CREATE INDEX IF NOT EXISTS idx_fills_filled_at ON fills(filled_at);
            CREATE INDEX IF NOT EXISTS idx_model_decisions_token_id ON model_decisions(token_id);
            CREATE INDEX IF NOT EXISTS idx_model_decisions_created_at ON model_decisions(created_at);
            CREATE INDEX IF NOT EXISTS idx_risk_events_token_id ON risk_events(token_id);
            CREATE INDEX IF NOT EXISTS idx_risk_events_created_at ON risk_events(created_at);
            CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at);
            CREATE INDEX IF NOT EXISTS idx_service_heartbeats_created_at ON service_heartbeats(created_at);
            CREATE INDEX IF NOT EXISTS idx_system_health_created_at ON system_health(created_at);
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

