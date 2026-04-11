import sqlite3
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

_instances: dict[str, "Database"] = {}
_instance_lock = threading.Lock()


class Database:
    def __new__(cls, db_path="logs/trading.db"):
        resolved = str(Path(db_path).resolve())
        with _instance_lock:
            if resolved in _instances:
                return _instances[resolved]
            inst = super().__new__(cls)
            inst._initialized = False
            _instances[resolved] = inst
            return inst

    def __init__(self, db_path="logs/trading.db"):
        if self._initialized:
            return
        self._initialized = True
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self._lock = threading.Lock()
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
        self._ensure_column("candidate_decisions", "cycle_id", "TEXT")
        self._ensure_column("candidate_decisions", "candidate_id", "TEXT")
        self._ensure_column("candidate_decisions", "condition_id", "TEXT")
        self._ensure_column("candidate_decisions", "outcome_side", "TEXT")
        self._ensure_column("candidate_decisions", "market_slug", "TEXT")
        self._ensure_column("candidate_decisions", "trader_wallet", "TEXT")
        self._ensure_column("candidate_decisions", "entry_intent", "TEXT")
        self._ensure_column("candidate_decisions", "model_action", "TEXT")
        self._ensure_column("candidate_decisions", "final_decision", "TEXT")
        self._ensure_column("candidate_decisions", "reject_reason", "TEXT")
        self._ensure_column("candidate_decisions", "reject_category", "TEXT")
        self._ensure_column("candidate_decisions", "gate", "TEXT")
        self._ensure_column("candidate_decisions", "confidence", "REAL")
        self._ensure_column("candidate_decisions", "p_tp_before_sl", "REAL")
        self._ensure_column("candidate_decisions", "expected_return", "REAL")
        self._ensure_column("candidate_decisions", "edge_score", "REAL")
        self._ensure_column("candidate_decisions", "calibrated_edge", "REAL")
        self._ensure_column("candidate_decisions", "calibrated_baseline", "REAL")
        self._ensure_column("candidate_decisions", "proposed_size_usdc", "REAL")
        self._ensure_column("candidate_decisions", "final_size_usdc", "REAL")
        self._ensure_column("candidate_decisions", "available_balance", "REAL")
        self._ensure_column("candidate_decisions", "order_id", "TEXT")
        self._ensure_column("candidate_decisions", "details_json", "TEXT")
        self._ensure_column("candidate_decisions", "market_family", "TEXT")
        self._ensure_column("candidate_decisions", "brain_id", "TEXT")
        self._ensure_column("candidate_decisions", "active_model_group", "TEXT")
        self._ensure_column("candidate_decisions", "active_model_kind", "TEXT")
        self._ensure_column("candidate_decisions", "active_regime", "TEXT")
        self._ensure_column("positions", "market", "TEXT")
        self._ensure_column("positions", "market_title", "TEXT")
        self._ensure_column("positions", "order_side", "TEXT")
        self._ensure_column("positions", "size_usdc", "REAL")
        self._ensure_column("positions", "negotiated_value_usdc", "REAL")
        self._ensure_column("positions", "max_payout_usdc", "REAL")
        self._ensure_column("positions", "current_value_usdc", "REAL")
        self._ensure_column("positions", "unrealized_pnl_pct", "REAL")
        self._ensure_column("positions", "avg_to_now_price_change", "REAL")
        self._ensure_column("positions", "avg_to_now_price_change_pct", "REAL")
        self._ensure_column("positions", "net_realized_pnl", "REAL")
        self._ensure_column("positions", "confidence", "REAL")
        self._ensure_column("positions", "confidence_at_entry", "REAL")
        self._ensure_column("positions", "signal_label", "TEXT")
        self._ensure_column("positions", "entry_signal_snapshot_json", "TEXT")
        self._ensure_column("positions", "entry_signal_snapshot_feature_count", "INTEGER")
        self._ensure_column("positions", "entry_signal_snapshot_version", "INTEGER")
        self._ensure_column("positions", "close_reason", "TEXT")
        self._ensure_column("positions", "exit_price", "REAL")
        self._ensure_column("positions", "close_fingerprint", "TEXT")
        self._ensure_column("positions", "is_reconciliation_close", "INTEGER")
        self._ensure_column("positions", "lifecycle_source", "TEXT")
        for column_name, column_type in [
            ("entry_model_family", "TEXT"),
            ("entry_model_version", "TEXT"),
            ("performance_governor_level", "INTEGER"),
            ("market_family", "TEXT"),
            ("brain_id", "TEXT"),
            ("active_model_group", "TEXT"),
            ("active_model_kind", "TEXT"),
            ("active_regime", "TEXT"),
            ("horizon_bucket", "TEXT"),
            ("liquidity_bucket", "TEXT"),
            ("volatility_bucket", "TEXT"),
            ("technical_regime_bucket", "TEXT"),
            ("entry_context_complete", "INTEGER"),
            ("learning_eligible", "INTEGER"),
            ("operational_close_flag", "INTEGER"),
            ("reconciliation_close_flag", "INTEGER"),
            ("exit_reason_family", "TEXT"),
            ("intended_exit_reason", "TEXT"),
            ("actual_execution_path", "TEXT"),
            ("exit_fill_latency_seconds", "REAL"),
            ("exit_cancel_count", "INTEGER"),
            ("exit_partial_fill_ratio", "REAL"),
            ("exit_realized_slippage_bps", "REAL"),
            ("market_slug", "TEXT"),
        ]:
            self._ensure_column("positions", column_name, column_type)
        self._ensure_column("fills", "condition_id", "TEXT")
        self._ensure_column("fills", "outcome_side", "TEXT")
        self._ensure_column("fills", "side", "TEXT")
        for column_name, column_type in [
            ("first_fill_at", "TEXT"),
            ("opened_at", "TEXT"),
            ("market", "TEXT"),
            ("market_title", "TEXT"),
            ("entry_signal_snapshot_json", "TEXT"),
            ("entry_signal_snapshot_feature_count", "INTEGER"),
            ("entry_signal_snapshot_version", "INTEGER"),
        ]:
            self._ensure_column("live_positions", column_name, column_type)

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
        try:
            cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                market TEXT,
                market_title TEXT,
                token_id TEXT,
                condition_id TEXT,
                outcome_side TEXT,
                order_side TEXT,
                status TEXT,
                entry_price REAL,
                current_price REAL,
                size_usdc REAL,
                negotiated_value_usdc REAL,
                shares REAL,
                max_payout_usdc REAL,
                market_value REAL,
                current_value_usdc REAL,
                realized_pnl REAL,
                net_realized_pnl REAL,
                unrealized_pnl REAL,
                unrealized_pnl_pct REAL,
                avg_to_now_price_change REAL,
                avg_to_now_price_change_pct REAL,
                confidence REAL,
                confidence_at_entry REAL,
                signal_label TEXT,
                entry_signal_snapshot_json TEXT,
                entry_signal_snapshot_feature_count INTEGER,
                entry_signal_snapshot_version INTEGER,
                close_reason TEXT,
                exit_price REAL,
                close_fingerprint TEXT,
                is_reconciliation_close INTEGER,
                lifecycle_source TEXT,
                entry_model_family TEXT,
                entry_model_version TEXT,
                performance_governor_level INTEGER,
                market_family TEXT,
                brain_id TEXT,
                active_model_group TEXT,
                active_model_kind TEXT,
                active_regime TEXT,
                horizon_bucket TEXT,
                liquidity_bucket TEXT,
                volatility_bucket TEXT,
                technical_regime_bucket TEXT,
                entry_context_complete INTEGER,
                learning_eligible INTEGER,
                operational_close_flag INTEGER,
                reconciliation_close_flag INTEGER,
                exit_reason_family TEXT,
                intended_exit_reason TEXT,
                actual_execution_path TEXT,
                exit_fill_latency_seconds REAL,
                exit_cancel_count INTEGER,
                exit_partial_fill_ratio REAL,
                exit_realized_slippage_bps REAL,
                market_slug TEXT,
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
                condition_id TEXT,
                outcome_side TEXT,
                side TEXT,
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
            CREATE TABLE IF NOT EXISTS candidate_decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id TEXT,
                candidate_id TEXT,
                token_id TEXT,
                condition_id TEXT,
                outcome_side TEXT,
                market TEXT,
                market_slug TEXT,
                trader_wallet TEXT,
                entry_intent TEXT,
                model_action TEXT,
                final_decision TEXT,
                reject_reason TEXT,
                reject_category TEXT,
                gate TEXT,
                confidence REAL,
                p_tp_before_sl REAL,
                expected_return REAL,
                edge_score REAL,
                calibrated_edge REAL,
                calibrated_baseline REAL,
                proposed_size_usdc REAL,
                final_size_usdc REAL,
                available_balance REAL,
                order_id TEXT,
                details_json TEXT,
                market_family TEXT,
                brain_id TEXT,
                active_model_group TEXT,
                active_model_kind TEXT,
                active_regime TEXT,
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
            CREATE TABLE IF NOT EXISTS live_positions (
                position_key TEXT PRIMARY KEY,
                token_id TEXT,
                condition_id TEXT,
                outcome_side TEXT,
                shares REAL,
                avg_entry_price REAL,
                realized_pnl REAL,
                first_fill_at TEXT,
                last_fill_at TEXT,
                opened_at TEXT,
                market TEXT,
                market_title TEXT,
                entry_signal_snapshot_json TEXT,
                entry_signal_snapshot_feature_count INTEGER,
                entry_signal_snapshot_version INTEGER,
                source TEXT,
                status TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS state_mismatches (
                mismatch_id TEXT PRIMARY KEY,
                severity TEXT,
                source TEXT,
                detail TEXT,
                created_at TEXT,
                resolved_at TEXT
            );
            CREATE TABLE IF NOT EXISTS external_position_syncs (
                sync_id TEXT PRIMARY KEY,
                position_key TEXT,
                token_id TEXT,
                condition_id TEXT,
                outcome_side TEXT,
                sync_type TEXT,
                local_shares_before REAL,
                exchange_shares REAL,
                delta_shares REAL,
                avg_entry_price REAL,
                fill_id TEXT,
                observed_at TEXT
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
            CREATE INDEX IF NOT EXISTS idx_candidate_decisions_created_at ON candidate_decisions(created_at);
            CREATE INDEX IF NOT EXISTS idx_candidate_decisions_cycle_id ON candidate_decisions(cycle_id);
            CREATE INDEX IF NOT EXISTS idx_candidate_decisions_final_decision ON candidate_decisions(final_decision);
            CREATE INDEX IF NOT EXISTS idx_candidate_decisions_reject_reason ON candidate_decisions(reject_reason);
            CREATE INDEX IF NOT EXISTS idx_risk_events_token_id ON risk_events(token_id);
            CREATE INDEX IF NOT EXISTS idx_risk_events_created_at ON risk_events(created_at);
            CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at);
            CREATE INDEX IF NOT EXISTS idx_service_heartbeats_created_at ON service_heartbeats(created_at);
            CREATE INDEX IF NOT EXISTS idx_system_health_created_at ON system_health(created_at);
            CREATE INDEX IF NOT EXISTS idx_live_positions_token_id ON live_positions(token_id);
            CREATE INDEX IF NOT EXISTS idx_live_positions_status ON live_positions(status);
            CREATE INDEX IF NOT EXISTS idx_state_mismatches_created_at ON state_mismatches(created_at);
            CREATE INDEX IF NOT EXISTS idx_external_position_syncs_position_key ON external_position_syncs(position_key);
            CREATE INDEX IF NOT EXISTS idx_external_position_syncs_observed_at ON external_position_syncs(observed_at);
            """
        )
            self.conn.commit()
        finally:
            cur.close()

    def execute(self, query, params=()):
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(query, params)
                self.conn.commit()
                return cur
            except Exception:
                cur.close()
                raise

    def query_all(self, query, params=()):
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
            finally:
                cur.close()

    def integrity_report(self):
        report = {"ok": True, "checks": {}, "errors": []}
        try:
            quick = self.conn.execute("PRAGMA quick_check;").fetchone()
            quick_val = quick[0] if quick else "unknown"
            report["checks"]["quick_check"] = quick_val
            if str(quick_val).lower() != "ok":
                report["ok"] = False
        except Exception as exc:
            report["ok"] = False
            report["errors"].append(f"quick_check_failed:{exc}")

        required_tables = [
            "orders",
            "fills",
            "live_positions",
            "model_decisions",
            "candidate_decisions",
            "state_mismatches",
        ]
        try:
            existing = {
                row["name"] if isinstance(row, sqlite3.Row) else row[0]
                for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            missing = [t for t in required_tables if t not in existing]
            report["checks"]["missing_tables"] = missing
            if missing:
                report["ok"] = False
        except Exception as exc:
            report["ok"] = False
            report["errors"].append(f"table_scan_failed:{exc}")
        return report

    def backup_and_reset_runtime_state(self, logs_dir="logs"):
        """
        Archive potentially-corrupted runtime state (DB + log CSVs) and rebuild a clean DB.
        Model artifacts in `weights/` are intentionally untouched.
        Clears the singleton cache so subsequent Database() calls get a fresh instance.
        """
        resolved = str(self.db_path.resolve())
        with _instance_lock:
            _instances.pop(resolved, None)
        self._initialized = False
        logs_path = Path(logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = logs_path / f"runtime_reset_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        self.conn.close()

        db_file = logs_path / "trading.db"
        if db_file.exists():
            shutil.copy2(db_file, backup_dir / "trading.db.bak")
            db_file.unlink()

        for csv_name in [
            "positions.csv",
            "closed_positions.csv",
            "live_orders.csv",
            "live_fills.csv",
            "execution_log.csv",
            "trade_events.csv",
            "signals.csv",
            "markets.csv",
            "alerts.csv",
            "candidate_decisions.csv",
            "candidate_cycle_stats.csv",
            "service_heartbeats.csv",
            "system_health.csv",
            "incidents.csv",
        ]:
            src = logs_path / csv_name
            if src.exists():
                shutil.copy2(src, backup_dir / f"{csv_name}.bak")

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
        self._ensure_column("candidate_decisions", "cycle_id", "TEXT")
        self._ensure_column("candidate_decisions", "candidate_id", "TEXT")
        self._ensure_column("candidate_decisions", "condition_id", "TEXT")
        self._ensure_column("candidate_decisions", "outcome_side", "TEXT")
        self._ensure_column("candidate_decisions", "market_slug", "TEXT")
        self._ensure_column("candidate_decisions", "trader_wallet", "TEXT")
        self._ensure_column("candidate_decisions", "entry_intent", "TEXT")
        self._ensure_column("candidate_decisions", "model_action", "TEXT")
        self._ensure_column("candidate_decisions", "final_decision", "TEXT")
        self._ensure_column("candidate_decisions", "reject_reason", "TEXT")
        self._ensure_column("candidate_decisions", "reject_category", "TEXT")
        self._ensure_column("candidate_decisions", "gate", "TEXT")
        self._ensure_column("candidate_decisions", "confidence", "REAL")
        self._ensure_column("candidate_decisions", "p_tp_before_sl", "REAL")
        self._ensure_column("candidate_decisions", "expected_return", "REAL")
        self._ensure_column("candidate_decisions", "edge_score", "REAL")
        self._ensure_column("candidate_decisions", "calibrated_edge", "REAL")
        self._ensure_column("candidate_decisions", "calibrated_baseline", "REAL")
        self._ensure_column("candidate_decisions", "proposed_size_usdc", "REAL")
        self._ensure_column("candidate_decisions", "final_size_usdc", "REAL")
        self._ensure_column("candidate_decisions", "available_balance", "REAL")
        self._ensure_column("candidate_decisions", "order_id", "TEXT")
        self._ensure_column("candidate_decisions", "details_json", "TEXT")
        self._ensure_column("candidate_decisions", "market_family", "TEXT")
        self._ensure_column("candidate_decisions", "brain_id", "TEXT")
        self._ensure_column("candidate_decisions", "active_model_group", "TEXT")
        self._ensure_column("candidate_decisions", "active_model_kind", "TEXT")
        self._ensure_column("candidate_decisions", "active_regime", "TEXT")
        for column_name, column_type in [
            ("entry_model_family", "TEXT"),
            ("entry_model_version", "TEXT"),
            ("performance_governor_level", "INTEGER"),
            ("market_family", "TEXT"),
            ("brain_id", "TEXT"),
            ("active_model_group", "TEXT"),
            ("active_model_kind", "TEXT"),
            ("active_regime", "TEXT"),
            ("horizon_bucket", "TEXT"),
            ("liquidity_bucket", "TEXT"),
            ("volatility_bucket", "TEXT"),
            ("technical_regime_bucket", "TEXT"),
            ("entry_context_complete", "INTEGER"),
            ("learning_eligible", "INTEGER"),
            ("operational_close_flag", "INTEGER"),
            ("reconciliation_close_flag", "INTEGER"),
            ("exit_reason_family", "TEXT"),
            ("intended_exit_reason", "TEXT"),
            ("actual_execution_path", "TEXT"),
            ("exit_fill_latency_seconds", "REAL"),
            ("exit_cancel_count", "INTEGER"),
            ("exit_partial_fill_ratio", "REAL"),
            ("exit_realized_slippage_bps", "REAL"),
            ("market_slug", "TEXT"),
        ]:
            self._ensure_column("positions", column_name, column_type)
        self._ensure_column("fills", "condition_id", "TEXT")
        self._ensure_column("fills", "outcome_side", "TEXT")
        self._ensure_column("fills", "side", "TEXT")
        return str(backup_dir)
