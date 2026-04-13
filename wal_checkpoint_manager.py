"""
wal_checkpoint_manager.py

Fix 3: SQLite WAL checkpoint manager.

A 938 MB WAL file is an operational risk:
  - Slows every read and write through the DB
  - Grows until disk is exhausted
  - Can corrupt on a hard crash

This module provides a simple WALCheckpointManager that:
  1. Runs PRAGMA wal_checkpoint(TRUNCATE) every N supervisor cycles
  2. Emits a WARNING when the WAL exceeds a configurable size threshold
  3. Is safe to call from a single-threaded supervisor loop without blocking

Usage (in supervisor loop):
    from wal_checkpoint_manager import WALCheckpointManager
    _wal_mgr = WALCheckpointManager()
    # ... inside each cycle:
    _wal_mgr.maybe_checkpoint()
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default thresholds (overridable via env vars)
_DEFAULT_DB_PATH = "logs/trading.db"
_DEFAULT_CYCLE_INTERVAL = 50          # run checkpoint every N supervisor cycles
_DEFAULT_WARN_SIZE_MB = 200           # warn when WAL > this many MB
_DEFAULT_TRUNCATE_SIZE_MB = 500       # force immediate checkpoint when WAL > this many MB


class WALCheckpointManager:
    """
    Tracks supervisor cycle count and periodically checkpoints the SQLite WAL.

    Thread safety: NOT thread-safe. Designed for a single supervisor loop thread.
    """

    def __init__(
        self,
        db_path: str | None = None,
        *,
        cycle_interval: int | None = None,
        warn_size_mb: float | None = None,
        truncate_size_mb: float | None = None,
    ) -> None:
        self.db_path = Path(
            db_path
            or os.getenv("TRADING_DB_PATH", _DEFAULT_DB_PATH)
        )
        self.cycle_interval = int(
            cycle_interval
            or os.getenv("WAL_CHECKPOINT_CYCLE_INTERVAL", str(_DEFAULT_CYCLE_INTERVAL))
        )
        self.warn_size_mb = float(
            warn_size_mb
            or os.getenv("WAL_WARN_SIZE_MB", str(_DEFAULT_WARN_SIZE_MB))
        )
        self.truncate_size_mb = float(
            truncate_size_mb
            or os.getenv("WAL_TRUNCATE_SIZE_MB", str(_DEFAULT_TRUNCATE_SIZE_MB))
        )
        self._cycle_count = 0
        self._wal_path = Path(str(self.db_path) + "-wal")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_checkpoint(self) -> bool:
        """
        Call once per supervisor cycle.
        Returns True if a checkpoint was actually run, False otherwise.
        """
        self._cycle_count += 1
        wal_mb = self._wal_size_mb()

        # Always warn on large WAL, regardless of cycle count
        if wal_mb > self.warn_size_mb:
            logger.warning(
                "WALCheckpointManager: WAL size is %.1f MB (threshold %.1f MB). "
                "Performance and stability may be affected.",
                wal_mb, self.warn_size_mb,
            )

        # Force immediate checkpoint if WAL is dangerously large
        if wal_mb > self.truncate_size_mb:
            logger.warning(
                "WALCheckpointManager: WAL %.1f MB exceeds truncate threshold %.1f MB. "
                "Running forced checkpoint now.",
                wal_mb, self.truncate_size_mb,
            )
            return self._run_checkpoint(mode="TRUNCATE")

        # Scheduled checkpoint
        if self._cycle_count % self.cycle_interval == 0:
            logger.info(
                "WALCheckpointManager: scheduled checkpoint at cycle %d (WAL=%.1f MB)",
                self._cycle_count, wal_mb,
            )
            return self._run_checkpoint(mode="PASSIVE")

        return False

    def force_checkpoint(self, mode: str = "TRUNCATE") -> bool:
        """Immediately run a WAL checkpoint. Use before shutdown or large retrain."""
        logger.info("WALCheckpointManager: forced checkpoint (mode=%s)", mode)
        return self._run_checkpoint(mode=mode)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _wal_size_mb(self) -> float:
        try:
            return self._wal_path.stat().st_size / (1024 * 1024)
        except FileNotFoundError:
            return 0.0
        except Exception:
            return 0.0

    def _run_checkpoint(self, mode: str = "PASSIVE") -> bool:
        if not self.db_path.exists():
            return False
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.db_path), timeout=30)
            try:
                result = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
                conn.close()
                # result: (busy, log, checkpointed)
                if result:
                    busy, log_frames, ckpt_frames = result
                    logger.info(
                        "WALCheckpointManager: checkpoint(%s) complete — "
                        "log_frames=%d checkpointed=%d busy=%d  WAL=%.1f MB",
                        mode, log_frames, ckpt_frames, busy, self._wal_size_mb(),
                    )
                return True
            except Exception as exc:
                conn.close()
                raise exc
        except Exception as exc:
            logger.warning("WALCheckpointManager: checkpoint failed: %s", exc)
            return False
