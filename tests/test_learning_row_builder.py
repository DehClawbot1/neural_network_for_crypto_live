import tempfile
from pathlib import Path

import pandas as pd

from learning_row_builder import LearningRowBuilder


class TestLearningRowBuilder:
    def setup_method(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.logs_dir = Path(self.test_dir.name)

    def teardown_method(self):
        self.test_dir.cleanup()

    def test_empty_logs_returns_empty(self):
        builder = LearningRowBuilder(logs_dir=str(self.logs_dir))
        result = builder.build()
        assert result.empty

    def test_builds_from_closed_positions(self):
        closed = pd.DataFrame([{
            "token_id": "tok1",
            "condition_id": "cond1",
            "outcome_side": "YES",
            "entry_price": 0.45,
            "exit_price": 0.52,
            "size_usdc": 5.0,
            "realized_pnl": 0.35,
            "opened_at": "2026-04-01T10:00:00Z",
            "closed_at": "2026-04-01T10:30:00Z",
            "close_reason": "take_profit",
            "status": "CLOSED",
            "signal_label": "STRONG",
            "confidence_at_entry": 0.72,
            "entry_signal_snapshot_json": '{"trader_wallet":"w1","btc_live_price":82000}',
        }])
        closed.to_csv(self.logs_dir / "closed_positions.csv", index=False)

        builder = LearningRowBuilder(logs_dir=str(self.logs_dir))
        result = builder.build()

        assert len(result) == 1
        assert result.iloc[0]["token_id"] == "tok1"
        assert result.iloc[0]["entry_price"] == 0.45
        # snapshot fields should be exploded
        assert result.iloc[0]["trader_wallet"] == "w1"
        assert float(result.iloc[0]["btc_live_price"]) == 82000

    def test_hold_time_computed(self):
        closed = pd.DataFrame([{
            "token_id": "tok2",
            "outcome_side": "NO",
            "opened_at": "2026-04-01T10:00:00Z",
            "closed_at": "2026-04-01T11:00:00Z",
            "status": "CLOSED",
        }])
        closed.to_csv(self.logs_dir / "closed_positions.csv", index=False)

        builder = LearningRowBuilder(logs_dir=str(self.logs_dir))
        result = builder.build()

        assert len(result) == 1
        assert abs(result.iloc[0]["hold_time_minutes"] - 60.0) < 0.01

    def test_write_creates_file(self):
        closed = pd.DataFrame([{
            "token_id": "tok3",
            "outcome_side": "YES",
            "opened_at": "2026-04-01T10:00:00Z",
            "closed_at": "2026-04-01T10:15:00Z",
            "status": "CLOSED",
        }])
        closed.to_csv(self.logs_dir / "closed_positions.csv", index=False)

        builder = LearningRowBuilder(logs_dir=str(self.logs_dir))
        result = builder.write()

        assert not result.empty
        assert (self.logs_dir / "learning_dataset.csv").exists()
