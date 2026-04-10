import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from blocker_profit_audit import blocker_replay_report, regime_performance_report


class TestBlockerReplayReport:
    def test_empty_when_no_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = blocker_replay_report(logs_dir=tmpdir)
            assert result.empty

    def test_replay_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            decisions = pd.DataFrame([
                {"token_id": "t1", "final_decision": "REJECTED", "reject_reason": "rule_veto", "gate": "rule"},
                {"token_id": "t2", "final_decision": "REJECTED", "reject_reason": "freeze", "gate": "freeze"},
                {"token_id": "t1", "final_decision": "ENTRY_FILLED", "reject_reason": None, "gate": None},
            ])
            targets = pd.DataFrame([
                {"token_id": "t1", "forward_return_15m": 0.05, "tp_before_sl_60m": 1},
                {"token_id": "t2", "forward_return_15m": -0.02, "tp_before_sl_60m": 0},
            ])
            decisions.to_csv(Path(tmpdir) / "candidate_decisions.csv", index=False)
            targets.to_csv(Path(tmpdir) / "contract_targets.csv", index=False)

            result = blocker_replay_report(logs_dir=tmpdir)
            assert not result.empty
            assert "reject_reason" in result.columns
            assert "mean_forward_return_15m" in result.columns


class TestRegimePerformanceReport:
    def test_empty_when_no_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = regime_performance_report(logs_dir=tmpdir)
            assert result.empty

    def test_regime_report_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            closed = pd.DataFrame([
                {"technical_regime_bucket": "calm", "net_realized_pnl": 0.5},
                {"technical_regime_bucket": "calm", "net_realized_pnl": -0.2},
                {"technical_regime_bucket": "volatile", "net_realized_pnl": -0.8},
            ])
            closed.to_csv(Path(tmpdir) / "closed_positions.csv", index=False)

            result = regime_performance_report(logs_dir=tmpdir)
            assert len(result) == 2
            calm = result[result["regime"] == "calm"]
            assert calm.iloc[0]["n_trades"] == 2
