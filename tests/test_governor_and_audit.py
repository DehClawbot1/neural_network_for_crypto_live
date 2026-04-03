from pathlib import Path

import pandas as pd

from performance_governor import PerformanceGovernor
from trade_lifecycle_audit import TradeLifecycleAuditor


def test_performance_governor_degrades_on_bad_recent_losses(tmp_path):
    logs_dir = Path(tmp_path)
    closed = pd.DataFrame(
        [
            {
                "closed_at": f"2026-04-03T00:{i:02d}:00Z",
                "realized_pnl": -0.5 if i < 40 else 0.1,
                "close_reason": "rl_exit",
                "exit_reason_family": "rl_discretionary",
                "learning_eligible": True,
                "entry_context_complete": True,
                "operational_close_flag": False,
                "signal_label": "HIGHEST-RANKED PAPER SIGNAL",
            }
            for i in range(50)
        ]
    )
    closed.to_csv(logs_dir / "closed_positions.csv", index=False)

    state = PerformanceGovernor(logs_dir=str(logs_dir)).evaluate()

    assert state["governor_level"] >= 1
    assert state["live_profit_factor"] < 1.0


def test_trade_lifecycle_audit_reports_operational_and_unknown_ratios(tmp_path):
    logs_dir = Path(tmp_path)
    closed = pd.DataFrame(
        [
            {
                "closed_at": "2026-04-03T00:00:00Z",
                "realized_pnl": -0.1,
                "close_reason": "external_manual_close",
                "signal_label": "UNKNOWN",
                "entry_context_complete": False,
                "learning_eligible": False,
                "operational_close_flag": True,
                "exit_reason_family": "operational",
            },
            {
                "closed_at": "2026-04-03T00:05:00Z",
                "realized_pnl": 0.2,
                "close_reason": "take_profit_roi",
                "signal_label": "HIGHEST-RANKED PAPER SIGNAL",
                "entry_context_complete": True,
                "learning_eligible": True,
                "operational_close_flag": False,
                "exit_reason_family": "profit_take",
            },
        ]
    )
    closed.to_csv(logs_dir / "closed_positions.csv", index=False)

    TradeLifecycleAuditor(logs_dir=str(logs_dir)).build_reports()
    audit_df = pd.read_csv(logs_dir / "trade_lifecycle_audit.csv")

    assert not audit_df.empty
    latest = audit_df.iloc[-1]
    assert latest["operational_close_ratio"] > 0
    assert latest["unknown_signal_label_ratio"] > 0
