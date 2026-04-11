from pathlib import Path

import pandas as pd

from duplicate_column_audit import audit_duplicate_columns, describe_duplicate_columns


def test_describe_duplicate_columns_reports_counts():
    frame = pd.DataFrame(
        [[1, 2, 3, 4]],
        columns=["alpha", "beta", "alpha", "beta"],
    )

    duplicates = describe_duplicate_columns(frame)

    assert duplicates == [
        {"column": "alpha", "count": 2},
        {"column": "beta", "count": 2},
    ]


def test_audit_duplicate_columns_writes_report(tmp_path):
    report_path = tmp_path / "duplicate_column_audit.csv"
    frame = pd.DataFrame(
        [[1, 2, 3]],
        columns=["signal", "signal", "market"],
    )

    duplicates = audit_duplicate_columns(
        frame,
        step_name="signals_add_macro_context",
        cycle_id="20260411T130000.000000Z",
        report_path=report_path,
        extra={"source": "macro_context"},
    )

    assert duplicates == [{"column": "signal", "count": 2}]
    written = pd.read_csv(report_path)
    assert len(written) == 1
    assert written.loc[0, "cycle_id"] == "20260411T130000.000000Z"
    assert written.loc[0, "step_name"] == "signals_add_macro_context"
    assert written.loc[0, "duplicate_column_count"] == 1
    assert written.loc[0, "duplicate_columns"] == "signal"
    assert written.loc[0, "source"] == "macro_context"


def test_audit_duplicate_columns_skips_clean_frames(tmp_path):
    report_path = tmp_path / "duplicate_column_audit.csv"
    frame = pd.DataFrame([{"signal": 1, "market": 2}])

    duplicates = audit_duplicate_columns(
        frame,
        step_name="clean_step",
        cycle_id="20260411T130100.000000Z",
        report_path=report_path,
    )

    assert duplicates == []
    assert not Path(report_path).exists()
