from __future__ import annotations

import json

import pandas as pd

from brain_coverage_report import (
    build_btc_brain_coverage_report,
    format_btc_brain_coverage_line,
)
from brain_paths import resolve_brain_context


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _seed_source_snapshots(shared_logs_dir, *, live_ready=True, regime_ready=True):
    live_rows = [
        {
            "timestamp": "2026-04-10T20:05:00Z",
            "btc_live_price_kalman": 100000.0 if live_ready else None,
            "btc_live_index_price_kalman": 99950.0 if live_ready else None,
            "btc_live_return_15m_kalman": None,
        },
        {
            "timestamp": "2026-04-10T20:06:00Z",
            "btc_live_price_kalman": 100010.0 if live_ready else None,
            "btc_live_index_price_kalman": 99960.0 if live_ready else None,
            "btc_live_return_15m_kalman": 0.002 if live_ready else None,
        },
    ]
    regime_rows = [
        {
            "timestamp": "2026-04-10T20:05:00Z",
            "btc_market_regime_score": 0.5 if regime_ready else None,
            "btc_market_regime_weight_stage1": None,
        },
        {
            "timestamp": "2026-04-10T20:07:00Z",
            "btc_market_regime_score": 0.55 if regime_ready else None,
            "btc_market_regime_weight_stage1": 0.6 if regime_ready else None,
        },
    ]
    _write_csv(shared_logs_dir / "btc_live_snapshot.csv", live_rows)
    _write_csv(shared_logs_dir / "technical_regime_snapshot.csv", regime_rows)


def test_brain_coverage_report_marks_not_ready_without_post_rollout_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("BTC_BRAIN_MIN_POST_ROLLOUT_CONTRACT_ROWS", "3")
    monkeypatch.setenv("BTC_BRAIN_MIN_POST_ROLLOUT_SEQUENCE_ROWS", "2")
    monkeypatch.setenv("BTC_BRAIN_MIN_CORE_FEATURE_COVERAGE_RATIO", "0.8")

    context = resolve_brain_context("btc", shared_logs_dir=tmp_path / "logs", shared_weights_dir=tmp_path / "weights")
    _seed_source_snapshots(context.shared_logs_dir, live_ready=True, regime_ready=True)

    _write_csv(
        context.logs_dir / "contract_targets.csv",
        [
            {
                "timestamp": "2026-04-10T19:55:00Z",
                "btc_live_price_kalman": 99900.0,
                "btc_live_index_price_kalman": 99850.0,
                "btc_live_return_15m_kalman": 0.001,
                "btc_market_regime_score": 0.4,
                "btc_market_regime_weight_stage1": 0.5,
            }
        ],
    )
    _write_csv(context.logs_dir / "historical_dataset.csv", [{"timestamp": "2026-04-10T19:55:00Z", "x": 1}])
    _write_csv(context.logs_dir / "sequence_dataset.csv", [{"timestamp": "2026-04-10T19:55:00Z", "y": 1}])

    report = build_btc_brain_coverage_report(brain_context=context)

    assert report["rollout_ready_from"] == "2026-04-10T20:07:00+00:00"
    assert report["post_rollout_contract_rows"] == 0
    assert report["retrain_confident_ready"] is False
    assert "collect 3 more post-rollout contract rows" in report["readiness_reason"]
    assert (context.logs_dir / "brain_coverage_report.csv").exists()
    assert (context.logs_dir / "brain_coverage_feature_report.csv").exists()
    summary = json.loads((context.logs_dir / "brain_coverage_summary.json").read_text(encoding="utf-8"))
    assert summary["retrain_confident_ready"] is False


def test_brain_coverage_report_marks_ready_with_sufficient_rows_and_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("BTC_BRAIN_MIN_POST_ROLLOUT_CONTRACT_ROWS", "3")
    monkeypatch.setenv("BTC_BRAIN_MIN_POST_ROLLOUT_SEQUENCE_ROWS", "2")
    monkeypatch.setenv("BTC_BRAIN_MIN_CORE_FEATURE_COVERAGE_RATIO", "0.8")

    context = resolve_brain_context("btc", shared_logs_dir=tmp_path / "logs", shared_weights_dir=tmp_path / "weights")
    _seed_source_snapshots(context.shared_logs_dir, live_ready=True, regime_ready=True)

    contract_rows = []
    for minute in range(8, 12):
        contract_rows.append(
            {
                "timestamp": f"2026-04-10T20:{minute:02d}:00Z",
                "btc_live_price_kalman": 100000.0 + minute,
                "btc_live_index_price_kalman": 99900.0 + minute,
                "btc_live_return_15m_kalman": 0.002,
                "btc_market_regime_score": 0.55,
                "btc_market_regime_weight_stage1": 0.60,
                "open_positions_count": 1,
            }
        )
    _write_csv(context.logs_dir / "contract_targets.csv", contract_rows)
    _write_csv(
        context.logs_dir / "historical_dataset.csv",
        [{"timestamp": "2026-04-10T20:08:00Z", "feature": 1}, {"timestamp": "2026-04-10T20:09:00Z", "feature": 2}],
    )
    _write_csv(
        context.logs_dir / "sequence_dataset.csv",
        [{"timestamp": "2026-04-10T20:08:00Z", "seq": 1}, {"timestamp": "2026-04-10T20:09:00Z", "seq": 2}],
    )

    report = build_btc_brain_coverage_report(brain_context=context)

    assert report["post_rollout_contract_rows"] == 4
    assert report["post_rollout_sequence_rows"] == 2
    assert report["core_feature_coverage_ratio"] == 1.0
    assert report["retrain_confident_ready"] is True
    assert report["readiness_reason"] == "btc brain has enough post-rollout coverage"
    assert "ready=yes" in format_btc_brain_coverage_line(report)


def test_brain_coverage_report_calls_out_missing_source_activation(tmp_path, monkeypatch):
    monkeypatch.setenv("BTC_BRAIN_MIN_POST_ROLLOUT_CONTRACT_ROWS", "1")
    monkeypatch.setenv("BTC_BRAIN_MIN_POST_ROLLOUT_SEQUENCE_ROWS", "1")
    monkeypatch.setenv("BTC_BRAIN_MIN_CORE_FEATURE_COVERAGE_RATIO", "0.5")

    context = resolve_brain_context("btc", shared_logs_dir=tmp_path / "logs", shared_weights_dir=tmp_path / "weights")
    _seed_source_snapshots(context.shared_logs_dir, live_ready=True, regime_ready=False)

    _write_csv(
        context.logs_dir / "contract_targets.csv",
        [{"timestamp": "2026-04-10T20:08:00Z", "btc_live_price_kalman": 100000.0}],
    )
    _write_csv(context.logs_dir / "historical_dataset.csv", [{"timestamp": "2026-04-10T20:08:00Z", "feature": 1}])
    _write_csv(context.logs_dir / "sequence_dataset.csv", [{"timestamp": "2026-04-10T20:08:00Z", "seq": 1}])

    report = build_btc_brain_coverage_report(brain_context=context)

    assert report["rollout_groups_ready"] is False
    assert report["rollout_ready_from"] == ""
    assert report["retrain_confident_ready"] is False
    assert "awaiting source activation: market_regime_source" in report["readiness_reason"]
