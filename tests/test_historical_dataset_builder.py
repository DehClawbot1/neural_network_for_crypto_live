import tempfile
from pathlib import Path

import pandas as pd
from trade_lifecycle import serialize_signal_snapshot

from historical_dataset_builder import HistoricalDatasetBuilder


def test_historical_dataset_builder_backfills_portfolio_context_from_equity_curve():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {
                    "market": "BTC Test",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "confidence": 0.42,
                }
            ]
        ).to_csv(logs / "signals.csv", index=False)
        pd.DataFrame(
            [
                {
                    "timestamp": "2026-04-09T04:59:00Z",
                    "open_positions": 2,
                    "gross_market_value": 5.42,
                    "entry_notional": 5.70,
                    "unrealized_pnl": -0.28,
                }
            ]
        ).to_csv(logs / "portfolio_equity_curve.csv", index=False)

        df = HistoricalDatasetBuilder(logs_dir=logs).build()

        assert not df.empty
        assert int(df.iloc[0]["open_positions_count"]) == 2
        assert round(float(df.iloc[0]["open_positions_negotiated_value_total"]), 2) == 5.70
        assert round(float(df.iloc[0]["open_positions_current_value_total"]), 2) == 5.42
        assert round(float(df.iloc[0]["open_positions_unrealized_pnl_total"]), 2) == -0.28


def test_historical_dataset_builder_merges_kalman_and_regime_context():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {
                    "market": "BTC Test",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "confidence": 0.42,
                }
            ]
        ).to_csv(logs / "signals.csv", index=False)
        pd.DataFrame(
            [
                {
                    "btc_live_timestamp": "2026-04-09T04:59:30Z",
                    "btc_live_mark_price_kalman": 68201.4,
                    "btc_live_return_15m_kalman": 0.0062,
                    "btc_live_confluence_kalman": 0.67,
                }
            ]
        ).to_csv(logs / "btc_live_snapshot.csv", index=False)
        pd.DataFrame(
            [
                {
                    "technical_timestamp": "2026-04-09T04:59:45Z",
                    "btc_market_regime_label": "trend",
                    "btc_market_regime_score": 0.74,
                    "btc_market_regime_trend_score": 0.81,
                    "btc_market_regime_confidence_multiplier": 1.08,
                }
            ]
        ).to_csv(logs / "technical_regime_snapshot.csv", index=False)

        df = HistoricalDatasetBuilder(logs_dir=logs).build()

        assert not df.empty
        assert round(float(df.iloc[0]["btc_live_mark_price_kalman"]), 1) == 68201.4
        assert round(float(df.iloc[0]["btc_live_return_15m_kalman"]), 4) == 0.0062
        assert df.iloc[0]["btc_market_regime_label"] == "trend"
        assert round(float(df.iloc[0]["btc_market_regime_confidence_multiplier"]), 2) == 1.08


def test_historical_dataset_builder_backfills_missing_signal_columns_from_entry_snapshots():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {
                    "market": "BTC Test",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                }
            ]
        ).to_csv(logs / "signals.csv", index=False)
        snapshot_json, feature_count = serialize_signal_snapshot(
            {
                "timestamp": "2026-04-09T05:00:00Z",
                "market": "BTC Test",
                "trader_wallet": "0xabc",
                "token_id": "tok-1",
                "condition_id": "cond-1",
                "outcome_side": "YES",
                "wallet_quality_score": 0.81,
                "btc_live_mark_price_kalman": 68222.4,
            }
        )
        pd.DataFrame(
            [
                {
                    "position_id": "tok-1|cond-1|YES",
                    "market": "BTC Test",
                    "market_title": "BTC Test",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "opened_at": "2026-04-09T05:00:00Z",
                    "status": "OPEN",
                    "entry_signal_snapshot_json": snapshot_json,
                    "entry_signal_snapshot_feature_count": feature_count,
                    "entry_signal_snapshot_version": 1,
                }
            ]
        ).to_csv(logs / "positions.csv", index=False)

        df = HistoricalDatasetBuilder(logs_dir=logs).build()

        assert not df.empty
        assert round(float(df.iloc[0]["wallet_quality_score"]), 2) == 0.81
        assert round(float(df.iloc[0]["btc_live_mark_price_kalman"]), 1) == 68222.4
        assert bool(df.iloc[0]["entry_snapshot_backfilled"]) is True
