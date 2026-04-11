import tempfile
from pathlib import Path

import pandas as pd

from sequence_feature_builder import SequenceFeatureBuilder


def test_sequence_builder_uses_historical_dataset_backbone_for_lag_features():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {
                    "timestamp": "2026-04-11T10:00:00Z",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "trader_wallet": "0xabc",
                    "market_title": "BTC Test",
                    "entry_price": 0.41,
                    "spread": 0.02,
                    "wallet_alpha_30d": 0.11,
                    "wallet_trade_count_30d": 4,
                },
                {
                    "timestamp": "2026-04-11T10:05:00Z",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "trader_wallet": "0xabc",
                    "market_title": "BTC Test",
                    "entry_price": 0.43,
                    "spread": 0.03,
                    "wallet_alpha_30d": 0.17,
                    "wallet_trade_count_30d": 5,
                },
                {
                    "timestamp": "2026-04-11T10:10:00Z",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "trader_wallet": "0xabc",
                    "market_title": "BTC Test",
                    "entry_price": 0.45,
                    "spread": 0.04,
                    "wallet_alpha_30d": 0.21,
                    "wallet_trade_count_30d": 6,
                },
            ]
        ).to_csv(logs / "historical_dataset.csv", index=False)

        pd.DataFrame(
            [
                {
                    "timestamp": "2026-04-11T10:10:00Z",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "trader_wallet": "0xabc",
                    "market_title": "BTC Test",
                    "forward_return_15m": 0.08,
                    "tp_before_sl_60m": 1,
                    "target_up": 1,
                    "entry_price": 0.45,
                }
            ]
        ).to_csv(logs / "contract_targets.csv", index=False)

        result = SequenceFeatureBuilder(logs_dir=logs).build(lags=(1,))

        assert len(result) == 1
        assert "wallet_alpha_30d_lag_1" in result.columns
        assert "wallet_trade_count_30d_lag_1" in result.columns
        assert "spread_lag_1" in result.columns
        assert float(result.iloc[0]["wallet_alpha_30d_lag_1"]) == 0.17
        assert float(result.iloc[0]["wallet_trade_count_30d_lag_1"]) == 5.0
        assert float(result.iloc[0]["spread_lag_1"]) == 0.03
        assert int(result.iloc[0]["tp_before_sl_60m"]) == 1


def test_sequence_builder_falls_back_to_contract_targets_when_no_historical_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {"timestamp": "2026-04-11T10:00:00Z", "token_id": "tok-1", "entry_price": 0.40, "spread": 0.02, "tp_before_sl_60m": 0},
                {"timestamp": "2026-04-11T10:05:00Z", "token_id": "tok-1", "entry_price": 0.41, "spread": 0.03, "tp_before_sl_60m": 0},
                {"timestamp": "2026-04-11T10:10:00Z", "token_id": "tok-1", "entry_price": 0.42, "spread": 0.04, "tp_before_sl_60m": 1},
            ]
        ).to_csv(logs / "contract_targets.csv", index=False)

        result = SequenceFeatureBuilder(logs_dir=logs).build(lags=(1,))

        assert len(result) == 2
        assert "entry_price_lag_1" in result.columns
