import tempfile
from pathlib import Path

import pandas as pd

from contract_target_builder import ContractTargetBuilder


def _write_common_files(logs: Path):
    pd.DataFrame([
        {"market": "BTC Test", "timestamp": "2026-03-22T00:00:00Z", "token_id": "1", "side": "YES", "confidence": 0.8}
    ]).to_csv(logs / "raw_candidates.csv", index=False)
    pd.DataFrame([
        {"question": "BTC Test", "yes_token_id": "1", "no_token_id": "2"}
    ]).to_csv(logs / "markets.csv", index=False)


def test_forward_minutes_changes_forward_return():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        _write_common_files(logs)
        pd.DataFrame([
            {"token_id": "1", "timestamp": "2026-03-22T00:00:00Z", "price": 0.40},
            {"token_id": "1", "timestamp": "2026-03-22T00:10:00Z", "price": 0.50},
            {"token_id": "1", "timestamp": "2026-03-22T00:20:00Z", "price": 0.60},
            {"token_id": "1", "timestamp": "2026-03-22T01:00:00Z", "price": 0.30},
        ]).to_csv(logs / "clob_price_history.csv", index=False)

        builder = ContractTargetBuilder(logs_dir=logs)
        df_15 = builder.build(forward_minutes=15, max_hold_minutes=60)
        df_30 = builder.build(forward_minutes=30, max_hold_minutes=60)

        assert not df_15.empty
        assert not df_30.empty
        assert round(float(df_15.iloc[0]["forward_return_15m"]), 4) == 0.25
        assert round(float(df_30.iloc[0]["forward_return_15m"]), 4) == 0.50


def test_uses_market_token_mapping_when_signal_token_missing():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame([
            {"market": "BTC Test", "timestamp": "2026-03-22T00:00:00Z", "side": "YES", "confidence": 0.8}
        ]).to_csv(logs / "raw_candidates.csv", index=False)
        pd.DataFrame([
            {"question": "BTC Test", "yes_token_id": "1", "no_token_id": "2"}
        ]).to_csv(logs / "markets.csv", index=False)
        pd.DataFrame([
            {"token_id": "1", "timestamp": "2026-03-22T00:00:00Z", "price": 0.40},
            {"token_id": "1", "timestamp": "2026-03-22T00:10:00Z", "price": 0.50},
        ]).to_csv(logs / "clob_price_history.csv", index=False)

        df = ContractTargetBuilder(logs_dir=logs).build(forward_minutes=15, max_hold_minutes=15)

        assert not df.empty
        assert str(df.iloc[0]["token_id"]) == "1"
