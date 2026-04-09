import tempfile
from pathlib import Path

import pandas as pd

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
