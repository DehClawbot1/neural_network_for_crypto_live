import tempfile
from pathlib import Path

import pandas as pd

from feature_ablation import FeatureAblationReporter


def test_feature_ablation_prefers_historical_dataset_backbone():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        historical_rows = []
        for idx in range(60):
            historical_rows.append(
                {
                    "timestamp": f"2026-04-11T10:{idx:02d}:00Z",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "trader_wallet": "0xabc",
                    "market_title": "BTC Test",
                    "wallet_trade_count_30d": 5 + (idx % 3),
                    "wallet_alpha_30d": 0.1 + (idx * 0.001),
                    "spread": 0.02,
                    "current_price": 0.45,
                    "target_up": int(idx % 2 == 0),
                    "forward_return_15m": 0.01 * (1 if idx % 2 == 0 else -1),
                }
            )
        pd.DataFrame(historical_rows).to_csv(logs / "historical_dataset.csv", index=False)

        contract_rows = []
        for idx in range(60):
            contract_rows.append(
                {
                    "timestamp": f"2026-04-11T10:{idx:02d}:00Z",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "trader_wallet": "0xabc",
                    "market_title": "BTC Test",
                    "wallet_trade_count_30d": None,
                    "wallet_alpha_30d": None,
                    "spread": None,
                    "target_up": int(idx % 2 == 0),
                    "forward_return_15m": 0.01 * (1 if idx % 2 == 0 else -1),
                }
            )
        pd.DataFrame(contract_rows).to_csv(logs / "contract_targets.csv", index=False)

        reporter = FeatureAblationReporter(logs_dir=logs)
        dataset = reporter._load_dataset()

        assert not dataset.empty
        assert dataset["wallet_trade_count_30d"].notna().all()
        assert dataset["wallet_alpha_30d"].notna().all()
        assert dataset["spread"].notna().all()


def test_feature_ablation_coerces_string_booleans_to_numeric():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        rows = []
        for idx in range(60):
            rows.append(
                {
                    "timestamp": f"2026-04-11T10:{idx:02d}:00Z",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "trader_wallet": "0xabc",
                    "market_title": "BTC Test",
                    "current_price": 0.45 + (idx * 0.001),
                    "spread": 0.02,
                    "wallet_watchlist_approved": "True" if idx % 2 == 0 else "False",
                    "forecast_ready": "False" if idx % 3 == 0 else "True",
                    "target_up": int(idx % 2 == 0),
                    "forward_return_15m": 0.01 * (1 if idx % 2 == 0 else -1),
                }
            )
        pd.DataFrame(rows).to_csv(logs / "historical_dataset.csv", index=False)
        pd.DataFrame(rows).to_csv(logs / "contract_targets.csv", index=False)

        reporter = FeatureAblationReporter(logs_dir=logs)
        df = reporter._load_dataset()
        numeric = reporter._numeric_feature_frame(
            df,
            ["current_price", "spread", "wallet_watchlist_approved", "forecast_ready"],
        )

        assert list(numeric.columns) == ["current_price", "spread", "wallet_watchlist_approved", "forecast_ready"]
        assert numeric["wallet_watchlist_approved"].isin([0.0, 1.0]).all()
        assert numeric["forecast_ready"].isin([0.0, 1.0]).all()
