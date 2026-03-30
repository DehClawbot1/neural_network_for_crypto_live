import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from shadow_doa_resurrection import DOAResurrector


class TestDOAResurrection:
    def setup_method(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmp_dir.name) / "shadow_results.csv"
        pd.DataFrame([
            {
                "timestamp": "2025-03-23T12:00:00+00:00",
                "market_title": "Test Market",
                "token_id": "tok-1",
                "shadow_entry_price": 0.50,
                "outcome": "DOA",
                "meta_prob": 0.85,
                "expected_slip_bps": 100,
                "entry_slippage_bps": 120,
            }
        ]).to_csv(self.log_path, index=False)

        mock_bundle = {"model": MagicMock(), "features": []}
        self.joblib_patcher = patch("shadow_purgatory.joblib.load", return_value=mock_bundle)
        self.joblib_patcher.start()
        self.resurrector = DOAResurrector(log_path=self.log_path)

    def teardown_method(self):
        self.joblib_patcher.stop()
        self.tmp_dir.cleanup()

    @patch("shadow_purgatory.ResilientCLOBClient.get_trades_with_retry")
    def test_resurrection_identifies_alpha_leak(self, mock_trades):
        mock_trades.return_value = [
            {"timestamp": 1742731500, "price": 0.55}
        ]

        with patch.object(self.resurrector, "_report") as mock_report:
            self.resurrector.run_audit()

        results = mock_report.call_args[0][0]
        assert results.iloc[0]["outcome"] == "TP"
        assert results.iloc[0]["pnl"] == 0.04
