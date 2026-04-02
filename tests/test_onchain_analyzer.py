import unittest
from unittest.mock import patch

from onchain_analyzer import OnChainAnalyzer


class _FakeResponse:
    def __init__(self, payload=None, text=None, status_code=200):
        self._payload = payload
        self.text = text or ""
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class TestOnChainAnalyzer(unittest.TestCase):
    @patch("onchain_analyzer.requests.get")
    def test_analyze_combines_hashrate_and_mempool_metrics(self, mock_get):
        def _fake_get(url, timeout=10):
            if "coinmetrics" in url:
                return _FakeResponse({"data": [{"HashRate": 850000000.0}]})
            if url.endswith("/mempool"):
                return _FakeResponse({"count": 180000, "vsize": 12000000, "total_fee": 6000000})
            if url.endswith("/v1/fees/recommended"):
                return _FakeResponse({"fastestFee": 15, "halfHourFee": 10, "hourFee": 8, "economyFee": 3})
            if url.endswith("/v1/difficulty-adjustment"):
                return _FakeResponse({"difficultyChange": 4.2, "progressPercent": 88.0})
            if url.endswith("/blocks/tip/height"):
                return _FakeResponse(text="943305")
            raise AssertionError(f"Unexpected URL: {url}")

        mock_get.side_effect = _fake_get
        analyzer = OnChainAnalyzer(cache_ttl_seconds=1, mempool_cache_ttl_seconds=1)
        context = analyzer.analyze()

        self.assertEqual(context["btc_block_height"], 943305)
        self.assertEqual(context["btc_fee_fastest_satvb"], 15.0)
        self.assertEqual(context["btc_difficulty_change_pct"], 4.2)
        self.assertGreater(context["btc_network_activity_score"], 0.0)
        self.assertGreater(context["btc_network_stress_score"], 0.0)
        self.assertIn(context["onchain_network_health"], {"HEALTHY", "BUSY", "CONGESTED", "ACTIVE"})


if __name__ == "__main__":
    unittest.main()
