from pathlib import Path

import pandas as pd

from leaderboard_service import PolymarketLeaderboardService


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return _FakeResponse(self.payload)


def test_fetch_leaderboard_writes_btc_audit_file(tmp_path):
    session = _FakeSession(
        [
            {"proxyWallet": "0xabc", "pnl": 1500, "username": "alpha"},
            {"proxyWallet": "0xdef", "pnl": 700, "username": "beta"},
        ]
    )
    service = PolymarketLeaderboardService(logs_dir=str(tmp_path), ttl_seconds=600, session=session)

    rows = service.fetch_leaderboard(category="CRYPTO", limit=2, approved_wallets={"0xabc"})

    assert len(rows.index) == 2
    assert rows.iloc[0]["wallet"] == "0xabc"
    assert bool(rows.iloc[0]["approved"]) is True
    assert bool(rows.iloc[1]["approved"]) is False
    assert (tmp_path / "leaderboard_wallets_btc.csv").exists()


def test_fetch_leaderboard_uses_ttl_cache(tmp_path):
    session = _FakeSession([{"proxyWallet": "0xabc", "pnl": 1}])
    service = PolymarketLeaderboardService(logs_dir=str(tmp_path), ttl_seconds=600, session=session)

    service.fetch_leaderboard(category="CRYPTO", limit=1)
    service.fetch_leaderboard(category="CRYPTO", limit=1)

    assert session.calls == 1


def test_merge_with_overrides_keeps_dynamic_and_manual_rows(tmp_path):
    service = PolymarketLeaderboardService(logs_dir=str(tmp_path), ttl_seconds=600, session=_FakeSession([]))
    dynamic = pd.DataFrame([{"wallet": "0xlive", "label": "live", "source": "leaderboard_api", "enabled": True}])
    overrides = pd.DataFrame([{"wallet": "0xmanual", "label": "manual", "source": "manual_override_csv", "enabled": True}])

    merged = service.merge_with_overrides(category="WEATHER", dynamic_rows=dynamic, override_rows=overrides)

    assert set(merged["wallet"].astype(str)) == {"0xlive", "0xmanual"}
    assert Path(tmp_path / "leaderboard_wallets_weather.csv").exists()
