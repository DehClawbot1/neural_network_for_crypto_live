from pathlib import Path

import pandas as pd

from btc_live_price_tracker import BTCLivePriceTracker


class _FakeCandleDataService:
    def refresh_latest_closed_candles(self, interval, limit=2, timezone_name="UTC"):
        return pd.DataFrame({"close": [100.0, 100.0]})


def test_live_index_ready_requires_real_mark_and_index_feeds(tmp_path: Path):
    tracker = BTCLivePriceTracker(candle_data_service=_FakeCandleDataService(), logs_dir=str(tmp_path))
    tracker._fetch_binance_premium_index = lambda: {"mark_price": None, "index_price": None, "funding_rate": 0.0}
    tracker._fetch_binance_spot_price = lambda: 100.0
    tracker._fetch_coinbase_spot_price = lambda: 100.2
    tracker._fetch_kraken_spot_price = lambda: 99.8

    context = tracker.analyze()

    assert context["btc_live_index_price"] == 100.0
    assert context["btc_live_mark_price"] == 100.0
    assert context["btc_live_index_ready"] is False
    assert context["btc_live_index_feed_available"] is False
    assert context["btc_live_mark_feed_available"] is False


def test_live_index_ready_turns_true_when_real_premium_fields_exist(tmp_path: Path):
    tracker = BTCLivePriceTracker(candle_data_service=_FakeCandleDataService(), logs_dir=str(tmp_path))
    tracker._fetch_binance_premium_index = lambda: {"mark_price": 101.0, "index_price": 100.5, "funding_rate": 0.0}
    tracker._fetch_binance_spot_price = lambda: 100.0
    tracker._fetch_coinbase_spot_price = lambda: 100.2
    tracker._fetch_kraken_spot_price = lambda: 99.8

    context = tracker.analyze()

    assert context["btc_live_index_price"] == 100.5
    assert context["btc_live_mark_price"] == 101.0
    assert context["btc_live_index_ready"] is True
    assert context["btc_live_index_feed_available"] is True
    assert context["btc_live_mark_feed_available"] is True
