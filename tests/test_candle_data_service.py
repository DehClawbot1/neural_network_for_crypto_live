import pandas as pd

from candle_data_service import CandleDataService


def test_drop_unfinished_candles_excludes_open_bar():
    service = CandleDataService()
    frame = pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                [
                    "2026-04-03T10:00:00Z",
                    "2026-04-03T10:15:00Z",
                    "2026-04-03T10:30:00Z",
                ],
                utc=True,
            ),
            "close_time": pd.to_datetime(
                [
                    "2026-04-03T10:14:59Z",
                    "2026-04-03T10:29:59Z",
                    "2026-04-03T10:44:59Z",
                ],
                utc=True,
            ),
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10.0, 12.0, 14.0],
            "quote_volume": [15.0, 20.0, 25.0],
            "trade_count": [10, 12, 14],
            "taker_buy_base_volume": [5.0, 6.0, 7.0],
            "taker_buy_quote_volume": [7.5, 8.0, 9.0],
        }
    )

    filtered = service._drop_unfinished_candles(
        frame,
        "15m",
        now_utc=pd.Timestamp("2026-04-03T10:37:00Z"),
    )

    assert len(filtered) == 2
    assert filtered["open_time"].iloc[-1] == pd.Timestamp("2026-04-03T10:15:00Z")


def test_convert_timezone_keeps_timezone_aware_timestamps():
    service = CandleDataService()
    frame = pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2026-04-03T10:00:00Z"], utc=True),
            "close_time": pd.to_datetime(["2026-04-03T10:14:59Z"], utc=True),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
            "quote_volume": [15.0],
            "trade_count": [10],
            "taker_buy_base_volume": [5.0],
            "taker_buy_quote_volume": [7.5],
        }
    )

    converted = service._convert_timezone(frame, "Europe/Lisbon")

    assert str(converted["open_time"].dt.tz) == "Europe/Lisbon"
    assert str(converted["close_time"].dt.tz) == "Europe/Lisbon"


def test_clean_frame_sorts_and_deduplicates_history():
    service = CandleDataService()
    raw = pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                [
                    "2026-04-03T10:15:00Z",
                    "2026-04-03T10:00:00Z",
                    "2026-04-03T10:15:00Z",
                ],
                utc=True,
            ),
            "close_time": pd.to_datetime(
                [
                    "2026-04-03T10:29:59Z",
                    "2026-04-03T10:14:59Z",
                    "2026-04-03T10:29:59Z",
                ],
                utc=True,
            ),
            "open": [2.0, 1.0, 2.1],
            "high": [3.0, 2.0, 3.1],
            "low": [1.5, 0.5, 1.6],
            "close": [2.5, 1.5, 2.6],
            "volume": [12.0, 10.0, 13.0],
            "quote_volume": [20.0, 15.0, 21.0],
            "trade_count": [12, 10, 13],
            "taker_buy_base_volume": [6.0, 5.0, 7.0],
            "taker_buy_quote_volume": [8.0, 7.5, 8.5],
        }
    )

    cleaned = service._clean_frame(raw)

    assert len(cleaned) == 2
    assert cleaned["open_time"].iloc[0] == pd.Timestamp("2026-04-03T10:00:00Z")
    assert cleaned["open_time"].iloc[1] == pd.Timestamp("2026-04-03T10:15:00Z")
    assert cleaned["close"].iloc[1] == 2.6
