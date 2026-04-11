from datetime import datetime, timezone

from weather_temperature_markets import (
    is_weather_temperature_market,
    parse_weather_temperature_market_text,
)


def test_parse_weather_temperature_threshold_market_in_fahrenheit():
    parsed = parse_weather_temperature_market_text(
        "Will the highest temperature in NYC be 64°F or higher on March 10?",
        reference_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )

    assert parsed["market_family"] == "weather_temperature_threshold"
    assert parsed["weather_parseable"] is True
    assert parsed["weather_location"] == "NYC"
    assert parsed["weather_question_type"] == "threshold"
    assert parsed["weather_temp_unit"] == "F"
    assert parsed["weather_event_date_local"] == "2026-03-10"
    assert round(parsed["weather_lower_c"], 2) == 17.78
    assert parsed["weather_upper_c"] is None


def test_parse_weather_temperature_range_market_in_fahrenheit():
    parsed = parse_weather_temperature_market_text(
        "Will the highest temperature in Dallas be between 52-53°F on January 8?",
        reference_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert parsed["market_family"] == "weather_temperature_range"
    assert parsed["weather_parseable"] is True
    assert parsed["weather_location"] == "Dallas"
    assert parsed["weather_question_type"] == "range"
    assert parsed["weather_temp_unit"] == "F"
    assert parsed["weather_event_date_local"] == "2026-01-08"
    assert round(parsed["weather_lower_c"], 2) == 11.11
    assert round(parsed["weather_upper_c"], 2) == 11.67
    assert round(parsed["weather_interval_width_c"], 2) == 0.56


def test_parse_weather_temperature_threshold_market_in_celsius():
    parsed = parse_weather_temperature_market_text(
        "Will the highest temperature in London be 10°C or higher on January 28?",
        reference_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert parsed["market_family"] == "weather_temperature_threshold"
    assert parsed["weather_parseable"] is True
    assert parsed["weather_temp_unit"] == "C"
    assert parsed["weather_lower_c"] == 10.0
    assert parsed["weather_event_date_local"] == "2026-01-28"


def test_parse_high_temperature_variation_market():
    parsed = parse_weather_temperature_market_text(
        "Will the high temperature in New York's Central Park be 60Â°F or higher on November 15th, 2026?",
        reference_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    )

    assert parsed["market_family"] == "weather_temperature_threshold"
    assert parsed["weather_parseable"] is True
    assert parsed["weather_location"] == "New York's Central Park"
    assert parsed["weather_temp_unit"] == "F"


def test_is_weather_temperature_market_accepts_high_temperature_phrase():
    assert is_weather_temperature_market(
        {"question": "Will the high temperature in Dallas be 70°F or higher on April 11?"}
    ) is True
    assert is_weather_temperature_market({"question": "Who will win the match?"}) is False
