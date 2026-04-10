from types import SimpleNamespace

from weather_temperature_guard import find_conflicting_weather_temperature_position, weather_positions_conflict


def test_weather_positions_conflict_detects_non_overlapping_intervals():
    candidate = {
        "market_family": "weather_temperature_threshold",
        "weather_location": "London",
        "weather_event_date_local": "2026-01-28",
        "weather_lower_c": 15.0,
        "outcome_side": "YES",
    }
    existing = {
        "market_family": "weather_temperature_range",
        "weather_location": "London",
        "weather_event_date_local": "2026-01-28",
        "weather_lower_c": 10.0,
        "weather_upper_c": 12.0,
        "outcome_side": "YES",
    }

    assert weather_positions_conflict(candidate, existing) is True


def test_find_conflicting_weather_temperature_position_enforces_cluster_cap():
    candidate = {
        "market_family": "weather_temperature_threshold",
        "weather_location": "Dallas",
        "weather_event_date_local": "2026-01-08",
        "weather_lower_c": 11.0,
        "outcome_side": "YES",
    }
    existing_trade = SimpleNamespace(
        market_family="weather_temperature_threshold",
        weather_location="Dallas",
        weather_event_date_local="2026-01-08",
        weather_lower_c=10.0,
        weather_upper_c=None,
        outcome_side="YES",
        market="Dallas threshold",
        condition_id="cond-1",
    )

    conflict = find_conflicting_weather_temperature_position(candidate, [existing_trade], cluster_cap=1)

    assert conflict is not None
    assert conflict["reason"] == "weather_temperature_cluster_cap_reached"
    assert conflict["cluster_cap"] == 1


def test_find_conflicting_weather_temperature_position_detects_interval_conflict_first():
    candidate = {
        "market_family": "weather_temperature_threshold",
        "weather_location": "NYC",
        "weather_event_date_local": "2026-03-10",
        "weather_lower_c": 18.0,
        "outcome_side": "YES",
    }
    existing_trade = SimpleNamespace(
        market_family="weather_temperature_threshold",
        weather_location="NYC",
        weather_event_date_local="2026-03-10",
        weather_lower_c=16.0,
        weather_upper_c=None,
        outcome_side="NO",
        market="NYC below 16C",
        condition_id="cond-2",
    )

    conflict = find_conflicting_weather_temperature_position(candidate, [existing_trade], cluster_cap=2)

    assert conflict is not None
    assert conflict["reason"] == "weather_temperature_interval_conflict"
