from weather_temperature_forecast import WeatherForecastService


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_market_forecast_returns_interval_metrics(monkeypatch):
    service = WeatherForecastService(timeout=1)

    def fake_geocode(location, country_hint=None):
        return {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timezone": "America/New_York",
            "name": "New York City",
            "country": "United States",
            "country_code": "US",
            "admin1": "New York",
        }

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(
            {
                "daily": {
                    "time": ["2026-03-10", "2026-03-11"],
                    "temperature_2m_max": [19.5, 21.0],
                }
            }
        )

    monkeypatch.setattr(service, "geocode_location", fake_geocode)
    monkeypatch.setattr(service.session, "get", fake_get)

    forecast = service.fetch_market_forecast(
        {
            "weather_location": "NYC",
            "weather_event_date_local": "2026-03-10",
            "weather_lower_c": 17.0,
            "weather_upper_c": None,
        }
    )

    assert forecast["forecast_ready"] is True
    assert forecast["forecast_source"] == "open_meteo"
    assert forecast["weather_country"] == "United States"
    assert forecast["weather_resolution_timezone"] == "America/New_York"
    assert forecast["forecast_max_temp_c"] == 19.5
    assert forecast["forecast_margin_to_lower_c"] == 2.5
    assert 0.0 <= forecast["forecast_p_hit_interval"] <= 1.0


def test_fetch_market_forecast_marks_missing_location_as_unready():
    service = WeatherForecastService(timeout=1)

    forecast = service.fetch_market_forecast({"weather_location": "", "weather_event_date_local": ""})

    assert forecast["forecast_ready"] is False
    assert forecast["forecast_missing_reason"] == "missing_location_or_event_date"
    assert forecast["forecast_stale"] is True
