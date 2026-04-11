from weather_temperature_markets import is_weather_temperature_market


def test_highest_temperature_question():
    assert is_weather_temperature_market({"question": "Will the highest temperature in NYC be 64F or higher?"}) is True


def test_temperature_will_be_question():
    assert is_weather_temperature_market({"question": "Will the temperature will be above 30C in London?"}) is True


def test_temperature_be_between_question():
    assert is_weather_temperature_market({"question": "Will the temperature be between 50-60F in Dallas?"}) is True


def test_unrelated_question():
    assert is_weather_temperature_market({"question": "will BTC go up"}) is False


def test_empty_dict():
    assert is_weather_temperature_market({}) is False


def test_none_input():
    assert is_weather_temperature_market(None) is False
