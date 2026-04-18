from family_live_runtime_policy import build_family_live_runtime_policy


def test_build_family_live_runtime_policy_respects_family_specific_live_requirements():
    payload = {
        "family_configs": {
            "btc": {
                "supported_models": ["entry_edge", "fill_probability", "slippage_liquidity", "exit_quality", "regime_calibration"],
                "required_live_entry_models": ["entry_edge", "fill_probability", "slippage_liquidity", "regime_calibration"],
                "required_live_exit_models": ["exit_quality"],
            },
            "weather_temperature": {
                "supported_models": ["entry_edge", "exit_quality", "regime_calibration"],
                "required_live_entry_models": ["entry_edge", "regime_calibration"],
                "required_live_exit_models": ["exit_quality"],
            },
        },
        "results": [
            {"family": "btc", "model_name": "entry_edge", "promoted": True},
            {"family": "btc", "model_name": "fill_probability", "promoted": True},
            {"family": "btc", "model_name": "slippage_liquidity", "promoted": True},
            {"family": "btc", "model_name": "exit_quality", "promoted": True},
            {"family": "btc", "model_name": "regime_calibration", "promoted": True},
            {"family": "weather_temperature", "model_name": "regime_calibration", "promoted": True},
        ],
    }

    states = build_family_live_runtime_policy(payload)

    btc = states["btc"]
    weather = states["weather_temperature"]

    assert btc.entry_ready is True
    assert btc.exit_ready is True
    assert btc.live_ready is True
    assert weather.entry_ready is False
    assert weather.exit_ready is False
    assert weather.missing_entry_models == ("entry_edge",)
    assert weather.missing_exit_models == ("exit_quality",)
    assert "missing_entry_models" in weather.reason


def test_build_family_live_runtime_policy_without_signal_marks_states_not_ready():
    states = build_family_live_runtime_policy({})

    assert states["btc"].signal_present is False
    assert states["btc"].entry_ready is False
    assert states["weather_temperature"].signal_present is False
    assert states["weather_temperature"].entry_ready is False
