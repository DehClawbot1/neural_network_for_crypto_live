import joblib
import pandas as pd

from weather_temperature_trainer import WeatherTemperatureTrainer


def test_weather_temperature_trainer_builds_market_family_specific_artifact(tmp_path):
    logs_dir = tmp_path / "logs"
    weights_dir = tmp_path / "weights"
    logs_dir.mkdir()
    weights_dir.mkdir()
    df = pd.DataFrame(
        [
            {
                "market_family": "weather_temperature_threshold",
                "target_up": 1,
                "wallet_temp_hit_rate_90d": 0.8,
                "wallet_temp_realized_pnl_90d": 120.0,
                "wallet_region_score": 0.7,
                "wallet_temp_range_skill": 0.6,
                "wallet_temp_threshold_skill": 0.9,
                "wallet_quality_score": 0.85,
                "wallet_state_confidence": 0.82,
                "wallet_state_freshness_score": 0.9,
                "wallet_size_change_score": 0.5,
                "wallet_agreement_score": 0.7,
                "current_price": 0.42,
                "spread": 0.03,
                "time_left": 0.6,
                "liquidity_score": 0.4,
                "volume_score": 0.3,
                "market_structure_score": 0.55,
                "execution_quality_score": 0.61,
                "forecast_p_hit_interval": 0.72,
                "forecast_margin_to_lower_c": 2.0,
                "forecast_margin_to_upper_c": 0.0,
                "forecast_uncertainty_c": 1.2,
                "forecast_drift_c": 0.1,
                "weather_fair_probability_yes": 0.72,
                "weather_fair_probability_side": 0.72,
                "weather_market_probability": 0.42,
                "weather_forecast_edge": 0.30,
                "weather_forecast_margin_score": 0.66,
                "weather_forecast_stability_score": 0.92,
            },
            {
                "market_family": "weather_temperature_range",
                "target_up": 0,
                "wallet_temp_hit_rate_90d": 0.55,
                "wallet_temp_realized_pnl_90d": 20.0,
                "wallet_region_score": 0.5,
                "wallet_temp_range_skill": 0.65,
                "wallet_temp_threshold_skill": 0.45,
                "wallet_quality_score": 0.58,
                "wallet_state_confidence": 0.51,
                "wallet_state_freshness_score": 0.8,
                "wallet_size_change_score": 0.2,
                "wallet_agreement_score": 0.4,
                "current_price": 0.67,
                "spread": 0.08,
                "time_left": 0.4,
                "liquidity_score": 0.18,
                "volume_score": 0.1,
                "market_structure_score": 0.25,
                "execution_quality_score": 0.32,
                "forecast_p_hit_interval": 0.31,
                "forecast_margin_to_lower_c": -1.1,
                "forecast_margin_to_upper_c": 0.4,
                "forecast_uncertainty_c": 1.8,
                "forecast_drift_c": 0.5,
                "weather_fair_probability_yes": 0.31,
                "weather_fair_probability_side": 0.31,
                "weather_market_probability": 0.67,
                "weather_forecast_edge": -0.36,
                "weather_forecast_margin_score": 0.22,
                "weather_forecast_stability_score": 0.45,
            },
            {
                "market_family": "btc_price_threshold",
                "target_up": 1,
                "wallet_temp_hit_rate_90d": 0.99,
            },
        ]
    )
    df.to_csv(logs_dir / "contract_targets.csv", index=False)

    trainer = WeatherTemperatureTrainer(logs_dir=str(logs_dir), weights_dir=str(weights_dir))
    model, features = trainer.train()

    assert model is not None
    assert features
    artifact = weights_dir / "weather_temperature_model.joblib"
    assert artifact.exists()
    payload = joblib.load(artifact)
    assert payload["market_family"] == "weather_temperature"
    assert set(features).issubset(set(payload["features"]))
