from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from strategy_layers import EntryRuleLayer
from weather_temperature_strategy import WeatherTemperatureStrategy


class _FakeProfileClient:
    def __init__(self, *, open_positions, closed_positions=None, trades=None):
        self._open_positions = list(open_positions or [])
        self._closed_positions = list(closed_positions or [])
        self._trades = list(trades or [])

    def get_positions(self, **kwargs):
        return list(self._open_positions)

    def get_closed_positions(self, **kwargs):
        return list(self._closed_positions)

    def get_trades(self, **kwargs):
        return list(self._trades)

    def get_activity(self, **kwargs):
        return []


class _FakeForecastService:
    def __init__(self, *, open_forecast: dict, exit_forecast: dict):
        self._open_forecast = dict(open_forecast)
        self._exit_forecast = dict(exit_forecast)

    def fetch_market_forecast(self, market_row: dict):
        if str(market_row.get("smoke_exit_mode", "")).lower() == "stale":
            return dict(self._exit_forecast)
        return dict(self._open_forecast)


class _SmokeTrade:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.state = "OPEN"
        self.close_reason = None

    def close(self, exit_price, reason):
        self.current_price = exit_price
        self.close_reason = reason
        self.state = "CLOSED"


def _write_watchlist(path: Path, wallet: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "wallet,label,enabled,min_wallet_score,region_scope\n"
        f"{wallet},Smoke Wallet,true,0.60,nyc\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Run a safe weather-temperature strategy smoke test.")
    parser.add_argument("--logs-dir", default="logs_weather_temperature_smoke")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    watchlist_path = logs_dir / "weather_wallet_watchlist.csv"
    wallet = "0xweatherwallet"
    _write_watchlist(watchlist_path, wallet)

    now_utc = datetime.now(timezone.utc)
    event_dt = now_utc + timedelta(days=1)
    historical_dt = now_utc - timedelta(days=30)
    event_day_phrase = event_dt.strftime("%B").replace(" 0", " ") + f" {event_dt.day}"
    event_day_iso = event_dt.date().isoformat()

    market_title = f"Will the highest temperature in NYC be 64°F or higher on {event_day_phrase}?"
    market_slug = f"weather-nyc-64f-{event_day_iso}"
    condition_id = "weather-cond-1"
    token_id = "weather-token-yes"
    markets_df = pd.DataFrame(
        [
            {
                "market_title": market_title,
                "question": market_title,
                "market_slug": market_slug,
                "slug": market_slug,
                "condition_id": condition_id,
                "yes_token_id": token_id,
                "no_token_id": "weather-token-no",
                "current_price": 0.41,
                "last_trade_price": 0.41,
                "best_bid": 0.40,
                "best_ask": 0.43,
                "liquidity": 25000.0,
                "volume": 14000.0,
                "end_date": f"{event_day_iso}T23:59:00+00:00",
                "url": f"https://polymarket.com/event/{market_slug}",
            }
        ]
    )
    open_positions = [
        {
            "title": market_title,
            "slug": market_slug,
            "conditionId": condition_id,
            "asset": token_id,
            "outcome": "YES",
            "price": 0.39,
            "curPrice": 0.41,
            "size": 25.0,
            "timestamp": now_utc.isoformat(),
            "endDate": f"{event_day_iso}T23:59:00+00:00",
        }
    ]
    closed_positions = [
        {
            "title": market_title,
            "slug": market_slug,
            "conditionId": condition_id,
            "asset": token_id,
            "outcome": "YES",
            "price": 0.18,
            "curPrice": 1.0,
            "size": 10.0,
            "realizedPnl": 8.2,
            "timestamp": historical_dt.isoformat(),
            "endDate": historical_dt.date().isoformat() + "T23:59:00+00:00",
        }
    ]
    trades = [
        {
            "title": market_title,
            "slug": market_slug,
            "conditionId": condition_id,
            "asset": token_id,
            "outcome": "YES",
            "price": 0.39,
            "curPrice": 0.41,
            "size": 25.0,
            "side": "BUY",
            "timestamp": now_utc.isoformat(),
            "endDate": f"{event_day_iso}T23:59:00+00:00",
        }
    ]

    profile_client = _FakeProfileClient(open_positions=open_positions, closed_positions=closed_positions, trades=trades)
    forecast_service = _FakeForecastService(
        open_forecast={
            "weather_country": "United States",
            "weather_resolution_timezone": "America/New_York",
            "forecast_ready": True,
            "forecast_source": "open_meteo",
            "forecast_stale": False,
            "forecast_missing_reason": None,
            "forecast_max_temp_c": 20.0,
            "forecast_p_hit_interval": 0.74,
            "forecast_margin_to_lower_c": 2.2,
            "forecast_margin_to_upper_c": None,
            "forecast_uncertainty_c": 1.4,
            "forecast_last_update_ts": now_utc.isoformat(),
            "forecast_drift_c": 0.1,
        },
        exit_forecast={
            "forecast_ready": False,
            "forecast_stale": True,
            "forecast_p_hit_interval": 0.50,
            "forecast_missing_reason": "smoke_forced_stale",
        },
    )

    snapshot_file = logs_dir / "weather_wallet_state_snapshot.csv"
    if snapshot_file.exists():
        snapshot_file.unlink()
    strategy = WeatherTemperatureStrategy(
        logs_dir=str(logs_dir),
        watchlist_path=str(watchlist_path),
        profile_client=profile_client,
        forecast_service=forecast_service,
    )

    signals_df = strategy.build_cycle_signals(markets_df)
    scored_df = strategy.score_candidates(signals_df, markets_df)
    open_candidates = scored_df[scored_df.get("entry_intent", pd.Series(dtype=str)).astype(str).str.upper() == "OPEN_LONG"].copy()
    rule_layer = EntryRuleLayer(min_score=0.25, max_spread=0.20, min_liquidity_score=0.05)
    rule_eval = rule_layer.evaluate(open_candidates.iloc[0].to_dict()) if not open_candidates.empty else {}

    smoke_trade = _SmokeTrade(
        token_id=token_id,
        condition_id=condition_id,
        outcome_side="YES",
        market=market_title,
        market_family="weather_temperature_threshold",
        current_price=0.44,
        entry_price=0.39,
        weather_location="NYC",
        weather_event_date_local=event_day_iso,
        smoke_exit_mode="stale",
    )
    trade_manager = SimpleNamespace(active_trades={token_id: smoke_trade})
    exit_events = strategy.apply_active_exit_rules(trade_manager, markets_df)

    report = {
        "signals_generated": int(len(signals_df.index)),
        "open_candidates": int(len(open_candidates.index)),
        "forecast_ready_candidates": int(open_candidates.get("forecast_ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not open_candidates.empty else 0,
        "rule_allow_first_open_candidate": bool(rule_eval.get("allow", False)),
        "first_open_candidate_confidence": float(open_candidates.iloc[0].get("confidence", 0.0)) if not open_candidates.empty else 0.0,
        "first_open_candidate_reason": str(open_candidates.iloc[0].get("reason", "")) if not open_candidates.empty else "",
        "first_open_candidate_wallet_gate_pass": bool(open_candidates.iloc[0].get("wallet_state_gate_pass", False)) if not open_candidates.empty else False,
        "first_open_candidate_wallet_gate_reason": str(open_candidates.iloc[0].get("wallet_state_gate_reason", "")) if not open_candidates.empty else "",
        "first_open_candidate_edge": float(open_candidates.iloc[0].get("weather_forecast_edge", 0.0)) if not open_candidates.empty else 0.0,
        "rule_eval": rule_eval,
        "exit_events": exit_events,
        "smoke_trade_close_reason": smoke_trade.close_reason,
    }
    report_path = logs_dir / "weather_temperature_smoke.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
