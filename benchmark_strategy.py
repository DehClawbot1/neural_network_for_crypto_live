from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class BenchmarkStrategy:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.logs_dir / "benchmark_vs_main.csv"

    def evaluate_cycle(self, technical_context: dict | None, governor_state: dict | None = None) -> dict:
        technical_context = technical_context or {}
        governor_state = governor_state or {}
        alligator = str(technical_context.get("alligator_alignment", "NEUTRAL") or "NEUTRAL").strip().upper()
        bias = str(technical_context.get("btc_trend_bias", "NEUTRAL") or "NEUTRAL").strip().upper()
        adx_value = float(technical_context.get("adx_value", 0.0) or 0.0)
        adx_threshold = float(technical_context.get("adx_threshold", 0.0) or 0.0)
        above_vwap = bool(technical_context.get("price_above_anchored_vwap", False))
        below_vwap = bool(technical_context.get("price_below_anchored_vwap", False))
        long_breakout = bool(technical_context.get("long_fractal_breakout", False))
        short_breakout = bool(technical_context.get("short_fractal_breakout", False))

        benchmark_action = "HOLD"
        if alligator == "BULLISH" and bias == "LONG" and adx_value >= adx_threshold and above_vwap and long_breakout:
            benchmark_action = "LONG"
        elif alligator == "BEARISH" and bias == "SHORT" and adx_value >= adx_threshold and below_vwap and short_breakout:
            benchmark_action = "SHORT"

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark_mode": "shadow",
            "benchmark_action": benchmark_action,
            "alligator_alignment": alligator,
            "btc_trend_bias": bias,
            "adx_value": round(adx_value, 6),
            "adx_threshold": round(adx_threshold, 6),
            "price_above_anchored_vwap": above_vwap,
            "price_below_anchored_vwap": below_vwap,
            "long_fractal_breakout": long_breakout,
            "short_fractal_breakout": short_breakout,
            "performance_governor_level": int(governor_state.get("governor_level", 0) or 0),
            "main_live_win_rate": float(governor_state.get("live_win_rate", 0.0) or 0.0),
            "main_live_profit_factor": float(governor_state.get("live_profit_factor", 0.0) or 0.0),
        }
        pd.DataFrame([row]).to_csv(self.output_file, mode="a", header=not self.output_file.exists(), index=False)
        return row
