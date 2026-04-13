"""
family_feature_store.py
Phase 4 — Family-Specific Feature Stores

Provides a unified, cache-backed interface for each family's features:
  BTCFeatureStore       — spot/funding/basis/vol/momentum/orderbook/regime/portfolio
  WeatherFeatureStore   — vintages/revision/ensemble/source-disagreement/threshold
  ExecutionFeatureStore — spread/depth/queue/fill-speed/aggressiveness
  MetaFeatureStore      — aggregates alpha+execution+portfolio+governor+drift state
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BRAIN_FAMILY_TO_DIR = {
    "btc": "btc",
    "weather_temperature": "weather_temperature",
}


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        num = float(value)
        return float(default) if not np.isfinite(num) else num
    except Exception:
        return float(default)


# ---------------------------------------------------------------------------
# BTC Feature Store
# ---------------------------------------------------------------------------
class BTCFeatureStore:
    """
    Reads BTC features from logs/btc and returns a merged snapshot
    suitable for training BTC alpha models.

    Columns: spot, index, mark, funding, basis, realized_vol, momentum,
             threshold_distance, poly_implied_prob, orderbook_state,
             regime_state, portfolio_state, timestamp.
    """

    _CACHE: pd.DataFrame | None = None
    _CACHE_TS: float = 0.0
    _CACHE_TTL: float = 300.0  # 5 min

    def __init__(self, shared_logs_dir: str = "logs", shared_weights_dir: str = "weights") -> None:
        self.logs_dir = Path(shared_logs_dir)
        self.btc_logs = self.logs_dir / "btc"
        self.btc_logs.mkdir(parents=True, exist_ok=True)

    def load(self, *, refresh: bool = False) -> pd.DataFrame:
        import time
        now = time.time()
        if (
            not refresh
            and BTCFeatureStore._CACHE is not None
            and (now - BTCFeatureStore._CACHE_TS) < BTCFeatureStore._CACHE_TTL
        ):
            return BTCFeatureStore._CACHE.copy()

        frames = []
        # Contract targets (main feature source)
        ct_path = self.btc_logs / "contract_targets.csv"
        ct = _safe_read(ct_path)
        if not ct.empty:
            frames.append(ct)

        # BTC price dataset
        price_path = self.logs_dir / "btc_price_dataset.csv"
        price = _safe_read(price_path)
        if not price.empty:
            frames.append(price)

        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for other in frames[1:]:
            # Try to merge on timestamp if both have it
            if "timestamp" in merged.columns and "timestamp" in other.columns:
                try:
                    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
                    other["timestamp"] = pd.to_datetime(other["timestamp"], utc=True, errors="coerce")
                    merged = pd.merge_asof(
                        merged.sort_values("timestamp"),
                        other.sort_values("timestamp"),
                        on="timestamp",
                        direction="nearest",
                        suffixes=("", "_btcprice"),
                    )
                except Exception:
                    pass

        BTCFeatureStore._CACHE = merged.copy()
        BTCFeatureStore._CACHE_TS = now
        logger.debug("BTCFeatureStore: loaded %d rows", len(merged))
        return merged.copy()

    def invalidate_cache(self) -> None:
        BTCFeatureStore._CACHE = None
        BTCFeatureStore._CACHE_TS = 0.0

    def feature_snapshot(self) -> dict[str, Any]:
        """Return the most recent feature row as a dict."""
        df = self.load()
        if df.empty:
            return {}
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values("timestamp")
        return df.iloc[-1].to_dict()


# ---------------------------------------------------------------------------
# Weather Feature Store
# ---------------------------------------------------------------------------
class WeatherFeatureStore:
    """
    Reads weather features from logs/weather_temperature.

    Columns: forecast_vintage, revision_path, ensemble_spread,
             source_disagreement, time_to_resolution, station_context,
             threshold_rarity, poly_implied_prob, orderbook_state, timestamp.
    """

    _CACHE: pd.DataFrame | None = None
    _CACHE_TS: float = 0.0
    _CACHE_TTL: float = 600.0  # 10 min

    def __init__(self, shared_logs_dir: str = "logs", shared_weights_dir: str = "weights") -> None:
        self.logs_dir = Path(shared_logs_dir)
        self.weather_logs = self.logs_dir / "weather_temperature"
        self.weather_logs.mkdir(parents=True, exist_ok=True)

    def load(self, *, refresh: bool = False) -> pd.DataFrame:
        import time
        now = time.time()
        if (
            not refresh
            and WeatherFeatureStore._CACHE is not None
            and (now - WeatherFeatureStore._CACHE_TS) < WeatherFeatureStore._CACHE_TTL
        ):
            return WeatherFeatureStore._CACHE.copy()

        ct_path = self.weather_logs / "contract_targets.csv"
        df = _safe_read(ct_path)

        WeatherFeatureStore._CACHE = df.copy() if not df.empty else pd.DataFrame()
        WeatherFeatureStore._CACHE_TS = now
        logger.debug("WeatherFeatureStore: loaded %d rows", len(df))
        return df.copy() if not df.empty else pd.DataFrame()

    def invalidate_cache(self) -> None:
        WeatherFeatureStore._CACHE = None
        WeatherFeatureStore._CACHE_TS = 0.0

    def feature_snapshot(self) -> dict[str, Any]:
        df = self.load()
        if df.empty:
            return {}
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values("timestamp")
        return df.iloc[-1].to_dict()


# ---------------------------------------------------------------------------
# Execution Feature Store
# ---------------------------------------------------------------------------
class ExecutionFeatureStore:
    """
    Reads execution feedback for training the Execution Engine models.

    Columns: spread, depth, queue_imbalance, requested_size, fill_speed,
             quote_instability, time_pressure, order_aggressiveness,
             fill_prob_5s, fill_prob_30s, expected_slippage,
             cancel_risk, liquidity_failure_risk.
    """

    _CACHE: pd.DataFrame | None = None
    _CACHE_TS: float = 0.0
    _CACHE_TTL: float = 120.0  # 2 min

    def __init__(self, shared_logs_dir: str = "logs") -> None:
        self.logs_dir = Path(shared_logs_dir)

    def load(self, *, refresh: bool = False) -> pd.DataFrame:
        import time
        now = time.time()
        if (
            not refresh
            and ExecutionFeatureStore._CACHE is not None
            and (now - ExecutionFeatureStore._CACHE_TS) < ExecutionFeatureStore._CACHE_TTL
        ):
            return ExecutionFeatureStore._CACHE.copy()

        exec_path = self.logs_dir / "execution_feedback.csv"
        df = _safe_read(exec_path)

        ExecutionFeatureStore._CACHE = df.copy() if not df.empty else pd.DataFrame()
        ExecutionFeatureStore._CACHE_TS = now
        return df.copy() if not df.empty else pd.DataFrame()

    def invalidate_cache(self) -> None:
        ExecutionFeatureStore._CACHE = None
        ExecutionFeatureStore._CACHE_TS = 0.0

    def has_enough_data(self, min_rows: int = 50) -> bool:
        df = self.load()
        return len(df) >= min_rows


# ---------------------------------------------------------------------------
# Meta Feature Store
# ---------------------------------------------------------------------------
class MetaFeatureStore:
    """
    Aggregates outputs from BTC & Weather alpha engines, the execution
    engine, portfolio state, governor state, recent precision per family,
    and drift state into a single row for the MetaDecisionEngine.

    The meta feature row is built at runtime from live state; this store
    also reads the policy_feedback.csv for meta-model training data.
    """

    def __init__(self, shared_logs_dir: str = "logs") -> None:
        self.logs_dir = Path(shared_logs_dir)

    def build_meta_row(
        self,
        *,
        btc_alpha_output: dict[str, Any] | None = None,
        weather_alpha_output: dict[str, Any] | None = None,
        execution_output: dict[str, Any] | None = None,
        portfolio_state: dict[str, Any] | None = None,
        governor_state: dict[str, Any] | None = None,
        drift_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Merge all engine outputs into a single meta feature dict.
        Missing values default to neutral (0.5 for probabilities, 0 for scores).
        """
        btc = btc_alpha_output or {}
        weather = weather_alpha_output or {}
        execution = execution_output or {}
        portfolio = portfolio_state or {}
        governor = governor_state or {}
        drift = drift_state or {}

        row: dict[str, Any] = {
            # BTC alpha
            "btc_fair_prob": _safe_float(btc.get("fair_prob"), 0.5),
            "btc_edge": _safe_float(btc.get("edge"), 0.0),
            "btc_ev": _safe_float(btc.get("ev"), 0.0),
            "btc_trade_quality": _safe_float(btc.get("trade_quality"), 0.0),
            "btc_regime": str(btc.get("regime", "unknown")),
            "btc_calibrated_prob": _safe_float(btc.get("calibrated_prob"), 0.5),
            # Weather alpha
            "weather_fair_prob": _safe_float(weather.get("fair_prob"), 0.5),
            "weather_revision_dir": _safe_float(weather.get("revision_dir"), 0.0),
            "weather_uncertainty_penalty": _safe_float(weather.get("uncertainty_penalty"), 0.0),
            "weather_ev": _safe_float(weather.get("ev"), 0.0),
            "weather_trade_quality": _safe_float(weather.get("trade_quality"), 0.0),
            # Execution
            "exec_fill_prob_5s": _safe_float(execution.get("fill_prob_5s"), 0.5),
            "exec_fill_prob_30s": _safe_float(execution.get("fill_prob_30s"), 0.5),
            "exec_slippage": _safe_float(execution.get("expected_slippage"), 0.0),
            "exec_liquidity_failure_risk": _safe_float(execution.get("liquidity_failure_risk"), 0.0),
            "exec_quality_score": _safe_float(execution.get("quality_score"), 0.5),
            # Portfolio
            "portfolio_open_positions": int(_safe_float(portfolio.get("open_positions"), 0)),
            "portfolio_usdc_balance": _safe_float(portfolio.get("usdc_balance"), 0.0),
            "portfolio_drawdown_pct": _safe_float(portfolio.get("drawdown_pct"), 0.0),
            "portfolio_concentration": _safe_float(portfolio.get("concentration"), 0.0),
            # Governor
            "governor_level": int(_safe_float(governor.get("governor_level"), 0)),
            "governor_size_multiplier": _safe_float(governor.get("size_multiplier"), 1.0),
            "governor_min_confidence": _safe_float(governor.get("min_confidence"), 0.0),
            "governor_top_signal_only": int(bool(governor.get("top_signal_only", False))),
            # Drift
            "btc_feature_drift": int(bool(drift.get("btc_feature_drift", False))),
            "weather_feature_drift": int(bool(drift.get("weather_feature_drift", False))),
            "calibration_drift": int(bool(drift.get("calibration_drift", False))),
            "slippage_drift": int(bool(drift.get("slippage_drift", False))),
        }
        return row

    def load_policy_training_set(self) -> pd.DataFrame:
        """Load policy_feedback.csv for meta-decision model training."""
        path = self.logs_dir / "policy_feedback.csv"
        return _safe_read(path)

    def recent_family_precision(self, family: str, lookback: int = 50) -> float:
        """Return recent alpha_success rate for a given family."""
        alpha_path = self.logs_dir / "alpha_feedback_clean.csv"
        df = _safe_read(alpha_path)
        if df.empty or "market_family" not in df.columns or "alpha_verdict" not in df.columns:
            return 0.5
        subset = df[df["market_family"].fillna("").astype(str).str.startswith(family)].tail(lookback)
        if subset.empty:
            return 0.5
        return float((subset["alpha_verdict"] == "alpha_success").mean())
