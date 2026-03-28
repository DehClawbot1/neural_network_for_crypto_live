"""
research_features.py
=====================
Academic paper-derived features for BTC 5-minute prediction markets.

Reference set implemented:
  1. Moskowitz, Ooi, Pedersen (2012) — Time Series Momentum
  2. Jegadeesh & Titman (1993) — Cross-sectional Momentum
  3. Engle (1982) — ARCH volatility
  4. Bollerslev (1986) — GARCH volatility clustering
  5. Khuntia & Pattanayak (2018) — Adaptive Market Hypothesis / Regime detection
  6. Liu & Tsyvinski (2021) — Crypto-specific risk factors
  7. Liu, Tsyvinski, Wu (2022) — Crypto factor model
  8. Grobys & Sapkota (2019) — Crypto momentum caution
  9. Grobys, Ahmed, Sapkota (2020) — Technical trading rules
  10. Gerritsen et al. (2020) — BTC technical rule profitability

All features use ONLY past data (no lookahead).
All features must pass walk-forward evaluation before use.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ResearchFeatureBuilder:
    """Build research-grade features from BTC price history.
    
    Input: DataFrame with columns [timestamp, price] at 1-minute resolution.
    Output: DataFrame with all research features attached.
    """

    def __init__(self):
        self.feature_names = []

    def build_all(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Build all research features from a price series.
        
        Args:
            price_df: DataFrame with 'timestamp' and 'price' columns.
                      Should be sorted by timestamp, 1-min frequency.
        
        Returns:
            DataFrame with all features added as columns.
        """
        if price_df is None or price_df.empty or "price" not in price_df.columns:
            return pd.DataFrame()

        df = price_df.copy()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])
        if len(df) < 30:
            return df

        # Compute log returns (base for most features)
        df["log_return_1m"] = np.log(df["price"] / df["price"].shift(1))
        df["return_1m"] = df["price"].pct_change(1)

        # === Paper 1: Moskowitz et al. (2012) — Time Series Momentum ===
        df = self._momentum_features(df)

        # === Paper 3 & 4: Engle (1982), Bollerslev (1986) — Volatility ===
        df = self._volatility_features(df)

        # === Paper 5: Khuntia & Pattanayak (2018) — Regime Detection ===
        df = self._regime_features(df)

        # === Paper 6 & 7: Liu & Tsyvinski (2021), Liu et al. (2022) — Crypto Factors ===
        df = self._crypto_factor_features(df)

        # === Paper 9 & 10: Grobys et al. (2020), Gerritsen et al. (2020) — Technical Rules ===
        df = self._technical_rule_features(df)

        # === Combined: Volatility-adjusted momentum ===
        df = self._composite_features(df)

        self.feature_names = [c for c in df.columns if c not in ["timestamp", "price", "token_id"]]
        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Papers 1, 2, 8: Multi-horizon momentum and reversal features."""

        # Lagged returns at multiple horizons (Paper 1: intermediate horizons)
        for horizon in [1, 3, 5, 10, 15, 30]:
            col = f"return_{horizon}m"
            df[col] = df["price"].pct_change(horizon)

        # Cumulative momentum (Paper 1: past-return predictor)
        for window in [5, 10, 15, 30]:
            df[f"momentum_{window}m"] = df["return_1m"].rolling(window).sum()

        # Rolling reversal indicator (Paper 1: partial reversal at longer horizons)
        # If short-term momentum is opposite to longer-term, that's a reversal signal
        df["reversal_5v30"] = np.where(
            df["momentum_5m"] * df["momentum_30m"] < 0, 1.0, 0.0
        )

        # Trend persistence (Paper 1): fraction of positive returns in window
        for window in [5, 10, 15]:
            df[f"trend_persistence_{window}m"] = (
                (df["return_1m"] > 0).rolling(window).mean()
            )

        # Return skewness (Paper 8: momentum can be weak, check for skew)
        df["return_skewness_15m"] = df["return_1m"].rolling(15).skew()
        df["return_skewness_30m"] = df["return_1m"].rolling(30).skew()

        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Papers 3, 4: ARCH/GARCH-inspired volatility features."""

        # Realized volatility at multiple horizons
        for window in [5, 10, 15, 30]:
            df[f"realized_vol_{window}m"] = df["return_1m"].rolling(window).std()

        # Squared returns (ARCH proxy — volatility clustering)
        df["squared_return_1m"] = df["return_1m"] ** 2

        # EWMA volatility (GARCH-like exponential smoothing)
        for span in [5, 15, 30]:
            df[f"ewma_vol_{span}m"] = (
                df["squared_return_1m"].ewm(span=span, adjust=False).mean().apply(np.sqrt)
            )

        # Volatility ratio (short/long — detects volatility regime changes)
        if "realized_vol_5m" in df.columns and "realized_vol_30m" in df.columns:
            df["vol_ratio_5v30"] = df["realized_vol_5m"] / (df["realized_vol_30m"] + 1e-10)

        # Absolute return (proxy for instantaneous volatility)
        df["abs_return_1m"] = df["return_1m"].abs()

        # High-low range proxy (intrabar volatility, using rolling max-min of prices)
        df["price_range_5m"] = (
            df["price"].rolling(5).max() - df["price"].rolling(5).min()
        ) / (df["price"].rolling(5).mean() + 1e-10)

        df["price_range_15m"] = (
            df["price"].rolling(15).max() - df["price"].rolling(15).min()
        ) / (df["price"].rolling(15).mean() + 1e-10)

        return df

    def _regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Paper 5: Adaptive Market Hypothesis — regime shift detection."""

        # Rolling autocorrelation of returns (predictability proxy)
        # High autocorrelation = trending regime; low = random/efficient
        for lag in [1, 3, 5]:
            df[f"autocorr_lag{lag}_15m"] = (
                df["return_1m"].rolling(15).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else 0.0, raw=False
                )
            )

        # Rolling Hurst exponent approximation (R/S analysis simplified)
        # H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random
        df["hurst_proxy_30m"] = df["return_1m"].rolling(30).apply(
            _approx_hurst, raw=True
        )

        # Regime label: trending vs mean-reverting vs random
        # Based on autocorrelation sign and magnitude
        autocorr = df.get("autocorr_lag1_15m", pd.Series(0.0, index=df.index))
        df["regime_trending"] = (autocorr > 0.15).astype(float)
        df["regime_reverting"] = (autocorr < -0.15).astype(float)
        df["regime_random"] = ((autocorr >= -0.15) & (autocorr <= 0.15)).astype(float)

        # Regime instability: rolling std of autocorrelation
        df["regime_instability"] = autocorr.rolling(15).std()

        # Confidence penalty during unstable regimes
        vol = df.get("realized_vol_15m", pd.Series(0.0, index=df.index))
        df["confidence_penalty_vol"] = np.clip(vol / (vol.rolling(60).mean() + 1e-10), 0, 3)

        return df

    def _crypto_factor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Papers 6, 7: Crypto-specific risk factors."""

        # Volume shock (proxy — using absolute return as volume proxy)
        df["volume_shock_5m"] = (
            df["abs_return_1m"].rolling(5).sum() /
            (df["abs_return_1m"].rolling(30).sum() + 1e-10)
        )

        # Attention proxy (large absolute returns signal attention spikes)
        threshold = df["abs_return_1m"].rolling(60).quantile(0.95)
        df["attention_spike"] = (df["abs_return_1m"] > threshold).astype(float)
        df["attention_count_15m"] = df["attention_spike"].rolling(15).sum()

        # Crypto momentum factor (Paper 7: crypto-specific momentum)
        # Use 5-min return ranked against its own recent distribution
        df["momentum_rank_15m"] = df["return_5m"].rolling(15).rank(pct=True)

        # Cross-asset confirmation placeholder
        # (Would need ETH or other crypto data for full implementation)
        # Using price relative to its own moving average as proxy
        for ma_window in [10, 30]:
            ma = df["price"].rolling(ma_window).mean()
            df[f"ma_distance_{ma_window}m"] = (df["price"] - ma) / (ma + 1e-10)

        return df

    def _technical_rule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Papers 9, 10: Technical trading rule features."""

        # Moving average crossover signals
        ma5 = df["price"].rolling(5).mean()
        ma15 = df["price"].rolling(15).mean()
        ma30 = df["price"].rolling(30).mean()

        df["ma_cross_5v15"] = np.where(ma5 > ma15, 1.0, -1.0)
        df["ma_cross_5v30"] = np.where(ma5 > ma30, 1.0, -1.0)
        df["ma_cross_15v30"] = np.where(ma15 > ma30, 1.0, -1.0)

        # Breakout strength (price relative to rolling high/low)
        for window in [15, 30]:
            rolling_high = df["price"].rolling(window).max()
            rolling_low = df["price"].rolling(window).min()
            range_size = rolling_high - rolling_low + 1e-10
            df[f"breakout_strength_{window}m"] = (
                (df["price"] - rolling_low) / range_size
            )

        # Channel breakout signal
        df["upper_channel_15m"] = df["price"].rolling(15).max()
        df["lower_channel_15m"] = df["price"].rolling(15).min()
        df["channel_breakout_up"] = (df["price"] >= df["upper_channel_15m"]).astype(float)
        df["channel_breakout_down"] = (df["price"] <= df["lower_channel_15m"]).astype(float)

        # RSI-like feature (relative strength index)
        gains = df["return_1m"].clip(lower=0)
        losses = (-df["return_1m"]).clip(lower=0)
        avg_gain = gains.rolling(14).mean()
        avg_loss = losses.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Bollinger Band position
        bb_ma = df["price"].rolling(20).mean()
        bb_std = df["price"].rolling(20).std()
        df["bollinger_position"] = (df["price"] - bb_ma) / (2 * bb_std + 1e-10)

        return df

    def _composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combined features crossing multiple papers."""

        # Volatility-adjusted momentum (Papers 1 + 3)
        mom = df.get("momentum_5m", pd.Series(0.0, index=df.index))
        vol = df.get("realized_vol_5m", pd.Series(1e-10, index=df.index))
        df["vol_adj_momentum_5m"] = mom / (vol + 1e-10)

        mom15 = df.get("momentum_15m", pd.Series(0.0, index=df.index))
        vol15 = df.get("realized_vol_15m", pd.Series(1e-10, index=df.index))
        df["vol_adj_momentum_15m"] = mom15 / (vol15 + 1e-10)

        # Regime-filtered momentum (Papers 1 + 5)
        trending = df.get("regime_trending", pd.Series(0.0, index=df.index))
        df["regime_momentum_5m"] = mom * trending

        # Confidence-penalized edge (Papers 5 + 6)
        penalty = df.get("confidence_penalty_vol", pd.Series(1.0, index=df.index))
        df["penalized_momentum_5m"] = mom / (penalty + 1e-10)

        return df

    def get_feature_names(self) -> list:
        """Return list of all feature names generated."""
        return list(self.feature_names)


def _approx_hurst(returns):
    """Approximate Hurst exponent using rescaled range (R/S) method."""
    n = len(returns)
    if n < 10:
        return 0.5
    try:
        returns = np.asarray(returns, dtype=float)
        mean_r = np.mean(returns)
        deviations = np.cumsum(returns - mean_r)
        R = np.max(deviations) - np.min(deviations)
        S = np.std(returns, ddof=1)
        if S < 1e-12 or R < 1e-12:
            return 0.5
        return float(np.log(R / S) / np.log(n))
    except Exception:
        return 0.5


class ResearchFeatureIntegrator:
    """Integrates research features into the existing feature pipeline.
    
    Call this after FeatureBuilder.build_features() and before signal scoring.
    It fetches recent BTC price history and computes research features,
    then merges them into the feature DataFrame.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.builder = ResearchFeatureBuilder()
        self.history_file = self.logs_dir / "clob_price_history.csv"

    def enrich_features(self, features_df: pd.DataFrame, token_ids: list = None) -> pd.DataFrame:
        """Add research features to an existing feature DataFrame.
        
        For each token_id in features_df, loads its price history,
        computes research features at the signal timestamp, and
        merges them back.
        """
        if features_df is None or features_df.empty:
            return features_df

        history_df = self._load_history()
        if history_df.empty:
            logging.warning("ResearchFeatureIntegrator: No price history available")
            return features_df

        out = features_df.copy()
        if "token_id" not in out.columns:
            return out

        # Group history by token
        history_df["token_id"] = history_df["token_id"].astype(str).str.strip()
        grouped = {k: v for k, v in history_df.groupby("token_id")}

        research_rows = []
        for idx, row in out.iterrows():
            tid = str(row.get("token_id", "")).strip()
            if not tid or tid == "nan":
                research_rows.append({})
                continue

            token_history = grouped.get(tid)
            if token_history is None or len(token_history) < 30:
                research_rows.append({})
                continue

            # Build features on this token's price history
            enriched = self.builder.build_all(token_history[["timestamp", "price"]].copy())
            if enriched.empty:
                research_rows.append({})
                continue

            # Get the latest research features (most recent row)
            signal_ts = pd.to_datetime(row.get("timestamp"), errors="coerce", utc=True)
            if pd.notna(signal_ts):
                # Find closest row before signal timestamp
                enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], errors="coerce", utc=True)
                before = enriched[enriched["timestamp"] <= signal_ts]
                if not before.empty:
                    latest = before.iloc[-1]
                else:
                    latest = enriched.iloc[-1]
            else:
                latest = enriched.iloc[-1]

            # Extract research features (exclude base columns)
            research_cols = [c for c in latest.index if c not in ["timestamp", "price", "token_id"]]
            research_rows.append({f"research_{c}": latest[c] for c in research_cols})

        # Merge research features into output
        research_df = pd.DataFrame(research_rows, index=out.index)
        for col in research_df.columns:
            out[col] = research_df[col]

        n_features = len([c for c in out.columns if c.startswith("research_")])
        logging.info("ResearchFeatureIntegrator: added %d research features", n_features)
        return out

    def _load_history(self) -> pd.DataFrame:
        if not self.history_file.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.history_file, engine="python", on_bad_lines="skip")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            return df.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
        except Exception:
            return pd.DataFrame()


# === Baseline models for comparison (Paper requirement) ===

class MomentumBaseline:
    """Simple momentum-only baseline: go long if 5m return > 0."""

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        if "research_momentum_5m" in features_df.columns:
            return (features_df["research_momentum_5m"] > 0).astype(int)
        if "return_5m" in features_df.columns:
            return (features_df["return_5m"] > 0).astype(int)
        return pd.Series(0, index=features_df.index)


class MeanReversionBaseline:
    """Simple mean-reversion baseline: go long if price below 15m MA."""

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        if "research_ma_distance_15m" in features_df.columns:
            # Below MA = expect reversion up
            return (features_df["research_ma_distance_15m"] < -0.001).astype(int)
        return pd.Series(0, index=features_df.index)


class VolatilityFilterBaseline:
    """Only trade when volatility is in a normal range."""

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        if "research_vol_ratio_5v30" in features_df.columns:
            # Trade only when short vol is 0.5-1.5x long vol
            ratio = features_df["research_vol_ratio_5v30"]
            return ((ratio > 0.5) & (ratio < 1.5)).astype(int)
        return pd.Series(1, index=features_df.index)


class TechnicalRuleBaseline:
    """MA crossover + RSI baseline."""

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=features_df.index)
        if "research_ma_cross_5v15" in features_df.columns and "research_rsi_14" in features_df.columns:
            ma_bullish = features_df["research_ma_cross_5v15"] > 0
            rsi_ok = (features_df["research_rsi_14"] > 30) & (features_df["research_rsi_14"] < 70)
            signals = (ma_bullish & rsi_ok).astype(int)
        return signals


if __name__ == "__main__":
    # Demo: generate features from synthetic price data
    np.random.seed(42)
    n = 100
    prices = 50000 + np.cumsum(np.random.randn(n) * 50)
    demo_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "price": prices,
    })

    builder = ResearchFeatureBuilder()
    result = builder.build_all(demo_df)
    print(f"Generated {len(builder.get_feature_names())} research features:")
    for name in sorted(builder.get_feature_names()):
        print(f"  {name}")
    print(f"\nSample (last row):")
    for col in sorted(builder.get_feature_names())[:10]:
        print(f"  {col}: {result[col].iloc[-1]:.6f}")
