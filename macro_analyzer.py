import logging
import time
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MacroAnalyzer:
    """
    Pillar 1: Macro & Fundamental Analysis (The Why)
    Tracks the overarching fiat liquidity environment using traditional finance proxies.
    - DXY (US Dollar Index): High DXY = Tight Liquidity (Bearish Crypto)
    - ^TNX (10-Year Treasury Yield): High Yields = Expensive Capital (Bearish Crypto)
    - ^GSPC (S&P 500): High Equities = Risk-On Appetite (Bullish Crypto)
    """

    def __init__(self, cache_ttl_seconds=3600):
        # Cache for an hour since macro data is daily/slow
        self.cache_ttl = cache_ttl_seconds
        self._cached_context = None
        self._last_fetch_time = 0

    def analyze(self) -> dict:
        now = time.time()
        if self._cached_context and (now - self._last_fetch_time) < self.cache_ttl:
            return self._cached_context

        context = {
            "macro_dxy": None,
            "macro_tnx": None,
            "macro_sp500": None,
            "macro_liquidity_state": "NEUTRAL",
            "macro_score": 0.5, # 0.0 (Liquidity Crunch) to 1.0 (Liquidity Expansion)
        }

        if yf is None:
            logging.warning("MacroAnalyzer: yfinance not installed. Returning empty macro context.")
            return context

        try:
            # Fetch last 5 days of data to get the most recent valid close 
            # (handles weekends/holidays when markets are closed)
            tickers = yf.Tickers("DX-Y.NYB ^TNX ^GSPC")
            dxy_hist = tickers.tickers["DX-Y.NYB"].history(period="5d")
            tnx_hist = tickers.tickers["^TNX"].history(period="5d")
            spx_hist = tickers.tickers["^GSPC"].history(period="5d")

            dxy_close = dxy_hist['Close'].iloc[-1] if not dxy_hist.empty else 105.0
            tnx_close = tnx_hist['Close'].iloc[-1] if not tnx_hist.empty else 4.5
            spx_close = spx_hist['Close'].iloc[-1] if not spx_hist.empty else 5000.0

            # Advanced: We can calculate RSI or distance to moving averages, 
            # but for now, we evaluate against rough historical pivoting baselines.
            # DXY Baseline: ~104
            # TNX Baseline: ~4.2%
            
            # Simple Scoring Model
            score = 0.5
            if dxy_close > 105: score -= 0.15
            elif dxy_close < 103: score += 0.15
            
            if tnx_close > 4.4: score -= 0.15
            elif tnx_close < 4.0: score += 0.15
            
            # S&P500 Trend
            spx_trend = "UP"
            if len(spx_hist) >= 5 and spx_hist['Close'].iloc[-1] > spx_hist['Close'].iloc[0]:
                score += 0.1
            else:
                score -= 0.1
                spx_trend = "DOWN"

            score = max(0.0, min(1.0, score)) # Clamp between 0 and 1

            if score >= 0.7:
                state = "EXPANSION (Risk-On)"
            elif score <= 0.3:
                state = "CONTRACTION (Risk-Off)"
            else:
                state = "NEUTRAL"

            context.update({
                "macro_dxy": round(dxy_close, 3),
                "macro_tnx": round(tnx_close, 3),
                "macro_sp500": round(spx_close, 2),
                "macro_liquidity_state": state,
                "macro_score": round(score, 3),
            })
            
            logging.info(
                f"MacroAnalyzer: DXY={dxy_close:.2f} | 10Y Yield={tnx_close:.2f}% | "
                f"SP500 Trend={spx_trend} | Liquidity={state}"
            )

            self._cached_context = context
            self._last_fetch_time = now
            return context

        except Exception as e:
            logging.warning(f"MacroAnalyzer: Failed to fetch yfinance data: {e}")
            return context
