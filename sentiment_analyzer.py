import logging
import time
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SentimentAnalyzer:
    """
    Pillar 4: Sentiment & Derivative Positioning (The Psychology)
    Tracks the market's emotional state via the Fear & Greed Index
    and leverage positioning via Binance Futures Funding Rates.
    """

    def __init__(self, cache_ttl_seconds=300):
        # Cache sentiment context for 5 minutes
        self.cache_ttl = cache_ttl_seconds
        self._cached_context = None
        self._last_fetch_time = 0

    def analyze(self) -> dict:
        """
        Fetches Fear & Greed Index and Binance BTC Funding Rate.
        Returns a dictionary expressing the current sentiment configuration.
        """
        now = time.time()
        if self._cached_context and (now - self._last_fetch_time) < self.cache_ttl:
            return self._cached_context

        # Safe defaults if API fails
        context = {
            "fgi_value": 50,
            "fgi_status": "Neutral",
            "btc_funding_rate": 0.0,
            "is_overheated_long": False,
            "sentiment_score": 0.5, # 0.0 (Extreme Fear) to 1.0 (Extreme Greed)
        }

        # 1. Fetch Fear & Greed Index
        try:
            fgi_url = "https://api.alternative.me/fng/?limit=1"
            fgi_res = requests.get(fgi_url, timeout=10)
            fgi_res.raise_for_status()
            fgi_data = fgi_res.json()
            if fgi_data and "data" in fgi_data and len(fgi_data["data"]) > 0:
                fgi_val = int(fgi_data["data"][0]["value"])
                fgi_status = fgi_data["data"][0]["value_classification"]
                context["fgi_value"] = fgi_val
                context["fgi_status"] = fgi_status
                context["sentiment_score"] = fgi_val / 100.0
        except Exception as e:
            logging.warning(f"SentimentAnalyzer: Error fetching Fear & Greed Index: {e}")

        # 2. Fetch Binance Funding Rate (Derivatives proxy for leverage bias)
        try:
            fund_url = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
            fund_res = requests.get(fund_url, timeout=10)
            fund_res.raise_for_status()
            fund_data = fund_res.json()
            if fund_data and "lastFundingRate" in fund_data:
                funding_rate = float(fund_data["lastFundingRate"])
                context["btc_funding_rate"] = funding_rate
                
                # Baseline Binance funding rate is typically highly positive when the market is overheated long
                # Standard neutral funding on Binance is often 0.01% per 8 hours. 
                # Spikes above 0.03% (0.0003) indicate heavy leverage longing.
                if context["fgi_value"] >= 80 and funding_rate > 0.00025:
                    context["is_overheated_long"] = True
        except Exception as e:
            logging.warning(f"SentimentAnalyzer: Error fetching Binance Funding Rate: {e}")

        logging.info(
            f"SentimentAnalyzer Evaluated: FGI={context['fgi_value']} ({context['fgi_status']}) | "
            f"Funding={context['btc_funding_rate']:.5f} | Overheated={context['is_overheated_long']}"
        )

        self._cached_context = context
        self._last_fetch_time = now
        return context
