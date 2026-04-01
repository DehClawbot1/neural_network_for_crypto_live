import logging
import time
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class OnChainAnalyzer:
    """
    Pillar 2: On-Chain Analysis (The Under the Hood)
    Tracks raw blockchain fundamentals like Hash Rate. Massive drawdowns
    in hash rate indicate miner capitulation (bearish short-term, bullish long-term).
    Relies on free public APIs.
    """

    def __init__(self, cache_ttl_seconds=86400):
        # Cache for 24 hours as on-chain metrics update very slowly
        self.cache_ttl = cache_ttl_seconds
        self._cached_context = None
        self._last_fetch_time = 0

    def analyze(self) -> dict:
        now = time.time()
        if self._cached_context and (now - self._last_fetch_time) < self.cache_ttl:
            return self._cached_context

        context = {
            "onchain_hashrate_ths": None,
            "onchain_network_health": "UNKNOWN",
        }

        try:
            # CoinMetrics Public Community API
            # HashRate: Mean hash rate in terahashes per second
            url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics=HashRate&frequency=1d"
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            if "data" in data and len(data["data"]) > 0:
                # Get the most recent valid day
                latest_entry = data["data"][-1]
                if "HashRate" in latest_entry:
                    hr = float(latest_entry["HashRate"])
                    context["onchain_hashrate_ths"] = round(hr, 2)
                    context["onchain_network_health"] = "HEALTHY" # Simplified 

                    logging.info(
                        f"OnChainAnalyzer: Network HashRate={hr/1_000_000:.2f} Exahashes/s | Health=HEALTHY"
                    )

            self._cached_context = context
            self._last_fetch_time = now
            return context

        except Exception as e:
            logging.warning(f"OnChainAnalyzer: Failed to fetch on-chain data: {e}")
            return context
