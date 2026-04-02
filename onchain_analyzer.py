import logging
import math
import os
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _clip01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _log_scaled(value, ceiling):
    try:
        value = float(value or 0.0)
        ceiling = float(ceiling or 1.0)
    except Exception:
        return 0.0
    if value <= 0.0 or ceiling <= 0.0:
        return 0.0
    return _clip01(math.log1p(value) / math.log1p(ceiling))


class OnChainAnalyzer:
    """
    Pillar 2: On-Chain Analysis (The Under the Hood)

    Existing context:
    - CoinMetrics community API for network hash rate

    New open-source monitoring:
    - mempool.space public REST API for fee pressure, mempool congestion,
      block cadence, and difficulty-adjustment state

    The mempool metrics move much faster than hashrate, so they use a shorter cache.
    """

    def __init__(self, cache_ttl_seconds=86400, mempool_cache_ttl_seconds=60):
        self.cache_ttl = max(60, int(os.getenv("ONCHAIN_HASHRATE_CACHE_TTL_SECONDS", str(cache_ttl_seconds)) or cache_ttl_seconds))
        self.mempool_cache_ttl = max(15, int(os.getenv("BTC_MEMPOOL_CACHE_TTL_SECONDS", str(mempool_cache_ttl_seconds)) or mempool_cache_ttl_seconds))
        self.mempool_api_base = str(os.getenv("BTC_MEMPOOL_API_BASE", "https://mempool.space/api") or "https://mempool.space/api").rstrip("/")
        self._cached_hashrate_context = None
        self._last_hashrate_fetch_time = 0.0
        self._cached_mempool_context = None
        self._last_mempool_fetch_time = 0.0

    def _default_context(self):
        return {
            "onchain_hashrate_ths": None,
            "onchain_network_health": "UNKNOWN",
            "btc_mempool_tx_count": None,
            "btc_mempool_vsize": None,
            "btc_mempool_total_fee": None,
            "btc_fee_fastest_satvb": None,
            "btc_fee_halfhour_satvb": None,
            "btc_fee_hour_satvb": None,
            "btc_fee_economy_satvb": None,
            "btc_difficulty_change_pct": None,
            "btc_difficulty_progress_pct": None,
            "btc_block_height": None,
            "btc_fee_pressure_score": 0.5,
            "btc_mempool_congestion_score": 0.5,
            "btc_network_activity_score": 0.5,
            "btc_network_stress_score": 0.5,
        }

    def _fetch_hashrate_context(self):
        context = {}
        url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics=HashRate&frequency=1d"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "data" in data and len(data["data"]) > 0:
            latest_entry = data["data"][-1]
            if "HashRate" in latest_entry:
                hr = float(latest_entry["HashRate"])
                context["onchain_hashrate_ths"] = round(hr, 2)
        return context

    def _fetch_mempool_context(self):
        context = {}
        mempool_res = requests.get(f"{self.mempool_api_base}/mempool", timeout=10)
        mempool_res.raise_for_status()
        mempool = mempool_res.json()

        fees_res = requests.get(f"{self.mempool_api_base}/v1/fees/recommended", timeout=10)
        fees_res.raise_for_status()
        fees = fees_res.json()

        diff_res = requests.get(f"{self.mempool_api_base}/v1/difficulty-adjustment", timeout=10)
        diff_res.raise_for_status()
        diff = diff_res.json()

        height_res = requests.get(f"{self.mempool_api_base}/blocks/tip/height", timeout=10)
        height_res.raise_for_status()
        height_text = str(height_res.text or "").strip()

        tx_count = float(mempool.get("count") or 0.0)
        vsize = float(mempool.get("vsize") or 0.0)
        total_fee = float(mempool.get("total_fee") or 0.0)
        fastest_fee = float(fees.get("fastestFee") or 0.0)
        halfhour_fee = float(fees.get("halfHourFee") or 0.0)
        hour_fee = float(fees.get("hourFee") or 0.0)
        economy_fee = float(fees.get("economyFee") or 0.0)
        difficulty_change_pct = float(diff.get("difficultyChange") or 0.0)
        difficulty_progress_pct = float(diff.get("progressPercent") or 0.0)
        block_height = int(height_text) if height_text.isdigit() else None

        fee_pressure_score = _clip01(
            (_log_scaled(max(fastest_fee, hour_fee), 40.0) * 0.75)
            + (_log_scaled(total_fee, 15_000_000.0) * 0.25)
        )
        congestion_score = _clip01(
            (_log_scaled(vsize, 20_000_000.0) * 0.65)
            + (_log_scaled(tx_count, 300_000.0) * 0.35)
        )
        retarget_heat = _clip01(max(difficulty_change_pct, 0.0) / 10.0)
        activity_score = _clip01(
            (_log_scaled(tx_count, 300_000.0) * 0.35)
            + (_log_scaled(vsize, 20_000_000.0) * 0.35)
            + (_log_scaled(total_fee, 15_000_000.0) * 0.20)
            + (retarget_heat * 0.10)
        )
        stress_score = _clip01(
            (fee_pressure_score * 0.45)
            + (congestion_score * 0.35)
            + (activity_score * 0.20)
        )

        context.update(
            {
                "btc_mempool_tx_count": int(tx_count),
                "btc_mempool_vsize": int(vsize),
                "btc_mempool_total_fee": int(total_fee),
                "btc_fee_fastest_satvb": round(fastest_fee, 3),
                "btc_fee_halfhour_satvb": round(halfhour_fee, 3),
                "btc_fee_hour_satvb": round(hour_fee, 3),
                "btc_fee_economy_satvb": round(economy_fee, 3),
                "btc_difficulty_change_pct": round(difficulty_change_pct, 3),
                "btc_difficulty_progress_pct": round(difficulty_progress_pct, 3),
                "btc_block_height": block_height,
                "btc_fee_pressure_score": round(fee_pressure_score, 4),
                "btc_mempool_congestion_score": round(congestion_score, 4),
                "btc_network_activity_score": round(activity_score, 4),
                "btc_network_stress_score": round(stress_score, 4),
            }
        )
        return context

    def analyze(self) -> dict:
        now = time.time()
        context = self._default_context()

        if self._cached_hashrate_context and (now - self._last_hashrate_fetch_time) < self.cache_ttl:
            context.update(self._cached_hashrate_context)
        else:
            try:
                fetched = self._fetch_hashrate_context()
                self._cached_hashrate_context = fetched
                self._last_hashrate_fetch_time = now
                context.update(fetched)
            except Exception as e:
                logging.warning(f"OnChainAnalyzer: Failed to fetch CoinMetrics hashrate: {e}")
                if self._cached_hashrate_context:
                    context.update(self._cached_hashrate_context)

        if self._cached_mempool_context and (now - self._last_mempool_fetch_time) < self.mempool_cache_ttl:
            context.update(self._cached_mempool_context)
        else:
            try:
                fetched = self._fetch_mempool_context()
                self._cached_mempool_context = fetched
                self._last_mempool_fetch_time = now
                context.update(fetched)
            except Exception as e:
                logging.warning(f"OnChainAnalyzer: Failed to fetch mempool.space monitoring data: {e}")
                if self._cached_mempool_context:
                    context.update(self._cached_mempool_context)

        network_stress = float(context.get("btc_network_stress_score", 0.5) or 0.5)
        network_activity = float(context.get("btc_network_activity_score", 0.5) or 0.5)
        if network_stress >= 0.72:
            context["onchain_network_health"] = "CONGESTED"
        elif network_stress >= 0.55:
            context["onchain_network_health"] = "BUSY"
        elif context.get("onchain_hashrate_ths") is not None:
            context["onchain_network_health"] = "HEALTHY"
        elif network_activity >= 0.45:
            context["onchain_network_health"] = "ACTIVE"
        else:
            context["onchain_network_health"] = "QUIET"

        hr_value = context.get("onchain_hashrate_ths")
        hr_display = f"{float(hr_value) / 1_000_000:.2f} EH/s" if hr_value is not None else "n/a"
        logging.info(
            "OnChainAnalyzer: HashRate=%s | MempoolTx=%s | VSize=%.2f MVB | Fees fast/hour=%s/%s sat/vB | Stress=%.2f | Health=%s",
            hr_display,
            context.get("btc_mempool_tx_count"),
            float(context.get("btc_mempool_vsize") or 0.0) / 1_000_000.0,
            context.get("btc_fee_fastest_satvb"),
            context.get("btc_fee_hour_satvb"),
            network_stress,
            context.get("onchain_network_health"),
        )
        return context
