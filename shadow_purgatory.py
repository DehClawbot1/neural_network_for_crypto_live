import logging
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
import requests

from config import TradingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] ShadowPurgatory: %(message)s")


class ResilientCLOBClient:
    def __init__(self, max_retries=5, base_delay=2, base_url="https://clob.polymarket.com"):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.trades_url = f"{base_url}/trades"

    def get_trades_with_retry(self, token_id, after_ts, limit=100):
        retries = 0
        while retries < self.max_retries:
            try:
                params = {"market": str(token_id), "after": int(after_ts), "limit": int(limit)}
                response = requests.get(self.trades_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return data if isinstance(data, list) else []
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    retries += 1
                    sleep_time = (self.base_delay * (2 ** (retries - 1))) + random.uniform(0, 1)
                    logging.warning("API %s for %s. Retry %s/%s in %.2fs", response.status_code, token_id, retries, self.max_retries, sleep_time)
                    time.sleep(sleep_time)
                    continue
                logging.error("Fatal API Error %s for %s. Aborting.", response.status_code, token_id)
                break
            except requests.exceptions.RequestException as e:
                retries += 1
                sleep_time = self.base_delay * retries
                logging.warning("Network Error for %s: %s. Retry %s/%s in %.2fs", token_id, e, retries, self.max_retries, sleep_time)
                time.sleep(sleep_time)
        return None


class ShadowPurgatory:
    def __init__(self, model_bundle_path=None, clob_client=None, log_path="logs/shadow_results.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.clob = clob_client or ResilientCLOBClient()
        self.lock = threading.Lock()
        bundle_path = self._resolve_bundle_path(model_bundle_path)
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.feature_cols = bundle["features"]

    @staticmethod
    def _resolve_bundle_path(model_bundle_path=None):
        if model_bundle_path:
            return Path(model_bundle_path)
        bundles = sorted(Path("weights").glob("meta_model_bundle_*.pkl"))
        if not bundles:
            raise FileNotFoundError("No meta_model_bundle_*.pkl found in weights/")
        return bundles[-1]

    def _get_bucket_slippage(self, meta_prob, window=50):
        try:
            with self.lock:
                if not self.log_path.exists():
                    return 20.0
                df = pd.read_csv(self.log_path, engine="python", on_bad_lines="skip")
            bucket_df = df[
                (df["meta_prob"] >= meta_prob - 0.05)
                & (df["meta_prob"] <= meta_prob + 0.05)
                & (df["outcome"] != "PENDING")
            ].tail(window)
            if bucket_df.empty:
                return 20.0
            return float(bucket_df["entry_slippage_bps"].median())
        except Exception:
            return 20.0

    def log_intent(self, signal, features_df):
        try:
            if features_df is None or features_df.empty:
                return 0.0
            missing = [f for f in self.feature_cols if f not in features_df.columns]
            if missing:
                logging.warning("Missing features for shadow intent: %s", missing[:10])
                return 0.0
            X = features_df[self.feature_cols].copy()
            meta_prob = float(self.model.predict_proba(X)[:, 1][0])
            expected_slip_bps = self._get_bucket_slippage(meta_prob)
            expected_slip_pct = expected_slip_bps / 10000.0
            ev_adj = (meta_prob * TradingConfig.SHADOW_TP_DELTA) + ((1 - meta_prob) * -TradingConfig.SHADOW_SL_DELTA) - expected_slip_pct
            is_doa = ev_adj < TradingConfig.VETO_EV_THRESHOLD

            token_id = signal.get("token_id")
            if not token_id:
                return 0.0 if is_doa else meta_prob

            signal_ts = int(datetime.fromisoformat(str(signal["timestamp"]).replace("Z", "+00:00")).timestamp())
            shadow_price, delay = self._get_reachable_price(token_id, signal_ts)
            scraper_price = float(signal.get("price", signal.get("entry_price", 0.0)) or 0.0)
            shadow_price = shadow_price or scraper_price
            slippage = int(((shadow_price - scraper_price) / scraper_price) * 10000) if scraper_price > 0 else 0

            new_row = {
                "timestamp": signal.get("timestamp"),
                "market_title": signal.get("market_title", signal.get("market", "Unknown")),
                "market_slug": signal.get("market_slug"),
                "token_id": token_id,
                "trader_wallet": signal.get("trader_wallet", signal.get("wallet_copied")),
                "scraper_price": scraper_price,
                "shadow_entry_price": shadow_price,
                "entry_slippage_bps": slippage,
                "entry_delay_sec": delay,
                "meta_prob": round(meta_prob, 6),
                "expected_slip_bps": expected_slip_bps,
                "ev_adj": round(ev_adj, 6),
                "outcome": "DOA" if is_doa else "PENDING",
                "realized_return": 0.0,
                "trades_in_window": 0,
            }
            with self.lock:
                pd.DataFrame([new_row]).to_csv(self.log_path, mode="a", header=not self.log_path.exists(), index=False)
            if is_doa:
                logging.warning("🚫 VETO (DOA): %s | EV_adj: %.2f%% | Exp Slip: %sbps", new_row["market_title"], ev_adj * 100, expected_slip_bps)
                return 0.0
            logging.info("👻 Shadow Intent: %s | Prob: %.2f%% | Slip: %sbps | EV_adj: %.2f%%", new_row["market_title"], meta_prob * 100, slippage, ev_adj * 100)
            return meta_prob
        except Exception as e:
            logging.error("Failed to log shadow intent: %s", e)
            return 0.0

    def resolve_purgatory(self):
        with self.lock:
            if not self.log_path.exists():
                return
            df = pd.read_csv(self.log_path, engine="python", on_bad_lines="skip")
        pending_indices = df[df["outcome"] == "PENDING"].index.tolist() if "outcome" in df.columns else []
        if not pending_indices:
            return
        updates = {}
        for idx in pending_indices:
            row = df.loc[idx]
            try:
                start_dt = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
            except Exception:
                continue
            if datetime.now(timezone.utc) - start_dt > timedelta(minutes=TradingConfig.SHADOW_WINDOW_MINUTES + 1):
                outcome, ret, count = self._check_path(row)
                if outcome != "PENDING":
                    updates[idx] = (outcome, ret, count)
        if updates:
            with self.lock:
                df = pd.read_csv(self.log_path, engine="python", on_bad_lines="skip")
                for idx, (outcome, ret, count) in updates.items():
                    if idx >= len(df):
                        continue
                    df.at[idx, "outcome"] = outcome
                    df.at[idx, "realized_return"] = ret
                    df.at[idx, "trades_in_window"] = count
                df.to_csv(self.log_path, index=False)

    def _check_path(self, row):
        start_ts = int(datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00")).timestamp())
        end_ts = start_ts + (TradingConfig.SHADOW_WINDOW_MINUTES * 60)
        entry_p = float(row["shadow_entry_price"])
        tp = entry_p + TradingConfig.SHADOW_TP_DELTA
        sl = entry_p - TradingConfig.SHADOW_SL_DELTA
        trades = self.clob.get_trades_with_retry(row["token_id"], start_ts, limit=1000)
        if trades is None:
            return "PENDING", 0.0, 0
        if not trades:
            return "EXPIRED", 0.0, 0
        trades = sorted(trades, key=lambda x: int(x["timestamp"]))
        last_p = entry_p
        count = 0
        for t in trades:
            ts = int(t["timestamp"])
            if ts > end_ts:
                break
            p = float(t["price"])
            last_p = p
            count += 1
            if p >= tp:
                return "TP", TradingConfig.SHADOW_TP_DELTA, count
            if p <= sl:
                return "SL", -TradingConfig.SHADOW_SL_DELTA, count
        return "EXPIRED", round(last_p - entry_p, 4), count

    def _get_reachable_price(self, token_id, signal_ts):
        trades = self.clob.get_trades_with_retry(token_id, signal_ts, limit=10)
        if not trades:
            return None, 0
        sorted_trades = sorted(trades, key=lambda x: int(x["timestamp"]))
        first = next((t for t in sorted_trades if int(t["timestamp"]) > signal_ts), None)
        if first:
            return float(first["price"]), int(first["timestamp"]) - signal_ts
        return None, 0

    def get_stats(self):
        with self.lock:
            if not self.log_path.exists():
                return "🔴 NO_DATA"
            df = pd.read_csv(self.log_path, engine="python", on_bad_lines="skip")
        resolved = df[df["outcome"] != "PENDING"] if "outcome" in df.columns else pd.DataFrame()
        top = resolved[resolved["meta_prob"] >= 0.85] if not resolved.empty and "meta_prob" in resolved.columns else pd.DataFrame()
        if len(top) < 10:
            return f"🟡 WARMING_UP (N={len(top)})"
        win_rate = (top["outcome"] == "TP").mean()
        ev = top["realized_return"].mean()
        ce = abs(0.90 - win_rate)
        status = "🟢 READY" if (len(top) >= 50 and ev >= 0.005 and ce <= 0.15) else "🟡 AUDITING"
        return f"{status} | EV: {ev:+.2%} | CE: {ce:.2%} | N: {len(top)}"
