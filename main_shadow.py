import logging
import threading
import time

import pandas as pd

from feature_builder import FeatureBuilder
from leaderboard_scraper import run_scraper_cycle
from market_monitor import fetch_btc_markets, fetch_markets_by_slugs, save_market_snapshot
from shadow_purgatory import ShadowPurgatory, ResilientCLOBClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] MainShadow: %(message)s')

BUNDLE_PATH = None  # latest bundle
LOG_PATH = "logs/shadow_results.csv"


def run_audit_loop(purgatory, interval_sec=300):
    while True:
        try:
            purgatory.resolve_purgatory()
            status = purgatory.get_stats()
            print(f"\r💓 Heartbeat: {status}", end="", flush=True)
        except Exception as e:
            logging.error("Audit Loop Error: %s", e)
        time.sleep(interval_sec)


def main():
    logging.info("🚀 Starting Shadow Lab Launcher...")
    clob = ResilientCLOBClient()
    purgatory = ShadowPurgatory(BUNDLE_PATH, clob, LOG_PATH)
    feature_builder = FeatureBuilder()

    audit_thread = threading.Thread(target=run_audit_loop, args=(purgatory,), daemon=True)
    audit_thread.start()

    logging.info("📡 Monitoring live whale signals...")
    while True:
        try:
            markets_df = fetch_btc_markets()
            signals_df = run_scraper_cycle()
            if signals_df is not None and not signals_df.empty and "market_slug" in signals_df.columns:
                scraped_slugs = set(signals_df["market_slug"].dropna().astype(str).unique())
                scraped_slugs.discard("")
                known_slugs = set(markets_df["slug"].dropna().astype(str).unique()) if markets_df is not None and not markets_df.empty and "slug" in markets_df.columns else set()
                missing_slugs = scraped_slugs - known_slugs
                if missing_slugs:
                    logging.info("Universe Gap: %s slugs missing. Synchronizing...", len(missing_slugs))
                    missing_df = fetch_markets_by_slugs(list(missing_slugs))
                    if missing_df is not None and not missing_df.empty:
                        markets_df = pd.concat([markets_df, missing_df], ignore_index=True).drop_duplicates(subset=["slug"])
                save_market_snapshot(markets_df)
            if signals_df is not None and not signals_df.empty:
                for _, signal_row in signals_df.iterrows():
                    signal = signal_row.to_dict()
                    features_df = feature_builder.build_features(pd.DataFrame([signal]), markets_df)
                    if not features_df.empty:
                        purgatory.log_intent(signal, features_df)
        except Exception as e:
            logging.error("Main shadow loop error: %s", e)
        time.sleep(5)


if __name__ == "__main__":
    main()
