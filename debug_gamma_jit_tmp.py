import pandas as pd
import requests
from pathlib import Path

RAW = "logs/raw_candidates.csv"
URL = "https://gamma-api.polymarket.com/markets"

if not Path(RAW).exists():
    print(f"Error: {RAW} not found.")
    raise SystemExit(0)

df = pd.read_csv(RAW, engine="python", on_bad_lines="skip")
sample_ids = df["condition_id"].dropna().astype(str).unique()[:3]
if len(sample_ids) == 0:
    print("No condition_ids found in raw_candidates.csv")
    raise SystemExit(0)

print(f"--- Testing {len(sample_ids)} IDs from Scraper ---\n")
for cid in sample_ids:
    print(f"Scraped ID: {cid}")
    try:
        resp = requests.get(URL, params={"condition_id": cid}, timeout=10)
        data = resp.json()
        if not data:
            print(" Result: [EMPTY LIST] - Gamma does not recognize this ID.")
        else:
            print(f" Result: [SUCCESS] - Found {len(data)} market(s)")
            market = data[0]
            print(f" Raw Keys found: {list(market.keys())[:15]}...")
            print(f" Value for 'id': {market.get('id')}")
            print(f" Value for 'conditionId': {market.get('conditionId')}")
            print(f" Value for 'condition_id': {market.get('condition_id')}")
    except Exception as e:
        print(f" Error querying Gamma: {e}")
    print("-" * 40)
