import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from shadow_purgatory import ResilientCLOBClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LimitOrderSim] %(levelname)s: %(message)s")


def _to_ts(value):
    return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())


def _simulate_limit_fill_and_path(row, clob, fill_window_sec=300, horizon_sec=3600):
    token_id = row["token_id"]
    signal_ts = _to_ts(row["timestamp"])
    limit_price = float(row["scraper_price"])
    tp = limit_price + 0.04
    sl = limit_price - 0.03

    trades = clob.get_trades_with_retry(token_id, signal_ts, limit=1000)
    if trades is None:
        return {
            "fill_status": "API_FAIL",
            "fill_delay_sec": None,
            "filled": False,
            "ghost_outcome": "PENDING",
            "ghost_return": 0.0,
            "trades_in_window": 0,
        }
    if not trades:
        return {
            "fill_status": "NO_TRADES",
            "fill_delay_sec": None,
            "filled": False,
            "ghost_outcome": "EXPIRED",
            "ghost_return": 0.0,
            "trades_in_window": 0,
        }

    trades = sorted(trades, key=lambda x: int(x["timestamp"]))
    fill_deadline = signal_ts + fill_window_sec
    end_ts = signal_ts + horizon_sec

    fill_ts = None
    fill_count = 0
    for trade in trades:
        ts = int(trade["timestamp"])
        price = float(trade["price"])
        if ts <= signal_ts:
            continue
        if ts > fill_deadline:
            break
        fill_count += 1
        if price <= limit_price:
            fill_ts = ts
            break

    if fill_ts is None:
        return {
            "fill_status": "UNFILLED",
            "fill_delay_sec": None,
            "filled": False,
            "ghost_outcome": "UNFILLED",
            "ghost_return": 0.0,
            "trades_in_window": fill_count,
        }

    last_p = limit_price
    path_count = 0
    for trade in trades:
        ts = int(trade["timestamp"])
        price = float(trade["price"])
        if ts <= fill_ts:
            continue
        if ts > end_ts:
            break
        last_p = price
        path_count += 1
        if price >= tp:
            return {
                "fill_status": "FILLED",
                "fill_delay_sec": fill_ts - signal_ts,
                "filled": True,
                "ghost_outcome": "TP",
                "ghost_return": 0.04,
                "trades_in_window": path_count,
            }
        if price <= sl:
            return {
                "fill_status": "FILLED",
                "fill_delay_sec": fill_ts - signal_ts,
                "filled": True,
                "ghost_outcome": "SL",
                "ghost_return": -0.03,
                "trades_in_window": path_count,
            }

    return {
        "fill_status": "FILLED",
        "fill_delay_sec": fill_ts - signal_ts,
        "filled": True,
        "ghost_outcome": "EXPIRED",
        "ghost_return": round(last_p - limit_price, 4),
        "trades_in_window": path_count,
    }


def run_limit_order_simulation(log_path="logs/shadow_results.csv", fill_window_sec=300, horizon_sec=3600, only_doa=True):
    log_file = Path(log_path)
    if not log_file.exists():
        print("❌ Shadow log not found.")
        return

    df = pd.read_csv(log_file, engine="python", on_bad_lines="skip")
    if only_doa:
        sample = df[df["outcome"] == "DOA"].copy()
    else:
        sample = df.copy()

    if sample.empty:
        print("✅ No matching trades found for limit-order simulation.")
        return

    clob = ResilientCLOBClient()
    results = []
    for _, row in sample.iterrows():
        sim = _simulate_limit_fill_and_path(row, clob, fill_window_sec=fill_window_sec, horizon_sec=horizon_sec)
        results.append({
            "market_title": row.get("market_title"),
            "meta_prob": row.get("meta_prob"),
            "expected_slip_bps": row.get("expected_slip_bps"),
            "scraper_price": row.get("scraper_price"),
            **sim,
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("⏳ No simulation results produced.")
        return

    filled = results_df[results_df["filled"] == True]
    fill_rate = len(filled) / len(results_df)
    ghost_win_rate = (filled["ghost_outcome"] == "TP").mean() if not filled.empty else 0.0
    avg_fill_delay = filled["fill_delay_sec"].mean() if not filled.empty else 0.0
    avg_ghost_return = filled["ghost_return"].mean() if not filled.empty else 0.0

    print("-" * 60)
    print(f"📘 LIMIT ORDER SIMULATOR ({'DOA only' if only_doa else 'all intents'})")
    print("-" * 60)
    print(f"Sample Size: {len(results_df)}")
    print(f"Fill Window: {fill_window_sec}s")
    print(f"Horizon: {horizon_sec}s")
    print(f"Fill Rate: {fill_rate:.2%}")
    print(f"Filled Ghost Win Rate: {ghost_win_rate:.2%}")
    print(f"Avg Fill Delay: {avg_fill_delay:.2f}s")
    print(f"Avg Filled Ghost Return: {avg_ghost_return:+.2%}")

    if not filled.empty:
        outcome_mix = filled["ghost_outcome"].value_counts(normalize=True) * 100
        print("\n--- FILLED OUTCOME MIX ---")
        for outcome, pct in outcome_mix.items():
            print(f"{outcome:.<15} {pct:.1f}%")

    unfilled = (results_df["fill_status"] == "UNFILLED").sum()
    api_fail = (results_df["fill_status"] == "API_FAIL").sum()
    print("\n--- FILL DIAGNOSTICS ---")
    print(f"Unfilled: {unfilled}")
    print(f"API Failures: {api_fail}")

    if fill_rate > 0.50 and ghost_win_rate > 0.55:
        print("\n💡 SIGNAL: Limit-entry alternative looks promising for vetoed trades.")
    elif fill_rate < 0.20:
        print("\n⚠️ SIGNAL: Most vetoed trades never came back to the scraper price.")
    else:
        print("\n📊 SIGNAL: Mixed evidence — keep collecting samples.")

    print("-" * 60)


if __name__ == "__main__":
    run_limit_order_simulation()
