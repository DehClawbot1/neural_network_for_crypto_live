import os
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from neural_network_for_crypto.shadow_purgatory import ResilientCLOBClient # Import the client

# Configure logging for this script
logging.basicConfig(level=logging.INFO, format="%(asctime)s [DOAResurrection] %(levelname)s: %(message)s")

def _simulate_path(row, clob):
    token_id = row['token_id']
    entry_p = float(row['shadow_entry_price'])
    start_ts = int(datetime.fromisoformat(str(row['timestamp']).replace("Z", "+00:00")).timestamp())
    
    tp, sl = entry_p + 0.04, entry_p - 0.03
    
    trades = clob.get_trades_with_retry(token_id, start_ts)
    if trades is None: # Handle API failure
        logging.warning(f"Failed to fetch trades for {token_id} at {start_ts}. Cannot simulate.")
        return "PENDING_API_FAIL", 0.0, 0 # Special outcome for API failures
    if not trades: 
        return "EXPIRED", 0.0, 0
    
    trades = sorted(trades, key=lambda x: int(x['timestamp']))
    last_p = entry_p
    count = 0
    for t in trades:
        ts, p = int(t['timestamp']), float(t['price'])
        if ts > start_ts + 3600: # 60 minutes after signal
            break
        last_p, count = p, count + 1
        if p >= tp: return "TP", 0.04, count
        if p <= sl: return "SL", -0.03, count
    
    # If no TP/SL hit within 60 minutes, return based on last price
    return "EXPIRED", round(last_p - entry_p, 4), count

def resurrect_doa_trades(log_path="logs/shadow_results.csv", clob_client=None):
    if not os.path.exists(log_path):
        logging.error("❌ Shadow log not found at %s", log_path)
        return

    df = pd.read_csv(log_path, engine="python", on_bad_lines="skip")
    doa_df = df[df['outcome'] == "DOA"].copy()
    
    if doa_df.empty:
        logging.info("✅ No DOA trades found to resurrect.")
        return

    clob = clob_client or ResilientCLOBClient()
    logging.info(f"👻 Resurrecting {len(doa_df)} DOA trades to check opportunity cost...")

    ghost_results = []

    for idx, row in doa_df.iterrows():
        outcome, realized_ret, count = _simulate_path(row, clob)
        ghost_results.append({
            "market": row['market_title'],
            "prob": row['meta_prob'],
            "exp_slip": row['expected_slip_bps'],
            "ghost_outcome": outcome,
            "ghost_return": realized_ret
        })

    results_df = pd.DataFrame(ghost_results)
    
    if results_df.empty:
        logging.info("No ghost results to analyze.")
        return

    win_rate = (results_df['ghost_outcome'] == "TP").mean()
    avg_ret = results_df['ghost_return'].mean()
    
    bullets_dodged = (results_df['ghost_outcome'] == "SL").sum()
    alpha_leaked = (results_df['ghost_outcome'] == "TP").sum()

    print("-" * 50)
    print("📈 DOA RESURRECTION REPORT (Opportunity Cost)")
    print("-" * 50)
    print(f"🏆 Ghost Win Rate: {win_rate:.2%}")
    print(f"💰 Avg Ghost Return: {avg_ret:+.2%}")
    
    print(f"\n🛡️ Bullets Dodged (SL): {bullets_dodged}")
    print(f"💸 Alpha Leaked (TP): {alpha_leaked}")

    if win_rate > 0.60 and avg_ret > 0.01:
        print("\n⚠️ VERDICT: OVER-VETOING. Your slippage thresholds are too tight.")
        print("You are killing high-quality alpha to save on execution costs.")
    elif win_rate < 0.40:
        print("\n✅ VERDICT: VETO WORKING. Most DOA trades were indeed toxic or flat.")
    else:
        print("\n📊 VERDICT: NEUTRAL. Veto is neither consistently over-vetoing nor perfectly shielding.")
    
    print("-" * 50)

if __name__ == "__main__":
    resurrect_doa_trades()
