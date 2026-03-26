"""
diagnose_balance_fix.py
========================
Full diagnostic: balance, signature type, deposit status, and order book test.

Usage:
    python diagnose_balance_fix.py
"""

import os
from pprint import pprint

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def check_env_file():
    """Check if .env file has the required fields and auto-fix if possible."""
    env_path = os.path.join(os.getcwd(), ".env")
    issues = []

    if not os.path.exists(env_path):
        print("  [!] No .env file found. Create one with your credentials.")
        return ["no_env_file"]

    with open(env_path, "r", encoding="utf-8") as f:
        text = f.read()

    if "POLYMARKET_SIGNATURE_TYPE" not in text:
        issues.append("missing_signature_type")
        print("  [!] POLYMARKET_SIGNATURE_TYPE is NOT in your .env file!")
        print("      Adding it now...")
        with open(env_path, "a", encoding="utf-8") as f:
            f.write("\n# Signature type: 1=email/Magic, 2=MetaMask, 0=EOA\n")
            f.write("POLYMARKET_SIGNATURE_TYPE=1\n")
        print("      [+] Added POLYMARKET_SIGNATURE_TYPE=1 to .env")
        os.environ["POLYMARKET_SIGNATURE_TYPE"] = "1"
    elif "POLYMARKET_SIGNATURE_TYPE=0" in text:
        issues.append("wrong_signature_type")
        print("  [!] POLYMARKET_SIGNATURE_TYPE=0 (wrong for email login)")
        print("      Fixing to 1...")
        text = text.replace("POLYMARKET_SIGNATURE_TYPE=0", "POLYMARKET_SIGNATURE_TYPE=1")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("      [+] Fixed to POLYMARKET_SIGNATURE_TYPE=1")
        os.environ["POLYMARKET_SIGNATURE_TYPE"] = "1"
    else:
        print("  [+] POLYMARKET_SIGNATURE_TYPE present in .env")

    if "TRADING_MODE" not in text:
        issues.append("missing_trading_mode")
        print("  [!] TRADING_MODE not in .env, adding TRADING_MODE=live")
        with open(env_path, "a", encoding="utf-8") as f:
            f.write("\nTRADING_MODE=live\n")

    return issues


def main():
    print("=" * 60)
    print("FULL BALANCE & CONFIG DIAGNOSTIC")
    print("=" * 60)

    # ── Step 0: Check and fix .env file ──
    print("\n--- STEP 0: .env FILE CHECK ---")
    env_issues = check_env_file()

    # Reload env after fixes
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    # ── Step 1: Show config ──
    print("\n--- STEP 1: CONFIG ---")
    pk = os.getenv("PRIVATE_KEY", "")
    funder = os.getenv("POLYMARKET_FUNDER", "")
    sig_type = os.getenv("POLYMARKET_SIGNATURE_TYPE", "not set")
    sig_labels = {"0": "EOA", "1": "Email/Magic/Google", "2": "MetaMask/Rabby"}

    print(f"  PRIVATE_KEY:               {'set (' + pk[:6] + '...' + pk[-4:] + ')' if len(pk) > 10 else 'MISSING'}")
    print(f"  POLYMARKET_FUNDER:         {funder or 'MISSING'}")
    print(f"  POLYMARKET_SIGNATURE_TYPE: {sig_type} ({sig_labels.get(sig_type, '???')})")
    print(f"  TRADING_MODE:              {os.getenv('TRADING_MODE', 'not set')}")

    if not pk:
        print("\n[!] PRIVATE_KEY is missing. Cannot continue.")
        print("    Add to your .env:  PRIVATE_KEY=0xYOUR_KEY_HERE")
        return

    # ── Step 2: Create client ──
    print("\n--- STEP 2: CLIENT CONNECTION ---")
    from execution_client import ExecutionClient

    try:
        client = ExecutionClient()
    except Exception as exc:
        print(f"  [!] Client creation failed: {exc}")
        return

    print(f"  Credential source: {client.credential_source}")
    print(f"  Funder:            {client.funder}")
    print(f"  Signature type:    {client.signature_type} ({sig_labels.get(str(client.signature_type), '???')})")

    # ── Step 3: CLOB balance ──
    print("\n--- STEP 3: CLOB API BALANCE ---")
    clob_balance = 0.0
    try:
        raw = client.get_balance_allowance(asset_type="COLLATERAL")
        if isinstance(raw, dict):
            raw_val = raw.get("balance", raw.get("amount", 0))
            clob_balance = client._normalize_usdc_balance(raw_val)
            print(f"  Raw API response:  balance='{raw_val}'")
            print(f"  Normalized:        ${clob_balance:.6f}")
        else:
            print(f"  Raw API response:  {raw}")
    except Exception as exc:
        print(f"  [!] Error: {exc}")
        raw = {}

    # ── Step 4: On-chain balance ──
    print("\n--- STEP 4: ON-CHAIN USDC ---")
    onchain_total = 0.0
    try:
        onchain = client.get_onchain_collateral_balance()
        onchain_total = float((onchain or {}).get("total", 0.0) or 0.0)
        for token_addr, bal in (onchain or {}).get("balances", {}).items():
            label = "USDC.e (bridged)" if "2791" in token_addr else "USDC (native)"
            print(f"  {label}: ${bal:.6f}")
        print(f"  Total on-chain:    ${onchain_total:.6f}")
    except Exception as exc:
        print(f"  [!] Error: {exc}")

    # ── Step 5: Diagnosis ──
    available = max(clob_balance, onchain_total)
    print("\n--- STEP 5: DIAGNOSIS ---")
    print(f"  +-------------------------------------------+")
    print(f"  |  CLOB/API Balance:    ${clob_balance:>12,.2f}   |")
    print(f"  |  On-chain USDC:       ${onchain_total:>12,.2f}   |")
    print(f"  |  -----------------------------------------|")
    print(f"  |  AVAILABLE TO TRADE:  ${available:>12,.2f}   |")
    print(f"  +-------------------------------------------+")

    if clob_balance > 0:
        print("\n  [+] CLOB balance is positive — everything is working!")
    elif onchain_total > 0 and clob_balance == 0:
        print(f"\n  [!] Your ${onchain_total:.2f} is ON-CHAIN but not in the CLOB.")
        print()
        print("  This means your USDC is in your wallet but Polymarket's")
        print("  trading engine doesn't see it as available for orders.")
        print()
        print("  ============= HOW TO FIX =============")
        print()
        print("  OPTION A: Deposit through Polymarket website")
        print("    1. Go to https://polymarket.com")
        print("    2. Log in with your email account")
        print("    3. Click 'Deposit' in the top right")
        print("    4. Your USDC should appear as available")
        print("    5. Try making one small manual trade")
        print("    6. Run this diagnostic again")
        print()
        print("  OPTION B: If you already see funds on polymarket.com")
        print("    The issue may be that your PRIVATE_KEY doesn't match")
        print("    your email account's internal signer key.")
        print("    Email/Magic accounts have a key managed by Magic —")
        print("    you need to export THAT key, not a MetaMask key.")
        print()
        print("  Your bot will try to use the on-chain fallback")
        print(f"  (${onchain_total:.2f}) but CLOB orders may still fail")
        print("  until funds are properly deposited.")
    else:
        print("\n  [!] No funds found anywhere.")
        print("  Deposit USDC to your Polymarket account first.")

    # ── Step 6: Quick order book test ──
    print("\n--- STEP 6: ORDER BOOK TEST ---")
    try:
        from orderbook_guard import OrderBookGuard
        guard = OrderBookGuard()
        test_token = None
        try:
            import pandas as pd
            markets_path = "logs/markets.csv"
            if os.path.exists(markets_path):
                markets = pd.read_csv(markets_path, engine="python", on_bad_lines="skip")
                if not markets.empty and "yes_token_id" in markets.columns:
                    tokens = markets["yes_token_id"].dropna().astype(str).tolist()
                    if tokens:
                        test_token = tokens[0]
        except Exception:
            pass

        if test_token:
            print(f"  Testing order book for: {test_token[:24]}...")
            analysis = guard.analyze_book(test_token, depth=5)
            if analysis["book_available"]:
                print(f"  Best Bid:  {analysis['best_bid']}")
                print(f"  Best Ask:  {analysis['best_ask']}")
                print(f"  Midpoint:  {analysis['midpoint']}")
                print(f"  Spread:    {analysis['spread']}")
                print(f"  Bid depth: {analysis['bid_depth']} levels")
                print(f"  Ask depth: {analysis['ask_depth']} levels")
                if analysis['spread'] is not None and analysis['spread'] > 0.10:
                    print(f"  [!] Spread is wide ({analysis['spread']:.4f} > 0.10)")
                    print(f"      Bot will skip this market.")
                else:
                    print(f"  [+] Order book looks healthy!")
            else:
                print("  [!] Could not fetch order book (token may be expired)")
        else:
            print("  [SKIP] No markets.csv found yet — run the bot first")
    except ImportError:
        print("  [SKIP] orderbook_guard.py not installed — apply the patch first")
    except Exception as exc:
        print(f"  [!] Order book test error: {exc}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
