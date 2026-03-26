"""
test_order_minimal.py
======================
MINIMAL $1 TEST ORDER — bypasses the entire bot pipeline.

This script does ONE thing: places a $1 BUY order on the most liquid
active Polymarket market. If this works, your credentials and execution
pipeline are fine — the problem is in the signal/filter chain.

Usage:
    python test_order_minimal.py
"""

import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    print("=" * 55)
    print("  MINIMAL $1 TEST ORDER")
    print("=" * 55)

    # ── 1. Verify credentials exist ──
    pk = os.getenv("PRIVATE_KEY", "")
    funder = os.getenv("POLYMARKET_FUNDER", "")
    sig_type = os.getenv("POLYMARKET_SIGNATURE_TYPE", "1")

    if not pk:
        print("\n[!] PRIVATE_KEY not set. Add it to .env or set it in environment.")
        sys.exit(1)

    print(f"\n  Key: {pk[:6]}...{pk[-4:]}")
    print(f"  Funder: {funder or 'not set'}")
    print(f"  Sig type: {sig_type}")

    # ── 2. Create client ──
    print("\n[1] Creating execution client...")
    from execution_client import ExecutionClient

    try:
        client = ExecutionClient()
        print(f"    OK — source: {client.credential_source}")
    except Exception as e:
        print(f"    FAILED: {e}")
        sys.exit(1)

    # ── 3. Check balance ──
    print("\n[2] Checking balance...")
    try:
        bal_raw = client.get_balance_allowance(asset_type="COLLATERAL")
        bal = client._normalize_usdc_balance(
            bal_raw.get("balance", 0) if isinstance(bal_raw, dict) else 0
        )
        print(f"    CLOB balance: ${bal:.4f}")
    except Exception as e:
        print(f"    Balance check failed: {e}")
        bal = 0.0

    try:
        onchain = client.get_onchain_collateral_balance()
        onchain_bal = float((onchain or {}).get("total", 0.0) or 0.0)
        print(f"    On-chain USDC: ${onchain_bal:.4f}")
    except Exception:
        onchain_bal = 0.0

    available = max(bal, onchain_bal)
    print(f"    Available: ${available:.4f}")

    if available < 1.0:
        print(f"\n[!] Need at least $1.00. You have ${available:.2f}.")
        print("    Deposit USDC on polymarket.com first.")
        sys.exit(1)

    # ── 4. Find best market ──
    print("\n[3] Finding a market to test with...")
    import requests

    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 20, "active": True, "closed": False,
                    "order": "volume24hr", "ascending": False},
            timeout=15,
        )
        resp.raise_for_status()
        markets = resp.json()
    except Exception as e:
        print(f"    Market fetch failed: {e}")
        sys.exit(1)

    import json

    best_token = None
    best_ask = None
    best_question = None

    for m in markets:
        question = m.get("question", "")
        clob_ids_raw = m.get("clobTokenIds", "[]")
        if isinstance(clob_ids_raw, str):
            try:
                clob_ids = json.loads(clob_ids_raw)
            except Exception:
                clob_ids = []
        else:
            clob_ids = clob_ids_raw or []

        if not clob_ids:
            continue

        yes_token = clob_ids[0]

        # Check order book
        try:
            book = client.get_order_book(str(yes_token))
            asks = getattr(book, "asks", []) or []
            if not asks:
                continue
            sorted_asks = sorted(asks, key=lambda x: float(x.price))
            ask_price = float(sorted_asks[0].price)
            if 0.01 < ask_price < 0.99:
                best_token = str(yes_token)
                best_ask = ask_price
                best_question = question
                print(f"    Found: {question[:55]}...")
                print(f"    Token: {best_token[:30]}...")
                print(f"    Best ask: {best_ask:.4f}")
                break
        except Exception:
            continue

    if not best_token:
        print("    [!] No tradeable market found. All order books are empty.")
        sys.exit(1)

    # ── 5. Place the order ──
    amount = 1.0
    print(f"\n[4] Placing $1.00 BUY order...")
    print(f"    Market: {best_question[:55]}")
    print(f"    Token: {best_token[:30]}...")
    print(f"    Method: FOK market order")
    print(f"    Amount: ${amount:.2f}")

    confirm = input(f"\n    ⚠️  This spends $1.00 real money. Continue? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("    Cancelled.")
        sys.exit(0)

    # Try FOK market order
    print("\n    Submitting FOK market order...")
    try:
        resp = client.create_and_post_market_order(
            token_id=best_token,
            amount=amount,
            side="BUY",
            order_type="FOK",
        )
        print(f"    Response: {resp}")
        oid = resp.get("orderID") or resp.get("order_id") or resp.get("id")
        status = resp.get("status", "?")
        print(f"    Order ID: {oid}")
        print(f"    Status: {status}")

        if oid and status not in ("REJECTED", "FAILED"):
            print("    Waiting 3s...")
            time.sleep(3)
            try:
                check = client.get_order(oid)
                print(f"    Final status: {check.get('status', '?')}")
            except Exception:
                pass
            print(f"\n    ✅ Order submitted! Your execution pipeline WORKS.")
            print(f"    The problem is in signal filtering, not order placement.")
            return

    except Exception as e:
        fok_error = str(e)
        print(f"    FOK failed: {fok_error}")

    # Fallback: GTC limit order at the ask
    print(f"\n    Trying GTC limit order at {best_ask:.4f}...")
    shares = amount / best_ask
    try:
        resp = client.create_and_post_order(
            token_id=best_token,
            price=best_ask,
            size=shares,
            side="BUY",
            order_type="GTC",
        )
        print(f"    Response: {resp}")
        oid = resp.get("orderID") or resp.get("order_id") or resp.get("id")
        status = resp.get("status", "?")
        print(f"    Order ID: {oid}")
        print(f"    Status: {status}")

        if oid:
            print("    Waiting 5s for fill...")
            time.sleep(5)
            try:
                check = client.get_order(oid)
                final = check.get("status", "?")
                print(f"    Final status: {final}")
                if final.upper() not in ("FILLED", "MATCHED"):
                    print(f"    Cancelling unfilled order...")
                    try:
                        client.cancel_order(oid)
                        print(f"    Cancelled.")
                    except Exception:
                        print(f"    Cancel failed — manually cancel on polymarket.com")
            except Exception:
                pass

        print(f"\n    ✅ Order submitted! Execution pipeline WORKS.")
        print(f"    Problem is in your bot's signal pipeline, not execution.")

    except Exception as e:
        err = str(e).lower()
        print(f"    GTC also failed: {e}")
        print()

        if "insufficient" in err or "balance" in err:
            print("    ❌ INSUFFICIENT FUNDS")
            print("    Your USDC is on-chain but not deposited to Polymarket CLOB.")
            print("    Go to polymarket.com → Deposit → deposit your USDC.")
        elif "signature" in err or "401" in err or "unauthorized" in err:
            print("    ❌ AUTH ERROR")
            print(f"    Current sig_type={sig_type}. Try 1 (email login) or 2 (MetaMask).")
        elif "tick_size" in err:
            print("    ❌ TICK SIZE BUG — run: python apply_tick_size_fix.py")
        else:
            print(f"    ❌ UNKNOWN ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
