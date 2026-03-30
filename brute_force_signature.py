"""
brute_force_signature.py
=========================
Tests EVERY signature_type + funder combination by actually POSTING
an order to the CLOB (not just local signing).

The previous diagnostic was wrong — create_order() always succeeds locally.
The "invalid signature" error happens at post_order() on the server side.

This script posts a $0.01 order at price 0.01 (will never fill, and gets
rejected for min-size before any money is spent) to find which combination
the CLOB actually accepts.

Usage:
    python brute_force_signature.py
"""

import os
import sys
import json
import time
import logging

logging.basicConfig(level=logging.WARNING)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def derive_eoa(pk):
    try:
        from eth_account import Account
        return Account.from_key(pk).address
    except Exception:
        return None


def find_any_token_id():
    """Get a real token_id from the Gamma API to test with."""
    import requests
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 5, "active": True, "closed": False,
                    "order": "volume24hr", "ascending": False},
            timeout=15,
        )
        resp.raise_for_status()
        for m in resp.json():
            raw = m.get("clobTokenIds", "[]")
            if isinstance(raw, str):
                ids = json.loads(raw)
            else:
                ids = raw or []
            if ids:
                return ids[0], m.get("question", "?")[:50]
    except Exception as e:
        print(f"  [!] Could not fetch token_id: {e}")
    return None, None


def test_post_order(pk, funder, sig_type, host, chain_id, token_id):
    """
    Actually POST a tiny order to the CLOB and return the server's response.
    Uses price=0.01 size=0.01 which will be rejected for min-size,
    NOT for signature (if signature is correct).
    """
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
    except ImportError:
        return {"status": "SKIP", "error": "py_clob_client not installed"}

    result = {
        "sig_type": sig_type,
        "funder": funder,
        "derive_ok": False,
        "post_status": None,
        "post_error": None,
        "is_signature_error": False,
        "is_size_error": False,
        "is_balance_error": False,
        "raw_response": None,
    }

    try:
        # Build client with this specific combination
        kwargs = {
            "key": pk,
            "chain_id": chain_id,
            "signature_type": sig_type,
        }
        if funder and sig_type in (1, 2):
            kwargs["funder"] = funder

        client = ClobClient(host, **kwargs)

        # Derive and set API creds
        try:
            creds = client.derive_api_key()
            client.set_api_creds(creds)
            result["derive_ok"] = True
        except Exception:
            try:
                creds = client.create_or_derive_api_creds()
                client.set_api_creds(creds)
                result["derive_ok"] = True
            except Exception as e:
                result["post_error"] = f"derive failed: {e}"
                return result

        # Create a tiny order (will never fill)
        args = OrderArgs(
            token_id=token_id,
            price=0.01,
            size=0.01,
            side=BUY,
        )

        try:
            signed = client.create_order(args)
        except Exception as e:
            # create_order can fail with tick_size issues
            err = str(e).lower()
            if "tick_size" in err:
                # Try with PartialCreateOrderOptions
                try:
                    from py_clob_client.clob_types import PartialCreateOrderOptions
                    opts = PartialCreateOrderOptions(tick_size="0.01")
                    signed = client.create_order(args, options=opts)
                except Exception as e2:
                    result["post_error"] = f"create_order failed: {e2}"
                    return result
            else:
                result["post_error"] = f"create_order failed: {e}"
                return result

        # Actually POST to the CLOB — this is where "invalid signature" happens
        try:
            response = client.post_order(signed, OrderType.GTC)
            result["post_status"] = "SUCCESS"
            result["raw_response"] = str(response)

            # If we get here, the signature was accepted!
            # Cancel the order immediately
            order_id = None
            if isinstance(response, dict):
                order_id = response.get("orderID") or response.get("order_id") or response.get("id")
            if order_id:
                try:
                    client.cancel(order_id)
                except Exception:
                    pass

        except Exception as e:
            err_str = str(e).lower()
            result["post_error"] = str(e)

            if "invalid signature" in err_str or "signature" in err_str:
                result["is_signature_error"] = True
                result["post_status"] = "SIGNATURE_ERROR"
            elif "insufficient" in err_str or "balance" in err_str:
                result["is_balance_error"] = True
                result["post_status"] = "BALANCE_ERROR"
                # Balance error means signature was ACCEPTED!
            elif "size" in err_str or "min" in err_str or "too small" in err_str:
                result["is_size_error"] = True
                result["post_status"] = "SIZE_ERROR"
                # Size error means signature was ACCEPTED!
            elif "not enough" in err_str:
                result["is_balance_error"] = True
                result["post_status"] = "BALANCE_ERROR"
            else:
                result["post_status"] = "OTHER_ERROR"

    except Exception as e:
        result["post_error"] = str(e)
        result["post_status"] = "CRASH"

    return result


def main():
    print("=" * 65)
    print("  BRUTE-FORCE SIGNATURE TYPE TESTER")
    print("  (Actually POSTS orders to find which sig_type works)")
    print("=" * 65)

    pk = os.getenv("PRIVATE_KEY", "")
    funder = os.getenv("POLYMARKET_FUNDER", "")
    host = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
    chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))

    if not pk:
        print("\n[!] PRIVATE_KEY not set.")
        sys.exit(1)

    eoa = derive_eoa(pk)
    print(f"\n  Private key:   {pk[:6]}...{pk[-4:]}")
    print(f"  Derived EOA:   {eoa}")
    print(f"  Funder:        {funder}")

    # Get a real token to test with
    print(f"\n  Finding a real token to test with...")
    token_id, market_name = find_any_token_id()
    if not token_id:
        print("  [!] Could not find any active market token. Exiting.")
        sys.exit(1)
    print(f"  Using: {market_name}")
    print(f"  Token: {token_id[:30]}...")

    # Test ALL combinations
    combos = [
        # (sig_type, funder_to_use, description)
        (0, None,    "EOA, no funder (key signs directly)"),
        (0, funder,  "EOA, with funder (unusual)"),
        (1, funder,  "Email/Magic proxy, funder=profile wallet"),
        (2, funder,  "MetaMask proxy, funder=profile wallet"),
    ]

    # Also test with derived EOA as funder if different
    if eoa and eoa.lower() != funder.lower():
        combos.extend([
            (1, eoa, "Email/Magic proxy, funder=derived EOA"),
            (2, eoa, "MetaMask proxy, funder=derived EOA"),
        ])

    print(f"\n─── TESTING {len(combos)} COMBINATIONS ───")
    print(f"  (Each test actually posts to the CLOB server)")
    print()

    results = []
    for sig_type, test_funder, desc in combos:
        print(f"  [{sig_type}] {desc}...", end=" ", flush=True)
        time.sleep(0.5)  # Rate limit

        r = test_post_order(pk, test_funder, sig_type, host, chain_id, token_id)
        results.append((sig_type, test_funder, desc, r))

        if r["post_status"] == "SUCCESS":
            print("✅ ORDER ACCEPTED!")
        elif r["is_balance_error"]:
            print("✅ SIGNATURE OK (balance error = auth passed!)")
        elif r["is_size_error"]:
            print("✅ SIGNATURE OK (size error = auth passed!)")
        elif r["is_signature_error"]:
            print("❌ invalid signature")
        else:
            err_short = str(r.get("post_error", "?"))[:60]
            print(f"❓ {r['post_status']}: {err_short}")

    # Find winners
    print("\n─── RESULTS ───")

    # "Winner" = any response that's NOT an invalid signature error
    winners = []
    for sig_type, test_funder, desc, r in results:
        if r["post_status"] in ("SUCCESS",) or r["is_balance_error"] or r["is_size_error"]:
            winners.append((sig_type, test_funder, desc, r))

    if winners:
        best = winners[0]
        sig_type, best_funder, desc, r = best

        print(f"\n  ✅ WORKING COMBINATION FOUND:")
        print(f"     sig_type = {sig_type}")
        print(f"     funder   = {best_funder or '(none)'}")
        print(f"     desc     = {desc}")
        if r["is_balance_error"]:
            print(f"     note     = Signature accepted, but CLOB balance is $0")
            print(f"                → Deposit USDC on polymarket.com first")

        # Update .env
        print(f"\n  Updating .env...")
        _update_env("POLYMARKET_SIGNATURE_TYPE", str(sig_type))
        if best_funder:
            _update_env("POLYMARKET_FUNDER", best_funder)
        elif funder:
            # sig_type=0 with no funder — remove the warning by keeping funder
            pass

        print(f"  Done!")
        print()

        if r["is_balance_error"]:
            print(f"  ╔════════════════════════════════════════════════════╗")
            print(f"  ║  NEXT STEP: DEPOSIT USDC TO CLOB                 ║")
            print(f"  ║                                                    ║")
            print(f"  ║  Your signature is CORRECT but CLOB balance = $0. ║")
            print(f"  ║                                                    ║")
            print(f"  ║  1. Go to https://polymarket.com                  ║")
            print(f"  ║  2. Log in → Profile → Deposit                   ║")
            print(f"  ║  3. Deposit your on-chain USDC                   ║")
            print(f"  ║  4. Make one manual trade on the website          ║")
            print(f"  ║  5. Then run: python test_order_minimal.py        ║")
            print(f"  ╚════════════════════════════════════════════════════╝")
        else:
            print(f"  Run: python test_order_minimal.py")
            print(f"  Then: python apply_no_trade_fix.py && python run_bot.py")

    else:
        print(f"\n  ❌ NO COMBINATION WORKS")
        print()
        print(f"  Every sig_type + funder combo returned 'invalid signature'.")
        print(f"  This means your PRIVATE_KEY is NOT the authorized signer")
        print(f"  for ANY Polymarket account.")
        print()
        print(f"  Your key derives to: {eoa}")
        print(f"  Your funder is:      {funder}")
        print()
        print(f"  ╔════════════════════════════════════════════════════════╗")
        print(f"  ║  HOW TO GET THE CORRECT PRIVATE KEY:                 ║")
        print(f"  ║                                                      ║")
        print(f"  ║  If you log in with EMAIL (Magic):                   ║")
        print(f"  ║    → Go to polymarket.com → Settings → Export Key    ║")
        print(f"  ║    → Or check Magic dashboard for your signer key    ║")
        print(f"  ║    → The exported key will match your proxy wallet   ║")
        print(f"  ║                                                      ║")
        print(f"  ║  If you log in with METAMASK:                        ║")
        print(f"  ║    → Export private key from MetaMask                ║")
        print(f"  ║    → Settings → Security → Reveal Private Key       ║")
        print(f"  ║    → Use the account that shows on polymarket.com    ║")
        print(f"  ║                                                      ║")
        print(f"  ║  VERIFY: The derived address from your key should    ║")
        print(f"  ║  match what polymarket.com shows in your profile     ║")
        print(f"  ║  (or be the recognized signer for your proxy).      ║")
        print(f"  ╚════════════════════════════════════════════════════════╝")
        print()

        # Show all errors for debugging
        print(f"  Detailed results:")
        for sig_type, test_funder, desc, r in results:
            err = str(r.get("post_error", ""))
            # Extract just the error message
            if "error_message" in err:
                try:
                    start = err.index("error_message=") + len("error_message=")
                    err = err[start:].strip("{}' ")
                except Exception:
                    pass
            print(f"    sig={sig_type} funder={str(test_funder or 'none')[:12]}... → {err[:70]}")


def _update_env(key, value):
    """Update a key in .env file."""
    import re
    env_path = ".env"
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write(f"{key}={value}\n")
        os.environ[key] = value
        return

    text = open(env_path, "r", encoding="utf-8").read()
    pattern = rf"^{re.escape(key)}\s*=.*$"
    if re.search(pattern, text, re.MULTILINE):
        text = re.sub(pattern, f"{key}={value}", text, flags=re.MULTILINE)
    else:
        text += f"\n{key}={value}\n"
    open(env_path, "w", encoding="utf-8").write(text)
    os.environ[key] = value


if __name__ == "__main__":
    main()
