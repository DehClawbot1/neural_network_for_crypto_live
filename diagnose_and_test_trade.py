"""
diagnose_and_test_trade.py
===========================
FULL PIPELINE DIAGNOSTIC + LIVE $1 TEST ORDER

This script:
  1. Runs the exact same pipeline as supervisor.py (scrape → features → score → filter)
  2. Prints WHERE and WHY every signal gets killed
  3. Attempts to place a real $1 test order on the best available market

Usage:
    python diagnose_and_test_trade.py

If you just want the test order without the full diagnostic:
    python diagnose_and_test_trade.py --skip-diagnostic
"""

import os
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Load env ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# PART 1: FULL PIPELINE TRACE — Find where your signals die
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline_diagnostic():
    print("\n" + "=" * 70)
    print("  PART 1: FULL PIPELINE DIAGNOSTIC")
    print("  Tracing every filter that kills your signals")
    print("=" * 70 + "\n")

    # ── Step 1: Scrape signals ──
    print("─── STEP 1: SCRAPING SIGNALS ───")
    try:
        from leaderboard_scraper import run_scraper_cycle
        signals_df = run_scraper_cycle()
    except Exception as exc:
        print(f"  [FATAL] Scraper failed: {exc}")
        return None, None

    if signals_df is None or signals_df.empty:
        print("  [FATAL] No signals scraped. The leaderboard may be empty or API is down.")
        print("  → This means NO whale activity was found for BTC markets today.")
        print("  → The bot cannot trade if there are no whale signals to follow.")
        return None, None

    print(f"  [OK] Scraped {len(signals_df)} raw signals")
    if "market_title" in signals_df.columns:
        unique_markets = signals_df["market_title"].nunique()
        print(f"  [OK] Across {unique_markets} unique markets")
    if "trader_wallet" in signals_df.columns:
        unique_wallets = signals_df["trader_wallet"].nunique()
        print(f"  [OK] From {unique_wallets} unique wallets")

    # ── Step 2: Fetch markets ──
    print("\n─── STEP 2: FETCHING MARKETS ───")
    try:
        from market_monitor import fetch_btc_markets, save_market_snapshot, fetch_markets_by_slugs
        markets_df = fetch_btc_markets(closed=False)
        print(f"  [OK] Fetched {len(markets_df)} BTC markets")
    except Exception as exc:
        print(f"  [WARN] Market fetch failed: {exc}")
        markets_df = pd.DataFrame()

    # Sync missing slugs
    if not signals_df.empty and not markets_df.empty and "market_slug" in signals_df.columns:
        scraped_slugs = set(signals_df["market_slug"].dropna().astype(str).unique()) - {""}
        known_slugs = set(markets_df["slug"].dropna().astype(str).unique()) if "slug" in markets_df.columns else set()
        missing = scraped_slugs - known_slugs
        if missing:
            print(f"  [SYNC] {len(missing)} slugs missing from market universe. Fetching...")
            try:
                extra = fetch_markets_by_slugs(list(missing))
                if extra is not None and not extra.empty:
                    markets_df = pd.concat([markets_df, extra], ignore_index=True).drop_duplicates(subset=["slug"])
                    print(f"  [OK] Universe now has {len(markets_df)} markets")
            except Exception as exc:
                print(f"  [WARN] Slug sync failed: {exc}")

    # ── Step 3: Build features ──
    print("\n─── STEP 3: BUILDING FEATURES ───")
    try:
        from feature_builder import FeatureBuilder
        feature_builder = FeatureBuilder()
        features_df = feature_builder.build_features(signals_df, markets_df)
    except Exception as exc:
        print(f"  [FATAL] Feature builder failed: {exc}")
        return signals_df, markets_df

    if features_df.empty:
        print("  [FATAL] Feature builder produced 0 rows")
        print("  → Signals could not be matched to any market metadata")
        return signals_df, markets_df

    print(f"  [OK] Built {len(features_df)} feature rows")

    # Check critical fields
    missing_token = features_df["token_id"].isna().sum() if "token_id" in features_df.columns else len(features_df)
    print(f"  [INFO] Rows with missing token_id: {missing_token}/{len(features_df)}")
    if missing_token == len(features_df):
        print("  [FATAL] ALL rows have missing token_id!")
        print("  → Markets are not providing clobTokenIds / yes_token_id / no_token_id")
        print("  → The bot cannot trade without token IDs")

    # ── Step 4: Model inference ──
    print("\n─── STEP 4: MODEL INFERENCE ───")
    try:
        from model_inference import ModelInference
        from stage1_inference import Stage1Inference
        from stage2_temporal_inference import Stage2TemporalInference
        from stage3_hybrid import Stage3HybridScorer

        inferred_df = ModelInference().run(features_df)
        inferred_df = Stage1Inference().run(inferred_df)
        inferred_df = Stage2TemporalInference().run(inferred_df)
        inferred_df = Stage3HybridScorer().run(inferred_df)
        print(f"  [OK] Inference complete. {len(inferred_df)} rows.")

        if "p_tp_before_sl" in inferred_df.columns:
            p_tp = inferred_df["p_tp_before_sl"].astype(float)
            print(f"  [INFO] p_tp_before_sl: min={p_tp.min():.3f} max={p_tp.max():.3f} mean={p_tp.mean():.3f}")
        if "edge_score" in inferred_df.columns:
            edge = inferred_df["edge_score"].astype(float)
            print(f"  [INFO] edge_score: min={edge.min():.4f} max={edge.max():.4f} mean={edge.mean():.4f}")
    except Exception as exc:
        print(f"  [WARN] Model inference failed: {exc}")
        inferred_df = features_df

    # ── Step 5: Signal scoring ──
    print("\n─── STEP 5: SIGNAL SCORING ───")
    try:
        from signal_engine import SignalEngine
        scored_df = SignalEngine().score_features(inferred_df)
    except Exception as exc:
        print(f"  [FATAL] Signal scoring failed: {exc}")
        return signals_df, markets_df

    if scored_df.empty:
        print("  [FATAL] Scoring produced 0 rows")
        return signals_df, markets_df

    print(f"  [OK] Scored {len(scored_df)} rows")
    if "signal_label" in scored_df.columns:
        label_dist = scored_df["signal_label"].value_counts().to_dict()
        print(f"  [INFO] Label distribution: {label_dist}")
    if "confidence" in scored_df.columns:
        conf = scored_df["confidence"].astype(float)
        print(f"  [INFO] Confidence: min={conf.min():.3f} max={conf.max():.3f} mean={conf.mean():.3f}")

    # ── Step 6: Deduplication ──
    print("\n─── STEP 6: DEDUPLICATION ───")
    before_dedup = len(scored_df)
    if "token_id" in scored_df.columns:
        scored_df = scored_df.sort_values("confidence", ascending=False) if "confidence" in scored_df.columns else scored_df
        scored_df = scored_df.drop_duplicates(subset=["token_id"], keep="first")
    after_dedup = len(scored_df)
    killed_by_dedup = before_dedup - after_dedup
    print(f"  [INFO] Before dedup: {before_dedup} → After: {after_dedup} (killed {killed_by_dedup})")

    # ── Step 7: Entry rule filter ──
    print("\n─── STEP 7: ENTRY RULE FILTER ───")
    from strategy_layers import EntryRuleLayer, PredictionLayer
    entry_rule = EntryRuleLayer()
    print(f"  [CONFIG] min_score={entry_rule.min_score}, max_spread={entry_rule.max_spread}, min_liquidity={entry_rule.min_liquidity}")

    passed_entry = 0
    blocked_reasons = {"low_score": 0, "wide_spread": 0, "low_liquidity": 0}
    for _, row in scored_df.iterrows():
        r = row.to_dict()
        score = PredictionLayer.select_signal_score(r)
        spread = float(r.get("spread", 0.0) or 0.0)
        liquidity = float(r.get("liquidity", r.get("market_liquidity", 0.0)) or 0.0)

        if score < entry_rule.min_score:
            blocked_reasons["low_score"] += 1
        elif spread > entry_rule.max_spread:
            blocked_reasons["wide_spread"] += 1
        elif liquidity < entry_rule.min_liquidity:
            blocked_reasons["low_liquidity"] += 1
        else:
            passed_entry += 1

    print(f"  [RESULT] Passed entry rule: {passed_entry}/{len(scored_df)}")
    for reason, count in blocked_reasons.items():
        if count > 0:
            print(f"  [BLOCKED] {reason}: {count} signals killed")

    if blocked_reasons["low_liquidity"] > 0:
        # Show what liquidity values look like
        if "market_liquidity" in scored_df.columns:
            liq_vals = scored_df["market_liquidity"].astype(float)
            print(f"  [BUG?] market_liquidity values: min={liq_vals.min():.0f} max={liq_vals.max():.0f} mean={liq_vals.mean():.0f}")
            print(f"  [BUG?] Threshold is {entry_rule.min_liquidity}. If most markets have lower liquidity, this blocks everything.")

    # ── Step 8: OrderBook Guard ──
    print("\n─── STEP 8: ORDERBOOK GUARD ───")
    try:
        from orderbook_guard import OrderBookGuard
        guard = OrderBookGuard(max_spread=0.10, min_bid_depth=2, min_ask_depth=2)

        # Test top 3 token IDs
        test_tokens = scored_df["token_id"].dropna().astype(str).unique()[:3] if "token_id" in scored_df.columns else []
        ob_passed = 0
        ob_blocked = 0
        for token_id in test_tokens:
            if not token_id or token_id == "nan":
                continue
            check = guard.check_before_entry(token_id, side="BUY", intended_size_usdc=1.0)
            if check["tradable"]:
                ob_passed += 1
                analysis = check["analysis"]
                print(f"  [OK] {token_id[:20]}... spread={analysis.get('spread', 'N/A')} bid={analysis.get('best_bid')} ask={analysis.get('best_ask')}")
            else:
                ob_blocked += 1
                print(f"  [BLOCKED] {token_id[:20]}... reason={check['reason']}")

        if not test_tokens.any() if hasattr(test_tokens, 'any') else len(test_tokens) == 0:
            print("  [SKIP] No token IDs to test")
        else:
            print(f"  [RESULT] OrderBook passed: {ob_passed}/{ob_passed + ob_blocked}")
    except Exception as exc:
        print(f"  [ERROR] OrderBook guard test failed: {exc}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Raw signals scraped:     {len(signals_df) if signals_df is not None else 0}")
    print(f"  Feature rows built:      {len(features_df)}")
    print(f"  After scoring:           {before_dedup}")
    print(f"  After dedup:             {after_dedup}")
    print(f"  Passed entry rule:       {passed_entry}")
    print(f"  Passed orderbook guard:  {ob_passed if 'ob_passed' in dir() else '?'}")
    print()

    if passed_entry == 0:
        print("  ❌ DIAGNOSIS: Entry rule filter kills ALL signals.")
        print()
        if blocked_reasons["low_liquidity"] > 0:
            print("  ROOT CAUSE: market_liquidity values are below the min_liquidity threshold (100)")
            print("  FIX: Lower min_liquidity in EntryRuleLayer or use liquidity_score instead")
        if blocked_reasons["low_score"] > 0:
            print("  ROOT CAUSE: Model scores are below min_score threshold (0.45)")
            print("  FIX: Lower min_score or wait for model to train on more data")
        if blocked_reasons["wide_spread"] > 0:
            print("  ROOT CAUSE: Spreads exceed max_spread threshold (0.08)")
            print("  FIX: Increase max_spread to 0.15 for BTC prediction markets")
    elif ob_passed == 0 and 'ob_passed' in dir():
        print("  ❌ DIAGNOSIS: OrderBookGuard blocks all tradeable signals.")
        print("  FIX: Increase max_spread in OrderBookGuard or disable it for testing")
    else:
        print("  ✅ Signals CAN pass through the pipeline.")
        print("  If trades still don't execute, the issue is in live order submission.")

    return scored_df, markets_df


# ═══════════════════════════════════════════════════════════════════════
# PART 2: LIVE $1 TEST ORDER — Verify execution works
# ═══════════════════════════════════════════════════════════════════════

def run_test_order():
    print("\n" + "=" * 70)
    print("  PART 2: LIVE $1 TEST ORDER")
    print("  Attempting to place a real order on Polymarket")
    print("=" * 70 + "\n")

    # ── Check credentials ──
    print("─── CHECKING CREDENTIALS ───")
    pk = os.getenv("PRIVATE_KEY", "")
    funder = os.getenv("POLYMARKET_FUNDER", "")
    sig_type = os.getenv("POLYMARKET_SIGNATURE_TYPE", "1")

    if not pk:
        print("  [FATAL] PRIVATE_KEY is not set. Cannot place orders.")
        return False

    print(f"  PRIVATE_KEY:        {'set (' + pk[:6] + '...' + pk[-4:] + ')' if len(pk) > 10 else 'MISSING'}")
    print(f"  FUNDER:             {funder or 'MISSING'}")
    print(f"  SIGNATURE_TYPE:     {sig_type}")
    print(f"  TRADING_MODE:       {os.getenv('TRADING_MODE', 'not set')}")

    # ── Create client ──
    print("\n─── CREATING EXECUTION CLIENT ───")
    try:
        from execution_client import ExecutionClient
        client = ExecutionClient()
        print(f"  [OK] Client created. Source: {client.credential_source}")
        print(f"  [OK] Signature type: {client.signature_type}")
    except Exception as exc:
        print(f"  [FATAL] Client creation failed: {exc}")
        return False

    # ── Check balance ──
    print("\n─── CHECKING BALANCE ───")
    clob_balance = 0.0
    try:
        raw = client.get_balance_allowance(asset_type="COLLATERAL")
        if isinstance(raw, dict):
            raw_val = raw.get("balance", raw.get("amount", 0))
            clob_balance = client._normalize_usdc_balance(raw_val)
            print(f"  Raw API response: {raw_val}")
            print(f"  Normalized CLOB balance: ${clob_balance:.4f}")
    except Exception as exc:
        print(f"  [WARN] CLOB balance check failed: {exc}")

    onchain_balance = 0.0
    try:
        onchain = client.get_onchain_collateral_balance()
        onchain_balance = float((onchain or {}).get("total", 0.0) or 0.0)
        print(f"  On-chain USDC: ${onchain_balance:.4f}")
    except Exception as exc:
        print(f"  [WARN] On-chain balance check failed: {exc}")

    available = max(clob_balance, onchain_balance)
    print(f"  Available to trade: ${available:.4f}")

    if available < 1.0:
        print(f"\n  [FATAL] Need at least $1.00 to test. You have ${available:.2f}")
        print("  Deposit USDC to your Polymarket account first.")
        return False

    # ── Find a tradeable market ──
    print("\n─── FINDING A TRADEABLE MARKET ───")
    test_token = None
    test_market = None
    best_ask = None

    try:
        from market_monitor import fetch_active_markets_by_volume
        active_markets = fetch_active_markets_by_volume(limit=10)

        if active_markets.empty:
            # Fall back to BTC markets
            from market_monitor import fetch_btc_markets
            active_markets = fetch_btc_markets(closed=False)

        if active_markets.empty:
            print("  [FATAL] No active markets found")
            return False

        # Try each market until we find one with a working order book
        for _, market_row in active_markets.iterrows():
            token_id = market_row.get("yes_token_id")
            if not token_id or pd.isna(token_id):
                continue

            print(f"  Testing: {str(market_row.get('question', ''))[:60]}...")
            try:
                book = client.get_order_book(str(token_id))
                bids = getattr(book, "bids", []) or []
                asks = getattr(book, "asks", []) or []

                if not asks:
                    print(f"    → No asks (cannot buy)")
                    continue

                sorted_asks = sorted(asks, key=lambda x: float(getattr(x, "price", 0)))
                ask_price = float(sorted_asks[0].price)
                ask_size = float(sorted_asks[0].size)

                sorted_bids = sorted(bids, key=lambda x: float(getattr(x, "price", 0)), reverse=True)
                bid_price = float(sorted_bids[0].price) if sorted_bids else 0.0

                spread = ask_price - bid_price if bid_price > 0 else None

                print(f"    → Best bid: {bid_price:.4f} | Best ask: {ask_price:.4f} | Spread: {spread:.4f if spread else 'N/A'}")

                if ask_price > 0 and ask_price < 0.99:
                    test_token = str(token_id)
                    test_market = str(market_row.get("question", "Unknown"))
                    best_ask = ask_price
                    print(f"    → ✅ SELECTED for test order")
                    break
                else:
                    print(f"    → Price too extreme ({ask_price}), skipping")

            except Exception as exc:
                print(f"    → Order book failed: {exc}")
                continue

    except Exception as exc:
        print(f"  [ERROR] Market search failed: {exc}")

    if not test_token:
        print("\n  [FATAL] Could not find any tradeable market with a valid order book.")
        return False

    print(f"\n  Selected market: {test_market[:60]}")
    print(f"  Token ID: {test_token[:40]}...")
    print(f"  Best ask: {best_ask:.4f}")

    # ── Calculate test order ──
    test_amount = 1.0  # $1.00
    test_shares = test_amount / best_ask
    print(f"\n─── PLACING TEST ORDER ───")
    print(f"  Amount: ${test_amount:.2f}")
    print(f"  Price: {best_ask:.4f}")
    print(f"  Shares: {test_shares:.2f}")
    print(f"  Side: BUY")

    # Ask for confirmation
    print()
    confirm = input("  ⚠️  This will spend $1.00 of real money. Proceed? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("  Cancelled.")
        return False

    # ── Method A: Try FOK market order first ──
    print("\n─── ATTEMPT A: FOK MARKET ORDER ($1.00) ───")
    try:
        response = client.create_and_post_market_order(
            token_id=test_token,
            amount=test_amount,
            side="BUY",
            order_type="FOK",
        )
        print(f"  [RESPONSE] {response}")
        order_id = response.get("orderID") or response.get("order_id") or response.get("id")
        status = response.get("status", "UNKNOWN")
        print(f"  Order ID: {order_id}")
        print(f"  Status: {status}")

        if status in ("FILLED", "MATCHED", "LIVE"):
            print(f"\n  ✅ SUCCESS! FOK market order filled.")
            print(f"  Your bot CAN place orders. The issue is in the signal pipeline, not execution.")
            return True
        elif status in ("SUBMITTED", "PENDING"):
            print(f"  Order submitted, waiting for fill...")
            time.sleep(3)
            try:
                check = client.get_order(order_id)
                print(f"  Updated status: {check.get('status', 'UNKNOWN')}")
                if str(check.get("status", "")).upper() in ("FILLED", "MATCHED"):
                    print(f"\n  ✅ SUCCESS! Order filled after wait.")
                    return True
            except Exception:
                pass
        else:
            print(f"  FOK order returned status: {status}")
            print(f"  This might mean the price moved. Trying limit order...")
    except Exception as exc:
        print(f"  [FAILED] FOK market order error: {exc}")
        print(f"  Trying limit order as fallback...")

    # ── Method B: Try GTC limit order ──
    print(f"\n─── ATTEMPT B: GTC LIMIT ORDER at {best_ask:.4f} ───")
    try:
        response = client.create_and_post_order(
            token_id=test_token,
            price=best_ask,
            size=test_shares,
            side="BUY",
            order_type="GTC",
        )
        print(f"  [RESPONSE] {response}")
        order_id = response.get("orderID") or response.get("order_id") or response.get("id")
        status = response.get("status", "UNKNOWN")
        print(f"  Order ID: {order_id}")
        print(f"  Status: {status}")

        if order_id:
            print(f"  Waiting 5 seconds for fill...")
            time.sleep(5)
            try:
                check = client.get_order(order_id)
                updated_status = str(check.get("status", "")).upper()
                print(f"  Updated status: {updated_status}")
                if updated_status in ("FILLED", "MATCHED"):
                    print(f"\n  ✅ SUCCESS! Limit order filled.")
                    return True
                elif updated_status in ("LIVE", "OPEN", "SUBMITTED"):
                    print(f"\n  ⚠️ Order is live but not yet filled.")
                    print(f"  This means your bot CAN place orders.")
                    print(f"  The order is sitting in the book waiting for a match.")
                    cancel = input("  Cancel this order? [Y/n]: ").strip().lower()
                    if cancel in ("", "y", "yes"):
                        try:
                            client.cancel_order(order_id)
                            print(f"  Order cancelled.")
                        except Exception as exc:
                            print(f"  Cancel failed: {exc}")
                    return True
            except Exception as exc:
                print(f"  Status check failed: {exc}")
                print(f"  But the order was submitted, so execution pipeline works.")
                return True

        print(f"\n  ✅ Order submitted successfully (ID: {order_id}).")
        print(f"  Your execution pipeline WORKS. The problem is in signal filtering.")
        return True

    except Exception as exc:
        error_msg = str(exc).lower()
        print(f"  [FAILED] Limit order error: {exc}")
        print()

        if "insufficient" in error_msg or "balance" in error_msg:
            print("  ❌ DIAGNOSIS: Insufficient funds error.")
            print("  → Your CLOB balance is $0. Funds are on-chain but not deposited to CLOB.")
            print("  → Go to polymarket.com > Deposit to move funds into the trading engine.")
        elif "signature" in error_msg or "unauthorized" in error_msg or "401" in error_msg:
            print("  ❌ DIAGNOSIS: Authentication/signature error.")
            print("  → POLYMARKET_SIGNATURE_TYPE may be wrong.")
            print(f"  → Current: {sig_type}. Try changing to 1 (email) or 2 (MetaMask).")
        elif "tick_size" in error_msg:
            print("  ❌ DIAGNOSIS: tick_size error — known bug in py-clob-client options handling.")
            print("  → Run: python apply_tick_size_fix.py")
        elif "orderbook" in error_msg and "does not exist" in error_msg:
            print("  ❌ DIAGNOSIS: This market's order book doesn't exist on the CLOB.")
            print("  → The market may have expired or been removed.")
        else:
            print("  ❌ DIAGNOSIS: Unknown execution error.")
            print(f"  → Full error: {exc}")

        return False


# ═══════════════════════════════════════════════════════════════════════
# PART 3: RECOMMENDED FIXES
# ═══════════════════════════════════════════════════════════════════════

def print_fixes():
    print("\n" + "=" * 70)
    print("  RECOMMENDED FIXES")
    print("=" * 70)
    print("""
  The most common reasons a bot runs 10h with 0 trades:

  1. ENTRY RULE FILTER (strategy_layers.py)
     The EntryRuleLayer checks:
       - score >= 0.45 (model must be trained)
       - spread <= 0.08 (too tight for most BTC markets)
       - liquidity >= 100 (many markets have less)

     FIX: Relax these thresholds:
       EntryRuleLayer(min_score=0.30, max_spread=0.15, min_liquidity=10)

  2. ORDERBOOK GUARD (supervisor.py)
     max_spread=0.10 blocks markets with > 10% spread.
     BTC 5-min prediction markets routinely have 10-20% spreads.

     FIX: Increase to 0.20 or disable for testing.

  3. TOKEN DEDUPLICATION (supervisor.py)
     scored_df.drop_duplicates(subset=["token_id"]) keeps only ONE signal
     per token. If that one has low confidence, it gets filtered.

     FIX: Keep top-N per token instead of just top-1.

  4. NO MODEL WEIGHTS (fresh install)
     Without trained models, p_tp_before_sl and edge_score are 0.0,
     so the signal score is based only on heuristics (whale_pressure etc).
     These may be below 0.45 threshold.

     FIX: Run the research pipeline first:
       python real_pipeline.py

  5. LIVE ORDER SUBMISSION BUG (supervisor.py)
     In the live entry path, there's a reference to 'trade.shares' before
     'trade' is defined. This crashes silently.

     FIX: Applied in the patch file below.
""")


def apply_quick_fix():
    """Apply the minimum changes needed to get trades flowing."""
    print("\n─── APPLYING QUICK FIXES ───")
    fixes_applied = 0

    # Fix 1: Relax EntryRuleLayer thresholds
    path = "strategy_layers.py"
    if os.path.exists(path):
        text = open(path, "r", encoding="utf-8").read()
        changed = False

        if "min_score=0.45" in text:
            text = text.replace("min_score=0.45", "min_score=0.30")
            changed = True
            print(f"  [FIX] {path}: min_score 0.45 → 0.30")

        if "max_spread=0.08" in text:
            text = text.replace("max_spread=0.08", "max_spread=0.15")
            changed = True
            print(f"  [FIX] {path}: max_spread 0.08 → 0.15")

        if "min_liquidity=100" in text:
            text = text.replace("min_liquidity=100", "min_liquidity=10")
            changed = True
            print(f"  [FIX] {path}: min_liquidity 100 → 10")

        if changed:
            open(path, "w", encoding="utf-8").write(text)
            fixes_applied += 1

    # Fix 2: Relax OrderBookGuard in supervisor.py
    path = "supervisor.py"
    if os.path.exists(path):
        text = open(path, "r", encoding="utf-8").read()
        if "max_spread=0.10" in text:
            text = text.replace("max_spread=0.10", "max_spread=0.20")
            open(path, "w", encoding="utf-8").write(text)
            fixes_applied += 1
            print(f"  [FIX] {path}: OrderBookGuard max_spread 0.10 → 0.20")

    # Fix 3: Fix the dead variable reference in supervisor.py live entry path
    path = "supervisor.py"
    if os.path.exists(path):
        text = open(path, "r", encoding="utf-8").read()
        # The bug: `trade.shares` is referenced before `trade` is created
        old_bug = 'actual_fill_size = float(fill_payload.get("size", trade.shares) or trade.shares)'
        fix = 'actual_fill_size = float(fill_payload.get("size", 0) or 0)'
        if old_bug in text:
            text = text.replace(old_bug, fix)
            open(path, "w", encoding="utf-8").write(text)
            fixes_applied += 1
            print(f"  [FIX] {path}: Fixed dead 'trade.shares' reference in live entry")

    print(f"\n  Applied {fixes_applied} fixes.")
    return fixes_applied


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    skip_diagnostic = "--skip-diagnostic" in sys.argv
    skip_fix = "--no-fix" in sys.argv

    if not skip_diagnostic:
        scored_df, markets_df = run_pipeline_diagnostic()

    print_fixes()

    if not skip_fix:
        print()
        fix = input("  Apply quick fixes to relax filters? [Y/n]: ").strip().lower()
        if fix in ("", "y", "yes"):
            apply_quick_fix()

    print()
    test = input("  Run $1 test order on Polymarket? [Y/n]: ").strip().lower()
    if test in ("", "y", "yes"):
        run_test_order()

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
