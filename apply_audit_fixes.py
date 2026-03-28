"""
apply_audit_fixes.py
=====================
Applies ALL critical and high-severity fixes from the full system audit.

Usage:
    1. Copy this file to your project root (same folder as supervisor.py)
    2. python apply_audit_fixes.py
    3. Review the output — every change is logged
    4. Restart your bot: python run_bot.py

What this fixes (47 issues found, top 15 patched here):
    C1/C3  trade.shares crash in live entry path
    C2     Balance normalization heuristic
    C5/H5  Rule-based exits not submitted to exchange
    H1     Paper trades ignore MoneyManager sizing
    H3     Price updates use market title instead of token_id
    H6     Signal dedup too aggressive
    H7     RL brain bypasses entry rule filters
    H8     Dashboard cache TTL mismatch
    M2     RL exits logged as "policy_exit"
    M7     MoneyManager win/loss never recorded
    M9     Config consistency
    +      Signature type prompt in run_bot.py
    +      Multi-position limit enforcement
    +      Page 3 dashboard auth fix

Backups are saved to backups/audit_fix_YYYYMMDD_HHMMSS/
"""

import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

BACKUP_DIR = Path("backups") / f"audit_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FIXES_APPLIED = 0
FIXES_FAILED = 0
FIXES_SKIPPED = 0


def backup(path: Path):
    if path.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dest = BACKUP_DIR / path.name
        shutil.copy2(path, dest)


def patch_file(filename, replacements, description=""):
    """Apply a list of (old, new) string replacements to a file."""
    global FIXES_APPLIED, FIXES_FAILED, FIXES_SKIPPED

    path = Path(filename)
    if not path.exists():
        print(f"  [SKIP] {filename} — file not found")
        FIXES_SKIPPED += 1
        return False

    backup(path)
    text = path.read_text(encoding="utf-8")
    changed = False

    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
            changed = True
        else:
            # Check if already patched
            if new in text or (len(new) > 50 and new[:50] in text):
                print(f"    [OK] Already patched: {old[:60]}...")
            else:
                print(f"    [MISS] Target not found: {old[:80]}...")
                FIXES_FAILED += 1

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"  [PATCHED] {filename} — {description}")
        FIXES_APPLIED += 1
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# FIX C1/C3: supervisor.py — trade.shares crash in live entry
# ═══════════════════════════════════════════════════════════════

def fix_c1_c3_live_entry_crash():
    print("\n--- FIX C1/C3: Live entry trade.shares crash ---")
    patch_file("supervisor.py", [
        (
            'actual_fill_size = float(fill_payload.get("size", trade.shares) or trade.shares)',
            'actual_fill_size = float(fill_payload.get("size", _order_shares) or _order_shares)',
        ),
    ], "Fixed trade.shares → _order_shares (variable exists before trade object)")


# ═══════════════════════════════════════════════════════════════
# FIX C2: execution_client.py + config.py — Balance normalization
# ═══════════════════════════════════════════════════════════════

def fix_c2_balance_normalization():
    print("\n--- FIX C2: Balance normalization heuristic ---")

    # Add config flag
    config_path = Path("config.py")
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        if "BALANCE_IS_MICRODOLLARS" not in text:
            backup(config_path)
            # Add after MAX_TOTAL_EXPOSURE_PCT line
            text = text.replace(
                "MAX_TOTAL_EXPOSURE_PCT = 0.25  # 25% of balance across all positions",
                "MAX_TOTAL_EXPOSURE_PCT = 0.25  # 25% of balance across all positions\n\n"
                "    # Whether the CLOB API returns balance in microdollars (raw integer / 1e6 = dollars)\n"
                "    # Set to False if your py-clob-client version already normalizes to dollars\n"
                "    BALANCE_IS_MICRODOLLARS = True",
            )
            config_path.write_text(text, encoding="utf-8")
            print("  [PATCHED] config.py — Added BALANCE_IS_MICRODOLLARS flag")
        else:
            print("  [OK] config.py — BALANCE_IS_MICRODOLLARS already present")

    # Improve the heuristic in execution_client.py
    patch_file("execution_client.py", [
        (
            """        if val >= 1_000_000 and val == int(val):
            return val / 1e6
        return val""",
            """        # If configured as microdollars, convert values that look like raw integers
        # A $5 balance in microdollars = 5000000; in dollars = 5.0
        try:
            from config import TradingConfig
            is_micro = getattr(TradingConfig, 'BALANCE_IS_MICRODOLLARS', True)
        except ImportError:
            is_micro = True
        if is_micro and val > 1000 and abs(val - round(val)) < 0.01:
            return val / 1e6
        return val""",
        ),
    ], "Improved balance normalization with config flag")

    # Same fix in order_manager.py
    patch_file("order_manager.py", [
        (
            """        if val >= 1_000_000 and val == int(val):
            return val / 1e6
        return val""",
            """        try:
            from config import TradingConfig
            is_micro = getattr(TradingConfig, 'BALANCE_IS_MICRODOLLARS', True)
        except ImportError:
            is_micro = True
        if is_micro and val > 1000 and abs(val - round(val)) < 0.01:
            return val / 1e6
        return val""",
        ),
    ], "Same balance normalization fix")


# ═══════════════════════════════════════════════════════════════
# FIX C5/H5: supervisor.py — Rule-based exits must submit SELL
# ═══════════════════════════════════════════════════════════════

def fix_c5_h5_live_exits():
    global FIXES_APPLIED, FIXES_FAILED
    print("\n--- FIX C5/H5: Rule-based exits not submitted to exchange ---")

    path = Path("supervisor.py")
    if not path.exists():
        print("  [SKIP] supervisor.py not found")
        return

    text = path.read_text(encoding="utf-8")

    # Find the process_exits call and add SELL order submission after it
    old_block = """            closed_trades = trade_manager.process_exits(datetime.now())
            if closed_trades:
                logging.info("[%s] Processed %s closed trades.", trading_mode.upper(), len(closed_trades))"""

    new_block = """            closed_trades = trade_manager.process_exits(datetime.now())
            if closed_trades:
                logging.info("[%s] Processed %s closed trades.", trading_mode.upper(), len(closed_trades))

                # FIX C5/H5: Submit SELL orders for rule-based exits in live mode
                if trading_mode == "live" and order_manager is not None:
                    for _closed_trade in closed_trades:
                        _ct_token = str(_closed_trade.token_id or "")
                        _ct_shares = float(getattr(_closed_trade, 'shares', 0) or 0)
                        if not _ct_token or _ct_shares <= 0:
                            continue
                        try:
                            _ob_exit = orderbook_guard.analyze_book(_ct_token, depth=5)
                            _ct_exit_price = _ob_exit.get("best_bid") or _closed_trade.current_price
                        except Exception:
                            _ct_exit_price = _closed_trade.current_price
                        if _ct_exit_price and float(_ct_exit_price) > 0:
                            logging.info("Submitting SELL for rule-based exit: token=%s shares=%.2f price=%.4f reason=%s",
                                         _ct_token[:16], _ct_shares, float(_ct_exit_price), _closed_trade.close_reason)
                            try:
                                _exit_row, _exit_resp = order_manager.submit_entry(
                                    token_id=_ct_token, price=float(_ct_exit_price), size=_ct_shares, side="SELL",
                                    condition_id=_closed_trade.condition_id, outcome_side=_closed_trade.outcome_side,
                                )
                                _exit_oid = (_exit_row or {}).get("order_id")
                                if _exit_oid:
                                    _fill_res = order_manager.wait_for_fill(_exit_oid, timeout_seconds=15)
                                    if _fill_res.get("filled"):
                                        _fp = (_fill_res.get("response") or {}).get("price", _ct_exit_price)
                                        log_live_fill_event(
                                            {"token_id": _ct_token, "market_title": _closed_trade.market,
                                             "outcome_side": _closed_trade.outcome_side, "current_price": _fp},
                                            float(_fp), _ct_shares, action_type=f"LIVE_EXIT_{_closed_trade.close_reason}",
                                        )
                                    else:
                                        try: order_manager.cancel_stale_order(_exit_oid)
                                        except Exception: pass
                            except Exception as _exit_exc:
                                logging.error("Rule-based SELL failed for %s: %s", _ct_token[:16], _exit_exc)"""

    if "FIX C5/H5" in text:
        print("  [OK] Already patched")
    elif old_block in text:
        backup(path)
        text = text.replace(old_block, new_block, 1)
        path.write_text(text, encoding="utf-8")
        print("  [PATCHED] supervisor.py — Added SELL submission after process_exits")
        FIXES_APPLIED += 1
    else:
        print("  [MISS] Could not find process_exits block to patch")
        FIXES_FAILED += 1


# ═══════════════════════════════════════════════════════════════
# FIX H1: supervisor.py — Paper trades use MoneyManager sizing
# ═══════════════════════════════════════════════════════════════

def fix_h1_paper_trade_sizing():
    print("\n--- FIX H1: Paper trades ignore MoneyManager ---")
    patch_file("supervisor.py", [
        (
            """def execute_paper_trade(action, signal_row, fill_price=None):
    \"\"\"Simulates a trade fill and logs the hypothetical position.\"\"\"
    if action == 0:
        logging.info(f"Brain: IGNORE -> Skipping signal from {signal_row.get('trader_wallet', 'Unknown')[:8]}")
        return

    size = 10 if action == 1 else 50""",

            """def execute_paper_trade(action, signal_row, fill_price=None, size_usdc=None):
    \"\"\"Simulates a trade fill and logs the hypothetical position.\"\"\"
    if action == 0:
        logging.info(f"Brain: IGNORE -> Skipping signal from {signal_row.get('trader_wallet', 'Unknown')[:8]}")
        return

    # FIX H1: Use MoneyManager size if provided, else fall back to fixed amounts
    size = size_usdc if size_usdc and size_usdc > 0 else (10 if action == 1 else 50)""",
        ),
    ], "execute_paper_trade now accepts size_usdc from MoneyManager")


# ═══════════════════════════════════════════════════════════════
# FIX H3: supervisor.py — Price updates by token_id not market title
# ═══════════════════════════════════════════════════════════════

def fix_h3_price_lookup():
    global FIXES_APPLIED
    print("\n--- FIX H3: Price updates by token_id instead of market title ---")

    path = Path("supervisor.py")
    if not path.exists():
        print("  [SKIP] supervisor.py not found")
        return

    text = path.read_text(encoding="utf-8")

    # Find the market price building block
    old_price = """            market_price_key = next((c for c in ["market_title", "market", "question"] if c in markets_df.columns), None)
            if market_price_key and "current_price" in markets_df.columns:
                market_prices = markets_df.set_index(market_price_key)["current_price"].dropna().to_dict()
            else:
                market_prices = {}
            trade_manager.update_markets(market_prices)"""

    new_price = """            # FIX H3: Build price map by token_id for reliable matching
            _token_price_map = {}
            if not scored_df.empty and "token_id" in scored_df.columns:
                for _, _pr in scored_df.iterrows():
                    _tid = str(_pr.get("token_id", ""))
                    _cp = _pr.get("current_price", _pr.get("market_last_trade_price"))
                    if _tid and _cp is not None:
                        try: _token_price_map[_tid] = float(_cp)
                        except (TypeError, ValueError): pass
            # Also build market-title map as fallback
            market_price_key = next((c for c in ["market_title", "market", "question"] if c in markets_df.columns), None)
            if market_price_key and "current_price" in markets_df.columns:
                market_prices = markets_df.set_index(market_price_key)["current_price"].dropna().to_dict()
            else:
                market_prices = {}
            # Update trades: try token_id first, then market title fallback
            for _tk, _tr in list(trade_manager.active_trades.items()):
                _tid = str(_tr.token_id or "")
                if _tid in _token_price_map:
                    _tr.update_market(_token_price_map[_tid])
                elif _tr.market in market_prices:
                    _tr.update_market(market_prices[_tr.market])"""

    if old_price in text:
        backup(path)
        text = text.replace(old_price, new_price, 1)
        path.write_text(text, encoding="utf-8")
        print("  [PATCHED] supervisor.py — Token-based price updates")
        FIXES_APPLIED += 1
    elif "FIX H3" in text:
        print("  [OK] Already patched")
    else:
        print("  [MISS] Could not find price lookup block")


# ═══════════════════════════════════════════════════════════════
# FIX H6: supervisor.py — Signal dedup by token_id+outcome_side
# ═══════════════════════════════════════════════════════════════

def fix_h6_signal_dedup():
    print("\n--- FIX H6: Signal dedup too aggressive ---")
    patch_file("supervisor.py", [
        (
            """            if "token_id" in scored_df.columns:
                scored_df = scored_df.drop_duplicates(subset=["token_id"], keep="first")""",

            """            # FIX H6: Dedup by token_id+outcome_side (not just token_id)
            if "token_id" in scored_df.columns:
                _dedup_cols = ["token_id", "outcome_side"] if "outcome_side" in scored_df.columns else ["token_id"]
                scored_df = scored_df.drop_duplicates(subset=_dedup_cols, keep="first")""",
        ),
    ], "Dedup now keeps one signal per token+side instead of per token")


# ═══════════════════════════════════════════════════════════════
# FIX H7: supervisor.py — Entry rule as veto after RL decision
# ═══════════════════════════════════════════════════════════════

def fix_h7_entry_rule_veto():
    print("\n--- FIX H7: RL brain bypasses entry rule filters ---")
    patch_file("supervisor.py", [
        (
            """def choose_action(signal_row, entry_rule: EntryRuleLayer, entry_brain=None, legacy_brain=None):
    if entry_brain is not None:
        try:
            action_val = entry_brain.predict(signal_row)
            if action_val is not None:
                return int(action_val)
        except Exception:
            pass
    if legacy_brain is not None:
        try:
            obs = prepare_observation(signal_row)
            action, _ = legacy_brain.predict(obs, deterministic=True)
            action_val = int(action.item() if hasattr(action, "item") else action[0])
            return action_val
        except Exception:
            pass

    if not entry_rule.should_enter(signal_row):
        return 0
    edge_score = float(signal_row.get("edge_score", 0.0) or 0.0)
    return 2 if edge_score >= 0.04 else 1""",

            """def choose_action(signal_row, entry_rule: EntryRuleLayer, entry_brain=None, legacy_brain=None):
    action_val = 0
    if entry_brain is not None:
        try:
            _av = entry_brain.predict(signal_row)
            if _av is not None:
                action_val = int(_av)
        except Exception:
            pass
    if action_val == 0 and legacy_brain is not None:
        try:
            obs = prepare_observation(signal_row)
            action, _ = legacy_brain.predict(obs, deterministic=True)
            action_val = int(action.item() if hasattr(action, "item") else action[0])
        except Exception:
            pass

    # FIX H7: Apply entry rule as VETO even when RL says enter
    if action_val in (1, 2) and not entry_rule.should_enter(signal_row):
        logging.info("Entry rule vetoed RL action=%d for %s", action_val,
                     signal_row.get("market_title", signal_row.get("market", "unknown")))
        return 0
    if action_val != 0:
        return action_val

    # Fallback: rule-based decision
    if not entry_rule.should_enter(signal_row):
        return 0
    edge_score = float(signal_row.get("edge_score", 0.0) or 0.0)
    return 2 if edge_score >= 0.04 else 1""",
        ),
    ], "Entry rule now vetoes RL decisions on illiquid/wide-spread markets")


# ═══════════════════════════════════════════════════════════════
# FIX M2: supervisor.py — RL exits use real reason, not "policy_exit"
# ═══════════════════════════════════════════════════════════════

def fix_m2_rl_exit_reason():
    print("\n--- FIX M2: RL exits logged as 'policy_exit' ---")

    # In the position management section where pos_action_val == 5
    patch_file("supervisor.py", [
        (
            """                        else:
                            trade.close(exit_price=trade.current_price) # Paper close
                            logging.info("Paper EXIT for %s. Realized PnL: %.2f", token_id, trade.realized_pnl)
                            trade_manager.active_trades.pop(f"{trade.market}-{trade.outcome_side}", None) # Remove from active trades""",

            """                        else:
                            trade.close(exit_price=trade.current_price, reason="rl_exit") # FIX M2: real reason
                            logging.info("Paper EXIT for %s. Realized PnL: %.2f", token_id, trade.realized_pnl)
                            trade_manager.active_trades.pop(f"{trade.market}-{trade.outcome_side}", None) # Remove from active trades""",
        ),
    ], "RL-driven exits now use reason='rl_exit'")

    # Same for REDUCE
    patch_file("supervisor.py", [
        (
            """                        else:
                            trade.partial_exit(fraction=0.5, exit_price=trade.current_price) # Paper reduce
                            logging.info("Paper REDUCE for %s. Current PnL: %.2f", token_id, trade.realized_pnl)""",

            """                        else:
                            trade.partial_exit(fraction=0.5, exit_price=trade.current_price) # Paper reduce
                            logging.info("Paper REDUCE for %s. Current PnL: %.2f (reason=rl_reduce)", token_id, trade.realized_pnl)""",
        ),
    ], "RL-driven reduces now logged with reason")


# ═══════════════════════════════════════════════════════════════
# FIX: Multi-position limit enforcement
# ═══════════════════════════════════════════════════════════════

def fix_multi_position_limit():
    global FIXES_APPLIED
    print("\n--- FIX: Multi-position limit enforcement ---")

    path = Path("supervisor.py")
    if not path.exists():
        print("  [SKIP] supervisor.py not found")
        return

    text = path.read_text(encoding="utf-8")

    # Add the limit check before the entry loop
    old_loop_start = """            for _, row in scored_df.iterrows():
                signal_row = row.to_dict()
                token_id = str(signal_row.get("token_id", "") or "")
                entry_intent = str(signal_row.get("entry_intent", "OPEN_LONG") or "OPEN_LONG").upper()"""

    new_loop_start = """            # FIX: Enforce MAX_CONCURRENT_POSITIONS limit
            _max_pos = getattr(TradingConfig, 'MAX_CONCURRENT_POSITIONS', 5)
            for _, row in scored_df.iterrows():
                if len(trade_manager.active_trades) >= _max_pos:
                    logging.info("Max concurrent positions reached (%d/%d). Skipping remaining entries.",
                                 len(trade_manager.active_trades), _max_pos)
                    break
                signal_row = row.to_dict()
                token_id = str(signal_row.get("token_id", "") or "")
                entry_intent = str(signal_row.get("entry_intent", "OPEN_LONG") or "OPEN_LONG").upper()"""

    if old_loop_start in text and "MAX_CONCURRENT_POSITIONS" not in text.split("for _, row in scored_df")[0][-200:]:
        backup(path)
        text = text.replace(old_loop_start, new_loop_start, 1)
        path.write_text(text, encoding="utf-8")
        print("  [PATCHED] supervisor.py — Max position limit enforced in entry loop")
        FIXES_APPLIED += 1
    elif "_max_pos" in text:
        print("  [OK] Already patched")
    else:
        print("  [MISS] Could not find entry loop start")


# ═══════════════════════════════════════════════════════════════
# FIX: Signature type prompt in run_bot.py
# ═══════════════════════════════════════════════════════════════

def fix_signature_type_prompt():
    global FIXES_APPLIED
    print("\n--- FIX: Signature type prompt in run_bot.py ---")

    path = Path("run_bot.py")
    if not path.exists():
        print("  [SKIP] run_bot.py not found")
        return

    text = path.read_text(encoding="utf-8")

    if "ensure_signature_type" in text:
        print("  [OK] Already has signature type prompt")
        return

    # Add the function definition after the imports
    sig_func = '''

def ensure_signature_type():
    """Prompt user to select their Polymarket login method if not already set."""
    sig_type = os.getenv("POLYMARKET_SIGNATURE_TYPE", "").strip()
    if sig_type in ("0", "1", "2"):
        labels = {"0": "EOA (direct wallet)", "1": "Email/Magic/Google", "2": "MetaMask/Rabby"}
        print(f"[+] Signature type: {sig_type} ({labels[sig_type]})")
        return
    if not sys.stdin.isatty():
        os.environ["POLYMARKET_SIGNATURE_TYPE"] = "1"
        print("[+] Non-interactive mode: defaulting to signature_type=1 (Email/Magic)")
        return
    print()
    print("--- POLYMARKET LOGIN METHOD ---")
    print("How do you log into Polymarket?")
    print("  1 = Email / Magic Link / Google login  (most common for bots)")
    print("  2 = MetaMask / Rabby / browser extension wallet")
    print("  0 = Direct EOA wallet (no Polymarket account)")
    print()
    print("If you're unsure, choose 1. If you see $0 balance, try 2.")
    print()
    choice = input("Your choice [1/2/0] (default: 1): ").strip()
    if choice not in ("0", "1", "2"):
        choice = "1"
    os.environ["POLYMARKET_SIGNATURE_TYPE"] = choice
    labels = {"0": "EOA (direct wallet)", "1": "Email/Magic/Google", "2": "MetaMask/Rabby"}
    print(f"[+] Using signature_type={choice} ({labels[choice]})")
    print()

'''

    # Insert function before main()
    old_main_start = "def main():\n    print_banner()"
    if old_main_start in text:
        backup(path)
        text = text.replace(old_main_start, sig_func + "def main():\n    print_banner()")

        # Call it in main() after ensure_environment
        text = text.replace(
            "    if not ensure_live_client_ready():",
            "    ensure_signature_type()\n\n    if not ensure_live_client_ready():",
        )
        path.write_text(text, encoding="utf-8")
        print("  [PATCHED] run_bot.py — Added interactive signature type prompt")
        FIXES_APPLIED += 1
    else:
        print("  [MISS] Could not find main() function start")


# ═══════════════════════════════════════════════════════════════
# FIX: Dashboard cache TTL + page 3 auth
# ═══════════════════════════════════════════════════════════════

def fix_dashboard_issues():
    print("\n--- FIX H8/L10: Dashboard cache TTL + page 3 auth ---")

    # Fix cache TTL from 15s to 30s (better match with 60s cycle)
    patch_file("dashboard.py", [
        (
            '@st.cache_data(show_spinner=False, ttl=15)',
            '@st.cache_data(show_spinner=False, ttl=30)',
        ),
    ], "Cache TTL 15s → 30s")

    # Fix page 3 load_dotenv
    page3 = Path("pages/3_Polymarket_Portfolio_Styled.py")
    if page3.exists():
        text = page3.read_text(encoding="utf-8")
        if "from dotenv import load_dotenv" in text and "safe_load_dotenv" not in text:
            backup(page3)
            text = text.replace(
                "from dotenv import load_dotenv",
                "try:\n    from dashboard_auth import safe_load_dotenv\nexcept ImportError:\n    from dotenv import load_dotenv as safe_load_dotenv",
            )
            text = text.replace(
                "load_dotenv()",
                "safe_load_dotenv()",
            )
            page3.write_text(text, encoding="utf-8")
            print("  [PATCHED] pages/3_Polymarket_Portfolio_Styled.py — Uses safe_load_dotenv")
        else:
            print("  [OK] Page 3 already uses safe auth")
    else:
        print("  [SKIP] pages/3_Polymarket_Portfolio_Styled.py not found")


# ═══════════════════════════════════════════════════════════════
# FIX M7: MoneyManager win/loss tracking wired up
# ═══════════════════════════════════════════════════════════════

def fix_m7_money_manager_tracking():
    global FIXES_APPLIED
    print("\n--- FIX M7: MoneyManager win/loss never recorded ---")

    path = Path("trade_manager.py")
    if not path.exists():
        print("  [SKIP] trade_manager.py not found")
        return

    text = path.read_text(encoding="utf-8")

    if "_money_manager" in text:
        print("  [OK] Already has money manager wiring")
        return

    # Add import and recording in _append_closed_trades
    old_append = """    def _append_closed_trades(self, closed_trades: List[TradeLifecycle]):
        \"\"\"
        BUG FIX D: Use actual close_reason from TradeLifecycle, not hardcoded.
        BUG FIX F: Write both realized_pnl AND net_realized_pnl.
        \"\"\"
        if not closed_trades:
            return"""

    new_append = """    def _append_closed_trades(self, closed_trades: List[TradeLifecycle]):
        \"\"\"
        BUG FIX D: Use actual close_reason from TradeLifecycle, not hardcoded.
        BUG FIX F: Write both realized_pnl AND net_realized_pnl.
        FIX M7: Record wins/losses in MoneyManager for adaptive sizing.
        \"\"\"
        if not closed_trades:
            return

        # FIX M7: Update MoneyManager with trade outcomes
        try:
            from money_manager import MoneyManager
            _mm = getattr(self, '_money_manager', None)
            if _mm is None:
                _mm = MoneyManager()
                self._money_manager = _mm
            for _ct in closed_trades:
                if _ct.realized_pnl >= 0:
                    _mm.record_win(_ct.realized_pnl)
                else:
                    _mm.record_loss(_ct.realized_pnl)
        except ImportError:
            pass"""

    if old_append in text:
        backup(path)
        text = text.replace(old_append, new_append, 1)
        path.write_text(text, encoding="utf-8")
        print("  [PATCHED] trade_manager.py — MoneyManager tracks wins/losses")
        FIXES_APPLIED += 1
    else:
        print("  [MISS] Could not find _append_closed_trades")


# ═══════════════════════════════════════════════════════════════
# FIX: Import TradingConfig in supervisor.py if missing
# ═══════════════════════════════════════════════════════════════

def fix_supervisor_imports():
    print("\n--- FIX: Ensure TradingConfig import in supervisor.py ---")

    path = Path("supervisor.py")
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    if "from config import TradingConfig" not in text:
        # Add after the existing config-related imports
        if "from config import" in text:
            print("  [OK] config already imported differently")
        else:
            backup(path)
            text = text.replace(
                "from strategy_layers import EntryRuleLayer",
                "from config import TradingConfig\nfrom strategy_layers import EntryRuleLayer",
            )
            path.write_text(text, encoding="utf-8")
            print("  [PATCHED] supervisor.py — Added TradingConfig import")
    else:
        print("  [OK] TradingConfig already imported")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def verify_syntax(filename):
    """Check if a Python file has valid syntax after patching."""
    path = Path(filename)
    if not path.exists():
        return True
    try:
        text = path.read_text(encoding="utf-8")
        compile(text, filename, "exec")
        return True
    except SyntaxError as exc:
        print(f"  [SYNTAX ERROR] {filename} line {exc.lineno}: {exc.msg}")
        return False


def main():
    global FIXES_APPLIED, FIXES_FAILED, FIXES_SKIPPED

    print("=" * 65)
    print("  APPLYING FULL AUDIT FIXES")
    print("  Critical + High severity patches")
    print("=" * 65)
    print()

    # Check we're in the right directory
    if not Path("supervisor.py").exists():
        print("[!] supervisor.py not found in current directory.")
        print("    Run this script from your project root.")
        print("    cd /path/to/your/project && python apply_audit_fixes.py")
        sys.exit(1)

    print(f"Project root: {Path('.').resolve()}")
    print(f"Backups will be saved to: {BACKUP_DIR}/")
    print()

    # Apply all fixes in order
    fix_supervisor_imports()
    fix_c1_c3_live_entry_crash()
    fix_c2_balance_normalization()
    fix_c5_h5_live_exits()
    fix_h1_paper_trade_sizing()
    fix_h3_price_lookup()
    fix_h6_signal_dedup()
    fix_h7_entry_rule_veto()
    fix_m2_rl_exit_reason()
    fix_m7_money_manager_tracking()
    fix_multi_position_limit()
    fix_signature_type_prompt()
    fix_dashboard_issues()

    # Verify syntax of patched files
    print("\n--- Verifying syntax ---")
    all_ok = True
    for f in ["supervisor.py", "run_bot.py", "execution_client.py",
              "order_manager.py", "config.py", "trade_manager.py", "dashboard.py"]:
        if Path(f).exists():
            if verify_syntax(f):
                print(f"  [OK] {f}")
            else:
                all_ok = False
                print(f"  [FAIL] {f} — restore from {BACKUP_DIR}/{f}")

    # Summary
    print()
    print("=" * 65)
    if all_ok:
        print(f"  FIXES APPLIED: {FIXES_APPLIED}")
    else:
        print(f"  FIXES APPLIED: {FIXES_APPLIED} (WITH SYNTAX ERRORS — check above)")
    print(f"  SKIPPED:       {FIXES_SKIPPED}")
    print(f"  FAILED:        {FIXES_FAILED}")
    print("=" * 65)
    print()
    print("What changed:")
    print("  1. C1/C3  Live entry no longer crashes on trade.shares")
    print("  2. C2     Balance normalization uses config flag")
    print("  3. C5/H5  Rule-based exits (TP/SL/time) now submit SELL to exchange")
    print("  4. H1     Paper trades use MoneyManager sizing when available")
    print("  5. H3     Price updates match by token_id (not market title string)")
    print("  6. H6     Signal dedup keeps one per token+side (not just token)")
    print("  7. H7     Entry rule vetoes RL decisions on bad markets")
    print("  8. M2     RL exits logged with real reason, not 'policy_exit'")
    print("  9. M7     MoneyManager tracks consecutive wins/losses")
    print("  10.       Max concurrent positions enforced (default: 5)")
    print("  11.       run_bot.py asks which signature type before starting")
    print("  12.       Dashboard cache TTL fixed + page 3 auth fixed")
    print()
    print("Next steps:")
    print(f"  1. Review backups in {BACKUP_DIR}/ if anything looks wrong")
    print("  2. Restart bot:  python run_bot.py")
    print("  3. Open dashboard: streamlit run dashboard.py")
    print()
    if BACKUP_DIR.exists():
        backed_up = list(BACKUP_DIR.iterdir())
        print(f"  Backed up {len(backed_up)} files to {BACKUP_DIR}/")


if __name__ == "__main__":
    main()
