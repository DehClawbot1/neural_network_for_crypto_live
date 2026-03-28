# Critical Fixes — Corrected Code Patches
# Apply these changes to fix the top 5 critical issues

# ============================================================
# FIX C1/C3: supervisor.py — Live entry path crash fix
# ============================================================
# LOCATION: supervisor.py, inside the live entry block
# FIND the block that starts with:
#   "if trading_mode == "live" and order_manager is not None:"
# REPLACE the fill processing section with:

"""
# --- CORRECTED LIVE ENTRY PATH ---
# (Replace the entire live entry block inside the scored_df loop)

if trading_mode == "live" and order_manager is not None:
    from pnl_engine import PNLEngine as _PNLEngine
    _order_shares = _PNLEngine.shares_from_capital(size_usdc, fill_price)
    
    entry_row, entry_response = order_manager.submit_entry(
        token_id=token_id,
        price=fill_price,
        size=size_usdc,
        side=signal_row.get("order_side", "BUY"),
        condition_id=signal_row.get("condition_id"),
        outcome_side=signal_row.get("outcome_side", signal_row.get("side")),
    )
    entry_order_id = (entry_row or {}).get("order_id") or \
                     (entry_response or {}).get("orderID") or \
                     (entry_response or {}).get("order_id") or \
                     (entry_response or {}).get("id")
    if not entry_order_id:
        logging.info("Live entry rejected for token_id=%s reason=%s",
                     token_id, (entry_row or {}).get("reason"))
        continue
    
    fill_result = order_manager.wait_for_fill(entry_order_id)
    if not fill_result.get("filled"):
        logging.info("Live entry not filled for token_id=%s", token_id)
        try:
            order_manager.cancel_stale_order(entry_order_id)
        except Exception:
            pass
        continue
    
    fill_payload = fill_result.get("response") or {}
    actual_fill_price = float(fill_payload.get("price", fill_price) or fill_price)
    # FIX C1/C3: Use _order_shares instead of trade.shares (trade doesn't exist yet)
    actual_fill_size = float(fill_payload.get("size", _order_shares) or _order_shares)
    
    log_live_fill_event(signal_row, actual_fill_price, size_usdc, action_type="LIVE_TRADE")
    
    # Create trade AFTER fill is confirmed (not before)
    trade = TradeLifecycle(
        market=signal_row.get("market_title", signal_row.get("market", "Unknown")),
        token_id=token_id,
        condition_id=signal_row.get("condition_id"),
        outcome_side=signal_row.get("outcome_side", signal_row.get("side", "YES")),
    )
    trade.on_signal(signal_row)
    trade.enter(size_usdc=size_usdc, entry_price=actual_fill_price)
    trade.shares = actual_fill_size
    trade_manager.active_trades[market_key] = trade
    logging.info("Live trade filled for %s at %s. Shares: %s",
                 token_id, actual_fill_price, actual_fill_size)
"""


# ============================================================
# FIX C5/H5: trade_manager.py — process_exits must submit SELL orders
# ============================================================
# LOCATION: supervisor.py, AFTER the call to trade_manager.process_exits()

"""
# --- ADD THIS BLOCK after process_exits ---
closed_trades = trade_manager.process_exits(datetime.now())
if closed_trades:
    logging.info("[%s] Processed %s closed trades.", trading_mode.upper(), len(closed_trades))
    
    # FIX C5/H5: Submit SELL orders for rule-based exits in live mode
    if trading_mode == "live" and order_manager is not None:
        for closed_trade in closed_trades:
            if closed_trade.shares <= 0:
                continue
            token_id = str(closed_trade.token_id or "")
            if not token_id:
                continue
            try:
                # Get real bid price from order book
                ob_exit = orderbook_guard.analyze_book(token_id, depth=5)
                exit_price = ob_exit.get("best_bid") or closed_trade.current_price
            except Exception:
                exit_price = closed_trade.current_price
            
            if exit_price and exit_price > 0:
                logging.info("Submitting SELL for rule-based exit: token=%s shares=%.2f price=%.4f reason=%s",
                             token_id[:16], closed_trade.shares, exit_price, closed_trade.close_reason)
                try:
                    exit_row, exit_response = order_manager.submit_entry(
                        token_id=token_id,
                        price=exit_price,
                        size=closed_trade.shares,
                        side="SELL",
                        condition_id=closed_trade.condition_id,
                        outcome_side=closed_trade.outcome_side,
                    )
                    exit_order_id = (exit_row or {}).get("order_id")
                    if exit_order_id:
                        fill_result = order_manager.wait_for_fill(exit_order_id, timeout_seconds=15)
                        if fill_result.get("filled"):
                            fill_payload = fill_result.get("response") or {}
                            actual_exit_price = float(fill_payload.get("price", exit_price) or exit_price)
                            log_live_fill_event(
                                {"token_id": token_id, "market_title": closed_trade.market,
                                 "outcome_side": closed_trade.outcome_side,
                                 "current_price": actual_exit_price},
                                actual_exit_price, closed_trade.shares,
                                action_type=f"LIVE_EXIT_{closed_trade.close_reason}",
                            )
                        else:
                            logging.warning("Rule-based SELL not filled for %s, cancelling", token_id[:16])
                            try:
                                order_manager.cancel_stale_order(exit_order_id)
                            except Exception:
                                pass
                except Exception as exc:
                    logging.error("Failed to submit rule-based SELL for %s: %s", token_id[:16], exc)
"""


# ============================================================
# FIX C2: execution_client.py — Replace balance heuristic with config flag
# ============================================================
# LOCATION: config.py — add new setting

"""
# Add to TradingConfig class in config.py:
    # Whether the CLOB API returns balance in microdollars (raw integer / 1e6 = dollars)
    # Set to False if your py-clob-client version already normalizes to dollars
    BALANCE_IS_MICRODOLLARS = True
"""

# LOCATION: execution_client.py — replace _normalize_usdc_balance

"""
def _normalize_usdc_balance(self, raw_balance):
    if raw_balance is None:
        return 0.0
    try:
        val = float(raw_balance)
    except (TypeError, ValueError):
        return 0.0
    
    from config import TradingConfig
    if TradingConfig.BALANCE_IS_MICRODOLLARS:
        # Only divide if value is clearly in microdollars range
        # A balance of $0.50 in microdollars = 500000
        # A balance of $0.50 in dollars = 0.5
        # Threshold: if > 1000 and integer-like, assume microdollars
        if val > 1000 and abs(val - round(val)) < 0.01:
            return val / 1e6
    return val
"""


# ============================================================
# FIX H3: supervisor.py — Use token_id for price lookups
# ============================================================
# LOCATION: supervisor.py, replace the market_prices building block

"""
# --- REPLACE the market_prices block ---
# OLD:
# market_price_key = next((c for c in ["market_title", ...] ...), None)
# market_prices = markets_df.set_index(market_price_key)["current_price"]...

# NEW: Build price map by token_id from scored signals
token_price_map = {}
if not scored_df.empty and "token_id" in scored_df.columns:
    for _, price_row in scored_df.iterrows():
        tid = str(price_row.get("token_id", ""))
        cp = price_row.get("current_price", price_row.get("market_last_trade_price"))
        if tid and cp is not None:
            token_price_map[tid] = float(cp)

# Also add from markets_df
if not markets_df.empty:
    for _, mkt_row in markets_df.iterrows():
        for tcol in ["yes_token_id", "no_token_id"]:
            tid = str(mkt_row.get(tcol, ""))
            ltp = mkt_row.get("last_trade_price")
            if tid and ltp is not None and tid not in token_price_map:
                token_price_map[tid] = float(ltp)

# Update trades by token_id instead of market title
for trade_key, trade in list(trade_manager.active_trades.items()):
    tid = str(trade.token_id or "")
    if tid in token_price_map:
        trade.update_market(token_price_map[tid])
"""


# ============================================================
# FIX H7: supervisor.py — Entry rule as veto after RL decision
# ============================================================
# LOCATION: supervisor.py → choose_action function

"""
def choose_action(signal_row, entry_rule: EntryRuleLayer, entry_brain=None, legacy_brain=None):
    action_val = 0
    
    if entry_brain is not None:
        try:
            action_val = entry_brain.predict(signal_row)
            if action_val is None:
                action_val = 0
            else:
                action_val = int(action_val)
        except Exception:
            action_val = 0
    
    if action_val == 0 and legacy_brain is not None:
        try:
            obs = prepare_observation(signal_row)
            action, _ = legacy_brain.predict(obs, deterministic=True)
            action_val = int(action.item() if hasattr(action, "item") else action[0])
        except Exception:
            action_val = 0
    
    if action_val == 0:
        # No RL brain available or both returned 0, use rules
        if not entry_rule.should_enter(signal_row):
            return 0
        edge_score = float(signal_row.get("edge_score", 0.0) or 0.0)
        return 2 if edge_score >= 0.04 else 1
    
    # FIX H7: Apply entry rule as VETO even when RL says enter
    if action_val in (1, 2) and not entry_rule.should_enter(signal_row):
        logging.info("Entry rule vetoed RL action=%d for %s (spread/liquidity/score filter)",
                     action_val, signal_row.get("market_title", "unknown"))
        return 0
    
    return action_val
"""


# ============================================================
# FIX: Multi-position limit enforcement
# ============================================================
# LOCATION: supervisor.py, before the entry loop over scored_df

"""
# ADD before "for _, row in scored_df.iterrows():"
max_positions = TradingConfig.MAX_CONCURRENT_POSITIONS
if len(current_active_trades) >= max_positions:
    logging.info("Max concurrent positions reached (%d/%d). Skipping new entries this cycle.",
                 len(current_active_trades), max_positions)
else:
    for _, row in scored_df.iterrows():
        # Check again inside loop since we may open multiple in one cycle
        if len(trade_manager.active_trades) >= max_positions:
            logging.info("Max positions reached during entry loop. Stopping.")
            break
        # ... rest of entry logic ...
"""


# ============================================================
# FIX: Signature type prompt in run_bot.py
# ============================================================
# LOCATION: run_bot.py, add new function and call it in main()

"""
def ensure_signature_type():
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

# Call in main() after ensure_environment() and before ensure_live_client_ready():
# def main():
#     ...
#     if not ensure_environment(): sys.exit(1)
#     ensure_signature_type()  # <-- ADD THIS
#     if not ensure_live_client_ready(): sys.exit(1)
"""


# ============================================================
# FIX M7: Wire MoneyManager win/loss recording into trade closes
# ============================================================
# LOCATION: supervisor.py, after process_exits and after RL-driven exits

"""
# Add to supervisor.py at module level:
_money_manager_instance = None
try:
    from money_manager import MoneyManager
    _money_manager_instance = MoneyManager()
except ImportError:
    pass

# Then after any trade close (both process_exits and RL exits):
if _money_manager_instance and closed_trades:
    for ct in closed_trades:
        if ct.realized_pnl >= 0:
            _money_manager_instance.record_win(ct.realized_pnl)
        else:
            _money_manager_instance.record_loss(ct.realized_pnl)
"""
