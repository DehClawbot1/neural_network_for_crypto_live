"""
apply_all_betting_fixes.py
===========================
Applies all fixes for the insufficient_funds bug + money management + market orders.

Usage:
    1. Copy all files from this directory to your project root
    2. python apply_all_betting_fixes.py
    3. python diagnose_balance_fix.py  (verify balance reads correctly)
    4. python run_bot.py               (start trading)

What this fixes:
    1. BALANCE BUG: CLOB API returns microdollars (5000000 = $5.00)
       but order_manager.py treated it as raw dollars
    2. ORDER TYPE: Added FOK market orders for fast Bitcoin markets
    3. MONEY MANAGEMENT: Bets are now sized as % of balance, not fixed $10/$50
    4. SUPERVISOR: Entry path now uses market orders + money management
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

BACKUP_DIR = Path("backups") / f"betting_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FIXES_DIR = Path(__file__).parent


def backup(path: Path):
    if path.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dest = BACKUP_DIR / path.name
        shutil.copy2(path, dest)
        print(f"  [BACKUP] {path.name}")


def copy_new_file(source_name, target_path=None):
    """Copy a new file from fixes directory to project root."""
    source = FIXES_DIR / source_name
    target = Path(target_path or source_name)
    if not source.exists():
        print(f"  [SKIP] {source_name} not found in fixes dir")
        return False
    if target.exists():
        backup(target)
    shutil.copy2(source, target)
    print(f"  [PATCH] {source_name}")
    return True


def patch_supervisor():
    """
    Patch supervisor.py to use market orders + money management
    for the live entry path.
    """
    path = Path("supervisor.py")
    if not path.exists():
        print("  [SKIP] supervisor.py not found")
        return False

    backup(path)
    text = path.read_text(encoding="utf-8")

    # 1. Add imports at top
    import_block = """from supervisor_betting_patch import apply_supervisor_betting_patch
from money_manager import MoneyManager
from config import TradingConfig"""

    # Add after existing config imports
    if "from config import TradingConfig" not in text:
        # Add before the first function definition
        text = text.replace(
            "from db import Database",
            "from db import Database\nfrom money_manager import MoneyManager",
        )

    if "from supervisor_betting_patch import" not in text:
        text = text.replace(
            "from db import Database",
            "from db import Database\nfrom supervisor_betting_patch import apply_supervisor_betting_patch",
        )

    # 2. Patch the main_loop to apply betting patch
    # Add after trade_manager initialization
    old_trade_manager_init = 'trade_manager = TradeManager(logs_dir="logs")'
    new_trade_manager_init = '''trade_manager = TradeManager(logs_dir="logs")
    _money_mgr = MoneyManager()'''

    if old_trade_manager_init in text and "_money_mgr = MoneyManager()" not in text:
        text = text.replace(old_trade_manager_init, new_trade_manager_init)

    # 3. Patch the live entry section to use market orders + money management
    # Find the section where it calculates size_usdc for live trades
    old_size_calc = "size_usdc = 10 if action_val == 1 else 50"

    new_size_calc = """# ── FIX: Use money management for bet sizing ──
                    available_bal = 0.0
                    if order_manager is not None:
                        try:
                            available_bal, _ = order_manager._get_available_balance(asset_type="COLLATERAL")
                        except Exception:
                            try:
                                available_bal = execution_client.get_available_balance(asset_type="COLLATERAL")
                            except Exception:
                                pass
                    current_exposure = sum(float(getattr(t, 'size_usdc', 0) or 0) for t in trade_manager.get_open_positions())
                    size_usdc = _money_mgr.calculate_bet_size(
                        available_balance=available_bal,
                        confidence=confidence,
                        current_exposure=current_exposure,
                    )
                    if size_usdc <= 0:
                        logging.info("Money manager says skip trade (balance=$%.2f, conf=%.2f)", available_bal, confidence)
                        continue"""

    if old_size_calc in text:
        text = text.replace(old_size_calc, new_size_calc)

    # 4. Patch the live order submission to use market orders
    old_live_submit = """entry_row, entry_response = order_manager.submit_entry(
                            token_id=token_id,
                            price=fill_price,
                            size=size_usdc,
                            side=signal_row.get("order_side", "BUY"),
                            condition_id=signal_row.get("condition_id"),
                            outcome_side=signal_row.get("outcome_side", signal_row.get("side")),
                        )"""

    new_live_submit = """# ── FIX: Use market orders for faster fills ──
                        if TradingConfig.USE_MARKET_ORDERS and hasattr(order_manager, 'submit_market_entry'):
                            logging.info("Submitting FOK market order: token=%s amount=$%.2f", token_id[:16], size_usdc)
                            entry_row, entry_response = order_manager.submit_market_entry(
                                token_id=token_id,
                                amount=size_usdc,
                                side="BUY",
                                condition_id=signal_row.get("condition_id"),
                                outcome_side=signal_row.get("outcome_side", signal_row.get("side")),
                            )
                        else:
                            entry_row, entry_response = order_manager.submit_entry(
                                token_id=token_id,
                                price=fill_price,
                                size=size_usdc,
                                side=signal_row.get("order_side", "BUY"),
                                condition_id=signal_row.get("condition_id"),
                                outcome_side=signal_row.get("outcome_side", signal_row.get("side")),
                            )"""

    if old_live_submit in text:
        text = text.replace(old_live_submit, new_live_submit)

    # 5. Also patch the paper trade path to use money management
    old_paper_size = """trade = trade_manager.handle_signal(signal_row=pd.Series(signal_row), confidence=confidence, size_usdc=size_usdc)"""
    # This should already use the patched size_usdc from step 3

    path.write_text(text, encoding="utf-8")
    print(f"  [PATCH] supervisor.py — market orders + money management")
    return True


def patch_run_bot():
    """
    Patch run_bot.py to show the balance diagnostic at startup.
    """
    path = Path("run_bot.py")
    if not path.exists():
        return False

    backup(path)
    text = path.read_text(encoding="utf-8")

    # Add import for money_manager
    if "from money_manager import" not in text:
        text = text.replace(
            "from execution_client import ExecutionClient",
            "from execution_client import ExecutionClient\nfrom money_manager import MoneyManager",
        )

    # Improve the balance display to show normalized values
    old_balance_display = """        clob_balance = 0.0
        for key in ["balance", "available", "available_balance", "amount"]:
            if collateral.get(key) is not None:
                clob_balance = float(collateral[key])
                break"""

    new_balance_display = """        # ── FIX: Normalize USDC balance (API returns microdollars) ──
        clob_balance = 0.0
        for key in ["balance", "available", "available_balance", "amount"]:
            if collateral.get(key) is not None:
                raw_val = collateral[key]
                clob_balance = client._normalize_usdc_balance(raw_val)
                logging.info("Raw balance from API: %s -> Normalized: $%.2f", raw_val, clob_balance)
                break"""

    if old_balance_display in text:
        text = text.replace(old_balance_display, new_balance_display)

    path.write_text(text, encoding="utf-8")
    print(f"  [PATCH] run_bot.py — normalized balance display")
    return True


def main():
    print("=" * 60)
    print("APPLYING ALL BETTING FIXES")
    print("=" * 60)
    print()
    print("This fixes:")
    print("  1. Balance normalization (CLOB returns microdollars)")
    print("  2. Market orders (FOK) for fast Bitcoin markets")
    print("  3. Money management (% of balance, not fixed amounts)")
    print()

    # Copy new/patched files
    copy_new_file("order_manager.py")
    copy_new_file("config.py")
    copy_new_file("money_manager.py")
    copy_new_file("supervisor_betting_patch.py")
    copy_new_file("diagnose_balance_fix.py")

    # Patch existing files
    patch_supervisor()
    patch_run_bot()

    print()
    print("=" * 60)
    print("ALL FIXES APPLIED")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run: python diagnose_balance_fix.py")
    print("     → Verify your balance reads correctly")
    print()
    print("  2. Run: python run_bot.py")
    print("     → Bot will now use market orders + smart bet sizing")
    print()
    print("What changed:")
    print("  - order_manager.py: Balance now divided by 1e6 when needed")
    print("  - order_manager.py: New submit_market_entry() for FOK orders")
    print("  - config.py: Money management settings (5% per trade max)")
    print("  - money_manager.py: Smart bet sizing based on balance + confidence")
    print("  - supervisor.py: Uses market orders + money management")
    print("  - run_bot.py: Shows normalized balance at startup")
    print()
    print(f"Backups saved to: {BACKUP_DIR}/")
    print()
    print("Money management defaults:")
    print(f"  Max per trade: {TradingConfig.MAX_RISK_PER_TRADE_PCT*100}% of balance")
    print(f"  High confidence (>70%): {TradingConfig.HIGH_CONFIDENCE_BET_PCT*100}% of balance")
    print(f"  Medium confidence (50-70%): {TradingConfig.MEDIUM_CONFIDENCE_BET_PCT*100}% of balance")
    print(f"  Low confidence (<50%): {TradingConfig.LOW_CONFIDENCE_BET_PCT*100}% of balance")
    print(f"  Min bet: ${TradingConfig.MIN_BET_USDC}")
    print(f"  Max bet: ${TradingConfig.MAX_BET_USDC}")
    print(f"  Max total exposure: {TradingConfig.MAX_TOTAL_EXPOSURE_PCT*100}% of balance")


if __name__ == "__main__":
    main()
