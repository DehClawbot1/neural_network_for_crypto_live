import os
import re
import shutil
from datetime import datetime

FILES_TO_PATCH = [
    "live_position_book.py",
    "reconciliation_service.py"
]

def backup_file(filepath):
    if not os.path.exists(filepath):
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.{timestamp}.bak"
    shutil.copy2(filepath, backup_path)
    print(f"[+] Backed up {filepath} -> {backup_path}")

def patch_file(filepath, patch_func):
    if not os.path.exists(filepath):
        print(f"[-] {filepath} not found. Skipping.")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        original_content = f.read()

    patched_content = patch_func(original_content)

    if patched_content != original_content:
        backup_file(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(patched_content)
        print(f"[+] Successfully fixed Phase 7 DB/Sync bugs in {filepath}")
    else:
        print(f"[-] No changes needed for {filepath} (or patterns didn't match)")

# --- Patching Functions ---

def patch_live_position_book(content):
    # BUG 10: "nan" String Typecast Vulnerability
    content = re.sub(
        r'token_id = str\(fill\.get\("token_id"\) or ""\)',
        r'tid = fill.get("token_id"); token_id = "" if pd.isna(tid) else str(tid or "") # BUG FIX 10',
        content
    )

    # BUG 6: Microscopic Dust Re-Opening
    content = re.sub(
        r'row\["status"\] = "OPEN" if float\(row\["shares"\]\) > 0 else "CLOSED"',
        r'row["status"] = "OPEN" if float(row["shares"]) > 1e-5 else "CLOSED" # BUG FIX 6: Prevent dust re-opening',
        content
    )

    # BUG 1: Limit Order Erasure (Protect balances locked in limit orders)
    target_1 = r'(if available_shares <= 1e-9:)'
    fix_1 = r"""cursor.execute("SELECT COUNT(*) FROM orders WHERE token_id = ? AND status IN ('OPEN', 'PARTIAL_FILLED', 'PENDING')", (token_id,))
            has_open_orders = cursor.fetchone()[0] > 0
            if available_shares <= 1e-9 and not has_open_orders: # BUG FIX 1: Do not erase position if tokens are locked in a limit order"""
    content = re.sub(target_1, fix_1, content)

    # BUG 5 & 7: Stale DB Read on Partial Sells & Unhandled exceptions
    target_5 = r'(local_shares = float\(row\.get\("shares"\) or 0\.0\)\n\s*row\["shares"\] = min\(local_shares, available_shares\)\n\s*verified_rows\.append\(row\))'
    fix_5 = r"""local_shares = float(row.get("shares") or 0.0)
            if available_shares < local_shares - 1e-5: # BUG FIX 5: Permanently save partial external sells to DB
                row["shares"] = available_shares
                cursor.execute("UPDATE live_positions SET shares = ?, updated_at = ? WHERE position_key = ?", (available_shares, now, row.get("position_key")))
                mutated = True
            else:
                row["shares"] = local_shares
            verified_rows.append(row)"""
    content = re.sub(target_5, fix_5, content)

    return content

def patch_reconciliation_service(content):
    # BUG 4: Split-Brain Reconciliation
    target_4 = r'local_orders = self\._safe_read_csv\("live_orders\.csv"\)\n\s*local_fills = self\._safe_read_csv\("live_fills\.csv"\)'
    fix_4 = r"""try:
            local_orders = pd.read_sql_query("SELECT * FROM orders", self.db.conn)
            local_fills = pd.read_sql_query("SELECT * FROM fills", self.db.conn)
        except Exception:
            local_orders = pd.DataFrame()
            local_fills = pd.DataFrame() # BUG FIX 4: Use DB instead of split-brain CSVs"""
    content = re.sub(target_4, fix_4, content)

    # BUG 3: Partial Fill Complete-Overwrite (Remove it completely)
    content = re.sub(
        r'if trade\["order_id"\]:\n\s*self\.db\.execute\("UPDATE orders SET status = \? WHERE order_id = \?", \("FILLED", trade\["order_id"\]\)\)',
        r'# BUG FIX 3: Removed blind FILLED overwrite. Open order sweep will handle terminal status naturally.',
        content
    )

    # BUG 2: Canceled Order Blindness (Sweep local open orders against remote open orders)
    target_2 = r'(synced_orders \+= 1\n\s*except Exception:.*?pass)'
    fix_2 = r"""\1
        
        # BUG FIX 2: Canceled Order Sweep
        try:
            if orders_payload and isinstance(orders_payload, (dict, list)) and not ("error" in str(orders_payload).lower()):
                remote_open_ids = [str(self._normalize_order(o)["order_id"]) for o in self._extract_items(orders_payload) if self._normalize_order(o)]
                if remote_open_ids:
                    placeholders = ",".join("?" for _ in remote_open_ids)
                    self.db.execute(f"UPDATE orders SET status = 'CANCELED' WHERE status = 'OPEN' AND order_id NOT IN ({placeholders})", tuple(remote_open_ids))
                else:
                    self.db.execute("UPDATE orders SET status = 'CANCELED' WHERE status = 'OPEN'")
                if hasattr(self.db.conn, "commit"): self.db.conn.commit()
        except Exception:
            pass"""
    content = re.sub(target_2, fix_2, content, flags=re.DOTALL)

    return content

if __name__ == "__main__":
    print("=== Commencing Phase 7 DB/CSV Sync & Ghost Trade Fixes ===")
    patch_file("live_position_book.py", patch_live_position_book)
    patch_file("reconciliation_service.py", patch_reconciliation_service)
    print("=== Done! ===")