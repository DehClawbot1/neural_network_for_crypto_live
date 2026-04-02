"""
cleanup_dead_tokens.py
----------------------
Purges dead/resolved token IDs from live_positions, external_position_syncs,
and markets.csv.

- Fills are preserved (PnL history is kept).
- A backup of trading.db is made before any changes.
- Safe to run while the bot is STOPPED.
"""
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOGS_DIR = Path("logs")
DB_PATH = LOGS_DIR / "trading.db"
MARKETS_CSV = LOGS_DIR / "markets.csv"

# ── Tokens identified as permanently dead (404 on /book) ──────────────────────
# Collected from repeated log spam. Add more as needed.
DEAD_TOKEN_IDS = {
    "87556431382382511265245585108323286175074347634442346217616431190076106000146",
    "20706359599436614643803070497508577462769358487573203754624359978925997745561",
    "66622160277413402520111643559730925144177947180775177354315014509188264363028",
    "104199344972770136726929562072076178281270863981769021856895025003810619296553",
    "10724132431211933233784140940537743719162423825862145124655220194913875746863",
    "110784729952747017297930577889826150569763155357163792700996394808965698222664",
    "5867351574834644863575978356417582834558823231609106645676692292844163805409",
    "33045360836863835191568040812490258945065651748558143875465083806714756580191",
    "63402960850134354122683356726747074358373208604995590731256571516454802713609",
    # From earlier balance-allowance spam:
    "114917445704065834902276711368994375262756331229653132513409186237228567904845",
    "95117371579895164314568764225488603233717855478132366261173768767976280090373",
    "42714724479899515670248535843375472632237301065191747370257876322524507784598",
    "87491344912703969480751446796760259510246739277847262759930095286453313330394",
    "51406339459238728300288185649419459347574681728934597015810284705282177435315",
    "54063056684987344987848558167978422151813428394423498436290120938736565161028",
}

def main():
    if not DB_PATH.exists():
        print(f"[ERROR] DB not found at {DB_PATH}")
        sys.exit(1)

    # ── 1. Backup ──────────────────────────────────────────────────────────────
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = LOGS_DIR / f"trading_backup_{stamp}.db"
    shutil.copy2(DB_PATH, backup_path)
    print(f"[OK] DB backed up → {backup_path}")

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    cursor = conn.cursor()

    placeholders = ",".join("?" for _ in DEAD_TOKEN_IDS)
    dead_list = list(DEAD_TOKEN_IDS)

    # ── 2. live_positions: mark dead tokens CLOSED ────────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        f"""
        UPDATE live_positions
        SET status = 'CLOSED', shares = 0, updated_at = ?
        WHERE token_id IN ({placeholders})
          AND status = 'OPEN'
        """,
        [now] + dead_list,
    )
    lp_updated = cursor.rowcount
    print(f"[OK] live_positions: marked {lp_updated} dead-token rows as CLOSED")

    # ── 3. Also close any OPEN position where shares = 0 (dust) ──────────────
    cursor.execute(
        """
        UPDATE live_positions
        SET status = 'CLOSED', updated_at = ?
        WHERE status = 'OPEN' AND (shares IS NULL OR shares <= 0)
        """,
        [now],
    )
    dust_updated = cursor.rowcount
    print(f"[OK] live_positions: closed {dust_updated} zero-share dust rows")

    # ── 4. external_position_syncs: remove rows for dead tokens ───────────────
    cursor.execute(
        f"DELETE FROM external_position_syncs WHERE token_id IN ({placeholders})",
        dead_list,
    )
    syncs_deleted = cursor.rowcount
    print(f"[OK] external_position_syncs: deleted {syncs_deleted} dead-token rows")

    conn.commit()
    conn.close()
    print("[OK] DB changes committed.")

    # ── 5. markets.csv: strip dead token rows ─────────────────────────────────
    if MARKETS_CSV.exists():
        try:
            df = pd.read_csv(MARKETS_CSV, engine="python", on_bad_lines="skip")
            before = len(df)
            if "token_id" in df.columns:
                df["token_id"] = df["token_id"].astype(str).str.strip()
                df = df[~df["token_id"].isin(DEAD_TOKEN_IDS)].copy()
            after = len(df)
            df.to_csv(MARKETS_CSV, index=False)
            print(f"[OK] markets.csv: removed {before - after} dead-token rows ({after} remaining)")
        except Exception as exc:
            print(f"[WARN] markets.csv cleanup failed: {exc}")
    else:
        print("[INFO] markets.csv not found — skipping")

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  Dead tokens targeted : {len(DEAD_TOKEN_IDS)}")
    print(f"  live_positions closed: {lp_updated}")
    print(f"  Dust rows closed     : {dust_updated}")
    print(f"  Sync rows deleted    : {syncs_deleted}")
    print(f"  DB backup            : {backup_path}")
    print("\n[DONE] Restart the bot. Dead tokens will be suppressed by the in-memory")
    print("       404 cache after their first probe this session.")


if __name__ == "__main__":
    main()
