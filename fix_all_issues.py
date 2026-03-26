"""
fix_all_issues.py
=================
Run this ONCE from your project root to fix:

  1. .env file → sets TRADING_MODE=live and cleans stale API creds
  2. Corrupted/oversized log CSVs → trims or removes broken files
  3. Prints a clear diagnostic of what was wrong

Usage:
    python fix_all_issues.py
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(".")
ENV_FILE = PROJECT_ROOT / ".env"
LOGS_DIR = PROJECT_ROOT / "logs"
BACKUP_DIR = PROJECT_ROOT / "backups" / f"fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def backup_file(path: Path):
    if path.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dest = BACKUP_DIR / path.name
        shutil.copy2(path, dest)
        print(f"  [BACKUP] {path.name} → {BACKUP_DIR.name}/")


# ─── FIX 1: .env file ───────────────────────────────────────────────────────

def fix_env_file():
    print("\n" + "=" * 60)
    print("FIX 1: .env file")
    print("=" * 60)

    if not ENV_FILE.exists():
        print("  [!] No .env file found. Creating one.")
        ENV_FILE.touch()

    backup_file(ENV_FILE)
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()

    # Parse existing key=value pairs
    existing = {}
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, value = stripped.partition("=")
            key = key.strip()
            # Remove surrounding quotes if present
            value = value.strip().strip('"').strip("'")
            existing[key] = value

    # Show current state
    print(f"\n  Current TRADING_MODE = '{existing.get('TRADING_MODE', '<missing>')}'")
    print(f"  Current POLYMARKET_API_KEY = '{existing.get('POLYMARKET_API_KEY', '<missing>')[:20]}...'")

    # Force TRADING_MODE=live
    existing["TRADING_MODE"] = "live"

    # Remove stale API creds that keep causing 401s
    # The bot will re-derive them on first startup
    stale_keys = ["POLYMARKET_API_KEY", "POLYMARKET_API_SECRET", "POLYMARKET_API_PASSPHRASE"]
    removed_stale = False
    for key in stale_keys:
        if key in existing:
            del existing[key]
            removed_stale = True

    # Make sure essential keys are present
    required_keys = {
        "TRADING_MODE": "live",
        "SIMULATED_STARTING_BALANCE": existing.get("SIMULATED_STARTING_BALANCE", "1000"),
        "MAX_RISK_PER_TRADE": existing.get("MAX_RISK_PER_TRADE", "50"),
    }
    for key, default in required_keys.items():
        if key not in existing:
            existing[key] = default

    # Write new .env
    new_lines = [
        "# PolyMarket Bot Configuration",
        f"# Fixed by fix_all_issues.py on {datetime.now().isoformat()}",
        "",
    ]
    for key, value in existing.items():
        # Quote values that contain spaces or special chars
        if " " in value or "=" in value:
            new_lines.append(f'{key}="{value}"')
        else:
            new_lines.append(f"{key}={value}")

    # Add commented-out API cred placeholders so the bot knows to derive them
    new_lines.extend([
        "",
        "# L2 API credentials will be auto-derived on first run.",
        "# Do NOT paste stale credentials here — the bot will refresh them.",
        "# POLYMARKET_API_KEY=",
        "# POLYMARKET_API_SECRET=",
        "# POLYMARKET_API_PASSPHRASE=",
    ])

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print(f"\n  [FIXED] TRADING_MODE = 'live'")
    if removed_stale:
        print(f"  [FIXED] Removed stale API creds (bot will re-derive fresh ones)")
    print(f"  [SAVED] {ENV_FILE}")


# ─── FIX 2: Corrupted / oversized log files ─────────────────────────────────

def fix_log_files():
    print("\n" + "=" * 60)
    print("FIX 2: Log file health check")
    print("=" * 60)

    if not LOGS_DIR.exists():
        print("  [OK] No logs/ directory yet — nothing to fix.")
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        return

    MAX_SIZE_MB = 50  # Files bigger than this are probably corrupt/bloated
    MAX_LINES = 500_000  # Trim files longer than this

    problem_files = []
    for csv_file in sorted(LOGS_DIR.glob("*.csv")):
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        if size_mb > MAX_SIZE_MB:
            problem_files.append((csv_file, size_mb, "oversized"))
        else:
            # Quick corruption check: try to read first and last lines
            try:
                with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
                    header = f.readline()
                    if "\0" in header or len(header) > 10000:
                        problem_files.append((csv_file, size_mb, "corrupted_header"))
            except Exception:
                problem_files.append((csv_file, size_mb, "unreadable"))

    if not problem_files:
        print("  [OK] All log files look healthy.")
        # Still check raw_candidates.csv specifically since that's what crashed
        raw = LOGS_DIR / "raw_candidates.csv"
        if raw.exists():
            size_mb = raw.stat().st_size / (1024 * 1024)
            print(f"  [INFO] raw_candidates.csv = {size_mb:.1f} MB")
            if size_mb > 10:
                print(f"  [WARN] raw_candidates.csv is large ({size_mb:.1f} MB) — trimming to last 50k lines")
                _trim_file(raw, 50000)
        return

    for csv_file, size_mb, issue in problem_files:
        print(f"\n  [PROBLEM] {csv_file.name}: {size_mb:.1f} MB ({issue})")
        backup_file(csv_file)

        if issue == "corrupted_header" or issue == "unreadable":
            print(f"  [FIX] Removing corrupted file (backed up)")
            csv_file.unlink()
        elif issue == "oversized":
            print(f"  [FIX] Trimming to last {MAX_LINES:,} lines")
            _trim_file(csv_file, MAX_LINES)

    # Always check raw_candidates.csv
    raw = LOGS_DIR / "raw_candidates.csv"
    if raw.exists():
        size_mb = raw.stat().st_size / (1024 * 1024)
        if size_mb > 5:
            print(f"\n  [TRIM] raw_candidates.csv is {size_mb:.1f} MB — trimming")
            backup_file(raw)
            _trim_file(raw, 50000)


def _trim_file(path: Path, max_lines: int):
    """Keep header + last N lines of a CSV file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            lines = f.readlines()

        if len(lines) <= max_lines:
            print(f"    File has {len(lines)} lines — no trim needed.")
            return

        kept = lines[-max_lines:]
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.writelines(kept)
        print(f"    Trimmed from {len(lines):,} to {len(kept):,} lines.")
    except Exception as exc:
        print(f"    [!] Trim failed: {exc}")


# ─── FIX 3: Diagnostic summary ──────────────────────────────────────────────

def print_diagnostic():
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    # Check .env
    if ENV_FILE.exists():
        content = ENV_FILE.read_text(encoding="utf-8")
        if "TRADING_MODE=live" in content:
            print("  [OK] TRADING_MODE=live in .env")
        else:
            print("  [!!] TRADING_MODE is NOT 'live' in .env")

        if "PRIVATE_KEY=" in content and not content.split("PRIVATE_KEY=")[1].startswith("\n"):
            pk_line = [l for l in content.splitlines() if l.startswith("PRIVATE_KEY=") and not l.startswith("#")]
            if pk_line:
                print("  [OK] PRIVATE_KEY is set")
            else:
                print("  [!!] PRIVATE_KEY is missing or commented out")

        if "POLYMARKET_FUNDER=" in content:
            print("  [OK] POLYMARKET_FUNDER is set")
        else:
            print("  [!!] POLYMARKET_FUNDER is missing")

        # Check for active (uncommented) API creds
        active_api = [l for l in content.splitlines()
                      if l.startswith("POLYMARKET_API_KEY=") and not l.startswith("#")]
        if active_api:
            print("  [WARN] Stored API creds found — bot will try these first")
            print("         If they're stale, the bot will re-derive (adds ~1s)")
        else:
            print("  [OK] No stored API creds — bot will derive fresh ones on startup")
    else:
        print("  [!!] No .env file found!")

    # Check wallet balance context
    print("\n  BALANCE CONTEXT:")
    print("  Your CLOB/API balance shows $0 because Polymarket's CLOB")
    print("  tracks 'deposited-to-CLOB' collateral separately from")
    print("  on-chain USDC. Your $21.45 is in on-chain USDC.")
    print("")
    print("  The bot already handles this with on-chain fallback,")
    print("  but $21.45 only supports $10 trades (not $50).")
    print("  With entry_price ~0.50, a $10 trade = 20 shares.")
    print("")
    print("  OPTIONS:")
    print("  a) Deposit more USDC to trade larger sizes")
    print("  b) The bot will work with $10 trades on your $21.45")

    # Check log health
    raw = LOGS_DIR / "raw_candidates.csv"
    if raw.exists():
        size_mb = raw.stat().st_size / (1024 * 1024)
        line_count = sum(1 for _ in open(raw, "r", encoding="utf-8", errors="replace")) if size_mb < 100 else "too large to count"
        print(f"\n  raw_candidates.csv: {size_mb:.1f} MB, {line_count} lines")
    else:
        print(f"\n  raw_candidates.csv: does not exist yet (will be created on first run)")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
  1. SECURITY: Your private key was exposed in the chat.
     Transfer funds to a new wallet and regenerate credentials.

  2. Restart the bot:
     python run_bot.py

  3. On first run, the bot will:
     - Derive fresh L2 API creds (saved to .env automatically)
     - Run in LIVE mode
     - Use on-chain USDC fallback for $21.45 balance
     - Place $10 trades (the only size your balance supports)

  4. If the research pipeline hangs again:
     Set this env var to skip it:
     $env:FORCE_RESEARCH_REFRESH="false"
     python run_bot.py

  5. Open dashboard in another terminal:
     streamlit run dashboard.py
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FIX ALL ISSUES")
    print("  Fixes .env, cleans corrupted logs, diagnoses balance")
    print("=" * 60)

    fix_env_file()
    fix_log_files()
    print_diagnostic()
