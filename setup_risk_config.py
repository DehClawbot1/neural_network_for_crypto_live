"""
setup_risk_config.py
────────────────────
Interactive setup for the risk/capital-pool configuration that used to be
set purely via environment variables.

Run once at deployment:

    python setup_risk_config.py

It prompts for each value (showing the current default), validates the
input, and writes `.env.risk` in the project root. `supervisor.py` loads
it at startup via `load_risk_env()`.

Non-interactive mode:

    python setup_risk_config.py --non-interactive   # writes defaults only
    python setup_risk_config.py --show              # prints current values

Design rules:
- NEVER touch the hard caps in `services/risk_service.py`. Those are the
  ceiling. The prompts here configure *soft* caps + per-family splits.
- Every prompt shows the hard ceiling so the user cannot silently exceed it.
- A confirmation step re-prints the full configuration before writing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from services.risk_service import (
    _HARD_MAX_BTC_NOTIONAL_USDC,
    _HARD_MAX_SINGLE_POSITION_USDC,
    _HARD_MAX_TOTAL_EXPOSURE_USDC,
    _HARD_MAX_WEATHER_NOTIONAL_USDC,
)

ENV_FILE = Path(__file__).resolve().parent / ".env.risk"

# (env_var, prompt_label, default, hard_ceiling_or_None, type_)
PROMPTS: list[tuple[str, str, float, float | None, type]] = [
    ("RISK_MAX_TOTAL_EXPOSURE_USDC",        "Soft total exposure cap (USDC)",
     500.0,  _HARD_MAX_TOTAL_EXPOSURE_USDC, float),
    ("RISK_MAX_SESSION_DRAWDOWN_USDC",      "Soft session drawdown halt (USDC)",
     150.0,  None, float),
    ("RISK_MIN_USDC_RESERVE",               "Minimum USDC reserve kept liquid",
     5.0,    None, float),

    ("RISK_MAX_BTC_POSITIONS",              "Max concurrent BTC positions",
     3,      None, int),
    ("RISK_HARD_MAX_BTC_NOTIONAL_USDC",     "Hard BTC pool notional (USDC)",
     _HARD_MAX_BTC_NOTIONAL_USDC, _HARD_MAX_TOTAL_EXPOSURE_USDC, float),
    ("RISK_SOFT_MAX_BTC_NOTIONAL_USDC",     "Soft BTC pool notional (USDC)",
     600.0,  _HARD_MAX_BTC_NOTIONAL_USDC, float),

    ("RISK_MAX_WEATHER_POSITIONS",          "Max concurrent weather positions",
     2,      None, int),
    ("RISK_HARD_MAX_WEATHER_NOTIONAL_USDC", "Hard weather pool notional (USDC)",
     _HARD_MAX_WEATHER_NOTIONAL_USDC, _HARD_MAX_TOTAL_EXPOSURE_USDC, float),
    ("RISK_SOFT_MAX_WEATHER_NOTIONAL_USDC", "Soft weather pool notional (USDC)",
     400.0,  _HARD_MAX_WEATHER_NOTIONAL_USDC, float),

    ("RISK_RECONCILIATION_PENALTY_FLOOR",   "Rec-rate where size penalty kicks in (0..1)",
     0.50,   1.0, float),
    ("RISK_RECONCILIATION_HALT_RATE",       "Rec-rate that halts trading (0..1)",
     0.80,   1.0, float),
]


def _parse_existing() -> dict[str, str]:
    values: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


def _prompt_once(label: str, default, ceiling: float | None, type_) -> str:
    ceiling_note = f" (hard ceiling: {ceiling})" if ceiling is not None else ""
    while True:
        raw = input(f"  {label}{ceiling_note} [{default}]: ").strip()
        if not raw:
            return str(default)
        try:
            value = type_(raw)
        except (TypeError, ValueError):
            print(f"    ! not a valid {type_.__name__}, try again")
            continue
        if ceiling is not None and value > ceiling:
            print(f"    ! value {value} exceeds hard ceiling {ceiling}")
            continue
        if value < 0:
            print("    ! value must be >= 0")
            continue
        return str(value)


def run_interactive() -> dict[str, str]:
    existing = _parse_existing()
    print("\nRisk configuration setup")
    print("========================")
    print("Press Enter to accept the shown default. Hard ceilings cannot be exceeded.\n")
    values: dict[str, str] = {}
    for key, label, default, ceiling, type_ in PROMPTS:
        shown_default = existing.get(key, default)
        try:
            shown_default = type_(shown_default)
        except (TypeError, ValueError):
            shown_default = default
        values[key] = _prompt_once(label, shown_default, ceiling, type_)

    print("\nReview:")
    for k, v in values.items():
        print(f"  {k}={v}")
    confirm = input("\nWrite to .env.risk? [Y/n]: ").strip().lower()
    if confirm in {"", "y", "yes"}:
        return values
    print("Aborted. Existing .env.risk (if any) unchanged.")
    sys.exit(1)


def run_non_interactive() -> dict[str, str]:
    return {key: str(default) for key, _, default, _, _ in PROMPTS}


def write_env(values: dict[str, str]) -> None:
    lines = [
        "# .env.risk — written by setup_risk_config.py",
        "# Edit via `python setup_risk_config.py` to keep hard-ceiling validation.",
        "",
    ]
    for key, _, _, _, _ in PROMPTS:
        lines.append(f"{key}={values[key]}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {ENV_FILE}")


def show() -> None:
    existing = _parse_existing()
    if not existing:
        print(f"{ENV_FILE} does not exist (defaults will be used).")
        return
    for key, _, default, _, _ in PROMPTS:
        print(f"{key}={existing.get(key, default)}")


def load_risk_env() -> None:
    """
    Idempotent loader called by supervisor.py on startup.
    Values in .env.risk are injected into os.environ only if not already set,
    so shell/CI overrides win over the file (standard .env precedence).
    """
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k and k not in os.environ:
            os.environ[k] = v.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--non-interactive", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.show:
        show()
        return
    values = run_non_interactive() if args.non_interactive else run_interactive()
    write_env(values)


if __name__ == "__main__":
    main()
