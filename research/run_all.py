#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
research/run_all.py

Runs all five research reports in sequence and writes output to logs/research/.
Set PYTHONUTF8=1 in your environment or run via this script which forces UTF-8.

Usage:
    python research/run_all.py
    python research/run_all.py --logs logs --out logs/research
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Make sure parent dir is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPORTS = [
    ("feature_importance_report", "generate_feature_importance_report"),
    ("ablation_study",            "generate_ablation_report"),
    ("fill_adjusted_returns",     "generate_fill_adjusted_report"),
    ("regime_dependence_eval",    "generate_regime_report"),
    ("sample_size_diagnostics",   "generate_report"),
]

OUT_FILES = {
    "feature_importance_report": "feature_importance.md",
    "ablation_study":            "ablation_study.md",
    "fill_adjusted_returns":     "fill_adjusted_returns.md",
    "regime_dependence_eval":    "regime_dependence.md",
    "sample_size_diagnostics":   "sample_size.md",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all research reports")
    ap.add_argument("--logs", default="logs", help="Path to logs directory")
    ap.add_argument("--out",  default="logs/research", help="Output directory for reports")
    args = ap.parse_args()

    logs_dir = Path(args.logs)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for module_name, fn_name in REPORTS:
        try:
            mod = importlib.import_module(f"research.{module_name}")
            fn  = getattr(mod, fn_name)
            out_md = out_dir / OUT_FILES[module_name]
            print(f"\n>>> Running {module_name} ...")
            fn(logs_dir, out_md)
        except Exception as exc:
            msg = f"FAILED {module_name}: {exc}"
            print(msg)
            errors.append(msg)

    print("\n" + "="*60)
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"All {len(REPORTS)} reports written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
