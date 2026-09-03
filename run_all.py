#!/usr/bin/env python3
"""Run all reproducibility scripts in the order used by the repository.

Usage:
    python run_all.py
    python run_all.py --quick

The --quick option skips the shooting/Floquet and large uncertainty scripts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ("00_memory_kernels_comparison_en.py", False),
    ("02_hyperlogistic_bifurcation_diagram_en.py", False),
    ("06_floquet_multipliers_hyperlogistic_en.py", True),
    ("08_applied_control_theta2_clear_figure_fixed.py", False),
    ("09_applied_control_theta2_floquet.py", True),
    ("10_growth_law_generality_en.py", True),
    ("11_parameter_sensitivity_en.py", False),
    ("12_exponential_vs_humped_memory_en.py", True),
    ("13_application_uncertainty_en.py", True),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the most computationally intensive scripts.",
    )
    args = parser.parse_args()

    start_all = time.perf_counter()
    for name, heavy in SCRIPTS:
        if args.quick and heavy:
            print(f"[skip] {name}")
            continue
        path = ROOT / "scripts" / name
        if not path.exists():
            print(f"[error] Missing script: {path}", file=sys.stderr)
            return 2
        print(f"\n[run] {name}", flush=True)
        start = time.perf_counter()
        completed = subprocess.run([sys.executable, str(path)], cwd=ROOT)
        elapsed = time.perf_counter() - start
        if completed.returncode != 0:
            print(f"[error] {name} failed after {elapsed:.1f} s", file=sys.stderr)
            return completed.returncode
        print(f"[ok]  {name} ({elapsed:.1f} s)")

    elapsed_all = time.perf_counter() - start_all
    print(f"\nAll selected scripts completed successfully in {elapsed_all:.1f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
