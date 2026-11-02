"""run_all.py -- one-shot runner for the full segmentation pipeline.

Runs make_windows -> train -> evaluate and prints the final raw vs smoothed
results table.

Run from repo root::

    python src/segmentation/run_all.py
    python src/segmentation/run_all.py --tag "rf200_baseline"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # repo root


def _run(cmd: list[str], step: str) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP: {step}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        print(f"\n[run_all] ERROR: '{step}' exited with code {result.returncode}.",
              file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run make_windows -> train -> evaluate end-to-end."
    )
    parser.add_argument("--tag", type=str, default="",
                        help="Free-text label forwarded to train.py (e.g. 'rf200_baseline').")
    parser.add_argument("--skip-windows", action="store_true",
                        help="Skip make_windows.py (re-use existing windows.csv).")
    args = parser.parse_args()

    python = sys.executable

    # ---- Step 1: make_windows ---------------------------------------------
    if not args.skip_windows:
        _run([python, "src/segmentation/make_windows.py"], "make_windows")
    else:
        print("\n[run_all] Skipping make_windows (--skip-windows).")

    # ---- Step 2: train ----------------------------------------------------
    train_cmd = [python, "src/segmentation/train.py"]
    if args.tag:
        train_cmd += ["--tag", args.tag]
    _run(train_cmd, "train")

    # ---- Step 3: evaluate -------------------------------------------------
    _run([python, "src/segmentation/evaluate.py"], "evaluate")

    print(f"\n{'='*60}")
    print("  run_all complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
