"""
Temporary helper: re-extract the three pilot videos at stride 2
to match the rest of the batch.  Run once then delete.

    python src/data_pipeline/reextract_pilots.py
"""
from pathlib import Path
import subprocess, sys

PILOTS = [
    "data/synthetic/train/synth_train_0001.mp4",
    "data/synthetic/val/synth_val_0001.mp4",
    "data/synthetic/test/synth_test_0001.mp4",
]

for p in PILOTS:
    print(f"\n--- Re-extracting {p} ---")
    result = subprocess.run(
        [
            sys.executable,
            "src/data_pipeline/extract_pose_features.py",
            "--video", p,
            "--stride", "2",
            "--force",
        ],
        check=False,
    )
    if result.returncode != 0:
        print(f"FAILED: {p}")
