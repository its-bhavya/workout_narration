"""
Temporary helper: re-extract the three pilot videos at stride 2
to match the rest of the batch.  Run once then delete.

    python src/data_pipeline/reextract_pilots.py
"""
from pathlib import Path
import subprocess, sys

def main():
    raise RuntimeError("Pilot force re-extraction retired; use extract_pose_features.py --all --stride 2 --workers 1")

if __name__ == "__main__":
    main()
