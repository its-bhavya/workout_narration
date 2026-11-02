"""Configuration and path constants for the segmentation module.

Run segmentation scripts from the repo root, e.g.::

    python src/segmentation/make_windows.py

All parameters live here (SEED = 42). Do not hardcode them elsewhere.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to repo root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "synthetic_index.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation"
WINDOWS_CSV = OUTPUT_DIR / "windows.csv"

# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
WINDOW_S = 2.0          # window length in seconds
STRIDE_S = 1.0          # step between successive window starts in seconds
MIN_ROWS_FRAC = 0.5     # drop windows with fewer than this fraction of expected rows
MIN_PURITY_TRAIN = 0.8  # min label purity for a window to be used as a TRAIN example
                        # (applied later when building the classifier training set,
                        # NOT in make_windows.py)

# ---------------------------------------------------------------------------
# Labels / reproducibility
# ---------------------------------------------------------------------------
SEED = 42
CLASSES = ["squat", "push_up", "jump_jack", "transition"]

# ---------------------------------------------------------------------------
# Feature channels (allow-list)
# ---------------------------------------------------------------------------
# Only these columns may be used as model features. Everything else in the
# angle CSVs is metadata and MUST be excluded:
#   seg_label, seg_type, subject_jump, n_poses, pose_detected, frame, time_s,
#   and any *_vis column.
#
# FEATURE_CHANNELS lists the base channels. The ``<channel>_3d`` variant of
# each is added at runtime by feature_columns() ONLY when that column actually
# exists in the CSV header (knee_best / elbow_best have no _3d variant).
FEATURE_CHANNELS = [
    "knee_best",
    "elbow_best",
    "hip_L",
    "hip_R",
    "arm_raise_L",
    "arm_raise_R",
    "leg_spread",
    "ankle_dist_norm",
    "torso_tilt",
]


def feature_columns(header) -> list[str]:
    """Resolve the full feature allow-list against a CSV header.

    Returns each base channel (if present) immediately followed by its
    ``<channel>_3d`` variant (only when that variant exists in ``header``).
    This is the single source of truth for which columns are valid model
    features; the classifier step must build its feature matrix from this
    list and nothing else.
    """
    header_set = set(header)
    cols: list[str] = []
    for base in FEATURE_CHANNELS:
        if base in header_set:
            cols.append(base)
        variant = f"{base}_3d"
        if variant in header_set:
            cols.append(variant)
    return cols
