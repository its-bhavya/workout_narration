"""features.py -- per-window feature extraction for the segmentation classifier.

Two stages, in order:

1. Per video (BEFORE windowing): linear-interpolate NaN gaps up to 0.5 s in each
   feature channel, using ``fill_short_gaps`` from ``pose_utils``. The 0.5 s
   threshold is converted to a row count via the median ``time_s`` spacing of
   that video (extraction used ``--stride 2``, so building from ``time_s`` is
   the only correct timeline).

2. Per window: ``window_features(rows_df)`` computes 7 nan-aware summary
   statistics per channel in the FEATURE_CHANNELS allow-list:
   mean, std, min, max, range, mean-absolute-change-per-second, NaN fraction.
   Nothing else is computed here.

Only columns returned by ``config.feature_columns`` may become features; all
metadata (seg_label, seg_type, subject_jump, n_poses, pose_detected, frame,
time_s, *_vis) is excluded by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURE_CHANNELS, feature_columns

# fill_short_gaps lives in the data_pipeline package (cross-package import).
# Match the repo convention: put the source dir on sys.path, then bare import.
_DP = Path(__file__).resolve().parents[2] / "src" / "data_pipeline"
if str(_DP) not in sys.path:
    sys.path.insert(0, str(_DP))
from pose_utils import fill_short_gaps  # noqa: E402

# Fill NaN gaps up to this many seconds before windowing.
MAX_FILL_GAP_S = 0.5

# The 7 statistics, in canonical order.
STAT_NAMES = ["mean", "std", "min", "max", "range", "macps", "nanfrac"]


def feature_names(channels: list[str]) -> list[str]:
    """Canonical ordered feature-column names for a given channel set."""
    return [f"{ch}_{stat}" for ch in channels for stat in STAT_NAMES]


# ---------------------------------------------------------------------------
# Stage 1: per-video NaN gap filling
# ---------------------------------------------------------------------------

def fill_video_nans(
    df: pd.DataFrame,
    channels: list[str] | None = None,
    max_gap_s: float = MAX_FILL_GAP_S,
) -> pd.DataFrame:
    """Fill short NaN gaps per channel for one video, before windowing.

    ``max_gap_s`` is converted to a row count using the median ``time_s``
    spacing of this video (floor, so a filled gap never exceeds ``max_gap_s``).
    Gaps longer than that, plus leading/trailing NaNs, are left as NaN.
    """
    df = df.sort_values("time_s").reset_index(drop=True).copy()
    if channels is None:
        channels = feature_columns(df.columns)

    t = df["time_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    max_gap_rows = max(1, int(max_gap_s / dt)) if dt > 0 else 1

    for ch in channels:
        if ch in df.columns:
            df[ch] = fill_short_gaps(
                df[ch].to_numpy(dtype=float), max_gap_frames=max_gap_rows
            )
    return df


# ---------------------------------------------------------------------------
# Stage 2: per-window feature statistics
# ---------------------------------------------------------------------------

def _mean_abs_change_per_s(x: np.ndarray, dt: float) -> float:
    """Mean absolute change per second over consecutive valid (non-NaN) pairs.

    Only adjacent samples that are both finite contribute; a diff spanning an
    unfilled gap is NaN and is dropped. Dividing by the median spacing turns
    the per-step change into a per-second rate.
    """
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    d = np.abs(np.diff(x))
    valid_d = d[~np.isnan(d)]
    if valid_d.size == 0:
        return float("nan")
    return float(np.mean(valid_d)) / dt


def window_features(rows_df: pd.DataFrame) -> dict:
    """Compute nan-aware summary stats per feature channel for one window.

    Parameters
    ----------
    rows_df : DataFrame slice for a single window (must contain ``time_s`` and
               the angle columns; metadata columns are ignored).

    Returns
    -------
    dict mapping ``f"{channel}_{stat}"`` to a float, for stat in
    ``STAT_NAMES``. Stats are NaN when a channel has no valid values in the
    window (except ``nanfrac``, which is always finite).
    """
    channels = feature_columns(rows_df.columns)
    t = rows_df["time_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")

    feats: dict = {}
    for ch in channels:
        x = rows_df[ch].to_numpy(dtype=float)
        n = x.size
        valid = x[~np.isnan(x)]
        n_valid = valid.size
        nanfrac = 1.0 - (n_valid / n) if n > 0 else 1.0

        if n_valid == 0:
            mean = std = mn = mx = rng = float("nan")
        else:
            mean = float(np.mean(valid))
            std = float(np.std(valid))
            mn = float(np.min(valid))
            mx = float(np.max(valid))
            rng = mx - mn
        macps = _mean_abs_change_per_s(x, dt)

        feats[f"{ch}_mean"] = mean
        feats[f"{ch}_std"] = std
        feats[f"{ch}_min"] = mn
        feats[f"{ch}_max"] = mx
        feats[f"{ch}_range"] = rng
        feats[f"{ch}_macps"] = macps
        feats[f"{ch}_nanfrac"] = nanfrac
    return feats
