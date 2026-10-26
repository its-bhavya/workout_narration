"""
pose_utils.py – Pure, testable pose/angle helper functions.

All functions are stateless and dependency-free (numpy only where needed).
They are called by extract_pose_features.py but can be unit-tested in
isolation without touching MediaPipe or video I/O.

IMAGE vs VIDEO mode tradeoff
-----------------------------
IMAGE mode (default, used here): each frame is processed independently.
  + No tracking state leaks across hard cuts in stitched clips.
  + Consistent with the screening script (quality_screen.py).
  - Pose identity is not tracked across frames; instead, the primary subject
    is chosen STATELESSLY as the pose with the largest landmark bounding box.

VIDEO mode: MediaPipe propagates tracking state between frames.
  + More stable landmark positions within a single continuous shot.
  - Tracking state from one clip leaks into the next after a hard cut,
    potentially producing a plausible-but-wrong pose for the first few frames
    of the new clip.  In stitched synthetic videos this would introduce a
    systematic artifact at every clip boundary – exactly the kind of boundary
    cue we want to avoid.  For that reason IMAGE mode is the correct default.

MediaPipe landmark indices (33 total, 0-based)
----------------------------------------------
  11 = left shoulder  12 = right shoulder
  13 = left elbow     14 = right elbow
  15 = left wrist     16 = right wrist
  23 = left hip       24 = right hip
  25 = left knee      26 = right knee
  27 = left ankle     28 = right ankle
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Angle computation
# ---------------------------------------------------------------------------

def angle_deg(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> float:
    """Angle at vertex *b* in the triangle a-b-c, in degrees.

    Returns NaN when any coordinate is non-finite or when either arm of the
    angle is degenerate (zero length).

    Parameters
    ----------
    a, b, c : sequences of 2 or 3 floats (x, y[, z])
        Points in pixel space (2-D) or world space (3-D).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return float("nan")

    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    denom = norm_ba * norm_bc
    if denom < 1e-8:
        return float("nan")

    cosine = np.dot(ba, bc) / denom
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def angle_deg_batch(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Vectorised angle at *b* for N points; shape (N,) in degrees.

    Parameters
    ----------
    a, b, c : (N, D) arrays where D is 2 or 3.
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    denom = norm_ba * norm_bc

    n = len(a)
    out = np.full(n, np.nan, dtype=float)
    valid = (
        np.isfinite(a).all(axis=1)
        & np.isfinite(b).all(axis=1)
        & np.isfinite(c).all(axis=1)
        & (denom > 1e-8)
    )
    cosine = np.full(n, np.nan, dtype=float)
    cosine[valid] = (
        np.einsum("ij,ij->i", ba[valid], bc[valid]) / denom[valid]
    )
    out[valid] = np.degrees(np.arccos(np.clip(cosine[valid], -1.0, 1.0)))
    return out


# ---------------------------------------------------------------------------
# 2-D pixel coordinate conversion
# ---------------------------------------------------------------------------

def norm_to_pixels(
    pts_norm: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert normalised MediaPipe coords (x∈[0,1], y∈[0,1]) to pixels.

    Parameters
    ----------
    pts_norm : (N, 2+) array  – first two columns are x_norm, y_norm.
    width, height : frame dimensions.

    Returns
    -------
    (N, 2+) array  – x and y columns scaled; remaining columns unchanged.
    """
    out = pts_norm.copy().astype(float)
    out[:, 0] *= width
    out[:, 1] *= height
    return out


# ---------------------------------------------------------------------------
# Primary-subject selection (stateless, per-frame)
# ---------------------------------------------------------------------------

def largest_bbox_pose_index(pose_landmarks_list) -> int:
    """Return the index of the pose with the largest 2-D bounding box area.

    Uses normalised image coordinates (no need to scale to pixels for a
    relative comparison).  Returns 0 if the list is empty or has one entry.

    Parameters
    ----------
    pose_landmarks_list : list of MediaPipe NormalizedLandmarkList
        As returned by PoseLandmarkerResult.pose_landmarks.
    """
    if not pose_landmarks_list:
        return 0
    if len(pose_landmarks_list) == 1:
        return 0

    best_idx = 0
    best_area = -1.0
    for i, lm_list in enumerate(pose_landmarks_list):
        xs = [lm.x for lm in lm_list]
        ys = [lm.y for lm in lm_list]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best_idx = i
    return best_idx


def hip_midpoint_norm(landmarks) -> tuple[float, float] | None:
    """Return (x, y) of the hip midpoint in normalised coordinates.

    Returns None if landmarks is None or visibility is 0 for both hips.
    Indices: left hip = 23, right hip = 24.
    """
    if landmarks is None:
        return None
    lh = landmarks[23]
    rh = landmarks[24]
    return (0.5 * (lh.x + rh.x), 0.5 * (lh.y + rh.y))


# ---------------------------------------------------------------------------
# Angle-vector jump (for boundary_jump_check)
# ---------------------------------------------------------------------------

def angle_vector_jump(
    v_prev: np.ndarray,
    v_curr: np.ndarray,
    min_components: int = 3,
) -> float:
    """L2 distance between two angle vectors, skipping NaN components.

    Parameters
    ----------
    v_prev, v_curr : 1-D arrays of angle values (degrees).  NaN means
        the component was not visible at that frame.
    min_components : minimum number of valid (non-NaN) shared components
        required; returns NaN if fewer are available.

    Returns
    -------
    L2 norm over valid components, or NaN.
    """
    v_prev = np.asarray(v_prev, dtype=float)
    v_curr = np.asarray(v_curr, dtype=float)
    valid = np.isfinite(v_prev) & np.isfinite(v_curr)
    if valid.sum() < min_components:
        return float("nan")
    diff = v_curr[valid] - v_prev[valid]
    return float(np.sqrt(np.dot(diff, diff)))


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------

def fill_short_gaps(series: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Linear interpolation of NaN runs up to *max_gap_frames* long.

    NaN runs longer than *max_gap_frames* are left as NaN.  Leading and
    trailing NaNs are never filled.

    Parameters
    ----------
    series : 1-D float array (may contain NaN).
    max_gap_frames : maximum gap length to interpolate (inclusive).

    Returns
    -------
    New array of the same shape with short gaps filled.
    """
    series = np.array(series, dtype=float)
    n = len(series)
    out = series.copy()

    i = 0
    while i < n:
        if not np.isnan(series[i]):
            i += 1
            continue
        # Find the start of a NaN run.
        gap_start = i
        while i < n and np.isnan(series[i]):
            i += 1
        gap_end = i  # exclusive; series[gap_end] is the first non-NaN after gap

        gap_len = gap_end - gap_start

        # Only fill if bounded on both sides and short enough.
        left_ok = gap_start > 0 and np.isfinite(series[gap_start - 1])
        right_ok = gap_end < n and np.isfinite(series[gap_end])

        if left_ok and right_ok and gap_len <= max_gap_frames:
            v_left = series[gap_start - 1]
            v_right = series[gap_end]
            for k in range(gap_len):
                alpha = (k + 1) / (gap_len + 1)
                out[gap_start + k] = v_left + alpha * (v_right - v_left)

    return out


# ---------------------------------------------------------------------------
# Best-side selection
# ---------------------------------------------------------------------------

def best_side_angle(
    left_angle: np.ndarray,
    right_angle: np.ndarray,
    left_vis: np.ndarray,
    right_vis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame: pick the side with higher visibility; tie goes to left.

    Parameters
    ----------
    left_angle, right_angle : (N,) float arrays (may contain NaN).
    left_vis, right_vis : (N,) float visibility arrays.

    Returns
    -------
    best_angle : (N,) array
    best_vis   : (N,) array
    side_mask  : (N,) bool array  – True = right side chosen
    """
    left_angle = np.asarray(left_angle, dtype=float)
    right_angle = np.asarray(right_angle, dtype=float)
    left_vis = np.asarray(left_vis, dtype=float)
    right_vis = np.asarray(right_vis, dtype=float)

    # right wins only when strictly better
    use_right = right_vis > left_vis

    best_angle = np.where(use_right, right_angle, left_angle)
    best_vis = np.where(use_right, right_vis, left_vis)
    return best_angle, best_vis, use_right
