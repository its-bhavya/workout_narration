"""
Rep counting from the shared pose features (squat, push_up, jump_jack).

Pipeline
    data/features/<video_id>_angles.csv   (written by extract_pose_features.py)
      -> slice one time window (one segment)
      -> pick the signal (best side / mean of both arms)
      -> clean: mask long gaps, interpolate short ones, smooth
      -> two-threshold state machine with hysteresis -> rep completion times

This module does NOT run MediaPipe. It reads the angle CSVs that the data
pipeline already produced (pixel-scaled angles, low-visibility values already
NaN), so every teammate counts reps from identical inputs.

Everything is expressed in SECONDS and uses the real `time_s` column, so it
works at any frame rate / stride (30 Hz, 15 Hz stride-2 features, ...).

Typical use (one call per segment from the segmentation module):

    df = load_features("data/features/synth_v2_train_f1_0001_angles.csv")
    out = count_for_exercise(df, "squat", start_s=4.2, end_s=19.8)
    out["reps"], out["rep_times"]
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration  (RETUNE on the train split; see evaluate_rep_counts.py)
# ---------------------------------------------------------------------

# Gaps up to this long (seconds) are interpolated; longer gaps stay NaN.
MAX_GAP_S = 0.4

# Warn if the feature sampling rate is below this (Hz).
MIN_FS_HZ = 8.0

# Segments with less than this fraction of valid samples are not counted.
MIN_VALID_FRACTION = 0.5

EXERCISES = {
    "squat": {
        "columns": ("knee_L", "knee_R"),
        "combine": "best",          # use the better-tracked side per segment
        "pattern": "down_up",       # high -> low -> high
        # Used only if mode == "fixed". These numbers came from OLD 2D angles
        # computed on normalised coordinates, so they are NOT comparable with
        # the pixel-scaled features: retune before using "fixed".
        "thresholds": (150.0, 168.0),   # (low_thresh, high_thresh) degrees
        "mode": "adaptive",
        "fracs": (0.35, 0.65),      # adaptive: fractions of the p10..p90 range
        "smooth_s": 0.5,
        "min_gap_s": 0.5,           # ignore reps closer than this
        "min_range_deg": 25.0,      # below this p10..p90 range = no real motion
        "confirm_s": 0.15,
    },
    "push_up": {
        "columns": ("elbow_L", "elbow_R"),
        "combine": "best",
        "pattern": "down_up",
        "thresholds": (135.0, 148.0),
        "mode": "adaptive",
        "fracs": (0.45, 0.75),
        # Local normalization follows changes in camera angle / rep amplitude.
        # Selected on source-disjoint training folds; see PUSHUP_COUNTER_UPDATE.md.
        "adaptive_window_s": 6.0,
        "arm_selection": "motion",
        # Training-fold selection among visibility/continuity/periodicity policies.
        "arm_quality_weights": {"visibility": 2.0, "continuity": 1.0, "periodicity": 4.0},
        "smooth_s": 0.1,
        "min_gap_s": 0.2,
        "min_range_deg": 20.0,
        "confirm_s": 0.0,
    },
    "jump_jack": {
        "columns": ("arm_raise_L", "arm_raise_R"),
        "combine": "mean",          # front view: average both arms
        "pattern": "low_high_low",  # low -> high -> low
        "thresholds": (60.0, 120.0),
        "mode": "adaptive",
        "fracs": (0.35, 0.65),
        "smooth_s": 0.3,
        "min_gap_s": 0.5,
        "min_range_deg": 40.0,
        "confirm_s": 0.15,
    },
}
# NOTE: min_range_deg values are placeholders. evaluate_rep_counts.py prints the
# 5th percentile of the angle range over ground-truth segments; set them from that.


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_features(path: str | Path) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"data_pipeline"))
    from v2_protocol import feature_frame
    df = feature_frame(path)

    if "time_s" not in df.columns:
        raise ValueError(
            f"{path}: no 'time_s' column. Columns: {list(df.columns)}"
        )

    return df.sort_values("time_s").reset_index(drop=True)


def sampling_rate(time_s: np.ndarray) -> float:
    """Median sampling rate in Hz (robust to the odd dropped frame)."""

    dt = np.diff(np.asarray(time_s, dtype=float))
    dt = dt[dt > 0]

    if len(dt) == 0:
        return 0.0

    return float(1.0 / np.median(dt))


# ---------------------------------------------------------------------
# Signal selection and cleaning
# ---------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, name: str, use_3d: bool) -> str:
    if use_3d:
        if f"{name}_3d" in df.columns:
            return f"{name}_3d"
        logger.warning("No %s_3d column; falling back to 2D.", name)

    if name not in df.columns:
        raise KeyError(
            f"Column '{name}' not found. Available: {list(df.columns)}"
        )

    return name


def _side_score(seg: pd.DataFrame, column: str) -> float:
    """How well a side was tracked inside this segment (higher is better)."""

    base = column[:-3] if column.endswith("_3d") else column
    vis = f"{base}_vis"

    if vis in seg.columns:
        return float(seg[vis].mean())

    return float(seg[column].notna().mean())


def candidate_signals(
    seg: pd.DataFrame,
    cfg: dict,
    use_3d: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """
    Signals to try, in order of preference: [(description, raw signal), ...].

    combine == "mean": one candidate, the average of both columns.
    combine == "best": both sides, the better-tracked side first. The second
        side is only used if the first one turns out to be unusable (see
        count_for_exercise), because visibility alone can pick a side that is
        confidently tracked but barely moves.
    """

    columns = [_resolve_column(seg, c, use_3d) for c in cfg["columns"]]

    if cfg["combine"] == "mean":
        # Average of the arms that are valid at each frame; falls back to the
        # single valid arm, NaN only if both are missing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            signal = np.nanmean(seg[columns].to_numpy(dtype=float), axis=1)

        return [("+".join(columns), signal)]

    scores = [_side_score(seg, c) for c in columns]
    order = np.argsort(-np.asarray(scores), kind="stable")

    return [(columns[i], seg[columns[i]].to_numpy(dtype=float)) for i in order]


def clean_signal(
    values: np.ndarray,
    fs: float,
    max_gap_s: float = MAX_GAP_S,
    smooth_s: float = 0.4,
) -> np.ndarray:
    """
    Interpolate short gaps, smooth, and keep long gaps (lost tracking) as NaN.

    Gaps longer than max_gap_s are masked COMPLETELY. (pandas interpolate(limit=)
    would fill only the first few samples of a long gap and leave a fake ramp.)
    """

    s = pd.Series(values, dtype=float)
    isnan = s.isna().to_numpy()

    if isnan.all():
        return np.full(len(s), np.nan)

    max_gap = int(round(max_gap_s * fs))

    run_id = np.cumsum(np.r_[True, isnan[1:] != isnan[:-1]])
    run_len = pd.Series(isnan).groupby(run_id).transform("sum").to_numpy()
    long_gap = isnan & (run_len > max_gap)

    vals = s.interpolate(limit_area="inside").bfill().ffill().to_numpy()

    window = max(5, int(round(smooth_s * fs)))
    if window % 2 == 0:
        window += 1

    if len(vals) >= window:
        vals = savgol_filter(vals, window, 2)

    vals = vals.copy()
    vals[long_gap] = np.nan

    return vals


def prepare_candidates(seg, cfg, use_3d=False, arm_selection=None):
    """Clean and rank arms without using annotations or detected rep counts."""
    try:
        from .arm_quality import motion_quality, quality_score
    except ImportError:  # compatibility with the historical standalone scripts
        from arm_quality import motion_quality, quality_score
    fs = sampling_rate(seg["time_s"].to_numpy())
    candidates = []
    for name, raw in candidate_signals(seg, cfg, use_3d):
        signal = clean_signal(raw, fs, MAX_GAP_S, cfg["smooth_s"])
        valid = float(np.isfinite(signal).mean())
        span = float(np.nanpercentile(signal, 90) - np.nanpercentile(signal, 10)) if valid else 0.
        visibility = _side_score(seg, name) if cfg["combine"] == "best" else float(np.isfinite(raw).mean())
        quality = motion_quality(signal, raw, visibility, fs, cfg["min_range_deg"])
        candidates.append(dict(signal=name, values=signal, valid_fraction=valid, range_deg=span,
                               usable=valid >= MIN_VALID_FRACTION and span >= cfg["min_range_deg"],
                               quality=quality, quality_score=quality_score(quality, cfg.get("arm_quality_weights", {}))))
    if (arm_selection or cfg.get("arm_selection")) == "motion" and cfg["combine"] == "best":
        # Stable ties preserve the established visibility preference. Unusable
        # arms can never displace one passing the existing motion/data guards.
        candidates.sort(key=lambda c: (c["usable"], c["quality_score"]), reverse=True)
    return candidates


# ---------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------

def adaptive_thresholds(
    sig: np.ndarray,
    fracs: tuple[float, float] = (0.35, 0.65),
) -> tuple[float, float]:
    """Thresholds at fixed fractions of this segment's own p10..p90 range."""

    lo = np.nanpercentile(sig, 10)
    hi = np.nanpercentile(sig, 90)

    return lo + fracs[0] * (hi - lo), lo + fracs[1] * (hi - lo)


def local_motion_signal(sig: np.ndarray, fs: float, window_s: float,
                        min_range_deg: float) -> np.ndarray:
    """Normalize against local p10/p90 without bridging lost-tracking gaps.

    Centered windows use offline lookahead, like the existing smoothing.
    Low-motion windows become missing so state cannot persist through a pause.
    """
    result = np.full(len(sig), np.nan)
    edges = np.flatnonzero(np.diff(np.r_[False, np.isfinite(sig), False]))
    for start, end in zip(edges[::2], edges[1::2]):
        values = pd.Series(sig[start:end])
        window = values.rolling(max(3, round(window_s * fs)), center=True, min_periods=1)
        low = window.quantile(.1).to_numpy()
        high = window.quantile(.9).to_numpy()
        span = high - low
        result[start:end] = np.where(span >= min_range_deg,
                                    (sig[start:end] - low) / np.maximum(span, 1e-9), np.nan)
    return result


# def run_state_machine(
#     sig: np.ndarray,
#     low_thresh: float,
#     high_thresh: float,
#     pattern: str,
# ) -> list[int]:
#     """
#     Indices at which a full cycle COMPLETES (hysteresis: two thresholds).

#     down_up       (squat, push_up):  high -> below low -> back above high
#     low_high_low  (jump_jack):       low -> above high -> back below low

#     NaN samples are skipped, so state is held across a lost-tracking gap.
#     The machine starts in the resting state; a segment that begins at the
#     bottom of a rep therefore counts that first partial rep when it returns.
#     """

#     completed: list[int] = []
#     active = False

#     for i, a in enumerate(sig):
#         if np.isnan(a):
#             continue

#         if pattern == "down_up":
#             if not active and a < low_thresh:
#                 active = True
#             elif active and a > high_thresh:
#                 active = False
#                 completed.append(i)
#         else:
#             if not active and a > high_thresh:
#                 active = True
#             elif active and a < low_thresh:
#                 active = False
#                 completed.append(i)

#     return completed

def run_state_machine(
    sig: np.ndarray,
    low_thresh: float,
    high_thresh: float,
    pattern: str,
    fs: float = 30.0,
    confirm_s: float = 0.15,
) -> list[int]:
    """
    Detect completed cycles using hysteresis and temporal confirmation.

    A completion threshold must remain crossed for confirm_s seconds.
    Missing pose data cancels a pending completion.
    """

    completed: list[int] = []
    active = False
    pending_idx = None

    confirm_frames = max(1, int(np.ceil(confirm_s * fs-1e-9)))

    for i, a in enumerate(sig):

        if np.isnan(a):

            active = False

            pending_idx = None

            continue

        if pattern == "down_up":

            if not active and a < low_thresh:
                active = True

            elif active and a > high_thresh:

                if pending_idx is None:
                    pending_idx = i

                if i - pending_idx + 1 >= confirm_frames:
                    completed.append(pending_idx)
                    active = False
                    pending_idx = None

            elif active:
                pending_idx = None

        else:
            if not active and a > high_thresh:
                active = True

            elif active and a < low_thresh:

                if pending_idx is None:
                    pending_idx = i

                if i - pending_idx + 1 >= confirm_frames:
                    completed.append(pending_idx)
                    active = False
                    pending_idx = None

            elif active:
                pending_idx = None

    return completed
def drop_close(times: list[float], min_gap_s: float) -> list[float]:
    kept: list[float] = []

    for t in times:
        if not kept or t - kept[-1] >= min_gap_s:
            kept.append(t)

    return kept


def count_for_exercise(
    df: pd.DataFrame,
    exercise: str,
    start_s: float,
    end_s: float,
    *,
    use_3d: bool = False,
    mode: str | None = None,
    fracs: tuple[float, float] | None = None,
    thresholds: tuple[float, float] | None = None,
    end_pad_s: float = 0.0,
    arm_selection: str | None = None,
) -> dict:
    """
    Count reps of `exercise` in [start_s, end_s + end_pad_s) of a feature table.

    mode: None -> exercise config; "adaptive" or "fixed" overrides it.
    thresholds: explicit (low, high) in degrees, overrides everything.
    end_pad_s: extend the window so a rep cut off at the end of a predicted
        segment can still complete (use ~0.3 for predicted segments, 0 for
        ground-truth segments).
    arm_selection: "visibility" reproduces the previous arm preference;
        "motion" uses the configured quality score. None uses the exercise default.

    Returns
        {exercise, reps, rep_times (s), thresholds, signal, range_deg, fs,
         valid_fraction, flag}
    where flag is "" or one of: too_short, mostly_missing, low_motion.
    """

    if arm_selection not in (None, "visibility", "motion"):
        raise ValueError("arm_selection must be 'visibility' or 'motion'")
    cfg = EXERCISES[exercise]

    result = {
        "exercise": exercise,
        "reps": 0,
        "rep_times": [],
        "thresholds": None,
        "signal": "",
        "range_deg": float("nan"),
        "fs": float("nan"),
        "valid_fraction": float("nan"),
        "flag": "",
    }

    mask = (df["time_s"] >= start_s) & (df["time_s"] < end_s + end_pad_s)
    seg = df.loc[mask]

    if len(seg) < 8:
        result["flag"] = "too_short"
        return result

    t = seg["time_s"].to_numpy(dtype=float)
    fs = sampling_rate(t)
    result["fs"] = round(fs, 2)

    if fs < MIN_FS_HZ:
        logger.warning("Sampling rate %.1f Hz is low for rep counting.", fs)

    # Push-ups rank usable arms by motion quality; other exercises retain
    # their existing visibility preference or mean signal.
    chosen = None
    first_problem = None

    candidates = prepare_candidates(seg, cfg, use_3d, arm_selection)
    result["arm_candidates"] = [{k: v for k, v in c.items() if k != "values"} for c in candidates]
    result["arm_selection"] = arm_selection or cfg.get("arm_selection", "visibility")
    for candidate in candidates:
        used, sig, valid = candidate["signal"], candidate["values"], candidate["valid_fraction"]

        if valid < MIN_VALID_FRACTION:
            problem = ("mostly_missing", used, valid, float("nan"))
        else:
            lo = np.nanpercentile(sig, 10)
            hi = np.nanpercentile(sig, 90)

            if hi - lo < cfg["min_range_deg"]:
                # Amplitude guard: a motionless or wrong segment must not
                # produce reps.
                problem = ("low_motion", used, valid, float(hi - lo))
            else:
                chosen = (used, sig, valid, float(hi - lo))
                break

        if first_problem is None:
            first_problem = problem

    if chosen is None:
        flag, used, valid, rng = first_problem
        result.update(
            flag=flag,
            signal=used,
            valid_fraction=round(valid, 3),
            range_deg=round(rng, 1) if not np.isnan(rng) else rng,
        )
        return result

    used, sig, valid, rng = chosen
    result.update(
        signal=used,
        valid_fraction=round(valid, 3),
        range_deg=round(rng, 1),
    )

    use_mode = mode or cfg["mode"]

    local_window = cfg.get("adaptive_window_s")
    counting_signal = sig
    if thresholds is None and use_mode == "adaptive" and local_window:
        counting_signal = local_motion_signal(sig, fs, local_window, cfg["min_range_deg"])
        t_low, t_high = fracs or cfg["fracs"]
        # No single pair of angle thresholds describes a locally normalized signal.
        result["thresholds"] = None
        result["threshold_mode"] = "local_adaptive"
        result["adaptive_window_s"] = local_window
        result["threshold_fractions"] = (t_low, t_high)
    elif thresholds is not None:
        t_low, t_high = thresholds
    elif use_mode == "adaptive":
        t_low, t_high = adaptive_thresholds(sig, fracs or cfg["fracs"])
    else:
        t_low, t_high = cfg["thresholds"]

    if result.get("threshold_mode") != "local_adaptive":
        result["thresholds"] = (round(float(t_low), 1), round(float(t_high), 1))

    #idx = run_state_machine(sig, t_low, t_high, cfg["pattern"])
    idx = run_state_machine(
    counting_signal, t_low, t_high, cfg["pattern"], fs=fs, confirm_s=cfg["confirm_s"],
    )
    times = drop_close([float(t[i]) for i in idx], cfg["min_gap_s"])

    result["reps"] = len(times)
    result["rep_times"] = [round(x, 3) for x in times]

    return result
