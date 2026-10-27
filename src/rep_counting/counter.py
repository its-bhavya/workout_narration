"""Rep counting from pose-based joint angles (squat, push_up, jump_jack).

Pipeline:  video -> joint angle per frame -> clean signal -> state machine -> reps

Every counting function works on any slice of an angle array, so it can be
called once per segment produced by the segmentation module.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# --------------------------------------------------------------------------
# Config: filled from observed angle ranges. Re-tune on the RepCount train split.
# MediaPipe indices: shoulder 11/12, elbow 13/14, wrist 15/16,
#                    hip 23/24, knee 25/26, ankle 27/28 (left/right)
# --------------------------------------------------------------------------
EXERCISES = {
    "squat": {
        "left": (23, 25, 27),      # hip, knee, ankle
        "right": (24, 26, 28),
        "side": "best",            # use the more visible side
        "pattern": "down_up",      # starts high, dips, returns high
        "thresholds": (150, 168),  # (down_thresh, up_thresh)
        "mode": "fixed",           # fixed beat adaptive on 10 train clips (MAE 2.3 vs 3.1)
        "min_gap": 0.5,            # seconds between counted reps
    },
    "push_up": {
        "left": (11, 13, 15),      # shoulder, elbow, wrist
        "right": (12, 14, 16),
        "side": "best",
        "pattern": "down_up",
        "thresholds": (135, 148),  # only used if mode == "fixed"
        "mode": "adaptive",        # camera angle changes the elbow range a lot
        "fracs": (0.35, 0.65),
        "min_gap": 0.5,
    },
    "jump_jack": {
        "left": (23, 11, 13),      # hip, shoulder, elbow (arm raise)
        "right": (24, 12, 14),
        "side": "mean",            # front view: average both arms
        "pattern": "low_high_low",  # starts low, rises, returns low
        "thresholds": (60, 120),   # (low_thresh, high_thresh), used if mode == "fixed"
        "mode": "adaptive",
        "fracs": (0.35, 0.65),
        "min_gap": 0.5,
    },
}


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def angle(a, b, c):
    """Angle in degrees at vertex b, formed by points a-b-c (2D, 0-180)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = (np.arctan2(c[1] - b[1], c[0] - b[0])
               - np.arctan2(a[1] - b[1], a[0] - b[0]))
    deg = np.abs(radians * 180.0 / np.pi)
    return 360 - deg if deg > 180.0 else deg


# --------------------------------------------------------------------------
# Pose extraction (needs mediapipe; imported lazily so the counting
# functions below work without it)
# --------------------------------------------------------------------------
def extract_angles(video_path, joints, model_path="pose_landmarker.task"):
    """Return (angles, visibility, fps) for one joint triple (a, b, c).

    angles / visibility have one value per frame; NaN / 0 if no pose found.
    A fresh landmarker is created per call because VIDEO mode requires
    increasing timestamps.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    a_i, b_i, c_i = joints
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)
    landmarker = vision.PoseLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    angles, vis, i = [], [], 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(img, int(i * 1000 / fps))
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            angles.append(angle([lm[a_i].x, lm[a_i].y],
                                [lm[b_i].x, lm[b_i].y],
                                [lm[c_i].x, lm[c_i].y]))
            vis.append(min(lm[a_i].visibility, lm[b_i].visibility,
                           lm[c_i].visibility))
        else:
            angles.append(np.nan)
            vis.append(0.0)
        i += 1
    cap.release()
    return np.array(angles), np.array(vis), fps


def extract_exercise_angles(video_path, exercise, model_path="pose_landmarker.task"):
    """Return (angle_signal, fps) for an exercise, following its config."""
    cfg = EXERCISES[exercise]
    left, v_left, fps = extract_angles(video_path, cfg["left"], model_path)
    right, v_right, _ = extract_angles(video_path, cfg["right"], model_path)
    if cfg["side"] == "mean":
        signal = np.nanmean([left, right], axis=0)
    else:  # "best": pick the side the model could see better
        signal = left if v_left.mean() >= v_right.mean() else right
    return signal, fps


# --------------------------------------------------------------------------
# Signal cleaning
# --------------------------------------------------------------------------
def clean_signal(sig, max_gap=5, window=9, poly=2):
    """Fill short gaps, smooth, and keep long gaps (lost tracking) as NaN."""
    s = pd.Series(sig, dtype=float)
    filled = s.interpolate(limit=max_gap, limit_area="inside")
    lost = filled.isna().values
    vals = filled.bfill().ffill().values
    if len(vals) >= window:
        vals = savgol_filter(vals, window, poly)
    vals = vals.copy()
    vals[lost] = np.nan
    return vals


# --------------------------------------------------------------------------
# State machines
# --------------------------------------------------------------------------
def count_reps(sig, down_thresh, up_thresh):
    """Squat / push-up: UP -> DOWN -> UP counts one rep (hysteresis)."""
    state, frames = "UP", []
    for i, a in enumerate(sig):
        if np.isnan(a):
            continue
        if state == "UP" and a < down_thresh:
            state = "DOWN"
        elif state == "DOWN" and a > up_thresh:
            state = "UP"
            frames.append(i)
    return len(frames), frames


def count_cycles(sig, low_thresh, high_thresh):
    """Jumping jack: LOW -> HIGH -> LOW counts one rep (hysteresis)."""
    state, frames = "LOW", []
    for i, a in enumerate(sig):
        if np.isnan(a):
            continue
        if state == "LOW" and a > high_thresh:
            state = "HIGH"
        elif state == "HIGH" and a < low_thresh:
            state = "LOW"
            frames.append(i)
    return len(frames), frames


def adaptive_thresholds(sig, low_frac=0.35, high_frac=0.65):
    """Thresholds placed at fixed fractions of the clip's own angle range."""
    lo, hi = np.nanpercentile(sig, 10), np.nanpercentile(sig, 90)
    return lo + low_frac * (hi - lo), lo + high_frac * (hi - lo)


def drop_close(times, min_gap=0.5):
    """Remove counted reps that follow the previous one by less than min_gap s."""
    kept = []
    for t in times:
        if not kept or t - kept[-1] >= min_gap:
            kept.append(t)
    return kept


def count_for_exercise(sig, exercise, fps, start=None, end=None, thresholds=None):
    """Count reps for `exercise` inside [start, end] seconds of a cleaned signal.

    Thresholds: pass `thresholds=(a, b)` to override; otherwise the exercise's
    mode decides ("fixed" uses the config numbers, "adaptive" derives them from
    the segment's own angle range, which is why start/end matter).

    Returns {"exercise", "reps", "rep_times", "thresholds"}; rep_times are in
    seconds from the start of the full signal.
    """
    cfg = EXERCISES[exercise]
    offset = int(start * fps) if start is not None else 0
    stop = int(end * fps) if end is not None else len(sig)
    segment = sig[offset:stop]

    if thresholds is not None:
        t1, t2 = thresholds
    elif cfg.get("mode") == "adaptive":
        t1, t2 = adaptive_thresholds(segment, *cfg.get("fracs", (0.35, 0.65)))
    else:
        t1, t2 = cfg["thresholds"]

    counter = count_reps if cfg["pattern"] == "down_up" else count_cycles
    _, frames = counter(segment, t1, t2)
    times = drop_close([(offset + f) / fps for f in frames], cfg.get("min_gap", 0.5))
    return {
        "exercise": exercise,
        "reps": len(times),
        "rep_times": [round(t, 2) for t in times],
        "thresholds": (round(float(t1), 1), round(float(t2), 1)),
    }
