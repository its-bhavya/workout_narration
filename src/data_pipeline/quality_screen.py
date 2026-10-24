"""
Quality screening for RepCount videos using MediaPipe Pose Landmarker.

Each RepCount video is a single-exercise clip, so frames are sampled
across the entire video.

For each video:
    - Check video readability, resolution, and rotation metadata
    - Sample frames
    - Run MediaPipe Pose Landmarker
    - Measure person detection rate and multi-person rate
    - Measure visibility of exercise-relevant landmarks

This script only MEASURES. Pass/fail thresholds are applied in a separate
step, after inspecting the distributions in the output CSVs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPCOUNT_ROOT = Path(
    r"C:\Users\Bhavya\Downloads\RepCountA (1)\LLSP"
)


MANIFEST_PATH = Path(
    "data/manifests/repcount_filtered.csv"
)

OUTPUT_PATH = Path(
    "data/manifests/repcount_quality.csv"
)

MODEL_PATH = Path(
    "models/pose_landmarker_full.task"
)

# Maximum number of frames sampled from each video / segment.
MAX_SAMPLED_FRAMES = 100

# Write partial results to disk every N videos.
SAVE_EVERY = 25

# If True, rows already present in the output CSV (without an error)
# are skipped, so an interrupted run can be restarted.
RESUME = True


# ---------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------

LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# squat and push_up only need ONE clean side to compute a knee/elbow
# angle. For each joint we take the better of left/right, so side-on
# clips (where the far limbs are occluded) are not penalized.
BEST_SIDE_JOINTS = {
    "squat": [
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ],
    "push_up": [
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
    ],
}

# jump_jack needs both arms and both legs in view, so every landmark
# is averaged (no best-side shortcut).
BOTH_SIDES_LANDMARKS = {
    "jump_jack": [
        "left_shoulder",
        "right_shoulder",
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
    ],
}


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_video_path(row: pd.Series) -> Path:
    """Return the local path of a RepCount video."""

    return (
        REPCOUNT_ROOT
        / "video"
        / row["split"]
        / row["filename"]
    )



def get_sample_indices(
    start_frame: int,
    end_frame: int,
    max_samples: int,
) -> list[int]:
    """
    Return approximately evenly spaced frame indices in
    [start_frame, end_frame)  (end exclusive).
    """

    span = end_frame - start_frame

    if span <= 0:
        return []

    if span <= max_samples:
        return list(range(start_frame, end_frame))

    step = span / max_samples

    return [
        min(start_frame + int(i * step), end_frame - 1)
        for i in range(max_samples)
    ]


def _vis(landmark) -> float:
    """Visibility can be None in some MediaPipe versions."""

    return landmark.visibility or 0.0


def get_pose_visibility(landmarks, exercise: str) -> float:
    """
    Exercise-aware visibility score for one detected pose.

    squat / push_up : mean over joints of max(left, right) visibility
    jump_jack       : mean over all relevant landmarks
    """

    if exercise in BEST_SIDE_JOINTS:
        values = [
            max(
                _vis(landmarks[LANDMARKS[left]]),
                _vis(landmarks[LANDMARKS[right]]),
            )
            for left, right in BEST_SIDE_JOINTS[exercise]
        ]
    else:
        values = [
            _vis(landmarks[LANDMARKS[name]])
            for name in BOTH_SIDES_LANDMARKS[exercise]
        ]

    if not values:
        return 0.0

    return sum(values) / len(values)


def pick_primary_pose(poses):
    """
    With num_poses > 1 the detection order is not guaranteed, so pick
    the pose with the largest landmark bounding box (the main subject,
    not a bystander in the background).
    """

    def bbox_area(landmarks) -> float:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    return max(poses, key=bbox_area)


def empty_result() -> dict:
    return {
        "video_readable": False,
        "width": 0,
        "height": 0,
        "rotation_meta": 0,
        "frame_count": 0,
        "fps": 0.0,
        "duration_seconds": 0.0,
        "sampled_frames": 0,
        "detected_frames": 0,
        "person_detection_rate": 0.0,
        "multi_person_rate": 0.0,
        "mean_landmark_visibility": 0.0,
        "p10_frame_visibility": 0.0,
        "error": "",
    }


# ---------------------------------------------------------------------
# Core frame sampling + pose scoring
# ---------------------------------------------------------------------

def read_video_metadata(cap, result: dict) -> None:
    """Fill resolution, rotation, fps, frame count, duration."""

    result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Rotation metadata (phone videos). Non-zero means the file is
    # stored rotated; if detection rates are oddly low on such clips,
    # rotation is the likely cause, not clip quality.
    orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if orientation_prop is not None:
        result["rotation_meta"] = int(cap.get(orientation_prop))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    result["frame_count"] = frame_count
    result["fps"] = fps

    if fps > 0:
        result["duration_seconds"] = frame_count / fps


def score_frames(
    cap,
    frame_indices: list[int],
    exercise: str,
    landmarker,
    result: dict,
) -> None:
    """
    Run pose detection on the given frame indices and fill the
    detection / visibility fields of `result`.
    """

    frame_visibilities = []
    sampled = 0
    detected = 0
    multi_person = 0

    for frame_index in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()

        if not success:
            continue

        sampled += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        detection_result = landmarker.detect(mp_image)

        if not detection_result.pose_landmarks:
            continue

        detected += 1

        poses = detection_result.pose_landmarks

        # Flag frames with more than one person detected.
        if len(poses) >= 2:
            multi_person += 1

        primary = pick_primary_pose(poses)

        frame_visibilities.append(
            get_pose_visibility(primary, exercise)
        )

    result["sampled_frames"] = sampled
    result["detected_frames"] = detected

    if sampled > 0:
        result["person_detection_rate"] = detected / sampled

    if detected > 0:
        result["multi_person_rate"] = multi_person / detected

    if frame_visibilities:
        result["mean_landmark_visibility"] = float(
            np.mean(frame_visibilities)
        )

        # 10th percentile instead of the minimum, so one bad frame
        # does not dominate the score.
        result["p10_frame_visibility"] = float(
            np.percentile(frame_visibilities, 10)
        )


# ---------------------------------------------------------------------
# RepCount: whole-video screening
# ---------------------------------------------------------------------

def screen_video(
    video_path: Path,
    exercise: str,
    landmarker,
) -> dict:
    """Screen one whole RepCount video."""

    result = empty_result()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return result

    try:
        read_video_metadata(cap, result)

        if result["frame_count"] <= 0 or result["fps"] <= 0:
            return result

        result["video_readable"] = True

        frame_indices = get_sample_indices(
            0,
            result["frame_count"],
            MAX_SAMPLED_FRAMES,
        )

        score_frames(cap, frame_indices, exercise, landmarker, result)

    finally:
        cap.release()

    return result


# ---------------------------------------------------------------------
# Persistence + generic screening loop
# ---------------------------------------------------------------------

def save_results(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output_path, index=False)


def run_screening(
    df: pd.DataFrame,
    output_path: Path,
    key_cols: list[str],
    meta_cols: list[str],
    screen_row: Callable[[pd.Series], dict],
    empty_fn: Callable[[], dict],
    label: str,
) -> pd.DataFrame:
    """
    Screen every row of `df` with `screen_row`, with resume support,
    per-row error handling and periodic saving.

    key_cols  : columns that uniquely identify a row (for resume)
    meta_cols : manifest columns copied into the output CSV
    empty_fn  : returns a blank result dict (used if screening crashes)
    """

    results: list[dict] = []
    done: set[tuple] = set()

    expected_cols = meta_cols + list(empty_fn().keys())

    if RESUME and output_path.exists():
        previous = pd.read_csv(output_path)

        missing = [c for c in expected_cols if c not in previous.columns]

        if missing:
            # Output file from an older version of this script. Its
            # rows used different metrics (e.g. the old visibility
            # score), so they are NOT comparable and must be redone.
            backup = output_path.with_name(
                output_path.stem + "_old_schema.csv"
            )
            output_path.replace(backup)

            logger.warning(
                "[%s] Existing %s is from an older script version "
                "(missing columns: %s). Moved to %s and re-screening "
                "everything.",
                label,
                output_path.name,
                missing,
                backup.name,
            )
        else:
            # Keep completed rows; rows that errored are re-attempted.
            previous = previous[previous["error"].isna()]
            previous = previous.assign(error="")

            results = previous.to_dict("records")
            done = set(
                previous[key_cols].itertuples(index=False, name=None)
            )

            logger.info(
                "[%s] Resuming: %d rows already screened",
                label,
                len(done),
            )

    processed_since_save = 0

    for position, (_, row) in enumerate(df.iterrows(), start=1):

        key = tuple(row[c] for c in key_cols)

        if key in done:
            continue

        logger.info("[%s %d/%d] %s", label, position, len(df), key[-1])

        try:
            quality = screen_row(row)
        except Exception as exc:
            logger.exception("Failed on %s", key)
            quality = empty_fn()
            quality["error"] = f"{type(exc).__name__}: {exc}"

        results.append({
            **{c: row[c] for c in meta_cols},
            **quality,
        })

        processed_since_save += 1

        if processed_since_save >= SAVE_EVERY:
            save_results(results, output_path)
            processed_since_save = 0

    save_results(results, output_path)

    # reindex guarantees every expected column exists, even when there
    # were no rows at all or every row came from a resumed file.
    return pd.DataFrame(results).reindex(columns=expected_cols)


def log_summary(
    quality_df: pd.DataFrame,
    label: str,
    output_path: Path,
    extra_cols: list[str] | None = None,
) -> None:

    logger.info("")
    logger.info("========== %s QUALITY SUMMARY ==========", label)

    logger.info("Rows in report: %d", len(quality_df))

    logger.info(
        "Unreadable: %d",
        (~quality_df["video_readable"].astype(bool)).sum(),
    )

    errors = quality_df["error"].fillna("")

    logger.info("Rows with errors: %d", (errors != "").sum())

    if (errors != "").any():
        logger.info(
            "Error breakdown:\n%s",
            errors[errors != ""].value_counts().to_string(),
        )

    logger.info(
        "Rows with rotation metadata: %d",
        (quality_df["rotation_meta"] != 0).sum(),
    )

    logger.info(
        "Rows with multi-person frames (>20%%): %d",
        (quality_df["multi_person_rate"] > 0.2).sum(),
    )

    # Per-exercise view, since thresholds should be chosen per exercise.
    columns = [
        "person_detection_rate",
        "mean_landmark_visibility",
        "p10_frame_visibility",
        "height",
    ] + (extra_cols or [])

    screened = quality_df[quality_df["sampled_frames"] > 0]

    if len(screened):
        summary = (
            screened
            .groupby("exercise")[columns]
            .agg(["mean", "min"])
            .round(3)
        )

        logger.info("\nPer-exercise summary:\n%s", summary)

    logger.info("Saved quality report: %s", output_path)


def safe_summary(*args, **kwargs) -> None:
    """A reporting failure must never block the next source from running."""

    try:
        log_summary(*args, **kwargs)
    except Exception:
        logger.exception(
            "Summary failed (results CSV was still saved)"
        )


def preflight_paths(
    df: pd.DataFrame,
    path_fn: Callable[[pd.Series], Path],
    label: str,
    only_downloaded: bool = False,
) -> bool:
    """
    Check up front how many source files actually resolve on disk, so a
    wrong path is reported loudly BEFORE a long run instead of showing
    up as hundreds of per-row warnings.

    Returns False (source should be skipped) if none are found.
    """

    if only_downloaded and "download_status" in df.columns:
        df = df[df["download_status"] == "downloaded"]

    paths = {path_fn(row) for _, row in df.iterrows()}
    found = sum(p.exists() for p in paths)

    logger.info(
        "[%s] source files found on disk: %d / %d",
        label,
        found,
        len(paths),
    )

    if found == 0 and paths:
        example = sorted(paths, key=str)[0]

        logger.error(
            "[%s] NO source files found. Example path tried:\n    %s\n"
            "    (resolved: %s)\n"
            "    Fix the root/path settings at the top of the script.",
            label,
            example,
            example.resolve(),
        )
        return False

    return True


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}\n"
            "Download pose_landmarker_full.task and place it "
            "inside the models/ directory."
        )

    repcount_df = None
    countix_df = None
    skipped: list[str] = []

    
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    repcount_df = pd.read_csv(MANIFEST_PATH)

    logger.info("Loaded %d RepCount videos", len(repcount_df))

    if not preflight_paths(repcount_df, get_video_path, "RepCount"):
        skipped.append("RepCount")
        repcount_df = None


    # -------------------------------------------------------------
    # Create MediaPipe Pose Landmarker
    # -------------------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,

        # IMAGE mode is intentional: we screen independently sampled
        # frames, not temporally tracked ones.
        running_mode=vision.RunningMode.IMAGE,

        # 2 poses so multi-person clips can be flagged. The main
        # subject is chosen by bounding-box size.
        num_poses=2,

        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.PoseLandmarker.create_from_options(
        options
    ) as landmarker:

        # ---------------------------------------------------------
        # RepCount
        # ---------------------------------------------------------

        if repcount_df is not None:

            def screen_repcount_row(row: pd.Series) -> dict:
                video_path = get_video_path(row)

                if not video_path.exists():
                    logger.warning("Video not found: %s", video_path)
                    result = empty_result()
                    result["error"] = "file_not_found"
                    return result

                return screen_video(
                    video_path=video_path,
                    exercise=row["exercise"],
                    landmarker=landmarker,
                )

            repcount_quality = run_screening(
                df=repcount_df,
                output_path=OUTPUT_PATH,
                key_cols=["split", "filename"],
                meta_cols=[
                    "filename",
                    "exercise",
                    "split",
                    "rep_count",
                ],
                screen_row=screen_repcount_row,
                empty_fn=empty_result,
                label="RepCount",
            )

            safe_summary(repcount_quality, "REPCOUNT", OUTPUT_PATH)


    # Repeat at the very end so it is not lost in the scrollback.
    for name in skipped:
        logger.error(
            "%s was SKIPPED (no source files found) - see the error above.",
            name,
        )


if __name__ == "__main__":
    main()