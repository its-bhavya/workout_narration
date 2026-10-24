
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/repcount_filtered.csv"
MODEL_PATH = PROJECT_ROOT / "models/pose_landmarker_full.task"
OUTPUT_DIR = PROJECT_ROOT / "data/features/repcount/pose"
SUMMARY_PATH = PROJECT_ROOT / "data/features/repcount/extraction_summary.csv"

LANDMARK_COUNT = 33
LANDMARK_TYPES = ("image", "world")
FIELDS = ("x", "y", "z", "visibility", "presence")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def find_video(video_root: Path, filename: str) -> Path:
    """Find a video by filename and reject ambiguous matches."""
    matches = list(video_root.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} under {video_root}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple files named {filename!r}: "
            + ", ".join(str(path) for path in matches)
        )

    return matches[0]


def landmark_value(landmark, field: str):
    """Return a landmark field, or None if the field is unavailable."""
    value = getattr(landmark, field, None)
    return float(value) if value is not None else None


def make_row(
    frame_index: int,
    timestamp_ms: int,
    result,
    metadata: dict,
) -> dict:
    row = {
        **metadata,
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "pose_detected": bool(result.pose_landmarks),
    }

    image_landmarks = (
        result.pose_landmarks[0] if result.pose_landmarks else None
    )
    world_landmarks = (
        result.pose_world_landmarks[0]
        if result.pose_world_landmarks
        else None
    )

    for kind, landmarks in (
        ("image", image_landmarks),
        ("world", world_landmarks),
    ):
        for i in range(LANDMARK_COUNT):
            landmark = landmarks[i] if landmarks else None

            for field in FIELDS:
                column = f"{kind}_{i:02d}_{field}"
                row[column] = (
                    landmark_value(landmark, field)
                    if landmark is not None
                    else None
                )

    return row


def output_is_valid(path: Path, filename: str) -> bool:
    """Basic resume check; reject empty, unreadable or mismatched outputs."""
    if not path.exists():
        return False

    try:
        parquet_file = pq.ParquetFile(path)
        if parquet_file.metadata.num_rows < 1:
            return False

        table = pq.read_table(
            path,
            columns=["filename", "frame_index", "pose_detected"],
        )
        frame_indices = table.column("frame_index").to_pylist()
        filenames = set(table.column("filename").to_pylist())

        return (
            filenames == {filename}
            and frame_indices == list(range(len(frame_indices)))
        )
    except Exception:
        logger.warning("Existing output failed validation: %s", path)
        return False


def extract_video(
    video_path: Path,
    output_path: Path,
    row_metadata: dict,
    model_path: Path,
) -> dict:
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS ({fps}) for {video_path}")

    metadata = {
        **row_metadata,
        "fps": fps,
        "video_width": width,
        "video_height": height,
    }

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=False,
    )

    rows = []
    frame_index = 0
    previous_timestamp_ms = -1
    detected_frames = 0

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                # Use the decoded frame index and source FPS.
                timestamp_ms = round(frame_index * 1000.0 / fps)

                # VIDEO mode requires strictly increasing timestamps.
                timestamp_ms = max(
                    timestamp_ms, previous_timestamp_ms + 1
                )
                previous_timestamp_ms = timestamp_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )
                result = landmarker.detect_for_video(
                    image, timestamp_ms
                )

                row = make_row(
                    frame_index,
                    timestamp_ms,
                    result,
                    metadata,
                )
                rows.append(row)

                if row["pose_detected"]:
                    detected_frames += 1

                frame_index += 1

    finally:
        capture.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.parquet")

    try:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, temp_path, compression="zstd")
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return {
        **row_metadata,
        "status": "success",
        "video_path": str(video_path),
        "output_path": str(output_path),
        "reported_frames": reported_frames,
        "decoded_frames": frame_index,
        "fps": fps,
        "pose_detected_frames": detected_frames,
        "detection_rate": detected_frames / frame_index,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-root",
        type=Path,
        required=True,
        help="Directory containing the RepCount videos",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Process one video from each exercise class",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract videos even when valid output files exist",
    )
    args = parser.parse_args()

    video_root = args.video_root.resolve()

    if not video_root.is_dir():
        raise SystemExit(f"Video directory does not exist: {video_root}")
    if not MANIFEST_PATH.is_file():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Pose model not found: {MODEL_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {"filename", "exercise", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest is missing columns: {sorted(missing)}")

    if args.pilot:
        # Pick one row per exercise, preferring a train example.
        manifest = (
            manifest.assign(
                _split_order=(manifest["split"] != "train").astype(int)
            )
            .sort_values("_split_order")
            .groupby("exercise", sort=True)
            .head(1)
            .drop(columns="_split_order")
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previous_summary = []

    if SUMMARY_PATH.exists():
        try:
            previous_summary = pd.read_csv(SUMMARY_PATH).to_dict("records")
        except Exception:
            logger.warning("Could not read prior summary; starting a new one.")

    results = []

    for _, record in manifest.iterrows():
        filename = str(record["filename"])
        output_path = OUTPUT_DIR / f"{Path(filename).stem}.parquet"

        row_metadata = {
            "filename": filename,
            "exercise": str(record["exercise"]),
            "split": str(record["split"]),
        }

        if (
            not args.force
            and output_is_valid(output_path, filename)
        ):
            logger.info("SKIP valid output: %s", filename)
            results.append({
                **row_metadata,
                "status": "skipped_valid_output",
                "output_path": str(output_path),
                "error": "",
            })
            continue

        try:
            video_path = find_video(video_root, filename)
            logger.info("Extracting %s", filename)

            result = extract_video(
                video_path=video_path,
                output_path=output_path,
                row_metadata=row_metadata,
                model_path=MODEL_PATH,
            )
            results.append(result)

            logger.info(
                "DONE %s | frames=%s | detection=%.3f",
                filename,
                result["decoded_frames"],
                result["detection_rate"],
            )

        except Exception as exc:
            logger.exception("FAILED: %s", filename)
            results.append({
                **row_metadata,
                "status": "failed",
                "output_path": str(output_path),
                "error": repr(exc),
            })

        # Persist progress after each video, so interruptions are recoverable.
        current = pd.DataFrame(previous_summary + results)
        current.to_csv(SUMMARY_PATH, index=False)

    summary = pd.DataFrame(previous_summary + results)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nExtraction run finished.")
    print(summary.tail(len(results)).to_string(index=False))
    print(f"\nSummary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
