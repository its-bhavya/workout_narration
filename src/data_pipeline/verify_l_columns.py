"""
verify_l_columns.py

For 6 random clips (2 per exercise) from splits.csv, save a contact-sheet
PNG of the frames at each L-value (rep boundary frame) to outputs/l_check/.

The frame number is printed on each thumbnail so the viewer can judge
whether L values mark rep starts, bottoms, or pairs.

Run from repo root:
    python src/data_pipeline/verify_l_columns.py
"""

from __future__ import annotations

import ast
import json
import logging
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPCOUNT_ROOT = Path(r"C:\Users\Bhavya\Downloads\RepCountA (1)\LLSP")
VIDEO_DIR = REPCOUNT_ROOT / "video"

SPLITS_CSV = Path("data/manifests/splits.csv")
FILTERED_CSV = Path("data/manifests/repcount_filtered.csv")
BOUNDARIES_JSON = Path("data/manifests/repcount_rep_boundaries.json")

OUTPUT_DIR = Path("outputs/l_check")

CLIPS_PER_EXERCISE = 2
SEED = 42
MAX_L_FRAMES = 12       # cap thumbnails per sheet (first N L-values shown)
THUMB_W, THUMB_H = 320, 240
COLS = 4                # thumbnails per row in the contact sheet

EXERCISES = ["jump_jack", "push_up", "squat"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_video(filename: str, original_split: str) -> Path | None:
    """Try to locate the video file. RepCount uses train/valid/test folders."""
    candidates = [original_split, "train", "valid", "test"]
    for split in candidates:
        p = VIDEO_DIR / split / filename
        if p.exists():
            return p
    return None


def read_frame(cap: cv2.VideoCapture, frame_no: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    return frame if ok else None


def annotate(frame: np.ndarray, label: str) -> np.ndarray:
    thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
    cv2.putText(
        thumb, label, (6, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        thumb, label, (6, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return thumb


def make_contact_sheet(thumbs: list[np.ndarray], title: str) -> np.ndarray:
    """Arrange thumbnails into a grid, add a title bar."""
    # Pad to full rows
    while len(thumbs) % COLS != 0:
        thumbs.append(np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8))

    rows = [
        np.hstack(thumbs[i : i + COLS]) for i in range(0, len(thumbs), COLS)
    ]
    sheet = np.vstack(rows)

    # Title bar
    bar = np.zeros((40, sheet.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar, title, (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return np.vstack([bar, sheet])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = pd.read_csv(SPLITS_CSV)
    filtered = pd.read_csv(FILTERED_CSV)
    with open(BOUNDARIES_JSON) as fh:
        boundaries: dict[str, list[int]] = json.load(fh)

    # Build lookup: filename -> original_split (for locating the video file)
    orig_split_map = dict(
        zip(filtered["filename"], filtered["split"])
    )
    # Also pull from splits.csv original_split
    for _, row in splits.iterrows():
        orig_split_map.setdefault(row["clip_id"], row["original_split"])

    rng = random.Random(SEED)
    generated: list[str] = []

    for exercise in EXERCISES:
        ex_clips = splits[splits["exercise"] == exercise]["clip_id"].tolist()
        sample = rng.sample(ex_clips, min(CLIPS_PER_EXERCISE, len(ex_clips)))

        for clip_id in sample:
            l_values = boundaries.get(clip_id)
            if not l_values:
                log.warning("No boundaries for %s — skipping", clip_id)
                continue

            original_split = orig_split_map.get(clip_id, "train")
            video_path = find_video(clip_id, original_split)

            if video_path is None:
                log.warning("Video not found for %s — skipping", clip_id)
                continue

            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            shown = l_values[:MAX_L_FRAMES]
            thumbs: list[np.ndarray] = []

            for i, fn in enumerate(shown, start=1):
                if fn >= total_frames:
                    log.warning(
                        "%s: L%d=%d >= total_frames=%d",
                        clip_id, i, fn, total_frames,
                    )
                frame = read_frame(cap, fn)
                if frame is None:
                    frame = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                label = f"L{i}={fn}"
                thumbs.append(annotate(frame, label))

            cap.release()

            if not thumbs:
                log.warning("No frames extracted for %s", clip_id)
                continue

            title = f"{exercise}  |  {clip_id}  |  {total_frames} frames @ {fps:.1f}fps"
            sheet = make_contact_sheet(thumbs, title)

            out_name = f"{exercise}__{clip_id.replace('.mp4','')}.png"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), sheet)
            log.info("Saved: %s", out_path)
            generated.append(str(out_path))

    log.info("\nDone. %d contact sheets saved to %s", len(generated), OUTPUT_DIR)
    for p in generated:
        log.info("  %s", p)


if __name__ == "__main__":
    main()
