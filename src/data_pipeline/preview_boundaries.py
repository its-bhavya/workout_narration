"""
preview_boundaries.py

Render an annotated preview MP4 for a synthetic video, overlaying:
  - segment label (exercise or transition) in the top-left
  - absolute frame index
  - a colour-coded timeline bar at the bottom

Usage
-----
    python src/data_pipeline/preview_boundaries.py --video-id synth_train_0001
    python src/data_pipeline/preview_boundaries.py --video-id synth_val_0001 \\
        --synthetic-dir data/synthetic --out outputs/previews

Run from repo root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Colour map  (BGR)
# ---------------------------------------------------------------------------
LABEL_COLOURS = {
    "squat":      (  0, 200,   0),   # green
    "push_up":    (200,   0,   0),   # blue
    "jump_jack":  (  0, 165, 255),   # orange
    "transition": (120, 120, 120),   # grey
}
DEFAULT_COLOUR = (180, 180, 180)

TIMELINE_H = 24   # pixels
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def colour_for(label: str) -> tuple[int, int, int]:
    return LABEL_COLOURS.get(label, DEFAULT_COLOUR)


def build_frame_label_map(segments: list[dict], n_frames: int) -> list[str]:
    """Return a list of length n_frames, each entry the label at that frame."""
    labels = ["transition"] * n_frames
    for seg in segments:
        for f in range(seg["start_frame"], min(seg["end_frame"], n_frames)):
            labels[f] = seg["label"]
    return labels


def make_timeline_bar(
    frame_idx: int,
    n_frames: int,
    frame_labels: list[str],
    width: int,
) -> np.ndarray:
    bar = np.zeros((TIMELINE_H, width, 3), dtype=np.uint8)

    # Colour every pixel column
    for x in range(width):
        f = int(x / width * n_frames)
        f = min(f, n_frames - 1)
        cv2.line(bar, (x, 0), (x, TIMELINE_H - 1), colour_for(frame_labels[f]), 1)

    # Playhead
    px = int(frame_idx / max(n_frames - 1, 1) * (width - 1))
    cv2.line(bar, (px, 0), (px, TIMELINE_H - 1), (255, 255, 255), 2)
    return bar


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", required=True)
    p.add_argument("--synthetic-dir", default="data/synthetic")
    p.add_argument("--out", default="outputs/previews")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_id = args.video_id
    syn_dir  = Path(args.synthetic_dir)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate the JSON — search all split subdirs
    json_path = None
    for candidate in syn_dir.glob(f"**/{video_id}.json"):
        json_path = candidate
        break
    if json_path is None:
        raise FileNotFoundError(f"No JSON found for video_id={video_id} under {syn_dir}")

    mp4_path = json_path.with_suffix(".mp4")
    if not mp4_path.exists():
        raise FileNotFoundError(f"Video not found: {mp4_path}")

    with open(json_path) as fh:
        meta = json.load(fh)

    segments  = meta["segments"]
    n_frames  = meta["n_frames"]
    fps       = meta["fps"]
    width     = meta["width"]
    height    = meta["height"]

    frame_labels = build_frame_label_map(segments, n_frames)

    cap = cv2.VideoCapture(str(mp4_path))
    out_path = out_dir / f"{video_id}_preview.mp4"
    writer = cv2.VideoWriter(
        str(out_path), FOURCC, float(fps),
        (width, height + TIMELINE_H),
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        label  = frame_labels[min(frame_idx, n_frames - 1)]
        colour = colour_for(label)

        # Overlay label and frame number
        text = f"{label}  f{frame_idx}"
        for col, thick in [((0, 0, 0), 3), (colour, 1)]:
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, thick, cv2.LINE_AA)

        timeline = make_timeline_bar(frame_idx, n_frames, frame_labels, width)
        combined = np.vstack([frame, timeline])
        writer.write(combined)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Preview saved: {out_path}  ({frame_idx} frames)")


if __name__ == "__main__":
    main()
