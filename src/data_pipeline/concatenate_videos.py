"""
concatenate_videos.py

Concatenate whole RepCount clips into synthetic multi-exercise videos
with ground-truth segment labels.

Usage
-----
    python src/data_pipeline/concatenate_videos.py \\
        --splits-file data/manifests/splits.csv \\
        --n-train 24 --n-val 6 --n-test 15 \\
        --seed 42 --out data/synthetic \\
        --pilot

Run from repo root.  Use --margin-label exercise to suppress transition
segments (the whole clip is labelled as its exercise).

Label rules (default --margin-label transition)
------------------------------------------------
  Per source clip in stitched timeline:
    [clip_start, clip_start + first_start)          -> transition
    [clip_start + first_start, clip_start + last_end + 1) -> exercise
    [clip_start + last_end + 1, clip_end)            -> transition

  Adjacent transition spans that touch are merged into one segment.
  Zero-length segments are never emitted.
  end_frame is EXCLUSIVE; segments cover [0, n_frames) contiguously.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_FPS  = 30
TARGET_W    = 640
TARGET_H    = 360
FOURCC      = cv2.VideoWriter_fourcc(*"mp4v")

K_MIN = 3
K_MAX = 5

INCLUDE_STATUSES = {"pass", "review"}
EXCLUDE_FPS_BELOW = 20.0

REPCOUNT_ROOT = Path(r"C:\Users\Bhavya\Downloads\RepCountA (1)\LLSP")
VIDEO_DIR = REPCOUNT_ROOT / "video"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClipInfo:
    clip_id: str
    exercise: str
    split: str
    first_start: int   # source frame index (0-based)
    last_end: int      # source frame index (inclusive)
    total_frames: int
    src_fps: float
    pairs: list[list[int]]
    rep_count: int


@dataclass
class RawSegment:
    """Segment in stitched-output frame space before merging."""
    seg_type: str          # "exercise" | "transition"
    label: str
    start: int             # inclusive
    end: int               # exclusive
    clip_id: str = ""
    rep_count_annotated: int = 0
    rep_pairs_out: list = field(default_factory=list)
    src_window_frames: list = field(default_factory=list)
    source_clips: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def find_video(clip_id: str, original_split: str) -> Path | None:
    for sp in [original_split, "train", "valid", "test"]:
        p = VIDEO_DIR / sp / clip_id
        if p.exists():
            return p
    return None


def letterbox_frame(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize frame to w×h with black letterboxing (no crop)."""
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def resample_index(out_frame: int, out_frame_offset: int,
                   src_fps: float, total_src: int) -> int:
    """Map an output frame index to a source frame index.

    out_frame         : absolute output frame number
    out_frame_offset  : output frame at which this source clip starts
    src_fps           : fps of the source clip
    total_src         : total frames in source clip
    """
    out_time = (out_frame - out_frame_offset) / TARGET_FPS
    src_idx  = round(out_time * src_fps)
    return min(max(src_idx, 0), total_src - 1)


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------

def load_pool(
    splits_path: Path,
    boundaries: dict[str, Any],
    exclusions: set[str],
) -> dict[str, list[ClipInfo]]:
    """Return {split -> [ClipInfo]} after applying all filters."""

    df = pd.read_csv(splits_path)
    df = df[df["status"].isin(INCLUDE_STATUSES)].copy()

    pool: dict[str, list[ClipInfo]] = defaultdict(list)
    skipped_missing = 0
    skipped_fps     = 0
    skipped_excl    = 0

    orig_map: dict[str, str] = dict(zip(df["clip_id"], df["original_split"]))

    for _, row in df.iterrows():
        clip_id = row["clip_id"]

        if clip_id in exclusions:
            skipped_excl += 1
            continue

        binfo = boundaries.get(clip_id)
        if binfo is None:
            skipped_missing += 1
            continue

        src_fps = binfo["fps"] or 0.0
        if src_fps < EXCLUDE_FPS_BELOW:
            skipped_fps += 1
            continue

        pairs = binfo["pairs"]
        if not pairs:
            skipped_missing += 1
            continue

        first_start = pairs[0][0]
        last_end    = pairs[-1][1]

        pool[row["split"]].append(ClipInfo(
            clip_id      = clip_id,
            exercise     = row["exercise"],
            split        = row["split"],
            first_start  = first_start,
            last_end     = last_end,
            total_frames = binfo["total_frames"],
            src_fps      = src_fps,
            pairs        = pairs,
            rep_count    = binfo["rep_count"],
        ))

    log.info("Pool loaded: %s", {s: len(v) for s, v in pool.items()})
    log.info("Skipped — missing boundaries: %d, low fps: %d, excluded: %d",
             skipped_missing, skipped_fps, skipped_excl)
    return pool


def load_exclusions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df["clip_id"].tolist())


# ---------------------------------------------------------------------------
# Sequence sampling
# ---------------------------------------------------------------------------

def sample_sequence(
    k: int,
    exercise_pool: dict[str, list[ClipInfo]],
    rng: random.Random,
    used_clips: set[str] | None,   # None = train (reuse OK)
) -> list[ClipInfo] | None:
    """Sample k clips with adjacent-exercise constraint and ≥2 distinct exercises."""

    exercises = list(exercise_pool.keys())
    if len(exercises) < 2:
        return None

    for _attempt in range(200):
        seq: list[ClipInfo] = []
        prev_ex = None

        for _ in range(k):
            # Build candidate exercises (must differ from previous)
            candidates = [e for e in exercises if e != prev_ex
                          and exercise_pool[e]]
            if not candidates:
                break
            ex = rng.choice(candidates)
            clips = [c for c in exercise_pool[ex]
                     if used_clips is None or c.clip_id not in used_clips]
            if not clips:
                break
            clip = rng.choice(clips)
            seq.append(clip)
            prev_ex = ex

        if len(seq) == k and len({c.exercise for c in seq}) >= 2:
            return seq

    return None


def balance_usage(
    clips: list[ClipInfo],
    usage: dict[str, int],
) -> list[ClipInfo]:
    """Sort clips so least-used come first (for greedy balanced selection)."""
    return sorted(clips, key=lambda c: usage.get(c.clip_id, 0))


# ---------------------------------------------------------------------------
# Segment building
# ---------------------------------------------------------------------------

def build_raw_segments(
    clips: list[ClipInfo],
    margin_label: str,
) -> list[RawSegment]:
    """Build un-merged segment list in output-frame space."""

    segments: list[RawSegment] = []
    out_offset = 0  # running output frame counter

    for clip in clips:
        # How many output frames does this source clip contribute?
        # We advance one output frame at a time up to the resampled last
        # source frame.  Simpler: just use total_src_frames rescaled.
        n_out = round(clip.total_frames * TARGET_FPS / clip.src_fps)
        clip_end_out = out_offset + n_out  # exclusive

        first_start_out = out_offset + round(
            clip.first_start * TARGET_FPS / clip.src_fps
        )
        last_end_out = out_offset + round(
            (clip.last_end + 1) * TARGET_FPS / clip.src_fps
        )
        # Clamp to clip bounds
        first_start_out = max(out_offset, min(first_start_out, clip_end_out))
        last_end_out    = max(first_start_out, min(last_end_out, clip_end_out))

        # Re-offset rep pairs into output frame space
        pairs_out: list[list[int]] = []
        for s, e in clip.pairs:
            s_out = out_offset + round(s * TARGET_FPS / clip.src_fps)
            e_out = out_offset + round((e + 1) * TARGET_FPS / clip.src_fps)
            s_out = max(out_offset, min(s_out, clip_end_out))
            e_out = max(s_out, min(e_out, clip_end_out))
            pairs_out.append([s_out, e_out])

        if margin_label == "exercise":
            # Whole clip = exercise
            segments.append(RawSegment(
                seg_type="exercise",
                label=clip.exercise,
                start=out_offset,
                end=clip_end_out,
                clip_id=clip.clip_id,
                rep_count_annotated=clip.rep_count,
                rep_pairs_out=pairs_out,
                src_window_frames=[0, clip.total_frames - 1],
            ))
        else:
            # pre-transition
            if first_start_out > out_offset:
                segments.append(RawSegment(
                    seg_type="transition",
                    label="transition",
                    start=out_offset,
                    end=first_start_out,
                    source_clips=[clip.clip_id],
                ))
            # exercise core
            if last_end_out > first_start_out:
                segments.append(RawSegment(
                    seg_type="exercise",
                    label=clip.exercise,
                    start=first_start_out,
                    end=last_end_out,
                    clip_id=clip.clip_id,
                    rep_count_annotated=clip.rep_count,
                    rep_pairs_out=pairs_out,
                    src_window_frames=[clip.first_start, clip.last_end],
                ))
            # post-transition
            if clip_end_out > last_end_out:
                segments.append(RawSegment(
                    seg_type="transition",
                    label="transition",
                    start=last_end_out,
                    end=clip_end_out,
                    source_clips=[clip.clip_id],
                ))

        out_offset = clip_end_out

    return segments, out_offset  # out_offset = total n_frames


def merge_transitions(segments: list[RawSegment]) -> list[RawSegment]:
    """Merge adjacent transition segments; drop zero-length segments."""
    merged: list[RawSegment] = []
    for seg in segments:
        if seg.end <= seg.start:
            continue  # drop zero-length
        if (merged
                and merged[-1].seg_type == "transition"
                and seg.seg_type == "transition"
                and merged[-1].end == seg.start):
            # merge
            merged[-1].end = seg.end
            merged[-1].source_clips = list(dict.fromkeys(
                merged[-1].source_clips + seg.source_clips
            ))
        else:
            merged.append(seg)
    return merged


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def seg_to_dict(seg: RawSegment) -> dict:
    d: dict[str, Any] = {
        "type":        seg.seg_type,
        "label":       seg.label,
        "start_frame": seg.start,
        "end_frame":   seg.end,
    }
    if seg.seg_type == "exercise":
        d["clip_id"]              = seg.clip_id
        d["rep_count_annotated"]  = seg.rep_count_annotated
        d["rep_pairs_out"]        = seg.rep_pairs_out
        d["src_window_frames"]    = seg.src_window_frames
    else:
        d["source_clips"] = seg.source_clips
    return d


# ---------------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------------

def write_video(
    clips: list[ClipInfo],
    out_path: Path,
    orig_map: dict[str, str],
) -> int:
    """Write stitched video; return actual frame count written."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), FOURCC, float(TARGET_FPS), (TARGET_W, TARGET_H)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

    total_written = 0

    for clip in clips:
        vp = find_video(clip.clip_id, orig_map.get(clip.clip_id, "train"))
        if vp is None:
            writer.release()
            raise FileNotFoundError(f"Video not found: {clip.clip_id}")

        cap = cv2.VideoCapture(str(vp))
        n_out = round(clip.total_frames * TARGET_FPS / clip.src_fps)

        out_frame_offset = total_written

        for out_f in range(n_out):
            src_idx = resample_index(
                out_frame_offset + out_f, out_frame_offset,
                clip.src_fps, clip.total_frames
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
            frame = letterbox_frame(frame, TARGET_W, TARGET_H)
            writer.write(frame)

        cap.release()
        total_written += n_out

    writer.release()
    return total_written


# ---------------------------------------------------------------------------
# Single-video generation
# ---------------------------------------------------------------------------

def make_video(
    video_id: str,
    clips: list[ClipInfo],
    split: str,
    out_dir: Path,
    margin_label: str,
    splits_file: str,
    seed: int,
    orig_map: dict[str, str],
) -> dict:
    """Generate one video + JSON; return index row dict."""

    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path  = out_dir / f"{video_id}.mp4"
    json_path = out_dir / f"{video_id}.json"

    log.info("Writing %s (%d clips)…", video_id, len(clips))
    n_frames = write_video(clips, mp4_path, orig_map)

    raw_segs, _ = build_raw_segments(clips, margin_label)
    segments     = merge_transitions(raw_segs)

    # Clip join frames (output frame where one clip ends, next begins)
    clip_join_frames: list[int] = []
    offset = 0
    for c in clips[:-1]:
        offset += round(c.total_frames * TARGET_FPS / c.src_fps)
        clip_join_frames.append(offset)

    # Transition frame share
    trans_frames = sum(
        s.end - s.start for s in segments if s.seg_type == "transition"
    )
    trans_share = trans_frames / n_frames if n_frames else 0.0

    # Direct cuts: clip join frames that land inside an exercise segment
    direct_cuts = 0
    for jf in clip_join_frames:
        for seg in segments:
            if seg.start <= jf < seg.end:
                if seg.seg_type == "exercise":
                    direct_cuts += 1
                break

    meta = {
        "video_id":    video_id,
        "split":       split,
        "seed":        seed,
        "fps":         TARGET_FPS,
        "width":       TARGET_W,
        "height":      TARGET_H,
        "n_frames":    n_frames,
        "splits_file": splits_file,
        "margin_label": margin_label,
        "clip_ids_in_order": [c.clip_id for c in clips],
        "clip_join_frames":  clip_join_frames,
        "segments":    [seg_to_dict(s) for s in segments],
    }

    with open(json_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    exercises_used = sorted({c.exercise for c in clips})
    index_row = {
        "video_id":               video_id,
        "split":                  split,
        "n_segments":             len(segments),
        "duration_s":             round(n_frames / TARGET_FPS, 3),
        "exercises":              "|".join(exercises_used),
        "source_clips":           "|".join(c.clip_id for c in clips),
        "transition_frame_share": round(trans_share, 4),
        "direct_cuts":            direct_cuts,
        "seed":                   seed,
    }
    return index_row


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def run_batch(
    pool: dict[str, list[ClipInfo]],
    counts: dict[str, int],
    out_base: Path,
    splits_file: str,
    margin_label: str,
    seed: int,
    pilot: bool,
    orig_map: dict[str, str],
) -> list[dict]:
    """Generate all synthetic videos; return index rows."""

    rng = random.Random(seed)
    index_rows: list[dict] = []

    # Global video counter per split
    video_counters: dict[str, int] = defaultdict(int)

    for split, n_videos in counts.items():
        split_pool = pool.get(split, [])
        if not split_pool:
            log.warning("No clips in pool for split=%s", split)
            continue

        # Group pool by exercise
        ex_pool: dict[str, list[ClipInfo]] = defaultdict(list)
        for c in split_pool:
            ex_pool[c.exercise].append(c)

        used_clips: set[str] | None = set() if split in ("val", "test") else None
        usage: dict[str, int] = defaultdict(int)

        if pilot:
            n_videos = 1

        for vid_idx in range(n_videos):
            k = rng.randint(K_MIN, K_MAX)

            # For balanced usage, sort each exercise pool by usage count
            balanced_ex_pool: dict[str, list[ClipInfo]] = {}
            for ex, clips in ex_pool.items():
                available = [c for c in clips
                             if used_clips is None or c.clip_id not in used_clips]
                balanced_ex_pool[ex] = balance_usage(available, usage)

            seq = sample_sequence(k, balanced_ex_pool, rng, used_clips)
            if seq is None:
                log.warning("Pool exhausted for split=%s after %d videos", split, vid_idx)
                break

            # Mark used (val/test only)
            if used_clips is not None:
                for c in seq:
                    used_clips.add(c.clip_id)

            # Update usage counters
            for c in seq:
                usage[c.clip_id] += 1

            video_counters[split] += 1
            nn = video_counters[split]
            video_id = f"synth_{split}_{nn:04d}"

            out_dir = out_base / split
            row = make_video(
                video_id=video_id,
                clips=seq,
                split=split,
                out_dir=out_dir,
                margin_label=margin_label,
                splits_file=splits_file,
                seed=seed,
                orig_map=orig_map,
            )
            index_rows.append(row)

        log.info("Split %s: generated %d videos", split, video_counters[split])

    return index_rows


# ---------------------------------------------------------------------------
# Index CSV
# ---------------------------------------------------------------------------

def write_index(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Index written: %s  (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concatenate RepCount clips into synthetic videos.")
    p.add_argument("--splits-file", default="data/manifests/splits.csv")
    p.add_argument("--boundaries-file", default="data/manifests/repcount_rep_boundaries.json")
    p.add_argument("--exclusions-file", default="data/manifests/manual_exclusions.csv")
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--n-val",   type=int, default=6)
    p.add_argument("--n-test",  type=int, default=15)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out",     default="data/synthetic")
    p.add_argument("--pilot",   action="store_true",
                   help="Generate 1 video per split then stop")
    p.add_argument("--margin-label", choices=["transition", "exercise"],
                   default="transition")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    splits_path = Path(args.splits_file)
    boundaries_path = Path(args.boundaries_file)
    exclusions_path = Path(args.exclusions_file)
    out_base = Path(args.out)

    with open(boundaries_path) as fh:
        boundaries = json.load(fh)

    exclusions = load_exclusions(exclusions_path)
    pool = load_pool(splits_path, boundaries, exclusions)

    # Build orig_map from splits CSV (for video file lookup)
    df = pd.read_csv(splits_path)
    orig_map = dict(zip(df["clip_id"], df["original_split"]))

    counts = {
        "train": args.n_train,
        "val":   args.n_val,
        "test":  args.n_test,
    }

    index_rows = run_batch(
        pool=pool,
        counts=counts,
        out_base=out_base,
        splits_file=args.splits_file,
        margin_label=args.margin_label,
        seed=args.seed,
        pilot=args.pilot,
        orig_map=orig_map,
    )

    write_index(index_rows, Path("data/manifests/synthetic_index.csv"))

    # Summary
    log.info("\n=== SUMMARY ===")
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in index_rows:
        by_split[r["split"]].append(r)
    for sp, rows in sorted(by_split.items()):
        trans_shares = [r["transition_frame_share"] for r in rows]
        dcuts = sum(r["direct_cuts"] for r in rows)
        log.info("  %s: %d videos | trans_share mean=%.3f | direct_cuts=%d",
                 sp, len(rows),
                 sum(trans_shares) / len(trans_shares) if trans_shares else 0,
                 dcuts)


if __name__ == "__main__":
    main()
