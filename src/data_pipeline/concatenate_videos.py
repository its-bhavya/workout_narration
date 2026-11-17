"""
concatenate_videos.py

Concatenate whole RepCount clips into synthetic multi-exercise videos
with ground-truth segment labels.

Usage
-----
    python src/data_pipeline/concatenate_videos.py \\
        --splits-file data/manifests/splits_v2.csv \\
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
import os
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

# Cumulative clip-use caps (batch 1 + batch 2 + …)
CUMULATIVE_CAPS = {"train": 5, "val": 2, "test": 1}

INCLUDE_STATUSES = {"pass", "review"}
EXCLUDE_FPS_BELOW = 20.0

REPCOUNT_ROOT = Path(os.environ.get(
    "REPCOUNT_ROOT", r"C:\Users\Bhavya\Downloads\RepCountA (1)\LLSP"))
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
    for sp in ["valid" if original_split == "val" else original_split, "train", "valid", "test"]:
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
        log.info("no manual exclusions file found")
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
    used_clips: set[str] | None,   # None = train (reuse OK across videos)
    used_in_video: set[str] | None = None,
) -> list[ClipInfo] | None:
    """Sample k clips with adjacent-exercise constraint and ≥2 distinct exercises.

    used_clips    : clips excluded entirely (None = no exclusion).
    used_in_video : clips already picked in this video — prevents the same
                    clip appearing twice in one video.  None = not checked
                    (preserves batch-1 behaviour where train clips could
                    repeat within a video).
    """
    exercises = list(exercise_pool.keys())
    if len(exercises) < 2:
        return None

    for _attempt in range(200):
        seq: list[ClipInfo] = []
        prev_ex = None
        local_used: set[str] = set() if used_in_video is None else set(used_in_video)

        for _ in range(k):
            # Build candidate exercises (must differ from previous)
            candidates = [e for e in exercises if e != prev_ex
                          and exercise_pool[e]]
            if not candidates:
                break
            ex = rng.choice(candidates)
            clips = [c for c in exercise_pool[ex]
                     if (used_clips is None or c.clip_id not in used_clips)
                     and c.clip_id not in local_used]
            if not clips:
                break
            clip = rng.choice(clips)
            seq.append(clip)
            local_used.add(clip.clip_id)
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
    try:
        for clip in clips:
            vp = find_video(clip.clip_id, orig_map.get(clip.clip_id, "train"))
            if vp is None:
                raise FileNotFoundError(f"Video not found: {clip.clip_id}")
            cap = cv2.VideoCapture(str(vp))
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open source video: {vp}")
                n_out = round(clip.total_frames * TARGET_FPS / clip.src_fps)
                decoded = -1
                frame = None
                for out_f in range(n_out):
                    src_idx = resample_index(out_f, 0, clip.src_fps, clip.total_frames)
                    # Sequential decode preserves the exact nearest-frame map
                    # without an expensive keyframe seek for every output frame.
                    while decoded < src_idx:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            raise RuntimeError(f"Source decode failed: {clip.clip_id} frame {decoded+1}; no footage substituted")
                        decoded += 1
                    writer.write(letterbox_frame(frame, TARGET_W, TARGET_H))
                total_written += n_out
            finally:
                cap.release()
    finally:
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
    batch_id: int = 1,
    run_id: str | None = None,
    fold: int | None = None,
    plan_hash: str | None = None,
) -> dict:
    """Generate one video + JSON; return index row dict."""

    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path  = out_dir / f"{video_id}.mp4"
    json_path = out_dir / f"{video_id}.json"

    # Abort if target files already exist (append-mode safety)
    if mp4_path.exists():
        raise FileExistsError(f"Target video already exists: {mp4_path}")
    if json_path.exists():
        raise FileExistsError(f"Target JSON already exists: {json_path}")

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
    if run_id is not None:
        meta.update(run_id=run_id, fold=fold, plan_hash=plan_hash)

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
        "batch":                  batch_id,
    }
    if run_id is not None:
        index_row.update(fold=fold, run_id=run_id)
    return index_row


# ---------------------------------------------------------------------------
# Resumability helpers
# ---------------------------------------------------------------------------

def append_index_row(index_path: Path, row: dict) -> None:
    """Append a single row to the index CSV (fast, no full rewrite)."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_path.exists()
    if write_header:
        fieldnames = list(row.keys())
    else:
        with open(index_path, "r", newline="") as fh:
            fieldnames = next(csv.reader(fh))
    with open(index_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    checkpoint_path: Path,
    batch_id: int,
    seed: int,
    splits_order: list[str],
    current_split_idx: int,
    current_vid_idx: int,
    video_counters: dict[str, int],
    batch_usage_all: dict[str, dict[str, int]],
    rng_state: tuple,
) -> None:
    """Persist RNG state and per-split batch usage for resumable rendering."""
    ckpt = {
        "batch_id": batch_id,
        "seed": seed,
        "splits_order": splits_order,
        "current_split_idx": current_split_idx,
        "current_vid_idx": current_vid_idx,
        "video_counters": dict(video_counters),
        "batch_usage": {s: dict(u) for s, u in batch_usage_all.items()},
        # rng.getstate() → (version, internalstate_tuple, gauss_next)
        "rng_state": [rng_state[0], list(rng_state[1]), rng_state[2]],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as fh:
        json.dump(ckpt, fh)


def load_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load checkpoint JSON, or return None if it doesn't exist."""
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path) as fh:
        return json.load(fh)


def reconstruct_row_from_json(json_path: Path, batch_id: int) -> dict:
    """Reconstruct an index row from an existing video .json file."""
    with open(json_path) as fh:
        meta = json.load(fh)
    segments = meta["segments"]
    n_frames = meta["n_frames"]
    fps = meta["fps"]

    trans_frames = sum(
        s["end_frame"] - s["start_frame"] for s in segments
        if s["type"] == "transition"
    )
    trans_share = trans_frames / n_frames if n_frames else 0.0

    clip_join_frames = meta.get("clip_join_frames", [])
    direct_cuts = 0
    for jf in clip_join_frames:
        for seg in segments:
            if seg["start_frame"] <= jf < seg["end_frame"]:
                if seg["type"] == "exercise":
                    direct_cuts += 1
                break

    exercises_used = sorted({s["label"] for s in segments if s["type"] == "exercise"})

    return {
        "video_id":               meta["video_id"],
        "split":                  meta["split"],
        "n_segments":             len(segments),
        "duration_s":             round(n_frames / fps, 3),
        "exercises":              "|".join(exercises_used),
        "source_clips":           "|".join(meta["clip_ids_in_order"]),
        "transition_frame_share": round(trans_share, 4),
        "direct_cuts":            direct_cuts,
        "seed":                   meta["seed"],
        "batch":                  batch_id,
    }


def load_indexed_video_ids(index_path: Path) -> set[str]:
    """Read synthetic_index.csv and return the set of video_ids already present."""
    if not index_path.exists():
        return set()
    with open(index_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        return {r["video_id"] for r in reader}


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
    append: bool = False,
    batch_id: int = 1,
    cumulative_usage: dict[str, int] | None = None,
    start_counters: dict[str, int] | None = None,
    # Resumability parameters (append mode only):
    checkpoint_path: Path | None = None,
    index_path: Path | None = None,
    usage_path: Path | None = None,
    manifest_path: Path | None = None,
) -> list[dict]:
    """Generate all synthetic videos; return index rows.

    append=False (batch-1 mode): original behaviour unchanged.

    append=True (batch-2+ mode): cumulative caps enforced, balancing
    uses cumulative+batch usage, "clip never twice in one video" is
    enforced for ALL splits, video numbering continues from
    start_counters, and batch_id is written to each index row.

    When *checkpoint_path* is provided (append mode), the function
    saves a checkpoint after every video (RNG state + batch usage +
    progress).  If interrupted and re-run with the same arguments,
    it resumes from the checkpoint.  The index, clip_usage and
    batch_manifest are also updated incrementally so they stay
    consistent with the rendered files.
    """

    splits_order = list(counts.keys())

    # ---- Load checkpoint or initialise fresh ----
    ckpt = None
    if append and checkpoint_path:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            if ckpt["batch_id"] != batch_id or ckpt["seed"] != seed:
                log.warning("Checkpoint batch_id/seed mismatch — starting fresh")
                ckpt = None
            elif ckpt["splits_order"] != splits_order:
                log.warning("Checkpoint splits_order mismatch — starting fresh")
                ckpt = None

    if ckpt is not None:
        rng = random.Random(seed)
        rng.setstate((ckpt["rng_state"][0],
                      tuple(ckpt["rng_state"][1]),
                      ckpt["rng_state"][2]))
        video_counters = ckpt["video_counters"]
        batch_usage_all: dict[str, dict[str, int]] = {
            s: defaultdict(int, u) for s, u in ckpt["batch_usage"].items()
        }
        start_split_idx = ckpt["current_split_idx"]
        start_vid_idx = ckpt["current_vid_idx"]
        log.info("Resumed from checkpoint: split=%s, next_vid_idx=%d",
                 splits_order[start_split_idx], start_vid_idx)
    else:
        rng = random.Random(seed)
        video_counters = (
            dict(start_counters) if start_counters else defaultdict(int)
        )
        batch_usage_all = {}
        start_split_idx = 0
        start_vid_idx = 0

    # Set of video_ids already in the index (to avoid duplicate rows)
    indexed_video_ids: set[str] = set()
    if append and index_path and index_path.exists():
        indexed_video_ids = load_indexed_video_ids(index_path)

    index_rows: list[dict] = []

    for split_idx in range(start_split_idx, len(splits_order)):
        split = splits_order[split_idx]
        n_videos = counts[split]
        split_pool = pool.get(split, [])
        if not split_pool:
            log.warning("No clips in pool for split=%s", split)
            continue

        cap = CUMULATIVE_CAPS[split]

        # Group pool by exercise
        ex_pool: dict[str, list[ClipInfo]] = defaultdict(list)
        for c in split_pool:
            ex_pool[c.exercise].append(c)

        if append:
            if split not in batch_usage_all:
                batch_usage_all[split] = defaultdict(int)
            batch_usage = batch_usage_all[split]
        else:
            used_clips: set[str] | None = set() if split in ("val", "test") else None
            usage: dict[str, int] = defaultdict(int)

        if pilot:
            n_videos = 1

        # Resume position within this split
        vid_start = start_vid_idx if split_idx == start_split_idx else 0
        start_vid_idx = 0  # reset for subsequent splits

        for vid_idx in range(vid_start, n_videos):
            k = rng.randint(K_MIN, K_MAX)

            if append:
                balanced_ex_pool: dict[str, list[ClipInfo]] = {}
                for ex, clips in ex_pool.items():
                    available = []
                    for c in clips:
                        total = (cumulative_usage or {}).get(c.clip_id, 0) \
                                + batch_usage.get(c.clip_id, 0)
                        if total < cap:
                            available.append(c)
                    available.sort(
                        key=lambda c: (cumulative_usage or {}).get(c.clip_id, 0)
                                      + batch_usage.get(c.clip_id, 0)
                    )
                    balanced_ex_pool[ex] = available
            else:
                balanced_ex_pool = {}
                for ex, clips in ex_pool.items():
                    available = [c for c in clips
                                 if used_clips is None or c.clip_id not in used_clips]
                    balanced_ex_pool[ex] = balance_usage(available, usage)

            if append:
                seq = sample_sequence(k, balanced_ex_pool, rng,
                                       used_clips=set(),
                                       used_in_video=set())
            else:
                seq = sample_sequence(k, balanced_ex_pool, rng, used_clips)

            if seq is None:
                log.warning("Pool exhausted for split=%s after %d videos", split, vid_idx)
                break


            if append:
                for c in seq:
                    batch_usage[c.clip_id] += 1
            else:
                if used_clips is not None:
                    for c in seq:
                        used_clips.add(c.clip_id)
                for c in seq:
                    usage[c.clip_id] += 1

            video_counters[split] = video_counters.get(split, 0) + 1
            nn = video_counters[split]
            video_id = f"fixture_{split}_{nn:04d}"

            out_dir = out_base / split
            mp4_path = out_dir / f"{video_id}.mp4"
            json_path = out_dir / f"{video_id}.json"

            _bid = batch_id if append else 1

            if mp4_path.exists() and json_path.exists():
                # ---- Skip: video already rendered (previous interrupted run) ----
                log.info("Skipping %s (files already exist)", video_id)
                row = reconstruct_row_from_json(json_path, _bid)
                index_rows.append(row)

                if append and checkpoint_path:
                    if video_id not in indexed_video_ids:
                        # Orphaned file — recover it into the index/manifests
                        append_index_row(index_path, row)
                        update_clip_usage(usage_path, [row], batch_id)
                        append_batch_manifest([row], manifest_path)
                        indexed_video_ids.add(video_id)
                        log.info("Recovered %s into index", video_id)
                    save_checkpoint(
                        checkpoint_path, batch_id, seed, splits_order,
                        split_idx, vid_idx + 1, video_counters,
                        batch_usage_all, rng.getstate(),
                    )
                continue

            if mp4_path.exists() and not json_path.exists():
                # Preserve interrupted footage before re-rendering.
                log.warning("Corrupted: %s exists without .json — re-rendering", mp4_path.name)
                from generate_v2 import archive_partial
                archive_partial(mp4_path, out_base)

            # ---- Normal path: render the video ----
            row = make_video(
                video_id=video_id,
                clips=seq,
                split=split,
                out_dir=out_dir,
                margin_label=margin_label,
                splits_file=splits_file,
                seed=seed,
                orig_map=orig_map,
                batch_id=_bid,
            )
            index_rows.append(row)

            # ---- Incremental persistence (append mode) ----
            if append and checkpoint_path:
                append_index_row(index_path, row)
                update_clip_usage(usage_path, [row], batch_id)
                append_batch_manifest([row], manifest_path)
                indexed_video_ids.add(video_id)
                save_checkpoint(
                    checkpoint_path, batch_id, seed, splits_order,
                    split_idx, vid_idx + 1, video_counters,
                    batch_usage_all, rng.getstate(),
                )

        generated = video_counters.get(split, 0) - (start_counters or {}).get(split, 0)
        log.info("Split %s: generated %d videos", split, generated)

    # ---- Clean up checkpoint when batch is complete ----
    if append and checkpoint_path and checkpoint_path.exists():
        from generate_v2 import archive_partial
        archive_partial(checkpoint_path, checkpoint_path.parent)
        log.info("Checkpoint archived (batch complete): %s", checkpoint_path)

    return index_rows


# ---------------------------------------------------------------------------
# Clip-usage tracking (cumulative across batches)
# ---------------------------------------------------------------------------

CLIP_USAGE_COLS = ["clip_id", "split", "exercise", "times_used", "batch"]


def init_clip_usage(index_path: Path, usage_path: Path) -> dict[str, int]:
    """Create or load clip_usage.csv from synthetic_index.csv.

    If *usage_path* exists it is loaded and returned (never reset).
    Otherwise it is built from *index_path* with every clip set to
    times_used = total appearances and batch = 1.

    Returns {clip_id: times_used} (flat, cross-split).
    """
    if usage_path.exists():
        df_u = pd.read_csv(usage_path)
        return dict(zip(df_u["clip_id"], df_u["times_used"]))

    if not index_path.exists():
        return {}

    df = pd.read_csv(index_path)
    splits_df = pd.read_csv("data/manifests/splits_v2.csv")
    ex_map = dict(zip(splits_df["clip_id"], splits_df["exercise"]))
    sp_map = dict(zip(splits_df["clip_id"], splits_df["split"]))

    usage: dict[str, dict] = {}
    for _, row in df.iterrows():
        for clip_id in str(row["source_clips"]).split("|"):
            if clip_id not in usage:
                usage[clip_id] = {
                    "clip_id": clip_id,
                    "split": sp_map.get(clip_id, ""),
                    "exercise": ex_map.get(clip_id, ""),
                    "times_used": 0,
                    "batch": 1,
                }
            usage[clip_id]["times_used"] += 1

    rows = list(usage.values())
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CLIP_USAGE_COLS).to_csv(usage_path, index=False)
    log.info("clip_usage.csv initialised: %s (%d clips)", usage_path, len(rows))
    return {r["clip_id"]: r["times_used"] for r in rows}


def update_clip_usage(
    usage_path: Path,
    new_rows: list[dict],
    batch_id: int,
) -> dict[str, int]:
    """Increment times_used for every clip in new_rows and write back."""
    df_u = pd.read_csv(usage_path)

    for row in new_rows:
        for clip_id in str(row["source_clips"]).split("|"):
            mask = df_u["clip_id"] == clip_id
            if mask.any():
                df_u.loc[mask, "times_used"] += 1
                df_u.loc[mask, "batch"] = batch_id
            else:
                splits_df = pd.read_csv("data/manifests/splits_v2.csv")
                ex_map = dict(zip(splits_df["clip_id"], splits_df["exercise"]))
                sp_map = dict(zip(splits_df["clip_id"], splits_df["split"]))
                new = pd.DataFrame([{
                    "clip_id": clip_id,
                    "split": sp_map.get(clip_id, ""),
                    "exercise": ex_map.get(clip_id, ""),
                    "times_used": 1,
                    "batch": batch_id,
                }], columns=CLIP_USAGE_COLS)
                df_u = pd.concat([df_u, new], ignore_index=True)

    df_u.to_csv(usage_path, index=False)
    log.info("clip_usage.csv updated: %s", usage_path)
    return dict(zip(df_u["clip_id"], df_u["times_used"]))


# ---------------------------------------------------------------------------
# Batch manifest
# ---------------------------------------------------------------------------

BATCH_MANIFEST_COLS = ["video_id", "split", "batch", "seed"]


def init_batch_manifest(index_path: Path, manifest_path: Path) -> None:
    """Create batch_manifest.csv from existing index if it doesn't exist."""
    if manifest_path.exists():
        return
    if not index_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=BATCH_MANIFEST_COLS).to_csv(manifest_path, index=False)
        return

    df = pd.read_csv(index_path)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "video_id": r["video_id"],
            "split":    r["split"],
            "batch":    r.get("batch", 1),
            "seed":     r["seed"],
        })
    pd.DataFrame(rows, columns=BATCH_MANIFEST_COLS).to_csv(manifest_path, index=False)
    log.info("batch_manifest.csv initialised: %s (%d rows)", manifest_path, len(rows))


def append_batch_manifest(
    new_rows: list[dict],
    manifest_path: Path,
) -> None:
    """Append new video rows to batch_manifest.csv."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in new_rows:
        rows.append({
            "video_id": r["video_id"],
            "split":    r["split"],
            "batch":    r["batch"],
            "seed":     r["seed"],
        })
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        combined = pd.concat([existing, pd.DataFrame(rows, columns=BATCH_MANIFEST_COLS)],
                              ignore_index=True)
    else:
        combined = pd.DataFrame(rows, columns=BATCH_MANIFEST_COLS)
    combined.to_csv(manifest_path, index=False)
    log.info("batch_manifest.csv updated: %s (%d total rows)", manifest_path, len(combined))


# ---------------------------------------------------------------------------
# Index CSV
# ---------------------------------------------------------------------------

def write_index(
    rows: list[dict],
    path: Path,
    append: bool = False,
) -> None:
    """Write (or append) index rows to CSV.

    When append=False the file is overwritten (batch-1 behaviour).
    When append=True existing rows are read, a ``batch`` column is
    added to them (default 1) if missing, and new rows are appended.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not append:
        return

    if append and path.exists():
        existing_df = pd.read_csv(path)
        if "batch" not in existing_df.columns:
            existing_df["batch"] = 1
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        fieldnames = list(combined.columns)
        combined.to_csv(path, index=False)
        log.info("Index appended: %s (%d total rows)", path, len(combined))
    else:
        fieldnames = list(rows[0].keys()) if rows else []
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info("Index written: %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Plan-only (no rendering)
# ---------------------------------------------------------------------------

def load_existing_usage(index_path: Path) -> dict[str, dict[str, int]]:
    """Read synthetic_index.csv → {split -> {clip_id -> times_used}}."""
    if not index_path.exists():
        return defaultdict(lambda: defaultdict(int))
    df = pd.read_csv(index_path)
    usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        split = row["split"]
        for clip_id in str(row["source_clips"]).split("|"):
            usage[split][clip_id] += 1
    return usage


def run_plan_only(
    pool: dict[str, list[ClipInfo]],
    index_path: Path,
) -> None:
    """Compute and print max additional videos per split under cumulative caps.

    No video files are written.
    """
    existing_usage = load_existing_usage(index_path)

    print("=" * 64)
    print("PLAN-ONLY: max additional videos under cumulative caps")
    print("=" * 64)

    for split in ["train", "val", "test"]:
        split_pool = pool.get(split, [])
        cap = CUMULATIVE_CAPS[split]
        usage_split = existing_usage.get(split, {})

        # remaining slots per clip
        remaining: dict[str, int] = {}
        for c in split_pool:
            rem = cap - usage_split.get(c.clip_id, 0)
            if rem > 0:
                remaining[c.clip_id] = rem

        # remaining slots per exercise
        ex_slots: dict[str, int] = defaultdict(int)
        for c in split_pool:
            rem = remaining.get(c.clip_id, 0)
            if rem > 0:
                ex_slots[c.exercise] += rem

        total_remaining = sum(ex_slots.values())
        upper = total_remaining // K_MIN

        sorted_vals = sorted(ex_slots.values(), reverse=True)
        minority_bound = sum(sorted_vals[1:]) if len(sorted_vals) >= 2 else 0
        max_v = min(upper, minority_bound)

        # current usage distribution
        used_clips = {cid: n for cid, n in usage_split.items() if n > 0}
        dist: dict[int, int] = defaultdict(int)
        for n in used_clips.values():
            dist[n] += 1

        print(f"\n--- {split.upper()}  (cap={cap}) ---")
        print(f"  Pool clips:       {len(split_pool)}")
        print(f"  Clips used (b1):  {len(used_clips)}")
        print(f"  Usage dist:       {dict(sorted(dist.items()))}")
        print(f"  Clips w/ remain:  {len(remaining)}")
        print(f"  Remaining slots:  {total_remaining}  by ex: {dict(ex_slots)}")
        print(f"  Upper (// {K_MIN}):    {upper}")
        print(f"  Minority bound:   {minority_bound}")
        print(f"  >>> MAX ADDITIONAL VIDEOS: {max_v}")

    print("\n" + "=" * 64)
    print("No videos written (--plan-only).")
    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concatenate RepCount clips into synthetic videos.")
    p.add_argument("--splits-file", default="data/manifests/splits_v2.csv")
    p.add_argument("--folds-file", default="data/manifests/folds.csv", help="Source-clip folds for the v2 planner")
    p.add_argument("--run-id", default="v2", help="Isolated run identifier (v2 planning only)")
    p.add_argument("--test-cap", type=int, default=1, help="Test uses per clip (v2 planning)")
    p.add_argument("--n-train-per-fold", type=int, default=None,
                   help="Plan this many videos per fold with seeded uniform K=3..5")
    p.add_argument("--redraw-candidates", type=int, default=0,
                   help="Compare this many new stratified fold seeds by margin balance only")
    p.add_argument("--validate-only", action="store_true", help="Validate the rendered v2 run without rendering")
    p.add_argument("--max-videos", type=int, default=None, help="Stop after this many newly rendered v2 videos; rerun to resume")
    p.add_argument("--boundaries-file", default="data/manifests/repcount_rep_boundaries.json")
    p.add_argument("--exclusions-file", default="data/manifests/manual_exclusions.csv")
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--n-val",   type=int, default=6)
    p.add_argument("--n-test",  type=int, default=15)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out",     default="data/synthetic")
    p.add_argument("--pilot",   action="store_true",
                   help="Generate 1 video per split then stop")
    p.add_argument("--plan-only", action="store_true",
                   help="Compute max additional videos per split under "
                        "cumulative caps; print plan and exit (no rendering)")
    p.add_argument("--append", action="store_true",
                   help="Append mode: read existing synthetic_index.csv for "
                        "cumulative clip usage, start video numbering after "
                        "existing, and append new rows to the index")
    p.add_argument("--batch-id", type=int, default=2,
                   help="Batch number for append mode (default 2)")
    p.add_argument("--margin-label", choices=["transition", "exercise"],
                   default="transition")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.folds_file or args.run_id:
        if not (args.folds_file and args.run_id):
            raise SystemExit("Fold-aware runs require --folds-file and --run-id")
        if args.append or args.margin_label != "transition" or args.test_cap < 1:
            raise SystemExit("v2 planning requires a fresh run, transition margins and positive --test-cap")
        if not args.plan_only:
            if args.n_train_per_fold is None or args.n_train_per_fold < 1:
                raise SystemExit("v2 generation requires positive --n-train-per-fold")
            from generate_v2 import run_v2
            run_v2(args)
            return
        from plan_folds import report_fold_plan
        if args.n_train_per_fold is not None:
            if args.n_train_per_fold < 1 or args.redraw_candidates < 0:
                raise SystemExit("Positive train target and nonnegative redraw candidate count required")
            from plan_targets import report_target_plan
            report_target_plan(args)
            return
        report_fold_plan(args)
        return

    splits_path = Path(args.splits_file)
    boundaries_path = Path(args.boundaries_file)
    exclusions_path = Path(args.exclusions_file)
    out_base = Path(args.out)

    index_path   = Path("data/manifests/synthetic_index.csv")
    usage_path   = Path("data/manifests/clip_usage.csv")
    manifest_path = Path("data/manifests/batch_manifest.csv")

    with open(boundaries_path) as fh:
        boundaries = json.load(fh)

    exclusions = load_exclusions(exclusions_path)
    pool = load_pool(splits_path, boundaries, exclusions)

    # Build orig_map from splits CSV (for video file lookup)
    df = pd.read_csv(splits_path)
    orig_map = dict(zip(df["clip_id"], df["original_split"]))

    if args.plan_only:
        run_plan_only(pool, index_path)
        return

    counts = {
        "train": args.n_train,
        "val":   args.n_val,
        "test":  args.n_test,
    }

    if args.append:
        # ---- Append mode (batch 2+) ----
        checkpoint_path = Path("data/manifests/batch_checkpoint.json")
        cumulative_usage = init_clip_usage(index_path, usage_path)
        init_batch_manifest(index_path, manifest_path)

        # Ensure existing index has a batch column
        if index_path.exists():
            idx_df = pd.read_csv(index_path)
            if "batch" not in idx_df.columns:
                idx_df["batch"] = 1
                idx_df.to_csv(index_path, index=False)
                log.info("Added batch column to existing index (all = 1)")

        # Compute start counters from index (unless resuming from checkpoint)
        ckpt = load_checkpoint(checkpoint_path)
        if (ckpt is not None
                and ckpt["batch_id"] == args.batch_id
                and ckpt["seed"] == args.seed):
            start_counters = ckpt["video_counters"]
            log.info("Resuming from checkpoint — start_counters: %s", start_counters)
        else:
            start_counters: dict[str, int] = {}
            if index_path.exists():
                idx_df = pd.read_csv(index_path)
                for split in ["train", "val", "test"]:
                    split_rows = idx_df[idx_df["split"] == split]
                    if len(split_rows) > 0:
                        nums = split_rows["video_id"].str.extract(r"synth_\w+_(\d+)")[0]
                        start_counters[split] = int(nums.astype(int).max())
                    else:
                        start_counters[split] = 0
            else:
                start_counters = {s: 0 for s in ["train", "val", "test"]}

        log.info("Start counters: %s", start_counters)
        log.info("Batch ID: %d, seed: %d", args.batch_id, args.seed)

        index_rows = run_batch(
            pool=pool,
            counts=counts,
            out_base=out_base,
            splits_file=args.splits_file,
            margin_label=args.margin_label,
            seed=args.seed,
            pilot=args.pilot,
            orig_map=orig_map,
            append=True,
            batch_id=args.batch_id,
            cumulative_usage=cumulative_usage,
            start_counters=start_counters,
            checkpoint_path=checkpoint_path,
            index_path=index_path,
            usage_path=usage_path,
            manifest_path=manifest_path,
        )

        # Index, clip_usage and batch_manifest are updated incrementally
        # inside run_batch — no bulk post-run writes needed.
    else:
        # ---- Batch-1 mode (original) ----
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

        write_index(index_rows, index_path)

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
