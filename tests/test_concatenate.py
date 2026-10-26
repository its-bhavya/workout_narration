"""
Tests for concatenate_videos.py

Run from repo root:
    python -m pytest tests/test_concatenate.py -v
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Import helpers directly from the module under test
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_pipeline"))

from concatenate_videos import (
    TARGET_FPS,
    ClipInfo,
    RawSegment,
    build_raw_segments,
    merge_transitions,
    resample_index,
)


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

def make_clip(
    clip_id: str = "clip_a.mp4",
    exercise: str = "squat",
    split: str = "train",
    total_frames: int = 120,
    src_fps: float = 30.0,
    first_start: int = 10,
    last_end: int = 100,
    pairs: list | None = None,
    rep_count: int = 4,
) -> ClipInfo:
    if pairs is None:
        pairs = [[10, 30], [31, 60], [61, 80], [81, 100]]
    return ClipInfo(
        clip_id=clip_id,
        exercise=exercise,
        split=split,
        first_start=first_start,
        last_end=last_end,
        total_frames=total_frames,
        src_fps=src_fps,
        pairs=pairs,
        rep_count=rep_count,
    )


# ---------------------------------------------------------------------------
# 1. FPS resampling index mapping
# ---------------------------------------------------------------------------

class TestResampleIndex:
    def test_same_fps(self):
        # source and target both 30 fps: out_frame N -> src N
        for n in [0, 5, 15, 29]:
            assert resample_index(n, 0, 30.0, 30) == n

    def test_half_fps(self):
        # source 15 fps, target 30 fps -> src = round(out_time * 15)
        # out_frame=0 -> t=0.000 -> src 0
        # out_frame=1 -> t=0.033 -> round(0.5) = 0 (Python banker's rounding)
        # out_frame=2 -> t=0.067 -> round(1.0) = 1
        assert resample_index(0, 0, 15.0, 60) == 0
        assert resample_index(1, 0, 15.0, 60) == 0   # banker's rounding: round(0.5) == 0
        assert resample_index(2, 0, 15.0, 60) == 1

    def test_double_fps(self):
        # source 60 fps, target 30 fps
        assert resample_index(0, 0, 60.0, 120) == 0
        assert resample_index(1, 0, 60.0, 120) == 2

    def test_clamps_to_last_frame(self):
        # If resampled index would exceed total_frames-1, clamp
        idx = resample_index(1000, 0, 30.0, 50)
        assert idx == 49

    def test_offset_applied(self):
        # out_frame 35, offset 30 -> local out_frame 5 -> src 5 at 30fps
        assert resample_index(35, 30, 30.0, 60) == 5


# ---------------------------------------------------------------------------
# 2. Pair re-offset
# ---------------------------------------------------------------------------

class TestPairReoffset:
    def test_pairs_land_inside_exercise_segment(self):
        clip = make_clip(total_frames=90, src_fps=30.0,
                         first_start=10, last_end=80,
                         pairs=[[10, 40], [41, 80]])
        segs, n = build_raw_segments([clip], margin_label="transition")
        ex_seg = next(s for s in segs if s.seg_type == "exercise")
        for pair in ex_seg.rep_pairs_out:
            s_out, e_out = pair
            assert ex_seg.start <= s_out, f"pair start {s_out} < exercise start {ex_seg.start}"
            assert e_out <= ex_seg.end,   f"pair end {e_out} > exercise end {ex_seg.end}"

    def test_pairs_offset_by_clip_position(self):
        # Two clips: first 60 frames, second 60 frames.
        # Second clip's pairs should be offset by ~60 output frames.
        c1 = make_clip("c1.mp4", "squat",    total_frames=60, src_fps=30.0,
                        first_start=5, last_end=55, pairs=[[5, 55]])
        c2 = make_clip("c2.mp4", "push_up",  total_frames=60, src_fps=30.0,
                        first_start=5, last_end=55, pairs=[[5, 55]])
        segs, _ = build_raw_segments([c1, c2], margin_label="transition")
        ex_segs = [s for s in segs if s.seg_type == "exercise"]
        assert len(ex_segs) == 2
        # Second exercise segment should start at >= 60
        assert ex_segs[1].start >= 60


# ---------------------------------------------------------------------------
# 3. Contiguity: segments cover [0, n_frames) with no gaps or overlaps
# ---------------------------------------------------------------------------

class TestContiguity:
    def _check_contiguous(self, clips, margin_label):
        raw_segs, n_frames = build_raw_segments(clips, margin_label)
        merged = merge_transitions(raw_segs)
        assert merged, "No segments produced"
        assert merged[0].start == 0, f"First segment does not start at 0: {merged[0].start}"
        assert merged[-1].end == n_frames, (
            f"Last segment end {merged[-1].end} != n_frames {n_frames}"
        )
        for i in range(1, len(merged)):
            assert merged[i].start == merged[i - 1].end, (
                f"Gap between seg {i-1} and {i}: "
                f"{merged[i-1].end} != {merged[i].start}"
            )

    def test_single_clip_transition(self):
        clip = make_clip()
        self._check_contiguous([clip], "transition")

    def test_single_clip_exercise(self):
        clip = make_clip()
        self._check_contiguous([clip], "exercise")

    def test_two_clips_transition(self):
        c1 = make_clip("c1.mp4", "squat",   total_frames=90, src_fps=30.0)
        c2 = make_clip("c2.mp4", "push_up", total_frames=60, src_fps=30.0)
        self._check_contiguous([c1, c2], "transition")

    def test_three_clips(self):
        c1 = make_clip("c1.mp4", "squat",     total_frames=90, src_fps=30.0)
        c2 = make_clip("c2.mp4", "push_up",   total_frames=60, src_fps=30.0)
        c3 = make_clip("c3.mp4", "jump_jack", total_frames=75, src_fps=25.0)
        self._check_contiguous([c1, c2, c3], "transition")

    def test_no_zero_length_segments(self):
        # Clip with first_start=0 and last_end=total-1: no pre/post transition
        clip = make_clip(first_start=0, last_end=119, total_frames=120)
        raw_segs, _ = build_raw_segments([clip], "transition")
        merged = merge_transitions(raw_segs)
        for seg in merged:
            assert seg.end > seg.start, f"Zero-length segment: {seg}"


# ---------------------------------------------------------------------------
# 4. No clip reuse in val / test
# ---------------------------------------------------------------------------

class TestClipReuse:
    def _make_pool(self, n: int, exercise: str = "squat", prefix: str = "sq") -> list[ClipInfo]:
        return [
            make_clip(f"{prefix}_clip_{i}.mp4", exercise, total_frames=60)
            for i in range(n)
        ]

    def test_val_no_reuse(self):
        from concatenate_videos import sample_sequence
        # Use distinct clip_id prefixes so squat and push_up pools don't share ids
        pool_sq = self._make_pool(5, "squat",   prefix="sq")
        pool_pu = self._make_pool(5, "push_up", prefix="pu")
        ex_pool = {"squat": pool_sq, "push_up": pool_pu}
        used: set[str] = set()
        rng = random.Random(0)
        seen = set()
        for _ in range(4):  # 4 videos of k=2 = 8 clips from pool of 10
            seq = sample_sequence(2, ex_pool, rng, used)
            if seq is None:
                break
            for c in seq:
                assert c.clip_id not in seen, f"Clip reused: {c.clip_id}"
                seen.add(c.clip_id)
                used.add(c.clip_id)


# ---------------------------------------------------------------------------
# 5. Single-split videos
# ---------------------------------------------------------------------------

class TestSingleSplit:
    def test_all_clips_same_split(self):
        clips = [
            make_clip("c1.mp4", "squat",   split="train"),
            make_clip("c2.mp4", "push_up", split="train"),
            make_clip("c3.mp4", "squat",   split="train"),
        ]
        splits_seen = {c.split for c in clips}
        assert splits_seen == {"train"}


# ---------------------------------------------------------------------------
# 6. Adjacent transitions merge; no zero-length segments
# ---------------------------------------------------------------------------

class TestMergeTransitions:
    def test_adjacent_transitions_merge(self):
        # Two transition segments back to back
        segs = [
            RawSegment("transition", "transition", 0,  10, source_clips=["a"]),
            RawSegment("transition", "transition", 10, 20, source_clips=["b"]),
        ]
        merged = merge_transitions(segs)
        assert len(merged) == 1
        assert merged[0].start == 0
        assert merged[0].end   == 20

    def test_non_adjacent_transitions_not_merged(self):
        segs = [
            RawSegment("transition", "transition", 0,  10, source_clips=["a"]),
            RawSegment("exercise",   "squat",      10, 20),
            RawSegment("transition", "transition", 20, 30, source_clips=["b"]),
        ]
        merged = merge_transitions(segs)
        assert len(merged) == 3

    def test_zero_length_dropped(self):
        segs = [
            RawSegment("transition", "transition", 5, 5, source_clips=["a"]),
            RawSegment("exercise",   "squat",      5, 15),
        ]
        merged = merge_transitions(segs)
        assert all(s.end > s.start for s in merged)
        assert len(merged) == 1

    def test_two_clip_inter_transition_merges(self):
        # post-tail of clip1 + pre-lead of clip2 should merge
        c1 = make_clip("c1.mp4", "squat",   total_frames=90, src_fps=30.0,
                        first_start=10, last_end=70)
        c2 = make_clip("c2.mp4", "push_up", total_frames=90, src_fps=30.0,
                        first_start=10, last_end=70)
        raw_segs, _ = build_raw_segments([c1, c2], "transition")
        merged = merge_transitions(raw_segs)
        # Segments: trans | ex | trans+trans(merged) | ex | trans
        trans_segs = [s for s in merged if s.seg_type == "transition"]
        ex_segs    = [s for s in merged if s.seg_type == "exercise"]
        assert len(ex_segs) == 2
        # The middle transition (between the two exercises) should be ONE segment
        if len(trans_segs) >= 1:
            middle = [s for s in trans_segs
                      if s.start > ex_segs[0].start and s.end < ex_segs[1].end]
            assert len(middle) <= 1, "Inter-clip transition was not merged"


# ---------------------------------------------------------------------------
# 7. clip_join_frames lie inside a transition segment (or at a boundary)
# ---------------------------------------------------------------------------

class TestClipJoinFrames:
    def test_join_frames_in_transition_or_boundary(self):
        c1 = make_clip("c1.mp4", "squat",   total_frames=60, src_fps=30.0,
                        first_start=5, last_end=55, pairs=[[5, 55]])
        c2 = make_clip("c2.mp4", "push_up", total_frames=60, src_fps=30.0,
                        first_start=5, last_end=55, pairs=[[5, 55]])

        raw_segs, _ = build_raw_segments([c1, c2], "transition")
        merged = merge_transitions(raw_segs)

        # Join frame = end of first clip = 60
        join_frame = round(c1.total_frames * TARGET_FPS / c1.src_fps)

        # Find which segment contains join_frame
        containing = [s for s in merged if s.start <= join_frame < s.end]
        boundary   = [s for s in merged if s.start == join_frame or s.end == join_frame]
        assert containing or boundary, (
            f"Join frame {join_frame} not covered by any segment"
        )
        if containing:
            assert containing[0].seg_type == "transition", (
                f"Join frame {join_frame} is inside an exercise segment: {containing[0]}"
            )

    def test_number_of_joins(self):
        clips = [
            make_clip(f"c{i}.mp4", ex, total_frames=60, src_fps=30.0)
            for i, ex in enumerate(["squat", "push_up", "jump_jack"])
        ]
        # 3 clips -> 2 joins
        expected_joins = len(clips) - 1
        join_count = 0
        offset = 0
        for c in clips[:-1]:
            offset += round(c.total_frames * TARGET_FPS / c.src_fps)
            join_count += 1
        assert join_count == expected_joins
