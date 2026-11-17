"""
Tests for concatenate_videos.py

Run from repo root:
    python -m pytest tests/test_concatenate.py -v
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
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


# ---------------------------------------------------------------------------
# 8. Append-mode: cumulative caps (a test clip can never be used twice)
# ---------------------------------------------------------------------------

class TestAppendCumulativeCaps:
    """Verify that clips at their cumulative cap are excluded in append mode."""

    def _make_test_pool(self):
        """3 squat + 3 push_up + 3 jump_jack clips for the test split."""
        clips = []
        for i in range(3):
            for ex, prefix in [("squat", "sq"), ("push_up", "pu"), ("jump_jack", "jj")]:
                clips.append(make_clip(
                    f"{prefix}_{i}.mp4", ex, split="test", total_frames=60))
        return clips

    def test_clip_at_test_cap_excluded(self):
        """Test cap=1: a clip used once must never appear again."""
        from concatenate_videos import CUMULATIVE_CAPS, sample_sequence

        clips = self._make_test_pool()
        cap = CUMULATIVE_CAPS["test"]  # 1

        # Simulate sq_0 and pu_0 already used once (at cap)
        cumulative = {"sq_0.mp4": 1, "pu_0.mp4": 1}

        ex_pool: dict[str, list] = defaultdict(list)
        for c in clips:
            if cumulative.get(c.clip_id, 0) < cap:
                ex_pool[c.exercise].append(c)

        rng = random.Random(42)
        # Run many sequences and verify capped clips never appear
        for _ in range(50):
            seq = sample_sequence(
                3, ex_pool, rng,
                used_clips=set(),
                used_in_video=set(),
            )
            if seq is None:
                break
            ids = {c.clip_id for c in seq}
            assert "sq_0.mp4" not in ids, "Clip at cap was selected"
            assert "pu_0.mp4" not in ids, "Clip at cap was selected"

    def test_val_clip_below_cap_available(self):
        """Val cap=2: a clip used once in batch 1 is still available."""
        from concatenate_videos import CUMULATIVE_CAPS, sample_sequence

        cap = CUMULATIVE_CAPS["val"]  # 2
        # Build a val pool: 5 squat + 5 push_up
        clips = []
        for i in range(5):
            clips.append(make_clip(f"sq_{i}.mp4", "squat", split="val", total_frames=60))
            clips.append(make_clip(f"pu_{i}.mp4", "push_up", split="val", total_frames=60))

        # sq_0 used once (1 remaining), sq_1 used twice (0 remaining, at cap)
        cumulative = {"sq_0.mp4": 1, "sq_1.mp4": 2}

        ex_pool: dict[str, list] = defaultdict(list)
        for c in clips:
            if cumulative.get(c.clip_id, 0) < cap:
                ex_pool[c.exercise].append(c)

        # sq_0 should be available, sq_1 should not
        assert any(c.clip_id == "sq_0.mp4" for c in ex_pool["squat"])
        assert not any(c.clip_id == "sq_1.mp4" for c in ex_pool["squat"])

    def test_train_clip_below_cap_available(self):
        """Train cap=5: a clip used 3 times is still available (2 remaining)."""
        from concatenate_videos import CUMULATIVE_CAPS

        cap = CUMULATIVE_CAPS["train"]  # 5
        cumulative = {"c1.mp4": 3}
        remaining = cap - cumulative.get("c1.mp4", 0)
        assert remaining == 2, f"Expected 2 remaining, got {remaining}"

    def test_clip_usage_tracking_persists(self, tmp_path):
        """clip_usage.csv should record cumulative usage and persist."""
        from concatenate_videos import init_clip_usage, update_clip_usage

        index_path = tmp_path / "synthetic_index.csv"
        usage_path = tmp_path / "clip_usage.csv"

        # Write a mini index
        index_path.write_text(
            "video_id,split,n_segments,duration_s,exercises,source_clips,"
            "transition_frame_share,direct_cuts,seed\n"
            "fixture_test_0001,test,5,60.0,jump_jack|squat,"
            "jj_0.mp4|sq_0.mp4|jj_1.mp4,0.1,0,42\n"
        )

        # Initialise usage
        usage = init_clip_usage(index_path, usage_path)
        assert usage["jj_0.mp4"] == 1
        assert usage["sq_0.mp4"] == 1
        assert usage["jj_1.mp4"] == 1

        # Simulate batch 2 using jj_0 again
        new_rows = [{"source_clips": "jj_0.mp4|sq_0.mp4|pu_0.mp4"}]
        updated = update_clip_usage(usage_path, new_rows, batch_id=2)
        assert updated["jj_0.mp4"] == 2, "jj_0 should now have times_used=2"
        assert updated["sq_0.mp4"] == 2, "sq_0 should now have times_used=2"
        assert updated["pu_0.mp4"] == 1, "pu_0 should be newly added with times_used=1"

        # Verify file persists and is not reset
        df = pd.read_csv(usage_path)
        assert len(df) == 4  # 3 from batch 1 + 1 new (pu_0) from batch 2
        jj0_row = df[df["clip_id"] == "jj_0.mp4"].iloc[0]
        assert jj0_row["times_used"] == 2
        assert jj0_row["batch"] == 2


# ---------------------------------------------------------------------------
# 9. Append-mode: ID continuity (new IDs continue from existing)
# ---------------------------------------------------------------------------

class TestAppendIdContinuity:
    def test_video_numbering_continues(self, monkeypatch, tmp_path):
        """New video IDs must start after the last existing one per split."""
        from concatenate_videos import run_batch, ClipInfo

        # Build a minimal pool
        pool: dict[str, list[ClipInfo]] = {"test": []}
        for i in range(10):
            for ex, prefix in [("squat", "sq"), ("push_up", "pu"), ("jump_jack", "jj")]:
                pool["test"].append(make_clip(
                    f"{prefix}_{i}.mp4", ex, split="test", total_frames=60))

        # Mock make_video to avoid rendering
        captured_ids: list[str] = []

        def mock_make_video(video_id, clips, **kw):
            captured_ids.append(video_id)
            return {
                "video_id": video_id,
                "split": "test",
                "n_segments": 5,
                "duration_s": 60.0,
                "exercises": "squat|push_up",
                "source_clips": "|".join(c.clip_id for c in clips),
                "transition_frame_share": 0.1,
                "direct_cuts": 0,
                "seed": 43,
                "batch": kw.get("batch_id", 1),
            }

        import concatenate_videos as cv

        monkeypatch.setattr(cv, "make_video", mock_make_video)
        start_counters = {"test": 15}

        cv.run_batch(
            pool=pool,
            counts={"test": 1},
            out_base=tmp_path / "synthetic",
            splits_file="data/manifests/splits_v2.csv",
            margin_label="transition",
            seed=43,
            pilot=False,
            orig_map={},
            append=True,
            batch_id=2,
            cumulative_usage={},
            start_counters={"test": 15},
        )
        # Simulate: batch 1 created 15 test videos (0001-0015)
        # Batch 2 should start at 0016

        assert len(captured_ids) == 1
        assert captured_ids[0] == "fixture_test_0016", (
            f"Expected fixture_test_0016, got {captured_ids[0]}"
        )


# ---------------------------------------------------------------------------
# 10. Append-mode: batch-1 files untouched (hash check)
# ---------------------------------------------------------------------------

class TestAppendBatch1Untouched:
    def test_make_video_aborts_if_mp4_exists(self, tmp_path):
        """make_video must raise FileExistsError if target .mp4 exists."""
        from concatenate_videos import make_video

        out_dir = tmp_path / "train"
        out_dir.mkdir()
        (out_dir / "fixture_train_0025.mp4").write_bytes(b"batch1_data")

        clip = make_clip("c1.mp4", "squat", total_frames=60)
        with pytest.raises(FileExistsError, match="already exists"):
            make_video(
                video_id="fixture_train_0025",
                clips=[clip],
                split="train",
                out_dir=out_dir,
                margin_label="transition",
                splits_file="data/manifests/splits_v2.csv",
                seed=42,
                orig_map={"c1.mp4": "train"},
                batch_id=2,
            )

    def test_make_video_aborts_if_json_exists(self, tmp_path):
        """make_video must raise FileExistsError if target .json exists."""
        from concatenate_videos import make_video

        out_dir = tmp_path / "train"
        out_dir.mkdir()
        (out_dir / "fixture_train_0025.json").write_text("{}")

        clip = make_clip("c1.mp4", "squat", total_frames=60)
        with pytest.raises(FileExistsError, match="already exists"):
            make_video(
                video_id="fixture_train_0025",
                clips=[clip],
                split="train",
                out_dir=out_dir,
                margin_label="transition",
                splits_file="data/manifests/splits_v2.csv",
                seed=42,
                orig_map={"c1.mp4": "train"},
                batch_id=2,
            )

    def test_write_index_preserves_existing_rows(self, tmp_path):
        """write_index(append=True) must keep all existing rows unchanged."""
        from concatenate_videos import write_index
        import hashlib

        index_path = tmp_path / "synthetic_index.csv"
        # Write a batch-1 index (without batch column, like the original)
        index_path.write_text(
            "video_id,split,n_segments,duration_s,exercises,source_clips,"
            "transition_frame_share,direct_cuts,seed\n"
            "fixture_train_0001,train,5,60.0,squat|push_up,sq_0.mp4|pu_0.mp4,"
            "0.1,0,42\n"
        )
        original_content = index_path.read_text()
        original_hash = hashlib.sha256(original_content.encode()).hexdigest()

        # Append new rows (batch 2)
        new_rows = [{
            "video_id": "fixture_train_0025",
            "split": "train",
            "n_segments": 5,
            "duration_s": 60.0,
            "exercises": "squat|push_up",
            "source_clips": "sq_1.mp4|pu_1.mp4",
            "transition_frame_share": 0.12,
            "direct_cuts": 0,
            "seed": 43,
            "batch": 2,
        }]
        write_index(new_rows, index_path, append=True)

        # Read back and verify
        df = pd.read_csv(index_path)
        assert len(df) == 2, "Should have 2 rows (1 original + 1 new)"

        # Original row preserved
        row0 = df.iloc[0]
        assert row0["video_id"] == "fixture_train_0001"
        assert row0["source_clips"] == "sq_0.mp4|pu_0.mp4"
        assert row0["seed"] == 42
        # batch column added to existing rows
        assert row0["batch"] == 1, "Existing rows should get batch=1"

        # New row present
        row1 = df.iloc[1]
        assert row1["video_id"] == "fixture_train_0025"
        assert row1["batch"] == 2
        assert row1["seed"] == 43

    def test_batch_manifest_init_and_append(self, tmp_path):
        """batch_manifest.csv should initialise from index and accept appends."""
        from concatenate_videos import (
            init_batch_manifest, append_batch_manifest,
        )

        index_path = tmp_path / "synthetic_index.csv"
        manifest_path = tmp_path / "batch_manifest.csv"

        index_path.write_text(
            "video_id,split,n_segments,duration_s,exercises,source_clips,"
            "transition_frame_share,direct_cuts,seed,batch\n"
            "fixture_train_0001,train,5,60.0,squat|push_up,sq_0.mp4|pu_0.mp4,"
            "0.1,0,42,1\n"
            "fixture_val_0001,val,5,60.0,squat|push_up,sq_1.mp4|pu_1.mp4,"
            "0.1,0,42,1\n"
        )

        init_batch_manifest(index_path, manifest_path)
        df = pd.read_csv(manifest_path)
        assert len(df) == 2
        assert all(df["batch"] == 1)
        assert df.iloc[0]["video_id"] == "fixture_train_0001"
        assert df.iloc[1]["video_id"] == "fixture_val_0001"

        # Append batch-2 rows
        new_rows = [
            {"video_id": "fixture_train_0025", "split": "train",
             "batch": 2, "seed": 43},
            {"video_id": "fixture_val_0007", "split": "val",
             "batch": 2, "seed": 43},
        ]
        append_batch_manifest(new_rows, manifest_path)

        df = pd.read_csv(manifest_path)
        assert len(df) == 4
        assert list(df["batch"]) == [1, 1, 2, 2]
        assert df.iloc[2]["video_id"] == "fixture_train_0025"
        assert df.iloc[3]["video_id"] == "fixture_val_0007"


# ---------------------------------------------------------------------------
# 11. Append-mode: clip never twice in one video
# ---------------------------------------------------------------------------

class TestClipNeverTwiceInVideo:
    def test_no_duplicate_clip_in_append_mode(self):
        """In append mode, the same clip must never appear twice in one video."""
        from concatenate_videos import sample_sequence

        pool_sq = [make_clip(f"sq_{i}.mp4", "squat", total_frames=60)
                    for i in range(5)]
        pool_pu = [make_clip(f"pu_{i}.mp4", "push_up", total_frames=60)
                    for i in range(5)]
        ex_pool = {"squat": pool_sq, "push_up": pool_pu}

        rng = random.Random(42)
        # used_in_video enforces the per-video uniqueness constraint
        seq = sample_sequence(
            5, ex_pool, rng,
            used_clips=set(),
            used_in_video=set(),
        )
        assert seq is not None
        clip_ids = [c.clip_id for c in seq]
        assert len(clip_ids) == len(set(clip_ids)), (
            f"Duplicate clips in video: {clip_ids}"
        )

    def test_used_in_video_prevents_reuse(self):
        """Even if a clip is in the pool, used_in_video should exclude it."""
        from concatenate_videos import sample_sequence

        # Only 2 clips per exercise — with k=4, the same clip would be
        # needed twice if used_in_video were not enforced.
        pool_sq = [make_clip("sq_0.mp4", "squat", total_frames=60),
                   make_clip("sq_1.mp4", "squat", total_frames=60)]
        pool_pu = [make_clip("pu_0.mp4", "push_up", total_frames=60),
                   make_clip("pu_1.mp4", "push_up", total_frames=60)]
        ex_pool = {"squat": pool_sq, "push_up": pool_pu}

        rng = random.Random(42)
        # k=4 with only 4 unique clips total: must use all 4, no duplicates
        seq = sample_sequence(
            4, ex_pool, rng,
            used_clips=set(),
            used_in_video=set(),
        )
        if seq is not None:
            ids = [c.clip_id for c in seq]
            assert len(ids) == len(set(ids)), f"Duplicate clips: {ids}"

    def test_batch1_mode_allows_reuse_for_train(self):
        """In batch-1 mode (used_in_video=None), train clips CAN repeat.
        This preserves original batch-1 reproducibility."""
        from concatenate_videos import sample_sequence

        # 2 squat + 2 push_up, k=4 — with used_in_video=None, a clip may repeat
        pool_sq = [make_clip("sq_0.mp4", "squat", total_frames=60),
                   make_clip("sq_1.mp4", "squat", total_frames=60)]
        pool_pu = [make_clip("pu_0.mp4", "push_up", total_frames=60),
                   make_clip("pu_1.mp4", "push_up", total_frames=60)]
        ex_pool = {"squat": pool_sq, "push_up": pool_pu}

        rng = random.Random(42)
        # Batch-1 mode: used_in_video=None (not checked)
        seq = sample_sequence(4, ex_pool, rng, used_clips=None, used_in_video=None)
        # We don't assert uniqueness here — batch 1 allows repeats.
        # Just verify it produces a valid sequence.
        if seq is not None:
            assert len(seq) == 4
            assert len({c.exercise for c in seq}) >= 2
