"""Tests for smooth.py

Run from repo root::

    python -m pytest tests/test_smooth.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "segmentation"))

from smooth import smooth, to_segments  # noqa: E402


# ---------------------------------------------------------------------------
# Local helpers (kept independent of smooth.py internals)
# ---------------------------------------------------------------------------

def _runs(labels):
    """Maximal runs as ``(label, start, end_exclusive)``."""
    runs = []
    i = 0
    n = len(labels)
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append((labels[i], i, j))
        i = j
    return runs


def _assert_tile(segs, video_end_s):
    """Segments must tile [0, video_end_s] with no gaps or overlaps."""
    assert segs, "expected at least one segment"
    assert segs[0]["start_time"] == pytest.approx(0.0)
    assert segs[-1]["end_time"] == pytest.approx(video_end_s)
    for a, b in zip(segs, segs[1:]):
        assert a["end_time"] == pytest.approx(b["start_time"]), (a, b)
    for s in segs:
        assert s["start_time"] <= s["end_time"]


# ---------------------------------------------------------------------------
# smooth: low-confidence step
# ---------------------------------------------------------------------------

class TestLowConfidence:
    def test_low_conf_replaced_with_previous(self):
        # Window 2 is low-confidence -> takes window 1's label.
        labels = ["squat", "squat", "push_up", "push_up"]
        confs = [0.9, 0.9, 0.2, 0.9]
        out = smooth(labels, confs, tau=0.5, min_len=1)
        assert out == ["squat", "squat", "squat", "push_up"]

    def test_first_window_keeps_own_even_if_low_conf(self):
        labels = ["push_up", "squat", "squat"]
        confs = [0.1, 0.9, 0.9]
        out = smooth(labels, confs, tau=0.5, min_len=1)
        # First window keeps push_up despite conf 0.1.
        assert out[0] == "push_up"

    def test_streak_of_low_conf_forward_fills(self):
        labels = ["squat", "push_up", "jump_jack", "jump_jack"]
        confs = [0.9, 0.1, 0.1, 0.1]
        out = smooth(labels, confs, tau=0.5, min_len=1)
        # All after the first inherit the first confident label.
        assert out == ["squat", "squat", "squat", "squat"]

    def test_conf_equal_to_tau_is_kept(self):
        # conf == tau is NOT low (strict <), so the label is kept.
        labels = ["squat", "push_up"]
        confs = [0.9, 0.5]
        out = smooth(labels, confs, tau=0.5, min_len=1)
        assert out == ["squat", "push_up"]


# ---------------------------------------------------------------------------
# smooth: min-length step
# ---------------------------------------------------------------------------

class TestMinLen:
    def test_short_run_absorbed_into_previous(self):
        labels = ["squat", "squat", "push_up", "squat", "squat", "squat"]
        confs = [0.9] * 6
        out = smooth(labels, confs, tau=0.5, min_len=2)
        # Middle push_up (len 1) absorbed into the previous squat run.
        assert out == ["squat"] * 6

    def test_first_run_short_absorbed_into_next(self):
        labels = ["push_up", "squat", "squat", "squat"]
        confs = [0.9] * 4
        out = smooth(labels, confs, tau=0.5, min_len=2)
        assert out == ["squat"] * 4

    def test_min_len_one_is_noop(self):
        labels = ["squat", "push_up", "squat"]
        confs = [0.9] * 3
        out = smooth(labels, confs, tau=0.5, min_len=1)
        assert out == ["squat", "push_up", "squat"]

    def test_all_too_short_collapses_to_one_run(self):
        labels = ["a", "b", "c"]
        confs = [0.9] * 3
        out = smooth(labels, confs, tau=0.5, min_len=10)
        # Everything collapses to a single run; length is preserved.
        assert len(set(out)) == 1
        assert len(out) == 3


# ---------------------------------------------------------------------------
# smooth: flickery sequence becomes clean
# ---------------------------------------------------------------------------

class TestFlickery:
    def test_flickery_via_confidence(self):
        # squat squat push_up squat push_up push_up (3rd window is a flicker)
        labels = ["squat", "squat", "push_up", "squat", "push_up", "push_up"]
        confs = [0.9, 0.9, 0.2, 0.9, 0.9, 0.9]
        out = smooth(labels, confs, tau=0.5, min_len=1)
        # Clean: one squat run then one push_up run.
        assert _runs(out) == [("squat", 0, 4), ("push_up", 4, 6)]

    def test_flickery_via_min_len(self):
        labels = ["squat", "squat", "push_up", "squat", "push_up", "push_up"]
        confs = [0.9] * 6
        out = smooth(labels, confs, tau=0.5, min_len=2)
        assert _runs(out) == [("squat", 0, 4), ("push_up", 4, 6)]

    def test_length_unchanged(self):
        labels = ["squat", "squat", "push_up", "squat", "push_up", "push_up"]
        confs = [0.9, 0.9, 0.2, 0.9, 0.9, 0.9]
        out = smooth(labels, confs, tau=0.5, min_len=2)
        assert len(out) == len(labels)

    def test_deterministic(self):
        labels = ["squat", "squat", "push_up", "squat", "push_up", "push_up"]
        confs = [0.9, 0.9, 0.2, 0.9, 0.9, 0.9]
        a = smooth(labels, confs, tau=0.5, min_len=2)
        b = smooth(labels, confs, tau=0.5, min_len=2)
        assert a == b


# ---------------------------------------------------------------------------
# to_segments
# ---------------------------------------------------------------------------

class TestToSegments:
    def test_merges_consecutive_identical(self):
        labels = ["squat", "squat", "push_up", "push_up"]
        t_centers = [1.0, 2.0, 3.0, 4.0]
        segs = to_segments(labels, t_centers, video_end_s=5.0)
        assert len(segs) == 2
        assert [s["exercise"] for s in segs] == ["squat", "push_up"]

    def test_tiles_zero_to_end_no_gaps_overlaps(self):
        labels = ["squat", "squat", "push_up", "jump_jack", "jump_jack"]
        t_centers = [1.0, 2.0, 3.0, 4.0, 5.0]
        segs = to_segments(labels, t_centers, video_end_s=6.0)
        _assert_tile(segs, 6.0)

    def test_first_segment_starts_at_zero(self):
        labels = ["squat", "push_up"]
        t_centers = [1.0, 2.0]
        segs = to_segments(labels, t_centers, video_end_s=3.0)
        assert segs[0]["start_time"] == pytest.approx(0.0)

    def test_last_segment_ends_at_video_end(self):
        labels = ["squat", "push_up"]
        t_centers = [1.0, 2.0]
        segs = to_segments(labels, t_centers, video_end_s=3.0)
        assert segs[-1]["end_time"] == pytest.approx(3.0)

    def test_boundary_is_midpoint_of_centers(self):
        labels = ["squat", "push_up"]
        t_centers = [1.0, 2.0]
        segs = to_segments(labels, t_centers, video_end_s=3.0)
        # Boundary between the two windows is the midpoint of centers 1.0, 2.0.
        assert segs[0]["end_time"] == pytest.approx(1.5)
        assert segs[1]["start_time"] == pytest.approx(1.5)

    def test_transition_is_normal_label(self):
        labels = ["squat", "transition", "push_up"]
        t_centers = [1.0, 2.0, 3.0]
        segs = to_segments(labels, t_centers, video_end_s=4.0)
        assert [s["exercise"] for s in segs] == ["squat", "transition", "push_up"]
        _assert_tile(segs, 4.0)

    def test_single_window(self):
        segs = to_segments(["squat"], [1.5], video_end_s=3.0)
        assert segs == [{"exercise": "squat", "start_time": 0.0, "end_time": 3.0}]

    def test_empty(self):
        assert to_segments([], [], video_end_s=5.0) == []

    def test_keys_present(self):
        segs = to_segments(["squat", "push_up"], [1.0, 2.0], video_end_s=3.0)
        for s in segs:
            assert set(s.keys()) == {"exercise", "start_time", "end_time"}

    def test_deterministic(self):
        labels = ["squat", "squat", "push_up", "squat", "push_up", "push_up"]
        t_centers = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        a = to_segments(labels, t_centers, video_end_s=7.0)
        b = to_segments(labels, t_centers, video_end_s=7.0)
        assert a == b


# ---------------------------------------------------------------------------
# End-to-end: smooth then to_segments
# ---------------------------------------------------------------------------

class TestSmoothThenSegments:
    def test_flickery_smooths_then_tiles(self):
        labels = ["squat", "squat", "push_up", "squat", "push_up", "push_up"]
        confs = [0.9, 0.9, 0.2, 0.9, 0.9, 0.9]
        smoothed = smooth(labels, confs, tau=0.5, min_len=2)
        t_centers = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        segs = to_segments(smoothed, t_centers, video_end_s=7.0)
        # Clean: exactly two segments, squat then push_up, tiling [0, 7].
        assert [s["exercise"] for s in segs] == ["squat", "push_up"]
        _assert_tile(segs, 7.0)
