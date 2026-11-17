"""Tests for metrics.py

Run from repo root::

    python -m pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "segmentation"))

from metrics import (  # noqa: E402
    _levenshtein,
    all_metrics,
    boundary_errors,
    edit_score,
    f1_at_iou,
    frame_accuracy,
    frame_f1_per_class,
    gt_from_json,
    segment_counts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _seg(label, start, end):
    return {"exercise": label, "start_time": float(start), "end_time": float(end)}


# Perfect two-class example (0-5 squat, 5-10 push_up)
GT2 = [_seg("squat", 0, 5), _seg("push_up", 5, 10)]
PR_PERFECT = [_seg("squat", 0, 5), _seg("push_up", 5, 10)]
PR_SWAPPED = [_seg("push_up", 0, 5), _seg("squat", 5, 10)]
PR_ONE = [_seg("squat", 0, 10)]  # one long squat, no push_up
VE = 10.0


# ---------------------------------------------------------------------------
# Levenshtein
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_equal(self):
        assert _levenshtein("abc", "abc") == 0

    def test_empty(self):
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3

    def test_one_edit(self):
        assert _levenshtein("abc", "axc") == 1

    def test_list_elements(self):
        assert _levenshtein(["squat", "push_up"], ["squat", "squat"]) == 1


# ---------------------------------------------------------------------------
# frame_accuracy
# ---------------------------------------------------------------------------

class TestFrameAccuracy:
    def test_perfect(self):
        assert frame_accuracy(GT2, PR_PERFECT, VE) == pytest.approx(1.0)

    def test_fully_wrong(self):
        # Swapped: every grid point has wrong label.
        acc = frame_accuracy(GT2, PR_SWAPPED, VE)
        assert acc == pytest.approx(0.0, abs=0.01)

    def test_half_correct(self):
        # PR_ONE labels everything squat; GT is 50% squat, 50% push_up.
        acc = frame_accuracy(GT2, PR_ONE, VE)
        assert 0.49 < acc < 0.51

    def test_range(self):
        acc = frame_accuracy(GT2, PR_PERFECT, VE)
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# frame_f1_per_class
# ---------------------------------------------------------------------------

class TestFrameF1:
    def test_perfect(self):
        f1 = frame_f1_per_class(GT2, PR_PERFECT, VE, ["squat", "push_up"])
        assert f1["squat"] == pytest.approx(1.0)
        assert f1["push_up"] == pytest.approx(1.0)

    def test_missing_class(self):
        # PR_ONE predicts only squat; push_up F1 should be 0.
        f1 = frame_f1_per_class(GT2, PR_ONE, VE, ["squat", "push_up"])
        assert f1["push_up"] == pytest.approx(0.0)
        assert f1["squat"] > 0.0


# ---------------------------------------------------------------------------
# edit_score
# ---------------------------------------------------------------------------

class TestEditScore:
    def test_perfect(self):
        assert edit_score(GT2, PR_PERFECT) == pytest.approx(100.0)

    def test_fully_wrong(self):
        # Both have 2 segments but all labels differ -> lev=2, max=2 -> score=0
        assert edit_score(GT2, PR_SWAPPED) == pytest.approx(0.0)

    def test_one_extra_segment(self):
        # PR has 3 segments vs GT 2 -> lev=1, max=3 -> score ~ 66.67
        pr3 = [_seg("squat", 0, 3), _seg("push_up", 3, 7), _seg("squat", 7, 10)]
        score = edit_score(GT2, pr3)
        assert score == pytest.approx((1 - 1 / 3) * 100, abs=0.1)

    def test_empty(self):
        assert edit_score([], []) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# f1_at_iou
# ---------------------------------------------------------------------------

class TestF1AtIoU:
    def test_perfect_match(self):
        # Exact overlap, same labels.
        assert f1_at_iou(GT2, PR_PERFECT, 0.5) == pytest.approx(1.0)

    def test_no_match_wrong_label(self):
        # Wrong labels -> IoU matches never made.
        assert f1_at_iou(GT2, PR_SWAPPED, 0.5) == pytest.approx(0.0)

    def test_partial_overlap_above_threshold(self):
        # Predicted squat [0,4] vs GT squat [0,5]: IoU = 4/5 = 0.8 >= 0.5 -> match.
        gt = [_seg("squat", 0, 5)]
        pr = [_seg("squat", 0, 4)]
        assert f1_at_iou(gt, pr, 0.5) == pytest.approx(1.0)

    def test_partial_overlap_below_threshold(self):
        # Predicted squat [0,2] vs GT squat [0,5]: IoU = 2/5 = 0.4 < 0.5 -> no match.
        gt = [_seg("squat", 0, 5)]
        pr = [_seg("squat", 0, 2)]
        assert f1_at_iou(gt, pr, 0.5) == pytest.approx(0.0)

    def test_each_gt_matched_once(self):
        # One GT, two predictions overlapping it; only one match should be made.
        gt = [_seg("squat", 0, 10)]
        pr = [_seg("squat", 0, 8), _seg("squat", 2, 10)]
        # tp=1, fp=1, fn=0 -> F1 = 2/(2+1+0) = 2/3
        assert f1_at_iou(gt, pr, 0.5) == pytest.approx(2 / 3, abs=0.01)


# ---------------------------------------------------------------------------
# boundary_errors
# ---------------------------------------------------------------------------

class TestBoundaryErrors:
    def test_perfect(self):
        be = boundary_errors(GT2, PR_PERFECT)
        assert be["mean_s"] == pytest.approx(0.0)
        assert be["median_s"] == pytest.approx(0.0)

    def test_off_by_one(self):
        # GT boundary at 5.0, PR boundary at 6.0 -> error 1.0.
        pr = [_seg("squat", 0, 6), _seg("push_up", 6, 10)]
        be = boundary_errors(GT2, pr)
        assert be["mean_s"] == pytest.approx(1.0)
        assert be["median_s"] == pytest.approx(1.0)

    def test_no_gt_boundaries(self):
        # Single GT segment -> no internal boundaries.
        be = boundary_errors([_seg("squat", 0, 10)], PR_PERFECT)
        assert be["mean_s"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# segment_counts
# ---------------------------------------------------------------------------

class TestSegmentCounts:
    def test_counts(self):
        c = segment_counts(GT2, [_seg("squat", 0, 10)])
        assert c == {"n_gt": 2, "n_pred": 1}


# ---------------------------------------------------------------------------
# all_metrics
# ---------------------------------------------------------------------------

class TestAllMetrics:
    def test_keys_present(self):
        m = all_metrics(GT2, PR_PERFECT, VE, ["squat", "push_up"])
        for key in ["frame_accuracy", "edit_score", "f1_at_10", "f1_at_25",
                    "f1_at_50", "boundary_mean_s", "boundary_median_s",
                    "n_pred", "n_gt"]:
            assert key in m, key

    def test_perfect_scores(self):
        m = all_metrics(GT2, PR_PERFECT, VE, ["squat", "push_up"])
        assert m["frame_accuracy"] == pytest.approx(1.0)
        assert m["edit_score"] == pytest.approx(100.0)
        assert m["f1_at_50"] == pytest.approx(1.0)
        assert m["boundary_mean_s"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# gt_from_json
# ---------------------------------------------------------------------------

class TestGtFromJson:
    def _make_meta(self):
        return {
            "fps": 30,
            "n_frames": 300,
            "segments": [
                {"label": "transition", "start_frame": 0, "end_frame": 9},
                {"label": "squat", "start_frame": 9, "end_frame": 291},
                {"label": "transition", "start_frame": 291, "end_frame": 300},
            ],
        }

    def test_video_end(self):
        meta = self._make_meta()
        segs, ve = gt_from_json(meta)
        assert ve == pytest.approx(10.0)

    def test_segments_cover_full_range(self):
        meta = self._make_meta()
        segs, ve = gt_from_json(meta)
        assert segs[0]["start_time"] == pytest.approx(0.0)
        assert segs[-1]["end_time"] == pytest.approx(ve)

    def test_labels(self):
        meta = self._make_meta()
        segs, _ = gt_from_json(meta)
        assert [s["exercise"] for s in segs] == ["transition", "squat", "transition"]
