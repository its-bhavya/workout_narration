"""
Unit tests for pose_utils.py

Run from repo root:
    python -m pytest tests/test_pose_utils.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_pipeline"))

from pose_utils import (
    angle_deg,
    angle_deg_batch,
    angle_vector_jump,
    best_side_angle,
    fill_short_gaps,
    norm_to_pixels,
)


# ---------------------------------------------------------------------------
# 1. Known geometry – angle_deg
# ---------------------------------------------------------------------------

class TestAngleDeg:
    def test_right_angle(self):
        # Classic L-shape: a=(1,0), b=(0,0), c=(0,1) -> 90°
        assert abs(angle_deg([1, 0], [0, 0], [0, 1]) - 90.0) < 1e-9

    def test_straight_line(self):
        # Points collinear: a=(-1,0), b=(0,0), c=(1,0) -> 180°
        assert abs(angle_deg([-1, 0], [0, 0], [1, 0]) - 180.0) < 1e-9

    def test_zero_angle(self):
        # a and c on same ray from b: a=(1,0), b=(0,0), c=(2,0) -> 0°
        assert abs(angle_deg([1, 0], [0, 0], [2, 0]) - 0.0) < 1e-9

    def test_45_degrees(self):
        # a=(1,0), b=(0,0), c=(1,1) -> 45°
        assert abs(angle_deg([1, 0], [0, 0], [1, 1]) - 45.0) < 1e-6

    def test_3d_right_angle(self):
        # a=(1,0,0), b=(0,0,0), c=(0,1,0) -> 90°
        assert abs(angle_deg([1, 0, 0], [0, 0, 0], [0, 1, 0]) - 90.0) < 1e-9

    def test_nan_on_inf(self):
        result = angle_deg([float("inf"), 0], [0, 0], [1, 0])
        assert math.isnan(result)

    def test_nan_on_nan(self):
        result = angle_deg([float("nan"), 0], [0, 0], [1, 0])
        assert math.isnan(result)

    def test_degenerate_zero_length(self):
        # a == b -> zero-length arm -> NaN
        result = angle_deg([0, 0], [0, 0], [1, 0])
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# 2. Batch angle – angle_deg_batch
# ---------------------------------------------------------------------------

class TestAngleDegBatch:
    def test_single_right_angle(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        c = np.array([[0.0, 1.0]])
        res = angle_deg_batch(a, b, c)
        assert abs(res[0] - 90.0) < 1e-9

    def test_multiple_angles(self):
        # [90°, 180°, 0°]
        a = np.array([[1., 0.], [-1., 0.], [1., 0.]])
        b = np.zeros((3, 2))
        c = np.array([[0., 1.], [1., 0.], [2., 0.]])
        res = angle_deg_batch(a, b, c)
        assert abs(res[0] - 90.0) < 1e-9
        assert abs(res[1] - 180.0) < 1e-9
        assert abs(res[2] - 0.0) < 1e-9

    def test_nan_propagated(self):
        a = np.array([[float("nan"), 0.0]])
        b = np.zeros((1, 2))
        c = np.array([[1.0, 0.0]])
        res = angle_deg_batch(a, b, c)
        assert np.isnan(res[0])

    def test_shape_preserved(self):
        n = 10
        a = np.random.rand(n, 2)
        b = np.random.rand(n, 2)
        c = np.random.rand(n, 2)
        res = angle_deg_batch(a, b, c)
        assert res.shape == (n,)


# ---------------------------------------------------------------------------
# 3. Pixel scaling – norm_to_pixels
# ---------------------------------------------------------------------------

class TestNormToPixels:
    def test_basic_scaling(self):
        pts = np.array([[0.5, 0.5, 0.0]])
        out = norm_to_pixels(pts, width=640, height=360)
        assert abs(out[0, 0] - 320.0) < 1e-9   # x
        assert abs(out[0, 1] - 180.0) < 1e-9   # y
        assert abs(out[0, 2] - 0.0) < 1e-9     # z unchanged

    def test_zero_stays_zero(self):
        pts = np.array([[0.0, 0.0]])
        out = norm_to_pixels(pts, width=800, height=600)
        assert out[0, 0] == 0.0
        assert out[0, 1] == 0.0

    def test_one_maps_to_full_size(self):
        pts = np.array([[1.0, 1.0]])
        out = norm_to_pixels(pts, width=640, height=480)
        assert out[0, 0] == 640.0
        assert out[0, 1] == 480.0

    def test_does_not_modify_original(self):
        pts = np.array([[0.5, 0.5]])
        orig = pts.copy()
        norm_to_pixels(pts, 640, 360)
        np.testing.assert_array_equal(pts, orig)

    def test_2d_angle_distortion_fix(self):
        """Angles computed in pixel space differ from normalised space when
        aspect ratio != 1:1.  Verify that scaling produces the correct 90°
        for a known geometry.
        """
        W, H = 640, 360
        # In normalised: a=(0.5,0.0), b=(0,0), c=(0,0.5) is a right angle at b.
        # After pixel scaling: a=(320,0), b=(0,0), c=(0,180) – still 90°.
        pts = np.array([[0.5, 0.0], [0.0, 0.0], [0.0, 0.5]])
        pix = norm_to_pixels(pts, W, H)
        a, b, c = pix
        ang = angle_deg(a, b, c)
        assert abs(ang - 90.0) < 1e-9


# ---------------------------------------------------------------------------
# 4. Best-side selection
# ---------------------------------------------------------------------------

class TestBestSideAngle:
    def test_right_wins_when_strictly_higher_vis(self):
        left_a = np.array([90.0, 100.0])
        right_a = np.array([85.0, 95.0])
        left_v = np.array([0.6, 0.5])
        right_v = np.array([0.9, 0.8])
        best, best_vis, use_right = best_side_angle(left_a, right_a, left_v, right_v)
        np.testing.assert_array_equal(best, right_a)
        np.testing.assert_array_equal(best_vis, right_v)
        assert use_right.all()

    def test_left_wins_on_tie(self):
        left_a = np.array([90.0])
        right_a = np.array([85.0])
        left_v = np.array([0.7])
        right_v = np.array([0.7])   # equal -> left wins
        best, best_vis, use_right = best_side_angle(left_a, right_a, left_v, right_v)
        assert best[0] == 90.0
        assert not use_right[0]

    def test_left_wins_when_strictly_higher_vis(self):
        left_v = np.array([0.9])
        right_v = np.array([0.5])
        left_a = np.array([45.0])
        right_a = np.array([50.0])
        best, _, use_right = best_side_angle(left_a, right_a, left_v, right_v)
        assert best[0] == 45.0
        assert not use_right[0]

    def test_per_frame_independence(self):
        left_a = np.array([90.0, 80.0])
        right_a = np.array([85.0, 95.0])
        left_v = np.array([0.9, 0.4])
        right_v = np.array([0.5, 0.8])
        best, _, use_right = best_side_angle(left_a, right_a, left_v, right_v)
        assert best[0] == 90.0   # left wins frame 0
        assert best[1] == 95.0   # right wins frame 1

    def test_nan_passthrough(self):
        left_a = np.array([float("nan")])
        right_a = np.array([90.0])
        left_v = np.array([0.3])
        right_v = np.array([0.9])
        best, _, _ = best_side_angle(left_a, right_a, left_v, right_v)
        assert best[0] == 90.0


# ---------------------------------------------------------------------------
# 5. NaN masking
# ---------------------------------------------------------------------------

class TestNaNMasking:
    """Verify that angle_deg_batch returns NaN when any point is NaN."""

    def test_nan_if_a_nan(self):
        a = np.array([[float("nan"), 0.0]])
        b = np.zeros((1, 2))
        c = np.array([[1.0, 0.0]])
        assert np.isnan(angle_deg_batch(a, b, c)[0])

    def test_nan_if_b_nan(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[float("nan"), 0.0]])
        c = np.array([[0.0, 1.0]])
        assert np.isnan(angle_deg_batch(a, b, c)[0])

    def test_nan_if_c_nan(self):
        a = np.array([[1.0, 0.0]])
        b = np.zeros((1, 2))
        c = np.array([[float("nan"), 1.0]])
        assert np.isnan(angle_deg_batch(a, b, c)[0])

    def test_valid_when_all_finite(self):
        a = np.array([[1.0, 0.0]])
        b = np.zeros((1, 2))
        c = np.array([[0.0, 1.0]])
        result = angle_deg_batch(a, b, c)
        assert not np.isnan(result[0])


# ---------------------------------------------------------------------------
# 6. fill_short_gaps
# ---------------------------------------------------------------------------

class TestFillShortGaps:
    def test_no_gap(self):
        s = np.array([1.0, 2.0, 3.0])
        out = fill_short_gaps(s, max_gap_frames=2)
        np.testing.assert_array_almost_equal(out, s)

    def test_fills_single_frame_gap(self):
        s = np.array([0.0, float("nan"), 2.0])
        out = fill_short_gaps(s, max_gap_frames=1)
        assert abs(out[1] - 1.0) < 1e-9

    def test_fills_multi_frame_gap_within_limit(self):
        s = np.array([0.0, float("nan"), float("nan"), 3.0])
        out = fill_short_gaps(s, max_gap_frames=2)
        assert abs(out[1] - 1.0) < 1e-9
        assert abs(out[2] - 2.0) < 1e-9

    def test_does_not_fill_gap_beyond_limit(self):
        s = np.array([0.0, float("nan"), float("nan"), float("nan"), 4.0])
        # 3-frame gap, max_gap=2 -> leave as NaN
        out = fill_short_gaps(s, max_gap_frames=2)
        assert np.isnan(out[1])
        assert np.isnan(out[2])
        assert np.isnan(out[3])

    def test_does_not_fill_leading_nan(self):
        s = np.array([float("nan"), 1.0, 2.0])
        out = fill_short_gaps(s, max_gap_frames=2)
        assert np.isnan(out[0])

    def test_does_not_fill_trailing_nan(self):
        s = np.array([1.0, 2.0, float("nan")])
        out = fill_short_gaps(s, max_gap_frames=2)
        assert np.isnan(out[2])

    def test_max_gap_exactly_filled(self):
        s = np.array([0.0, float("nan"), float("nan"), 6.0])
        out = fill_short_gaps(s, max_gap_frames=2)
        assert abs(out[1] - 2.0) < 1e-9
        assert abs(out[2] - 4.0) < 1e-9

    def test_original_not_mutated(self):
        s = np.array([1.0, float("nan"), 3.0])
        orig = s.copy()
        fill_short_gaps(s, max_gap_frames=1)
        np.testing.assert_array_equal(s, orig)

    def test_all_nan_returns_all_nan(self):
        s = np.array([float("nan"), float("nan"), float("nan")])
        out = fill_short_gaps(s, max_gap_frames=5)
        assert np.isnan(out).all()

    def test_zero_max_gap_fills_nothing(self):
        s = np.array([1.0, float("nan"), 3.0])
        out = fill_short_gaps(s, max_gap_frames=0)
        assert np.isnan(out[1])


# ---------------------------------------------------------------------------
# 7. angle_vector_jump
# ---------------------------------------------------------------------------

class TestAngleVectorJump:
    def test_identical_vectors(self):
        v = np.array([90.0, 45.0, 180.0, 30.0, 60.0, 10.0])
        assert angle_vector_jump(v, v) == 0.0

    def test_known_jump(self):
        # v_prev=[0,0,0,0,0,0], v_curr=[3,4,0,0,0,0] -> sqrt(9+16)=5
        v_prev = np.zeros(6)
        v_curr = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        assert abs(angle_vector_jump(v_prev, v_curr) - 5.0) < 1e-9

    def test_nan_components_skipped(self):
        v_prev = np.array([0.0, 0.0, float("nan"), 0.0, 0.0, 0.0])
        v_curr = np.array([3.0, 4.0, 99.0, 0.0, 0.0, 0.0])
        # NaN at position 2 is ignored; jump = sqrt(9+16+0+0+0) = 5
        result = angle_vector_jump(v_prev, v_curr)
        assert abs(result - 5.0) < 1e-9

    def test_returns_nan_below_min_components(self):
        # Only 2 valid components, min_components=3 -> NaN
        v_prev = np.array([0.0, 0.0, float("nan"), float("nan"), float("nan"), float("nan")])
        v_curr = np.array([1.0, 1.0, float("nan"), float("nan"), float("nan"), float("nan")])
        result = angle_vector_jump(v_prev, v_curr, min_components=3)
        assert math.isnan(result)

    def test_exactly_min_components(self):
        # Exactly 3 valid components
        v_prev = np.array([0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan")])
        v_curr = np.array([1.0, 0.0, 0.0, float("nan"), float("nan"), float("nan")])
        result = angle_vector_jump(v_prev, v_curr, min_components=3)
        assert abs(result - 1.0) < 1e-9

    def test_nan_in_curr_also_skipped(self):
        v_prev = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        v_curr = np.array([1.0, 1.0, float("nan"), 0.0, 0.0, 0.0])
        # Position 2 is NaN in curr -> excluded; remaining 5 components all equal -> 0
        result = angle_vector_jump(v_prev, v_curr, min_components=3)
        assert abs(result - 0.0) < 1e-9
