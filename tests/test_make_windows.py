"""Tests for make_windows.py

Run from repo root::

    python -m pytest tests/test_make_windows.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "segmentation"))

from make_windows import make_windows_for_video  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_csv(path: Path, duration_s: float = 10.0, dt: float = 0.1,
              boundary_s: float = 5.0) -> None:
    """Write a fake angle CSV: ``squat`` then ``push_up`` around ``boundary_s``.

    Includes the metadata columns that must never be used as features
    (frame, time_s, seg_type, seg_label, *_vis) plus a dummy angle column, so
    the file looks like a real extraction output.
    """
    frames = np.arange(0.0, duration_s + 1e-9, dt)
    rows = []
    for i, t in enumerate(frames):
        label = "squat" if t < boundary_s else "push_up"
        rows.append({
            "frame": i * 2,            # stride-2 source frame index
            "time_s": round(float(t), 6),
            "pose_detected": True,
            "n_poses": 1,
            "subject_jump": False,
            "seg_type": "exercise",
            "seg_label": label,
            "knee_best": 160.0 + i,     # dummy angle
            "knee_best_vis": 0.9,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMakeWindows:
    def test_fake_10s_two_segments(self, tmp_path):
        csv_path = tmp_path / "synth_test_9999_angles.csv"
        _fake_csv(csv_path, duration_s=10.0, dt=0.1, boundary_s=5.0)

        df = pd.read_csv(csv_path)
        wins = make_windows_for_video(df, "synth_test_9999", "test")

        # 10 s, window 2.0 s, stride 1.0 s -> starts at 0,1,...,8 -> 9 windows.
        assert len(wins) == 9

        # No window crosses the end: t_end must not exceed the last timestamp.
        last_t = float(df["time_s"].max())  # 10.0
        assert all(w["t_end"] <= last_t + 1e-6 for w in wins)
        assert max(w["t_end"] for w in wins) == pytest.approx(10.0, abs=1e-6)

        # win_idx is contiguous from 0.
        assert [w["win_idx"] for w in wins] == list(range(9))

        by_start = {w["t_start"]: w for w in wins}

        # Windows fully inside the squat segment: label squat, purity 1.0.
        for t_start in [0.0, 1.0, 2.0, 3.0]:
            w = by_start[t_start]
            assert w["label"] == "squat", (t_start, w)
            assert w["purity"] == pytest.approx(1.0)

        # Windows fully inside the push_up segment: label push_up, purity 1.0.
        for t_start in [5.0, 6.0, 7.0, 8.0]:
            w = by_start[t_start]
            assert w["label"] == "push_up", (t_start, w)
            assert w["purity"] == pytest.approx(1.0)

        # The single crossing window [4, 6) is mixed: 50% squat / 50% push_up.
        cross = by_start[4.0]
        assert cross["purity"] == pytest.approx(0.5)
        assert cross["label"] in {"squat", "push_up"}

    def test_no_window_crosses_end_short_video(self, tmp_path):
        """A 3.5 s video: only windows that fully fit (<= last timestamp) survive."""
        csv_path = tmp_path / "synth_test_short_angles.csv"
        _fake_csv(csv_path, duration_s=3.5, dt=0.1, boundary_s=2.0)

        df = pd.read_csv(csv_path)
        wins = make_windows_for_video(df, "synth_test_short", "test")
        last_t = float(df["time_s"].max())
        assert last_t == pytest.approx(3.5, abs=1e-6)
        # Every window's end is within the timeline.
        assert all(w["t_end"] <= last_t + 1e-6 for w in wins)
        # Starts are at integer seconds (stride 1.0). The window starting at
        # 2.0 would end at 4.0 > 3.5 (dropped), so the last kept starts at 1.0
        # (ends at 3.0 <= 3.5). Two windows survive.
        assert len(wins) == 2
        assert max(w["t_start"] for w in wins) == pytest.approx(1.0, abs=1e-6)

    def test_row_slice_recovers_window(self, tmp_path):
        """df.iloc[row_start:row_end] must give exactly the window's timestamps."""
        csv_path = tmp_path / "synth_test_slice_angles.csv"
        _fake_csv(csv_path, duration_s=10.0, dt=0.1, boundary_s=5.0)
        df = pd.read_csv(csv_path)
        wins = make_windows_for_video(df, "synth_test_slice", "test")
        for w in wins:
            sub = df.iloc[w["row_start"]:w["row_end"]]
            tmin = float(sub["time_s"].min())
            tmax = float(sub["time_s"].max())
            assert tmin >= w["t_start"] - 1e-6
            assert tmax < w["t_end"] + 1e-6
