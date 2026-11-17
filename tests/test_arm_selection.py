import numpy as np
import pandas as pd
import pytest

from src.rep_counting.arm_quality import motion_quality
from src.rep_counting.rep_counter import count_for_exercise
from src.rep_counting.validate_arm_selection import choose


def test_prefers_periodic_arm_over_more_visible_noisy_arm():
    rng = np.random.default_rng(42)
    t = np.arange(900) / 15
    frame = pd.DataFrame(dict(time_s=t, elbow_L=120 + rng.normal(0, 30, len(t)),
                              elbow_R=120 + 35 * np.cos(2 * np.pi * t / 2),
                              elbow_L_vis=1., elbow_R_vis=.85))
    result = count_for_exercise(frame, "push_up", 0, 60)
    baseline = count_for_exercise(frame, "push_up", 0, 60, arm_selection="visibility")
    assert result["signal"] == "elbow_R"
    assert result["reps"] == 30
    assert baseline["signal"] == "elbow_L"
    assert baseline["reps"] > result["reps"]  # selection does not maximize rep count
    assert len(result["arm_candidates"]) == 2


def test_unusable_arm_cannot_win_by_visibility_or_periodicity():
    t = np.arange(300) / 15
    frame = pd.DataFrame(dict(time_s=t, elbow_L=np.nan,
                              elbow_R=120 + 35 * np.cos(2 * np.pi * t / 2),
                              elbow_L_vis=1., elbow_R_vis=.6))
    result = count_for_exercise(frame, "push_up", 0, 20)
    assert result["signal"] == "elbow_R" and result["reps"] == 10


def test_quality_ignores_gaps_and_handles_nan_visibility():
    q = motion_quality(np.full(100, np.nan), np.full(100, np.nan), np.nan, 15)
    assert all(np.isfinite(v) for v in q.values())
    assert q["periodicity"] == 0


def test_policy_ties_preserve_visibility_and_ignore_counts():
    candidates = [dict(signal="left", usable=True, quality={"periodicity": .8}, predicted=4),
                  dict(signal="right", usable=True, quality={"periodicity": .8}, predicted=50)]
    assert choose(candidates, {"periodicity": 1})["signal"] == "left"


def test_unknown_selection_policy_is_rejected():
    with pytest.raises(ValueError, match="arm_selection"):
        count_for_exercise(pd.DataFrame(), "push_up", 0, 10, arm_selection="typo")
