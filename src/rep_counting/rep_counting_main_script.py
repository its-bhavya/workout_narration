import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from counter import (  # noqa: E402
    clean_signal, count_for_exercise, drop_close, run_state_machine,
    sampling_rate,
)


def make_df(angle_fn, duration, fs, column="knee_L", extra=None):
    # Real feature files always have both sides; the untracked side is all-NaN.
    t = np.arange(0, duration, 1.0 / fs)
    data = {"time_s": t, "knee_L": np.nan, "knee_R": np.nan}
    data[column] = angle_fn(t)
    data.update(extra or {})
    return pd.DataFrame(data)


def squat_wave(period):
    return lambda t: 130 + 40 * np.cos(2 * np.pi * t / period)


@pytest.mark.parametrize("fs", [30.0, 15.0])
def test_count_is_independent_of_sampling_rate(fs):
    df = make_df(squat_wave(2.5), 15.0, fs)
    out = count_for_exercise(df, "squat", 0.0, 15.0)
    assert out["reps"] == 6
    assert out["flag"] == ""


def test_jump_jack_mean_of_arms():
    fn = lambda t: 85 + 65 * np.cos(2 * np.pi * t / 0.9 + np.pi)  # starts low
    t = np.arange(0, 9.0, 1 / 15)
    df = pd.DataFrame({"time_s": t, "arm_raise_L": fn(t), "arm_raise_R": fn(t)})
    out = count_for_exercise(df, "jump_jack", 0.0, 9.0)
    assert out["reps"] == 10


def test_jump_jack_survives_one_missing_arm():
    fn = lambda t: 85 + 65 * np.cos(2 * np.pi * t / 0.9 + np.pi)
    t = np.arange(0, 9.0, 1 / 15)
    right = fn(t).copy()
    right[:] = np.nan
    df = pd.DataFrame({"time_s": t, "arm_raise_L": fn(t), "arm_raise_R": right})
    assert count_for_exercise(df, "jump_jack", 0.0, 9.0)["reps"] == 10


def test_long_gap_stays_nan_and_is_not_ramped():
    fs = 15.0
    x = np.full(150, 100.0)
    x[50:90] = np.nan                       # 2.7 s gap >> MAX_GAP_S
    cleaned = clean_signal(x, fs, max_gap_s=0.4, smooth_s=0.4)
    assert np.isnan(cleaned[50:90]).all()
    assert not np.isnan(cleaned[:50]).any()


def test_short_gap_is_interpolated():
    fs = 15.0
    x = np.linspace(100, 120, 100)
    x[40:43] = np.nan                       # 0.2 s gap
    cleaned = clean_signal(x, fs, max_gap_s=0.4, smooth_s=0.4)
    assert not np.isnan(cleaned).any()


def test_low_motion_segment_gives_zero_reps_and_flag():
    rng = np.random.default_rng(0)
    df = make_df(lambda t: 170 + rng.normal(0, 1.0, len(t)), 10.0, 15.0)
    out = count_for_exercise(df, "squat", 0.0, 10.0)
    assert out["reps"] == 0
    assert out["flag"] == "low_motion"


def test_mostly_missing_segment_is_flagged():
    df = make_df(squat_wave(2.5), 10.0, 15.0)
    df.loc[10:, "knee_L"] = np.nan
    out = count_for_exercise(df, "squat", 0.0, 10.0)
    assert out["flag"] == "mostly_missing" and out["reps"] == 0


def test_best_side_chosen_by_visibility():
    good = squat_wave(2.5)
    t = np.arange(0, 15.0, 1 / 15)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "time_s": t,
        "knee_L": 170 + rng.normal(0, 0.5, len(t)),      # flat = untracked side
        "knee_R": good(t),
        "knee_L_vis": 0.2, "knee_R_vis": 0.9,
    })
    out = count_for_exercise(df, "squat", 0.0, 15.0)
    assert out["signal"] == "knee_R" and out["reps"] == 6


def test_window_slicing_counts_only_that_segment():
    df = make_df(squat_wave(2.5), 30.0, 15.0)        # 12 reps in total
    out = count_for_exercise(df, "squat", 0.0, 15.0)
    assert out["reps"] == 6


def test_drop_close():
    assert drop_close([1.0, 1.2, 2.0, 2.3, 3.5], 0.5) == [1.0, 2.0, 3.5]


def test_state_machine_hysteresis_ignores_jitter_near_threshold():
    sig = np.array([170, 169, 145, 151, 144, 152, 170, 171], dtype=float)
    idx = run_state_machine(sig, 150, 168, "down_up")
    assert len(idx) == 1


def test_sampling_rate():
    assert sampling_rate(np.arange(0, 10, 1 / 15)) == pytest.approx(15.0, rel=0.01)


def test_falls_back_to_other_side_when_preferred_side_does_not_move():
    # Both sides report high visibility, but only the right knee actually moves.
    good = squat_wave(2.5)
    t = np.arange(0, 15.0, 1 / 15)
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "time_s": t,
        "knee_L": 170 + rng.normal(0, 0.5, len(t)),
        "knee_R": good(t),
        "knee_L_vis": 0.9, "knee_R_vis": 0.9,
    })
    out = count_for_exercise(df, "squat", 0.0, 15.0)
    assert out["signal"] == "knee_R" and out["reps"] == 6 and out["flag"] == ""


def test_both_sides_unusable_reports_flag_of_preferred_side():
    t = np.arange(0, 10.0, 1 / 15)
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "time_s": t,
        "knee_L": 170 + rng.normal(0, 0.5, len(t)),
        "knee_R": 170 + rng.normal(0, 0.5, len(t)),
        "knee_L_vis": 0.9, "knee_R_vis": 0.8,
    })
    out = count_for_exercise(df, "squat", 0.0, 10.0)
    assert out["reps"] == 0 and out["flag"] == "low_motion" and out["signal"] == "knee_L"