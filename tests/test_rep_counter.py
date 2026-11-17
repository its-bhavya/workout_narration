import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "rep_counting"))

from rep_counter import run_state_machine
from src.rep_counting.rep_counter import count_for_exercise, local_motion_signal
import pandas as pd


class TestSquatStateMachine:

    def test_counts_one_complete_squat(self):
        signal = np.array([
            160, 130, 110, 105, 115, 130, 150, 155, 160
        ])

        reps = run_state_machine(
            signal, 118, 140, "down_up", fs=10
        )

        assert len(reps) == 1

    def test_does_not_count_brief_upper_threshold_crossing(self):
        signal = np.array([
            160, 110, 105, 115, 130, 150, np.nan, np.nan
        ])

        reps = run_state_machine(
            signal, 118, 140, "down_up", fs=10
        )

        assert len(reps) == 0


    def test_counts_two_complete_squats(self):
        signal = np.array([
            160, 110, 105, 115, 130, 150, 155, 160,
            160, 110, 105, 115, 130, 150, 155, 160
        ])

        reps = run_state_machine(
            signal, 118, 140, "down_up", fs=10
        )

        assert len(reps) == 2

    def test_nan_during_confirmation_does_not_complete_rep(self):
        signal = np.array([
            160, 110, 105, 115, 130, 150, np.nan, 155, np.nan
        ])

        reps = run_state_machine(
            signal, 118, 140, "down_up", fs=10
        )

        assert len(reps) == 0


class TestPushupLocalThresholds:
    @staticmethod
    def frame(signal, fs=15):
        return pd.DataFrame({"time_s": np.arange(len(signal)) / fs,
                             "elbow_L": signal, "elbow_R": signal})

    def test_counts_cycles_when_elbow_baseline_drifts(self):
        t = np.arange(900) / 15
        signal = 70 + 70 * t / 60 + 30 * np.cos(2 * np.pi * t / 2)
        result = count_for_exercise(self.frame(signal), "push_up", 0, 60)
        assert result["reps"] == 30
        assert result["thresholds"] is None
        assert result["threshold_mode"] == "local_adaptive"
        assert all(a < b for a, b in zip(result["rep_times"], result["rep_times"][1:]))
        assert result["rep_times"][0] > 1  # counted on recovery, after the bottom

    def test_does_not_normalize_motionless_noise_into_reps(self):
        t = np.arange(150) / 15
        signal = 130 + 2 * np.cos(2 * np.pi * t)
        result = count_for_exercise(self.frame(signal), "push_up", 0, 10)
        assert result["reps"] == 0
        assert result["flag"] == "low_motion"

    def test_lost_tracking_resets_local_scale(self):
        signal = np.r_[np.linspace(150, 60, 40), np.full(15, np.nan),
                       np.linspace(120, 170, 40)]
        normalized = local_motion_signal(signal, 15, 6, 20)
        assert np.isnan(normalized[40:55]).all()
        assert normalized[55] < .45
        assert run_state_machine(normalized, .45, .75, "down_up", fs=15, confirm_s=0) == [
            next(i for i in range(55, len(signal)) if normalized[i] > .75)]

    def test_explicit_angle_thresholds_keep_legacy_behavior(self):
        t = np.arange(150) / 15
        result = count_for_exercise(self.frame(120 + 40 * np.cos(2 * np.pi * t / 2)),
                                    "push_up", 0, 10, thresholds=(100, 140))
        assert result["thresholds"] == (100., 140.)
        assert "threshold_mode" not in result
