import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "rep_counting"))

from rep_counter import run_state_machine


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