import numpy as np
import pandas as pd
import pytest

from src.segmentation.exercise_gate_sweep import exercise_gate, gate_segments, choose_threshold


def test_gate_uses_unnormalized_exercise_maximum_and_inclusive_threshold():
    probabilities=np.array([[.25,.1,.05],[.05,.2,.1],[.05,.1,.4]])
    original=probabilities.copy()
    assert exercise_gate(probabilities,.25)==['jump_jack','transition','squat']
    # First row has 0.60 probability on transition; it does not compete.
    assert exercise_gate(probabilities,.2)==['jump_jack','push_up','squat']
    np.testing.assert_array_equal(probabilities,original)
    with pytest.raises(ValueError): exercise_gate([[np.nan,.1,.2]],.2)


def test_gate_replaces_old_confidence_cutoff_instead_of_rejecting_twice():
    rows=pd.DataFrame(dict(t_center=[1.,2.,3.],prob_jump_jack=[.3]*3,prob_push_up=[.1]*3,prob_squat=[.1]*3))
    assert gate_segments(rows,.25,4)==[dict(exercise='jump_jack',start_time=0.,end_time=4.)]
    assert gate_segments(rows,.35,4)==[dict(exercise='transition',start_time=0.,end_time=4.)]


def test_threshold_selection_excludes_target_fold_and_uses_conservative_ties():
    scores=pd.DataFrame(dict(threshold=[.1,.2,.3],fold_0_mae=[100.,0.,0.],fold_1_mae=[1.,2.,2.],fold_2_mae=[1.,2.,2.],fold_3_mae=[1.,2.,2.],fold_4_mae=[1.,2.,2.]))
    assert choose_threshold(scores,heldout=0)==.1
    assert choose_threshold(scores)==.3
