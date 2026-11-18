import numpy as np
import pandas as pd
from copy import deepcopy

from src.rep_counting.analyze_peak_errors import assign_reps, position_bin, count_statistics
from src.rep_counting.peak_improvements import confidence_average, clean, count_built, ImprovementSettings
from src.rep_counting.peak_improvements import count_improved_peaks
from src.rep_counting.rep_counter import EXERCISES


def test_error_assignment_retains_original_indices_and_prevents_double_matches():
    matches, missed, extra = assign_reps([5., 1., 9.], [(5., 6.), (0., 7.)], tolerance=0.)
    assert matches == [(0, 0), (1, 1)]
    assert missed == [] and extra == [2]
    matches, missed, extra = assign_reps([1.], [(0., 1.), (1., 2.)], tolerance=0.)
    assert len(matches) == 1 and len(missed) == 1 and not extra


def test_normalized_position_and_signed_video_metrics():
    assert position_bin(10.5, 10, 20) == "first_10_percent"
    assert position_bin(15, 10, 20) == "middle_80_percent"
    assert position_bin(19, 10, 20) == "last_10_percent"
    stats = count_statistics(pd.DataFrame(dict(annotated=[5, 5, 5], predicted=[3, 6, 5], error=[-2, 1, 0])))
    assert stats["bias"] == -1 / 3
    assert stats["within_one_fraction"] == 2 / 3
    assert stats["overcount_fraction"] == stats["undercount_fraction"] == 1 / 3


def test_confidence_fusion_never_treats_missing_joint_as_zero():
    frame = pd.DataFrame(dict(elbow_L=[10., np.nan, 20.], elbow_R=[30., 40., np.nan],
                              elbow_L_vis=[.9, .9, .9], elbow_R_vis=[.5, .5, .5]))
    before = frame.copy(deep=True)
    result = confidence_average(frame, ["elbow_L", "elbow_R"])
    np.testing.assert_allclose(result, [(10 * .9 + 30 * .5) / 1.4, 40., 20.])
    pd.testing.assert_frame_equal(frame, before)


def test_smoothing_off_is_real_and_long_gaps_stay_missing():
    values = np.sin(np.arange(100))
    np.testing.assert_allclose(clean(values, 15, 0), values)
    values[20:35] = np.nan
    result = clean(values, 15, .3)
    assert np.isnan(result[20:35]).all()
    assert not np.allclose(result[:20], values[:20])


def test_median_ensemble_returns_an_actual_timeline_without_phantom_events():
    t = np.arange(900) / 15
    slow = -np.cos(np.pi * t)
    fast = -np.cos(2 * np.pi * t)
    settings = ImprovementSettings(signal="median", min_gap_s=.25)
    built = dict(baseline=slow, components=dict(a=slow, b=slow, noisy=fast))
    result = count_built(built, t, "jump_jack", settings)
    assert len(result) == 30
    baseline_result = count_built(dict(baseline=slow), t, "jump_jack", ImprovementSettings(signal="baseline", min_gap_s=.25))
    assert result == baseline_result


def test_median_falls_back_when_a_voter_is_unusable():
    t = np.arange(900) / 15
    signal = -np.cos(np.pi * t)
    built = dict(baseline=signal, components=dict(a=signal, b=signal, missing=np.full(len(t), np.nan)))
    assert len(count_built(built, t, "jump_jack", ImprovementSettings(signal="median", min_gap_s=.25))) == 30


def test_experimental_api_preserves_features_and_production_configuration():
    t = np.arange(300) / 15
    angle = 120 + 40 * np.cos(np.pi * t)
    frame = pd.DataFrame(dict(time_s=t, knee_L=angle, knee_R=angle, hip_L=angle, hip_R=angle))
    before, configuration = frame.copy(deep=True), deepcopy(EXERCISES)
    result = count_improved_peaks(frame, "squat", 0, 20, settings=ImprovementSettings(signal="fused"))
    assert result["reps"] == 10
    assert all(a < b for a, b in zip(result["rep_times"], result["rep_times"][1:]))
    pd.testing.assert_frame_equal(frame, before)
    assert EXERCISES == configuration
