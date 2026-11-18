import numpy as np
import pandas as pd
import pytest

from src.segmentation.periodicity_context_ablation import periodic_features, context_bounds
from src.segmentation.complete_ablation_counts import merge_raw_frames, RAW


def test_two_hz_motion_has_band_power_and_acf_without_mutating_input():
    times=np.arange(30)/15
    values=20*np.sin(2*np.pi*2*times)
    original=values.copy()
    result=periodic_features(values,15)
    assert abs(result['dominant_hz']-2)<.01
    assert result['power_1_3_fraction']>.95
    assert result['acf_peak']>.4
    np.testing.assert_array_equal(values,original)


def test_out_of_band_motion_and_linear_trend_are_not_one_to_three_hz_evidence():
    times=np.arange(30)/15
    result=periodic_features(np.sin(2*np.pi*5*times),15)
    assert result['dominant_hz']==5
    assert result['power_1_3_fraction']<.01
    assert periodic_features(times*10+3,15)==dict(dominant_hz=0.,power_1_3_fraction=0.,acf_peak=0.)


def test_periodicity_does_not_bridge_large_missing_runs():
    values=np.sin(np.arange(30))
    values[10:20]=np.nan
    assert all(np.isnan(value) for value in periodic_features(values,15).values())
    # Even sufficient total coverage cannot join two disconnected runs.
    values=np.sin(np.arange(30)); values[14:17]=np.nan
    assert all(np.isnan(value) for value in periodic_features(values,15).values())


def test_centered_context_is_clipped_at_video_edges():
    times=np.arange(20)
    np.testing.assert_array_equal(context_bounds(times,10),[6,14])
    np.testing.assert_array_equal(context_bounds(times,1),[0,5])
    np.testing.assert_array_equal(context_bounds(times,19),[15,20])


def test_all_class_cache_reuse_preserves_primary_observations_and_missing_frames():
    primary=pd.DataFrame({column:[100.,101.] for column in RAW})
    primary['frame']=[0,2]
    secondary=pd.DataFrame({column:[900.,901.] for column in RAW})
    secondary['frame']=[2,4]
    result=merge_raw_frames(primary,secondary,{0,2,4,6})
    assert result.frame.tolist()==[0,2,4]
    assert result.nose_x.tolist()==[100.,101.,901.]
    with pytest.raises(ValueError,match='Ambiguous'):
        merge_raw_frames(pd.concat([primary,primary]),secondary,{0,2,4})


def test_empty_raw_cache_does_not_turn_numeric_pose_arrays_into_objects():
    empty=pd.DataFrame(columns=RAW)
    secondary=pd.DataFrame({column:[1.,2.] for column in RAW})
    secondary['frame']=[0,2]
    result=merge_raw_frames(empty,secondary,{0,2})
    assert np.issubdtype(result.nose_x.dtype,np.number)
    assert np.linalg.norm(result[['nose_x','nose_y']].to_numpy(),axis=1).shape==(2,)
