import numpy as np
import pandas as pd
import pytest

from src.rep_counting.extract_pushup_diagnostic import coordinate_signals, run
from src.rep_counting.rep_counter import drop_close


def coordinates():
    t = np.arange(60) / 15
    displacement = 10 * np.cos(np.pi * t)
    data = dict(time_s=t)
    for name, y in [('nose',70),('shoulder_L',100),('shoulder_R',100),
                    ('hip_L',180),('hip_R',180),('elbow_L',150),('elbow_R',150),
                    ('wrist_L',220),('wrist_R',220)]:
        data[name+'_x'] = np.full(len(t),100.)
        data[name+'_y'] = np.full(len(t),y,dtype=float) if name.startswith('wrist') else y+displacement
        data[name+'_vis'] = np.ones(len(t))
    return pd.DataFrame(data)


def test_coordinate_signals_use_fixed_torso_scale_without_mutating_source():
    frame = coordinates();before=frame.copy(deep=True)
    signals, scale = coordinate_signals(frame)
    assert scale == 80.
    np.testing.assert_allclose(signals['shoulder_to_wrist_height'],(220-frame.shoulder_L_y)/80.)
    assert np.isfinite(signals['head_height']).all()
    pd.testing.assert_frame_equal(frame,before)


def test_low_visibility_head_is_masked_and_wrist_uses_available_side():
    frame=coordinates();frame.loc[0,'nose_vis']=.1
    frame['wrist_L_vis']=.1
    signals, _ = coordinate_signals(frame)
    assert np.isnan(signals['head_height'][0])
    assert np.isfinite(signals['shoulder_to_wrist_height']).all()


def test_long_minimum_gap_can_suppress_correct_fast_repetitions():
    completions=[.5,1.,1.5,2.,2.5]
    assert drop_close(completions,.2)==completions
    assert len(drop_close(completions,.7))==3


def test_pose_pilot_cannot_write_outside_experiment_outputs(tmp_path):
    with pytest.raises(ValueError,match='outputs'):
        run(tmp_path/'not_an_experiment_output',[])
