import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.rep_counting import default_counter as default
from src.rep_counting.coordinate_counter import count_coordinate_peaks
from src.rep_counting.extract_coordinate_training import add_signals
from src.rep_counting.rep_counter import EXERCISES


def raw_coordinates():
    frames=np.arange(0,720,2); times=frames/30
    body=200+40*np.cos(2*np.pi*times/2)
    data=dict(frame=frames,time_s=times,detected=True,n_poses=1)
    for point in default.RAW_POINTS:
        side=-1 if point.endswith('_L') else 1
        data[point+'_x']=np.full(len(frames),side*50.)
        data[point+'_y']=body+(100 if point.startswith('hip') else -40 if point=='nose' else 0)
        data[point+'_vis']=np.ones(len(frames))
    for side in ('L','R'):
        data[f'wrist_{side}_y']=np.full(len(frames),320.)
        data[f'elbow_{side}_x']=np.full(len(frames),-110. if side=='L' else 110.)
        data[f'elbow_{side}_y']=np.full(len(frames),260.)
    return pd.DataFrame(data)


def test_frozen_peak_dispatch_matches_selected_parameters(monkeypatch,tmp_path):
    seen=[]
    def counted(frame,exercise,start,end,settings):
        seen.append((exercise,start,end,settings))
        return dict(reps=1,rep_times=[start+1],flag='')
    monkeypatch.setattr(default,'count_improved_peaks',counted)
    with default.ExperimentalCounter(tmp_path/'not_needed.mp4',30,720) as session:
        session.count(pd.DataFrame(),'jump_jack',2,8)
        session.count(pd.DataFrame(),'squat',2,8)
    assert seen[0][1:3]==(2,8.3)
    assert seen[0][3].smooth_s==0 and seen[0][3].min_gap_s==.25
    assert seen[1][3].signal=='alternate' and seen[1][3].smooth_s==.6
    assert seen[1][3].prominence_fraction==.35


def test_raw_pushup_dispatch_matches_frozen_api_and_preserves_inputs(tmp_path):
    raw=raw_coordinates(); before=raw.copy(deep=True); original=deepcopy(EXERCISES)
    with default.ExperimentalCounter(tmp_path/'not_needed.mp4',30,720,raw) as session:
        actual=session.count(pd.DataFrame(),'push_up',6,12)
        local=raw[(raw.time_s>=6)&(raw.time_s<12.3)].copy()
        derived,_=add_signals(local)
        expected=count_coordinate_peaks(derived,session.pushup_settings)
        assert actual['rep_times']==expected['rep_times']
        assert actual['reps']>0 and actual['method']=='coordinate_median_peaks'
        assert all(6<=t<12.3 for t in actual['rep_times'])
        assert actual['counter_settings']['smooth_s']==.5
    pd.testing.assert_frame_equal(raw,before)
    assert EXERCISES==original


def test_missing_or_incomplete_raw_coordinates_never_silently_fall_back(tmp_path):
    with default.ExperimentalCounter(tmp_path/'absent.mp4',30,720) as session:
        with pytest.raises(FileNotFoundError,match='no angle-only fallback'):
            session.count(pd.DataFrame(),'push_up',6,12)
    raw=raw_coordinates().drop(index=100)
    with default.ExperimentalCounter(tmp_path/'absent.mp4',30,720,raw) as session:
        with pytest.raises(ValueError,match='Incomplete or misaligned'):
            session.count(pd.DataFrame(),'push_up',6,12)


def test_video_pose_handle_is_lazy_reused_and_closed_on_exception(tmp_path,monkeypatch):
    from mediapipe.tasks.python import vision
    from src.rep_counting import extract_pushup_diagnostic
    calls=[]
    class FakePose:
        def close(self): calls.append('closed')
    monkeypatch.setattr(vision.PoseLandmarker,'create_from_options',lambda options:calls.append('created') or FakePose())
    raw=raw_coordinates()
    def extract(handle,video,start,end,fps):
        return raw[(raw.frame>=start)&(raw.frame<end)].copy(),[],(400,400)
    monkeypatch.setattr(extract_pushup_diagnostic,'extract_pass',extract)
    video=tmp_path/'fixture.mp4'; video.write_bytes(b'fixture')
    with pytest.raises(RuntimeError,match='after counting'):
        with default.ExperimentalCounter(video,30,720) as session:
            assert calls==[]
            session.count(pd.DataFrame(),'push_up',6,12)
            session.count(pd.DataFrame(),'push_up',12,18)
            raise RuntimeError('after counting')
    assert calls==['created','closed']


def test_pipeline_defaults_to_combo_preserves_segments_and_narrates(synthetic_bundle,tmp_path):
    from src.pipeline import WorkoutPipeline,analyze_workout
    from src.narration.integration import narrate_analysis
    artifact=synthetic_bundle
    vid=next(v for v,m in artifact['videos'].items() if m['split']=='train')
    frame=pd.read_csv(artifact['features']/f'{vid}_angles.csv')
    # Supply extra hip and ankle measurements in memory only; canonical files/provenance stay intact.
    for side in ('L','R'): frame[f'hip_{side}']=120+35*np.cos(2*np.pi*frame.time_s/2)
    frame['ankle_dist_norm']=0.5+0.3*np.cos(2*np.pi*frame.time_s/2)
    options=dict(model_path=artifact['model'],windows_path=artifact['windows'],features_dir=artifact['features'],synth_dir=artifact['synth'])
    result=analyze_workout(frame,coordinate_features=raw_coordinates(),**options)
    legacy=WorkoutPipeline(**options,rep_method='state_machine').analyze(frame)
    assert result.rep_counter=='experimental'
    assert result.rep_counter_config['id']=='experimental_combo_v1'
    assert result.rep_counter_config['pushup_freeze_id']=='coordinate-voting-v1-20261118'
    assert [(s.exercise,s.start_time,s.end_time) for s in result.segments]==[(s.exercise,s.start_time,s.end_time) for s in legacy.segments]
    methods={s.exercise:s.counter_method for s in result.segments if s.is_exercise}
    assert methods==dict(squat='unguided_peaks',push_up='coordinate_median_peaks',jump_jack='unguided_peaks')
    assert result.summary.total_reps==sum(len(s.rep_times) for s in result.segments)>0
    json.dumps(result.to_dict(),allow_nan=False)
    report=narrate_analysis(result,tmp_path/'narration',synth=lambda text:'fake.mp3',dur=lambda p:.2,render=lambda *args:None)
    assert report['rep_counter']=='experimental'
    assert report['rep_counter_config']==result.rep_counter_config
    assert sum(len(s['rep_times']) for s in report['segments'])==result.summary.total_reps


def test_unknown_rep_method_rejected_before_loading_model():
    from src.pipeline import WorkoutPipeline
    with pytest.raises(ValueError,match='rep_method'):
        WorkoutPipeline(rep_method='typo')
