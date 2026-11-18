import pandas as pd
import pytest
import json

from src.rep_counting import evaluate_segmented_cv as evaluation


def test_frame_windows_apply_padding_clip_video_and_keep_global_stride():
    assert evaluation.even_frames(.1,.3,30,12)==[4,6,8,10]
    assert evaluation.even_frames(.1,.3,30,12,padding=0)==[4,6,8]


def test_only_missing_contiguous_frames_are_extracted():
    requested=set(range(0,20,2))
    cached={0,2,8,10,18}
    runs=evaluation.missing_runs(requested-cached)
    recovered={frame for start,end in runs for frame in range(start,end,2)}
    assert recovered==requested-cached
    assert not recovered & cached
    assert evaluation.missing_runs([])==[]


def test_oof_guard_rejects_seen_fold_and_incomplete_video_coverage(monkeypatch):
    monkeypatch.setattr(evaluation.workflow,'validate_windows',lambda *args:None)
    windows=pd.DataFrame(dict(video_id=['a','b'],win_idx=[0,0]))
    predictions=windows.assign(split='train',fold=[0,1],cv_fold=[0,1])
    evaluation.verify_oof(predictions,windows,{})
    with pytest.raises(ValueError,match='coverage'):
        evaluation.verify_oof(predictions.iloc[:1],windows,{})
    with pytest.raises(ValueError,match='held-out'):
        evaluation.verify_oof(predictions.assign(cv_fold=[1,1]),windows,{})


def test_missing_extractor_rejects_test_before_video_access():
    with pytest.raises(ValueError,match='No test frames'):
        evaluation.extract_missing(dict(split='test'))


def test_undefined_boundary_distances_are_null_without_changing_count_metrics():
    values={'mae':2.5,'boundary_mean_s':float('inf'),'folds':[float('nan'),0.]}
    cleaned=evaluation.finite_json(values)
    assert cleaned=={'mae':2.5,'boundary_mean_s':None,'folds':[None,0.]}
    json.dumps(cleaned,allow_nan=False)
    assert values['boundary_mean_s']==float('inf')
