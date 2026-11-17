"""Provenance rejection, source-clip CV isolation, and pose-gap regression tests."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'/'data_pipeline'))
sys.path.insert(0,str(ROOT/'src'/'segmentation'))
from v2_protocol import active_path, catalog, feature_frame
from cv import get_cv_split, validate_windows, count_windows
from extract_pose_features import exercise_gap_stats


@pytest.fixture
def dataset(tmp_path):
    synth=tmp_path/'synthetic'
    index=[]; splits=[]; folds=[]; videos={}; windows=[]
    for number in range(6):
        split='train' if number<5 else 'test'
        fold=number if split=='train' else None
        video_id=f'synth_v2_{split}_fixture_{number}'
        ids=[]
        for exercise in ['jump_jack','push_up','squat']:
            clip=f'clip_{number}_{exercise}.mp4';ids.append(clip)
            splits.append(dict(clip_id=clip,exercise=exercise,split=split,original_split='train' if split=='train' else 'val'))
            if split=='train':folds.append(dict(clip_id=clip,fold=fold))
        meta=dict(video_id=video_id,run_id='v2',split=split,fold=fold,clip_ids_in_order=ids)
        videos[video_id]=meta
        directory=synth/split;directory.mkdir(parents=True,exist_ok=True)
        (directory/f'{video_id}.json').write_text(json.dumps(meta))
        (directory/f'{video_id}.mp4').write_bytes(b'fixture-existence-only')
        index.append(dict(video_id=video_id,run_id='v2',split=split,fold=fold,source_clips='|'.join(ids)))
        for purity,label in [(1.,'squat'),(.5,'transition')]:
            windows.append(dict(video_id=video_id,run_id='v2',split=split,fold=fold,label=label,purity=purity))
    paths=[synth,tmp_path/'index.csv',tmp_path/'splits_v2.csv',tmp_path/'folds.csv']
    for path,rows in zip(paths[1:],[index,splits,folds]):pd.DataFrame(rows).to_csv(path,index=False)
    return paths,videos,pd.DataFrame(windows)


def test_catalog_rejects_wrong_run_and_cross_fold_sources(dataset):
    paths,videos,windows=dataset
    assert len(catalog(*paths))==6
    index=pd.read_csv(paths[1]);index.loc[0,'run_id']='v1';index.to_csv(paths[1],index=False)
    with pytest.raises(ValueError,match='run_id != v2'):catalog(*paths)
    index['run_id']='v2';index.to_csv(paths[1],index=False)
    assignment=pd.read_csv(paths[3]);assignment.loc[0,'fold']=1;assignment.to_csv(paths[3],index=False)
    with pytest.raises(AssertionError):catalog(*paths)


def test_catalog_rejects_clip_in_both_splits(dataset):
    paths,videos,_=dataset
    splits=pd.read_csv(paths[2]);splits.loc[len(splits)]=splits.iloc[0];splits.to_csv(paths[2],index=False)
    with pytest.raises(AssertionError):catalog(*paths)


def test_archive_is_rejected_before_any_read(tmp_path):
    with pytest.raises(ValueError,match='Archived v1'):
        active_path(tmp_path/'_archive_v1'/'missing.csv')


def test_cv_never_contains_test_and_detects_source_overlap(dataset):
    _,videos,windows=dataset
    for fold in range(5):
        train,val=get_cv_split(windows,fold,videos)
        assert len(train)==4 and len(val)==2
        assert train.split.eq('train').all() and val.split.eq('train').all()
        assert train.fold.ne(fold).all() and val.fold.eq(fold).all()
    wrong=windows.copy();wrong.loc[0,'fold']=3
    with pytest.raises(AssertionError):validate_windows(wrong,videos)
    train_ids=[vid for vid,meta in videos.items() if meta['split']=='train']
    videos[train_ids[1]]['clip_ids_in_order'][0]=videos[train_ids[0]]['clip_ids_in_order'][0]
    with pytest.raises(AssertionError,match='Source-clip overlap'):
        get_cv_split(windows,0,videos)


def test_feature_guard_and_test_counts_unchanged(dataset,tmp_path,monkeypatch):
    _,videos,windows=dataset
    meta=next(iter(videos.values()))
    frame=pd.DataFrame(dict(video_id=[meta['video_id']],run_id=['v1'],split=['train'],fold=[0]))
    path=tmp_path/'features.csv';frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='run_id != v2'):feature_frame(path,meta)
    frame['run_id']='v2';frame.to_csv(path,index=False)
    assert len(feature_frame(path,meta))==1
    import cv
    monkeypatch.setattr(cv,'catalog',lambda:videos)
    counts=count_windows(windows)
    test=counts.loc[counts.split=='test'].groupby('stage').windows.sum()
    assert test.to_dict()=={'before':2,'after_train_purity':2}
    assert counts.loc[(counts.stage=='before') & (counts.label=='transition'),'windows'].sum()==6


def test_push_up_labels_and_stride2_gap_seconds():
    frame=pd.DataFrame({'frame':np.arange(0,120,2),'time_s':np.arange(0,120,2)/30,
                        'elbow_best':np.r_[np.full(20,np.nan),np.ones(40)],
                        'knee_best':np.ones(60),'arm_raise_L':np.ones(60),'leg_spread':np.ones(60)})
    meta={'segments':[dict(type='exercise',label='push_up',start_frame=0,end_frame=120)]}
    result=exercise_gap_stats(frame,meta,30,2)
    assert result['frac_nan_push_up']==pytest.approx(1/3,abs=.0001)
    assert result['longest_nan_s_push_up']==1.333
    assert result['has_long_gap_push_up']
    # Two separate 10-row NaN runs must not be joined across exercise segments.
    meta['segments']=[dict(type='exercise',label='push_up',start_frame=0,end_frame=20),
                      dict(type='exercise',label='push_up',start_frame=20,end_frame=40)]
    result=exercise_gap_stats(frame,meta,30,2)
    assert result['longest_nan_s_push_up']==.667 and not result['has_long_gap_push_up']


def test_existing_window_builder_carries_fold_and_run_id(dataset,tmp_path,monkeypatch):
    _,videos,_=dataset
    import make_windows
    monkeypatch.setattr(make_windows,'catalog',lambda *args:videos)
    features=tmp_path/'features';features.mkdir()
    for meta in videos.values():
        times=np.arange(0,10,2/30)
        frame=pd.DataFrame(dict(time_s=times,seg_label=['squat']*len(times),
                                video_id=meta['video_id'],run_id='v2',split=meta['split'],fold=meta['fold']))
        frame.to_csv(features/f'{meta["video_id"]}_angles.csv',index=False)
    output=tmp_path/'windows.csv'
    monkeypatch.setattr(sys,'argv',['make_windows','--features-dir',str(features),'--out',str(output)])
    make_windows.main()
    windows=pd.read_csv(output)
    validate_windows(windows,videos)
    assert set(windows.video_id)==set(videos)
    assert windows.run_id.eq('v2').all()
    assert windows.loc[windows.split=='test','fold'].isna().all()


def test_contributor_retired_and_cv_entrypoints_available():
    import build_splits
    import train
    import evaluate
    with pytest.raises(RuntimeError,match='deprecated'):
        build_splits.group_key(pd.Series({'clip_id':'stu1_example.mp4'}))
    with pytest.raises(SystemExit) as train_exit:
        train.main(["--help"])
    assert train_exit.value.code==0
    with pytest.raises(SystemExit) as eval_exit:
        evaluate.main(["--help"])
    assert eval_exit.value.code==0


def test_stride2_boundary_interval_mapping():
    from boundary_jump_check import _sampled_boundary_frames, _compute_jumps, ANGLE_COLS
    assert _sampled_boundary_frames({0,1,2,3,7},np.array([0,2,4,6]))==[2,4]
    frame=pd.DataFrame({'frame':[0,2,4]})
    for channel in ANGLE_COLS:frame[channel]=[1.,2.,3.]
    jumps=_compute_jumps(frame)
    assert jumps.index.tolist()==[0,2,4]
    assert np.isnan(jumps.loc[0]) and np.isfinite(jumps.loc[4])


def test_postprocessing_runner_without_models(dataset,tmp_path,monkeypatch):
    """Exercise summaries, diagnostics, real window code, CV and final audit."""
    paths,videos,_=dataset
    import phase4
    import boundary_jump_check
    import make_windows
    import cv
    import v2_protocol
    features=tmp_path/'features';features.mkdir()
    output=tmp_path/'outputs'
    source=tmp_path/'data'/'synthetic'
    for meta in videos.values():
        meta.update(fps=30,n_frames=300,clip_join_frames=[100,200],
                    segments=[dict(type='exercise',label='squat',start_frame=0,end_frame=300)])
        directory=source/meta['split'];directory.mkdir(parents=True,exist_ok=True)
        json_path=directory/f'{meta["video_id"]}.json';json_path.write_text(json.dumps(meta))
        frames=np.arange(0,300,2)
        frame=pd.DataFrame(dict(frame=frames,time_s=frames/30,video_id=meta['video_id'],
                                 run_id='v2',split=meta['split'],fold=meta['fold'],seg_label='squat'))
        for channel in boundary_jump_check.ANGLE_COLS:frame[channel]=np.sin(frames/10)+1
        path=features/f'{meta["video_id"]}_angles.csv';frame.to_csv(path,index=False)
        summary=dict(video_id=meta['video_id'],run_id='v2',split=meta['split'],fold=meta['fold'],
                     elapsed_s=.1,has_long_gap_squat=False)
        sidecar=dict(run_id='v2',stride=2,mode='image',csv_sha256=v2_protocol.sha256(path),
                     source_json_sha256=v2_protocol.sha256(json_path),summary=summary)
        path.with_suffix('.meta.json').write_text(json.dumps(sidecar))
    (tmp_path/'docs').mkdir();(tmp_path/'docs'/'PROJECT_CONTEXT.md').write_text('# Fixture protocol\n')
    for module in [phase4,boundary_jump_check,make_windows,cv,v2_protocol]:
        if hasattr(module,'catalog'):monkeypatch.setattr(module,'catalog',lambda *args:videos)
    monkeypatch.setattr(phase4,'ROOT',tmp_path)
    monkeypatch.setattr(phase4,'FEATURES_DIR',features)
    monkeypatch.setattr(phase4,'OUTPUT_DIR',output)
    monkeypatch.setattr(phase4,'SUMMARY',tmp_path/'summary.csv')
    def window_subprocess(command,**kwargs):
        monkeypatch.setattr(sys,'argv',[command[2]]+command[3:])
        make_windows.main()
    monkeypatch.setattr(phase4.subprocess,'run',window_subprocess)
    phase4.finish_phase4(phase4.archive_fingerprint())
    report=json.loads((output/'phase4_consistency.json').read_text())
    assert report['status']=='complete' and report['videos']==6
    assert not report['model_training'] and not report['model_evaluation']
    assert len(pd.read_csv(output/'cv_counts.csv'))==5
    assert (output/'boundary_jump_check.csv').is_file()
