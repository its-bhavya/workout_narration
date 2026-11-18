"""Learn only on temporary synthetic fixtures; real benchmark remains untouched."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import workflow
from cv import get_cv_split
from classifiers import _build_classifiers


def test_selection_opens_no_test_features(synthetic_cv_data,monkeypatch):
    data=synthetic_cv_data
    original=workflow.feature_frame
    opened=[]
    def guarded(path,meta):
        assert meta['split']=='train'
        opened.append(meta['video_id'])
        return original(path,meta)
    monkeypatch.setattr(workflow,'feature_frame',guarded)
    windows=pd.read_csv(data['windows'])
    windows.loc[windows.split=='test','label']='POISONED_TEST_LABEL'
    windows.to_csv(data['windows'],index=False)
    selection=workflow.run_selection(data['windows'],data['features'],data['out'],['lr'],[.8],
                                     [.5],[1],42,'selection_only',data['videos'])
    config=json.loads(selection.read_text())
    assert config['test_features_read'] is False
    assert set(opened)=={vid for vid,meta in data['videos'].items() if meta['split']=='train'}
    assert not list(data['out'].rglob('*test*.csv'))
    assert not list(data['out'].rglob('*.joblib'))
    ranking=pd.read_csv(selection.parent/'cv_results.csv')
    assert not any(col.startswith('test') for col in ranking.columns)
    oof=pd.read_csv(selection.parent/'oof_predictions.csv')
    assert oof.split.eq('train').all() and oof.cv_fold.eq(oof.fold).all()
    assert not oof.duplicated(['video_id','win_idx']).any()


def test_imputer_and_scaler_fit_only_other_source_folds(synthetic_cv_data):
    data=synthetic_cv_data
    windows,videos=workflow.read_training(data['windows'],data['features'],data['videos'])
    values=windows.fold.to_numpy()*100.
    values[::7]=np.nan
    X=pd.DataFrame(dict(signal=values,empty=np.full(len(windows),np.nan)),index=windows.index)
    fitted=[]
    def factory(key,seed):
        model=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                        ('scaler',StandardScaler()),('classifier',DummyClassifier(strategy='prior'))])
        fitted.append(model)
        return model
    oof,_=workflow.oof_predictions(windows,X,'fixture',.8,videos,42,factory)
    assert len(oof)==len(windows)
    for fold,model in enumerate(fitted):
        training,validation=get_cv_split(windows,fold,videos)
        expected=float(X.loc[training.index,'signal'].median())
        assert model.named_steps['imputer'].statistics_[0]==expected
        assert model.named_steps['imputer'].statistics_[1]==0.
        filled=X.loc[training.index].fillna({'signal':expected,'empty':0.})
        assert model.named_steps['scaler'].mean_==pytest.approx(filled.mean().to_numpy())


def test_refit_activation_and_frozen_final_evaluation(synthetic_bundle):
    data=synthetic_bundle
    bundle,path=workflow.load_bundle(None,data['out'])
    assert path==data['model'].resolve()
    assert bundle['fit_scope']=='all_train'
    windows=pd.read_csv(data['windows'])
    assert bundle['n_fit_windows']==int((windows.split.eq('train') & windows.purity.ge(.8)).sum())
    original_selection=data['selection'].read_bytes()
    report=workflow.evaluate_final(path,data['windows'],data['features'],videos=data['videos'])
    assert report['scope']=='final_test' and report['parameters_frozen']
    assert report['n_videos']==1 and report['n_source_clips']==3
    assert data['selection'].read_bytes()==original_selection
    with pytest.raises(FileExistsError):
        workflow.evaluate_final(path,data['windows'],data['features'],videos=data['videos'])
    with pytest.raises(FileExistsError):
        workflow.refit_selection(data['selection'],data['windows'],data['features'],videos=data['videos'])


def test_bundle_provenance_and_smoothing_mismatches_rejected(synthetic_bundle,tmp_path):
    data=synthetic_bundle
    bundle,path=workflow.load_bundle(data['model'])
    params=path.parent/'chosen_params.json'
    assert workflow.bundle_smoothing(bundle,path,params)==(.5,1)
    wrong=json.loads(params.read_text());wrong['experiment_id']='another_run'
    wrong_path=tmp_path/'params.json';wrong_path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match='different model'):
        workflow.bundle_smoothing(bundle,path,wrong_path)
    old=dict(bundle,run_id='v1');old_path=tmp_path/'old.joblib';joblib.dump(old,old_path)
    with pytest.raises(ValueError,match='run_id=v2'):workflow.load_bundle(old_path)
    fold_model=dict(bundle,fit_scope='one_fold');fold_path=tmp_path/'fold.joblib';joblib.dump(fold_model,fold_path)
    with pytest.raises(ValueError,match='all-train refit'):workflow.load_bundle(fold_path)
    feature=next(p for p in data['features'].glob('*') if 'train' in p.name)
    frame=pd.read_csv(feature);frame.loc[0,'knee_best']+=1;frame.to_csv(feature,index=False)
    with pytest.raises(ValueError,match='provenance'):
        workflow.verify_bundle_training(bundle,data['windows'],data['features'],data['videos'])


def test_prediction_integration_parity_and_gt_independence(synthetic_bundle):
    data=synthetic_bundle
    from src.pipeline import WorkoutPipeline
    import predict_video
    bundle,_=workflow.load_bundle(data['model'])
    vid=next(vid for vid,meta in data['videos'].items() if meta['split']=='test')
    meta=data['videos'][vid]
    frame=workflow.feature_frame(data['features']/f'{vid}_angles.csv',meta)
    pred,raw,segments,end=workflow.infer_frame(bundle,frame,meta)
    assert 'label' not in pred and 'purity' not in pred
    frame['seg_label']='wrong labels must not affect predictions'
    again=workflow.infer_frame(bundle,frame,meta)
    assert pred.pred_label.tolist()==again[0].pred_label.tolist()
    assert segments==again[2] and end==24.
    pipeline=WorkoutPipeline(model_path=data['model'],windows_path=data['windows'],
                             features_dir=data['features'],synth_dir=data['synth'],rep_method='state_machine')
    result=pipeline.analyze(vid)
    assert [(s.exercise,s.start_time,s.end_time) for s in result.segments]==[
        (s['exercise'],s['start_time'],s['end_time']) for s in segments]
    standalone=predict_video.predict(vid,data['features'],data['synth'],data['model'],windows_path=data['windows'])
    assert standalone['segments']==segments and standalone['duration_s']==24.


def test_cv_report_detects_changed_oof_predictions(synthetic_bundle):
    import evaluate
    data=synthetic_bundle
    assert len(evaluate.evaluate_cv(data['selection'],data['windows'],data['features']))==5
    path=data['selection'].parent/'oof_predictions.csv'
    frame=pd.read_csv(path);frame.loc[0,'pred_label']='transition';frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='OOF predictions changed'):
        evaluate.evaluate_cv(data['selection'],data['windows'],data['features'])


def test_no_implicit_window_level_early_stopping():
    models=_build_classifiers(42)
    assert models['mlp'].early_stopping is False
    assert models['hist_gbt'].early_stopping is False


def test_missing_permitted_channel_is_nan_without_rewriting_features(synthetic_cv_data):
    data=synthetic_cv_data
    windows,videos=workflow.read_training(data['windows'],data['features'],data['videos'])
    before={p:p.read_bytes() for p in data['features'].glob('*.csv')}
    X,channels=workflow.build_feature_matrix(windows,data['features'],videos)
    assert len(channels)==16 and X.shape[1]==112
    assert X.ankle_dist_norm_3d_mean.isna().all()
    assert X.ankle_dist_norm_3d_nanfrac.eq(1.).all()
    assert {p:p.read_bytes() for p in before}==before


def test_refit_rejects_classifier_drift(synthetic_bundle,monkeypatch):
    data=synthetic_bundle
    original=workflow._build_classifiers
    def changed(seed):
        models=original(seed)
        models['lr'].set_params(max_iter=17)
        return models
    monkeypatch.setattr(workflow,'_build_classifiers',changed)
    with pytest.raises(ValueError,match='Classifier settings changed'):
        workflow.refit_selection(data['selection'],data['windows'],data['features'],videos=data['videos'])


def test_record_results_uses_only_completed_frozen_artifacts(synthetic_bundle,tmp_path,monkeypatch):
    import record_results
    data=synthetic_bundle
    workflow.evaluate_final(data['model'],data['windows'],data['features'],videos=data['videos'])
    experiment=data['selection'].parent
    (experiment/'run_manifest.json').write_text(json.dumps({'status':'complete'}))
    report_root=tmp_path/'report_repository';report_root.mkdir()
    paths=['README.md','PROJECT_CONTEXT.md','docs/PROJECT_CONTEXT.md','docs/V2_TRAINING.md',
           'src/segmentation/README.md','src/segmentation/USAGE.md','src/pipeline/README.md',
           'paper/PROJECT_REPORT_AND_PAPER_CONTEXT.md','paper/scope_proposal.md','paper/related_work_notes.md']
    for relative in paths:
        path=report_root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# Fixture context\n')
    monkeypatch.setattr(record_results,'ROOT',report_root)
    monkeypatch.setattr(record_results,'OUTPUT_DIR',data['out'])
    monkeypatch.setattr(record_results,'catalog',lambda:data['videos'])
    report=record_results.record(experiment)
    assert 'CV winner: lr' in report.read_text()
    snapshot=report_root/'docs'/'results'/'fixture'/'model_comparison.csv'
    assert len(pd.read_csv(snapshot))==1
    context=(report_root/'README.md').read_text()
    assert 'Test was not used for model selection or tuning' in context
    record_results.record(experiment)
    assert (report_root/'README.md').read_text().count('<!-- v2-model-results:start -->')==1
