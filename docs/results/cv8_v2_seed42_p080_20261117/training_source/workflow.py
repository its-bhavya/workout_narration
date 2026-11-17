"""Train-only source-clip CV, frozen all-train refit, and final evaluation.

run_id is the dataset protocol (v2); experiment_id identifies a model run.
Selection never opens test angle files or produces test predictions.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
for folder in [HERE,HERE.parent/'data_pipeline']:
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
from classifiers import _build_classifiers
from config import (CLASSES, FEATURES_DIR, OUTPUT_DIR, WINDOWS_CSV, WINDOW_S,
                    STRIDE_S, MIN_ROWS_FRAC, MIN_PURITY_TRAIN, SEED, FEATURE_CHANNELS, feature_columns)
from cv import get_cv_split, validate_windows
from features import MAX_FILL_GAP_S, feature_names, fill_video_nans, window_features
from make_windows import make_windows_for_video
from metrics import all_metrics, gt_from_json
from smooth import smooth, to_segments
from v2_protocol import active_path, catalog, feature_frame, require_v2, sha256

SCHEMA_VERSION=1
MODEL_CHANNELS=[channel for base in FEATURE_CHANNELS
                for channel in ([base] if base in {'knee_best','elbow_best'} else [base,base+'_3d'])]
PREPROCESSING=dict(window_s=WINDOW_S,stride_s=STRIDE_S,min_rows_frac=MIN_ROWS_FRAC,
                   max_fill_gap_s=MAX_FILL_GAP_S,extraction_stride=2,extraction_mode='image',
                   channels=MODEL_CHANNELS)


def align_channels(frame,channels=MODEL_CHANNELS):
    if not feature_columns(frame.columns):raise ValueError('No permitted feature channels')
    frame=frame.copy()
    for channel in channels:
        if channel not in frame:frame[channel]=np.nan
    return frame


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def signature(videos, windows, features_dir):
    """Hash train-only inputs, including feature bytes, without reading test features."""
    features_dir=active_path(features_dir)
    training=windows.loc[windows.split.eq('train')]
    ids=sorted(training.video_id.unique())
    content=dict(videos={vid:videos[vid] for vid in ids},
                 windows=training.to_json(orient='split',double_precision=15),
                 features={vid:sha256(features_dir/f'{vid}_angles.csv') for vid in ids},
                 preprocessing=PREPROCESSING)
    return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()


def read_training(windows_path=WINDOWS_CSV, features_dir=FEATURES_DIR, videos=None):
    videos=catalog() if videos is None else videos
    windows=pd.read_csv(active_path(windows_path))
    validate_windows(windows,videos)
    training=windows.loc[windows.split.eq('train')].reset_index(drop=True)
    if set(training.fold)!=set(range(5)):
        raise ValueError('All five source folds are required')
    expected={vid for vid,m in videos.items() if m['split']=='train'}
    if set(training.video_id)!=expected:
        raise ValueError('Training windows must cover every indexed train video')
    if not set(training.label)<=set(CLASSES):raise ValueError('Unknown training label')
    return training, videos


def build_feature_matrix(windows, features_dir=FEATURES_DIR, videos=None):
    videos=validate_windows(windows,videos)
    if windows.empty:raise ValueError('No windows')
    cache={}; channels=list(MODEL_CHANNELS); rows=[]
    for _,window in windows.iterrows():
        vid=window.video_id
        if vid not in cache:
            frame=feature_frame(active_path(features_dir)/f'{vid}_angles.csv',videos[vid])
            cache[vid]=fill_video_nans(align_channels(frame,channels),channels=channels)
        start,end=int(window.row_start),int(window.row_end)
        if not 0<=start<end<=len(cache[vid]):raise ValueError('Invalid window row slice')
        rows.append(window_features(cache[vid].iloc[start:end]))
    if not channels:raise ValueError('No permitted feature channels')
    return pd.DataFrame(rows,index=windows.index).reindex(columns=feature_names(channels)).replace([np.inf,-np.inf],np.nan),channels


def estimator(model_key, seed=SEED):
    classifiers=_build_classifiers(seed)
    if model_key not in classifiers:raise ValueError(f'Unknown classifier: {model_key}')
    steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if model_key in {'lr','svm','knn','mlp'}:steps.append(('scaler',StandardScaler()))
    steps.append(('classifier',classifiers[model_key]))
    return Pipeline(steps)


def prediction_table(model, X, rows):
    probabilities=model.predict_proba(X.to_numpy())
    out=rows.copy()
    out['pred_label']=model.classes_[probabilities.argmax(axis=1)]
    out['max_prob']=probabilities.max(axis=1)
    for label in CLASSES:
        out[f'prob_{label}']=probabilities[:,list(model.classes_).index(label)] if label in model.classes_ else 0.
    return out


def oof_predictions(windows, X, model_key, purity, videos, seed=SEED, factory=estimator):
    predictions=[]; diagnostics=[]
    for fold in range(5):
        started=time.perf_counter()
        training,validation=get_cv_split(windows,fold,videos,min_purity_train=purity)
        if set(training.label)!=set(CLASSES):raise ValueError(f'Fold {fold} training lacks a class')
        model=factory(model_key,seed)
        model.fit(X.loc[training.index].to_numpy(),training.label.to_numpy())
        pred=prediction_table(model,X.loc[validation.index],validation)
        pred['cv_fold']=fold
        predictions.append(pred)
        majority=training.label.value_counts().idxmax()
        diagnostics.append(dict(fold=fold,n_train=len(training),n_validation=len(validation),
                                elapsed_s=round(time.perf_counter()-started,3),
                                accuracy=accuracy_score(validation.label,pred.pred_label),
                                macro_f1=f1_score(validation.label,pred.pred_label,labels=CLASSES,average='macro',zero_division=0),
                                majority_accuracy=float(validation.label.eq(majority).mean()),
                                confusion_matrix=confusion_matrix(validation.label,pred.pred_label,labels=CLASSES).tolist()))
        print(f'  {model_key} fold {fold}: fit/predict complete in {diagnostics[-1]["elapsed_s"]:.1f}s; '
              f'train={len(training)}, validation={len(validation)}, source overlap=0',flush=True)
    out=pd.concat(predictions).sort_index()
    if len(out)!=len(windows) or not out.index.is_unique:raise ValueError('Incomplete/duplicated OOF coverage')
    return out,diagnostics


def score_predictions(predictions,videos,tau,min_len):
    """Descriptive per-video scores and equally weighted fold means."""
    per_video=[]
    for vid,rows in predictions.groupby('video_id',sort=True):
        meta=require_v2(videos[vid]); rows=rows.sort_values('win_idx')
        truth,end=gt_from_json(meta)
        labels=smooth(rows.pred_label.tolist(),rows.max_prob.tolist(),tau=tau,min_len=min_len)
        segments=to_segments(labels,rows.t_center.tolist(),end)
        metrics=all_metrics(truth,segments,end,CLASSES)
        per_video.append(dict(video_id=vid,fold=meta['fold'],**metrics))
    table=pd.DataFrame(per_video)
    numeric=table.select_dtypes(include='number').columns.difference(['fold'])
    fold_means=table.groupby('fold',dropna=False)[numeric].mean()
    return table,fold_means


def validate_grid(purities,taus,min_lens):
    if not purities or any(not 0<=p<=1 for p in purities):raise ValueError('Purity must be in [0,1]')
    if not taus or any(not 0<=t<=1 for t in taus):raise ValueError('tau must be in [0,1]')
    if not min_lens or any(int(n)!=n or n<1 for n in min_lens):raise ValueError('min_len must be a positive integer')


def run_selection(windows_path=WINDOWS_CSV,features_dir=FEATURES_DIR,out_dir=OUTPUT_DIR,
                  model_keys=('rf',),purities=(MIN_PURITY_TRAIN,),taus=(.4,.5,.6,.7),
                  min_lens=(1,2,3,4),seed=SEED,experiment_id=None,videos=None):
    validate_grid(purities,taus,min_lens)
    if not model_keys or not set(model_keys)<=set(_build_classifiers(seed)):
        raise ValueError('Unknown/empty classifier list')
    windows,videos=read_training(windows_path,features_dir,videos)
    train_signature=signature(videos,windows,features_dir)
    X,channels=build_feature_matrix(windows,features_dir,videos)
    experiment_id=experiment_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',experiment_id):raise ValueError('Invalid experiment ID')
    destination=active_path(out_dir)/'experiments'/experiment_id
    destination.mkdir(parents=True,exist_ok=False)
    results=[]; best=None
    for key in sorted(set(model_keys)):
        for purity in sorted(set(purities),reverse=True):
            print(f'CV {key}, training purity {purity}: five source folds; test features untouched',flush=True)
            oof,diagnostics=oof_predictions(windows,X,key,purity,videos,seed)
            candidate_dir=destination/f'{key}_purity_{purity:g}'
            candidate_dir.mkdir()
            write_json(candidate_dir/'window_metrics.json',diagnostics)
            oof.to_csv(candidate_dir/'oof_predictions.csv',index=False)
            for tau in sorted(set(taus)):
                for min_len in sorted(set(min_lens)):
                    per_video,folds=score_predictions(oof,videos,tau,min_len)
                    row=dict(model_key=key,min_purity_train=purity,tau=tau,min_len=min_len,
                             cv_edit_mean=float(folds.edit_score.mean()),
                             cv_edit_std=float(folds.edit_score.std(ddof=1)),
                             cv_frame_accuracy_mean=float(folds.frame_accuracy.mean()),
                             cv_f1_at_25_mean=float(folds.f1_at_25.mean()))
                    results.append(row)
                    # Deterministic train-only selection: Edit, accuracy, then
                    # lexical model, higher purity, smaller smoothing parameters.
                    rank=(-row['cv_edit_mean'],-row['cv_frame_accuracy_mean'],key,-purity,tau,min_len)
                    if best is None or rank<best[0]:best=(rank,row.copy(),oof.copy(),per_video,folds)
    ranking=pd.DataFrame(results).sort_values(
        ['cv_edit_mean','cv_frame_accuracy_mean','model_key','min_purity_train','tau','min_len'],
        ascending=[False,False,True,False,True,True]).reset_index(drop=True)
    ranking.to_csv(destination/'cv_results.csv',index=False)
    best[2].to_csv(destination/'oof_predictions.csv',index=False)
    best[3].to_csv(destination/'cv_video_metrics.csv',index=False)
    best[4].to_csv(destination/'cv_fold_metrics.csv')
    selected=dict(run_id='v2',schema_version=SCHEMA_VERSION,experiment_id=experiment_id,
                  selection_scope='train_source_clip_cv',selection_metric='mean_fold_video_edit',
                  best=best[1],seed=seed,train_signature=train_signature,
                  feature_channels=channels,feature_cols=list(X.columns),preprocessing=PREPROCESSING,
                  classifier_params=_build_classifiers(seed)[best[1]['model_key']].get_params(),
                  software=dict(python=sys.version.split()[0],sklearn=sklearn.__version__,
                                numpy=np.__version__,pandas=pd.__version__),
                  n_train_windows=len(windows),n_train_videos=windows.video_id.nunique(),
                  oof_sha256=sha256(destination/'oof_predictions.csv'),
                  test_features_read=False,cv_estimate='selection CV; not an unbiased nested-CV estimate')
    write_json(destination/'selection.json',selected)
    print(f'Train-only selection saved: {destination/"selection.json"}',flush=True)
    return destination/'selection.json'


def refit_selection(selection_path,windows_path=WINDOWS_CSV,features_dir=FEATURES_DIR,
                    activate=False,out_dir=OUTPUT_DIR,videos=None):
    selection_path=active_path(selection_path)
    selected=require_v2(json.loads(selection_path.read_text()))
    if selected.get('selection_scope')!='train_source_clip_cv':raise ValueError('Expected train-only CV selection')
    if selected['schema_version']!=SCHEMA_VERSION or selected['preprocessing']!=PREPROCESSING:
        raise ValueError('Incompatible selection schema/preprocessing')
    current_params=_build_classifiers(selected['seed'])[selected['best']['model_key']].get_params()
    if json.dumps(current_params,sort_keys=True)!=json.dumps(selected['classifier_params'],sort_keys=True):
        raise ValueError('Classifier settings changed since CV selection')
    windows,videos=read_training(windows_path,features_dir,videos)
    if signature(videos,windows,features_dir)!=selected['train_signature']:
        raise ValueError('Training inputs changed since CV selection')
    X,channels=build_feature_matrix(windows,features_dir,videos)
    if list(X.columns)!=selected['feature_cols']:raise ValueError('Feature schema changed')
    eligible=windows.purity.ge(selected['best']['min_purity_train'])
    if set(windows.loc[eligible,'label'])!=set(CLASSES):raise ValueError('Final training lacks a class')
    destination=selection_path.parent/'final'
    destination.mkdir(exist_ok=False)
    model=estimator(selected['best']['model_key'],selected['seed'])
    model.fit(X.loc[eligible].to_numpy(),windows.loc[eligible,'label'].to_numpy())
    bundle=dict(run_id='v2',schema_version=SCHEMA_VERSION,experiment_id=selected['experiment_id'],
                fit_scope='all_train',model=model,classes=list(model.classes_),
                feature_cols=selected['feature_cols'],feature_channels=channels,
                preprocessing=PREPROCESSING,train_signature=selected['train_signature'],
                selection_sha256=sha256(selection_path),
                classifier_params=selected['classifier_params'],software=selected['software'],seed=selected['seed'],
                selection=selected['best'],n_fit_windows=int(eligible.sum()),
                fill_values=dict(zip(X.columns,model.named_steps['imputer'].statistics_.tolist())))
    model_path=destination/'model.joblib'
    joblib.dump(bundle,model_path)
    params=dict(run_id='v2',experiment_id=selected['experiment_id'],model_sha256=sha256(model_path),
                tau=selected['best']['tau'],min_len=selected['best']['min_len'])
    write_json(destination/'chosen_params.json',params)
    if activate:
        root=active_path(out_dir);root.mkdir(parents=True,exist_ok=True)
        pointer=root/'active_model.json'
        if pointer.exists():
            history=root/'activation_history';history.mkdir(exist_ok=True)
            pointer.rename(history/f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}.json')
        write_json(pointer,dict(run_id='v2',experiment_id=selected['experiment_id'],
                                model_path=str(model_path.resolve()),model_sha256=sha256(model_path)))
    return model_path


def load_bundle(model_path=None,out_dir=OUTPUT_DIR):
    pointer=None
    if model_path is None:
        pointer=require_v2(json.loads((active_path(out_dir)/'active_model.json').read_text()))
        model_path=pointer['model_path']
    path=active_path(model_path)
    if pointer is not None and sha256(path)!=pointer['model_sha256']:raise ValueError('Active model hash mismatch')
    bundle=require_v2(joblib.load(path))
    if bundle.get('schema_version')!=SCHEMA_VERSION or bundle.get('preprocessing')!=PREPROCESSING:
        raise ValueError('Incompatible model schema/preprocessing')
    if bundle.get('fit_scope')!='all_train':raise ValueError('Integration/final test requires an all-train refit')
    if set(bundle['classes'])!=set(CLASSES):raise ValueError('Incompatible model labels')
    if bundle.get('feature_channels')!=MODEL_CHANNELS or bundle.get('feature_cols')!=feature_names(MODEL_CHANNELS):
        raise ValueError('Incompatible feature schema/order')
    if list(bundle['model'].classes_)!=bundle['classes']:raise ValueError('Model class-order mismatch')
    if pointer is not None and pointer['experiment_id']!=bundle['experiment_id']:
        raise ValueError('Active model experiment mismatch')
    return bundle,path


def verify_bundle_training(bundle,windows_path=WINDOWS_CSV,features_dir=FEATURES_DIR,videos=None):
    training,videos=read_training(windows_path,features_dir,videos)
    if signature(videos,training,features_dir)!=bundle['train_signature']:
        raise ValueError('Model training provenance does not match active dataset')
    return videos


def bundle_smoothing(bundle,model_path,params_path=None):
    params=bundle['selection']
    if params_path is not None:
        external=require_v2(json.loads(active_path(params_path).read_text()))
        if external.get('experiment_id')!=bundle['experiment_id'] or external.get('model_sha256')!=sha256(model_path):
            raise ValueError('Smoothing parameters belong to a different model')
        if external['tau']!=params['tau'] or external['min_len']!=params['min_len']:
            raise ValueError('Smoothing parameters differ from frozen CV selection')
    return float(params['tau']),int(params['min_len'])


def infer_frame(bundle,frame,meta):
    """Canonical inference: GT labels do not participate in window creation."""
    require_v2(meta)
    if 'run_id' not in frame or not frame.run_id.eq('v2').all():raise ValueError('Feature run_id != v2')
    if frame.video_id.nunique()!=1 or frame.video_id.iloc[0]!=meta['video_id']:
        raise ValueError('Feature video mismatch')
    if not frame.split.eq(meta['split']).all():raise ValueError('Feature split mismatch')
    if not (frame.fold.eq(meta['fold']).all() if meta['fold'] is not None else frame.fold.isna().all()):
        raise ValueError('Feature fold mismatch')
    if frame.frame.tolist()!=list(range(0,meta['n_frames'],2)):
        raise ValueError('Incomplete or incompatible feature timeline')
    if bundle['feature_channels']!=MODEL_CHANNELS:raise ValueError('Feature channel mismatch')
    filled=fill_video_nans(align_channels(frame,bundle['feature_channels']),channels=bundle['feature_channels'])
    filled['seg_label']='transition'  # placeholder metadata only, never a feature
    windows=make_windows_for_video(filled,meta['video_id'],meta['split'])
    if not windows:raise ValueError('Video too short for configured windows')
    rows=pd.DataFrame(windows)
    rows['fold']=meta['fold'];rows['run_id']='v2'
    X=pd.DataFrame([window_features(filled.iloc[w['row_start']:w['row_end']]) for w in windows])
    X=X.reindex(columns=bundle['feature_cols']).replace([np.inf,-np.inf],np.nan)
    predictions=prediction_table(bundle['model'],X,rows)
    end=meta['n_frames']/meta['fps']
    tau,min_len=bundle_smoothing(bundle,None)
    labels=smooth(predictions.pred_label.tolist(),predictions.max_prob.tolist(),tau=tau,min_len=min_len)
    raw=to_segments(predictions.pred_label.tolist(),predictions.t_center.tolist(),end)
    segments=to_segments(labels,predictions.t_center.tolist(),end)
    return predictions,raw,segments,end


def evaluate_final(model_path,windows_path=WINDOWS_CSV,features_dir=FEATURES_DIR,out_dir=None,videos=None):
    """Explicit final-test scoring only; never selects or changes parameters."""
    bundle,model_path=load_bundle(model_path)
    videos=verify_bundle_training(bundle,windows_path,features_dir,videos)
    destination=active_path(out_dir) if out_dir is not None else model_path.parent/'test_evaluation'
    if destination.exists():raise FileExistsError('Evaluation output exists; preserve it and use a new directory')
    destination.mkdir(parents=True)
    results=[]; raw_results=[]; predictions=[]; segment_output={}
    for vid,meta in sorted(videos.items()):
        if meta['split']!='test':continue
        frame=feature_frame(features_dir/f'{vid}_angles.csv',meta)
        pred,raw,segments,end=infer_frame(bundle,frame,meta)
        truth,_=gt_from_json(meta)
        results.append(dict(video_id=vid,**all_metrics(truth,segments,end,CLASSES)))
        raw_results.append(dict(video_id=vid,**all_metrics(truth,raw,end,CLASSES)))
        predictions.append(pred);segment_output[vid]=segments
    if not results:raise ValueError('No test videos')
    table=pd.DataFrame(results);table.to_csv(destination/'video_metrics.csv',index=False)
    pd.DataFrame(raw_results).to_csv(destination/'raw_video_metrics.csv',index=False)
    pd.concat(predictions).to_csv(destination/'window_predictions_test.csv',index=False)
    write_json(destination/'test_segments.json',dict(run_id='v2',experiment_id=bundle['experiment_id'],videos=segment_output))
    numbers=table.select_dtypes(include='number')
    report=dict(run_id='v2',experiment_id=bundle['experiment_id'],scope='final_test',
                model_sha256=sha256(model_path),parameters_frozen=True,n_videos=len(table),
                n_source_clips=len({c for m in videos.values() if m['split']=='test' for c in m['clip_ids_in_order']}),
                mean=numbers.mean().to_dict(),std_across_videos=numbers.std(ddof=1).to_dict())
    write_json(destination/'results.json',report)
    return report
