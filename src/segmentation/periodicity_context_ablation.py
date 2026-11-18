"""Opt-in train-only RF feature ablations; canonical segmenter stays unchanged."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate, detrend, find_peaks, periodogram

from src.segmentation import workflow
from src.segmentation.diagnose_oof_classes import class_metrics, duration_confusion, LABELS
from src.data_pipeline.v2_protocol import ROOT, FEATURES_DIR, SYNTH_DIR, sha256, feature_frame
from src.rep_counting.evaluate_segmented_cv import finite_json
from src.rep_counting.peak_improvements import ImprovementSettings, count_improved_peaks

EXPERIMENT=ROOT/'outputs/segmentation_v2/experiments/cv8_v2_seed42_p080_20261117'
COUNTER_PLAN=ROOT/'outputs/pushup_coordinates/segmented_training_cv_20261118/evaluation_plan.json'
FAILED_WINDOWS=ROOT/'outputs/pushup_coordinates/jumping_jack_window_failures_20261118/gt_windows.csv'
PERIOD_CHANNELS=['arm_raise_L','arm_raise_R','ankle_dist_norm','hip_L','hip_R','knee_best']
PERIOD_STATS=['dominant_hz','power_1_3_fraction','acf_peak']
CONTEXT_SECONDS=8.


def periodic_features(values,fs):
    """Use a long finite run only; no interpolation across remaining gaps."""
    values=np.asarray(values,dtype=float)
    invalid=dict.fromkeys(PERIOD_STATS,float('nan'))
    if len(values)<12 or not np.isfinite(fs) or fs<=0 or np.isfinite(values).mean()<.8: return invalid
    edges=np.flatnonzero(np.diff(np.r_[False,np.isfinite(values),False]))
    runs=[values[a:b] for a,b in zip(edges[::2],edges[1::2])]
    signal=max(runs,key=len) if runs else np.array([])
    if len(signal)<12 or len(signal)<.8*len(values): return invalid
    signal=detrend(signal,type='linear')
    if np.max(np.abs(signal))<1e-8: return dict.fromkeys(PERIOD_STATS,0.)
    frequencies,power=periodogram(signal,fs=fs,window='hann',detrend=False,scaling='spectrum')
    positive=frequencies>0
    total=float(power[positive].sum())
    dominant=float(frequencies[positive][np.argmax(power[positive])])
    band=float(power[(frequencies>=1)&(frequencies<=3)].sum()/total) if total>0 else 0.
    acf=correlate(signal,signal,mode='full',method='auto')[len(signal)-1:]
    acf=acf/acf[0]
    peaks,_=find_peaks(acf)
    peaks=peaks[(peaks>=max(1,int(np.ceil(fs/6))))&(peaks<=min(len(signal)-2,int(np.floor(fs*1.5))))]
    height=max(0.,float(acf[peaks].max())) if len(peaks) else 0.
    return dict(dominant_hz=dominant,power_1_3_fraction=band,acf_peak=height)


def context_bounds(times,center,width=CONTEXT_SECONDS):
    """Centered offline context, clipped only to video extent, never GT labels."""
    return np.searchsorted(times,[center-width/2,center+width/2],side='left')


def build_features(windows,videos):
    base_rows={}; periodic_rows={}; context_rows={}; frames={}
    channels=workflow.MODEL_CHANNELS
    columns=workflow.feature_names(channels)
    for number,(vid,rows) in enumerate(windows.groupby('video_id',sort=True),1):
        meta=videos[vid]
        if meta['split']!='train': raise ValueError('No test feature reads in ablation')
        frame=feature_frame(FEATURES_DIR/f'{vid}_angles.csv',meta); frames[vid]=frame
        filled=workflow.fill_video_nans(workflow.align_channels(frame,channels),channels=channels)
        times=filled.time_s.to_numpy(); fs=1/float(np.median(np.diff(times)))
        for index,window in rows.iterrows():
            start,end=int(window.row_start),int(window.row_end)
            if not 0<=start<end<=len(filled): raise ValueError('Invalid base window')
            local=filled.iloc[start:end]
            base_rows[index]=workflow.window_features(local)
            periodic_rows[index]={f'period2_{channel}_{stat}':value for channel in PERIOD_CHANNELS
                                  for stat,value in periodic_features(local[channel].to_numpy() if channel in local else np.full(len(local),np.nan),fs).items()}
            left,right=context_bounds(times,float(window.t_center))
            context_rows[index]={'context8_'+key:value for key,value in workflow.window_features(filled.iloc[left:right]).items()}
        if number%10==0 or number==120: print(f'Feature extraction {number}/120 videos',flush=True)
    base=pd.DataFrame.from_dict(base_rows,orient='index').reindex(index=windows.index,columns=columns)
    period=pd.DataFrame.from_dict(periodic_rows,orient='index').reindex(windows.index)
    context=pd.DataFrame.from_dict(context_rows,orient='index').reindex(index=windows.index,columns=['context8_'+c for c in columns])
    matrices=dict(baseline=base,periodicity=pd.concat([base,period],axis=1),context=pd.concat([base,context],axis=1),both=pd.concat([base,period,context],axis=1))
    return {name:x.replace([np.inf,-np.inf],np.nan) for name,x in matrices.items()},frames


def evaluate(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to((ROOT/'outputs').resolve()): raise ValueError('Use a new directory under outputs')
    selection=json.loads((EXPERIMENT/'selection.json').read_text())
    windows,videos=workflow.read_training()
    if workflow.signature(videos,windows,FEATURES_DIR)!=selection['train_signature']: raise ValueError('Training signature differs')
    saved=pd.read_csv(EXPERIMENT/'oof_predictions.csv')
    if sha256(EXPERIMENT/'oof_predictions.csv')!=selection['oof_sha256']: raise ValueError('Baseline OOF changed')
    if selection['best']['model_key']!='rf': raise ValueError('Expected fixed RF baseline')
    counter_plan=json.loads(COUNTER_PLAN.read_text())
    failed=pd.read_csv(FAILED_WINDOWS); failed=failed[failed.mostly_mislabeled]
    protected={str(p):sha256(p) for p in (EXPERIMENT/'selection.json',EXPERIMENT/'oof_predictions.csv',COUNTER_PLAN,FAILED_WINDOWS,workflow.WINDOWS_CSV)}
    for vid in windows.video_id.unique():
        for p in (FEATURES_DIR/f'{vid}_angles.csv',SYNTH_DIR/'train'/f'{vid}.json'): protected[str(p)]=sha256(p)
    for p,h in counter_plan['protected_hashes'].items():
        if sha256(p)!=h: raise ValueError('Prior counter comparison inputs changed')
    source_paths=[Path(__file__),Path(workflow.__file__),Path(workflow.__file__).with_name('features.py'),Path(workflow.__file__).with_name('smooth.py'),ROOT/'src/rep_counting/peak_improvements.py',ROOT/'src/rep_counting/rep_counter.py',ROOT/'src/rep_counting/peak_counter.py']
    source_hashes={str(p):sha256(p) for p in source_paths}
    output.mkdir(parents=True)
    plan=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),split='train',n_videos=windows.video_id.nunique(),n_windows=len(windows),
        model='Same RF: 200 balanced trees, seed 42; original train-purity 0.8 and smoothing tau 0.5/min_len 3.',
        groups=['baseline','periodicity','context','both'],period_channels=PERIOD_CHANNELS,period_stats=PERIOD_STATS,context_seconds=CONTEXT_SECONDS,
        spectral_policy='2 s original window; Hann periodogram after linear detrending; positive-frequency dominant bin and 1-3 Hz fraction; highest positive local ACF peak at lags 1/6 to 1.5 s. At least 80% finite in one run, >=12 samples; flat signals return zero; unavailable signals remain NaN.',
        context_policy='112 original summary statistics over centered 8 s interval at same 1 s stride; clipped only to current video. Future lookahead is offline.',
        no_new_hyperparameter_search=True,source_hashes=source_hashes,protected_inputs=protected,
        counter_settings_by_fold={f:cfg['jump_jack'] for f,cfg in counter_plan['counter_settings_by_heldout_fold'].items()},
        software={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn']})
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    matrices,frames=build_features(windows,videos)
    summaries={}; importance_rows=[]
    for name,X in matrices.items():
        directory=output/name; directory.mkdir()
        print(f'Ablation {name}: {len(X.columns)} features, five grouped folds',flush=True)
        def factory(model_key,seed):
            return workflow.estimator(model_key,seed)
        oof,diagnostics=workflow.oof_predictions(windows,X,'rf',selection['best']['min_purity_train'],videos,selection['seed'],factory=factory)
        if not oof.cv_fold.eq(oof.fold).all(): raise ValueError('Invalid held-out fold')
        if name=='baseline':
            key=['video_id','win_idx']; left=oof.sort_values(key); right=saved.sort_values(key)
            if left[key].to_numpy().tolist()!=right[key].to_numpy().tolist() or left.pred_label.tolist()!=right.pred_label.tolist() or not np.allclose(left.max_prob,right.max_prob,atol=1e-10,rtol=0):
                raise ValueError('Baseline refit does not reproduce saved OOF: stop ablation interpretation')
        per_video,folds=workflow.score_predictions(oof,videos,selection['best']['tau'],selection['best']['min_len'])
        stage_rows=[]; temporal=pd.DataFrame(0.,index=LABELS,columns=LABELS); cases=[]; jj_counts=[]
        for vid,rows in oof.groupby('video_id',sort=True):
            meta=videos[vid]; rows=rows.sort_values('win_idx').copy()
            labels=workflow.smooth(rows.pred_label.tolist(),rows.max_prob.tolist(),selection['best']['tau'],selection['best']['min_len'])
            rows['smoothed_label']=labels; stage_rows.append(rows)
            truth,end=workflow.gt_from_json(meta)
            segments=workflow.to_segments(labels,rows.t_center.tolist(),end)
            temporal+=duration_confusion(truth,segments)
            annotated=sum(s['rep_count_annotated'] for s in meta['segments'] if s.get('label')=='jump_jack')
            count=0; times=[]
            for seg in segments:
                if seg['exercise']=='jump_jack':
                    counted=count_improved_peaks(frames[vid],'jump_jack',seg['start_time'],seg['end_time']+.3,settings=ImprovementSettings(**plan['counter_settings_by_fold'][str(meta['fold'])]))
                    count+=counted['reps']; times.extend(counted['rep_times'])
            jj_counts.append(dict(video_id=vid,fold=meta['fold'],annotated=annotated,predicted=count,error=count-annotated,rep_times=times))
            for case in failed[failed.video_id.eq(vid)].to_dict('records'):
                gt=[dict(exercise='jump_jack',start_time=case['gt_start_s'],end_time=case['gt_end_s'])]
                durations=duration_confusion(gt,segments).loc['jump_jack']
                cases.append(dict(video_id=vid,clip_id=case['clip_id'],dominant_label=durations.idxmax(),jj_coverage_fraction=float(durations['jump_jack']/case['gt_length_s'])))
        stage=pd.concat(stage_rows)
        confusion=pd.crosstab(stage.label,stage.smoothed_label).reindex(index=LABELS,columns=LABELS,fill_value=0)
        counts=pd.DataFrame(jj_counts)
        if name=='baseline' and not np.isclose(counts.error.abs().mean(),5.241666666666666): raise ValueError('Fixed counter baseline differs from previous run')
        summaries[name]=dict(n_features=len(X.columns),edit_mean=float(folds.edit_score.mean()),edit_fold_sd=float(folds.edit_score.std(ddof=1)),
            frame_accuracy_mean=float(folds.frame_accuracy.mean()),f1_at_25_mean=float(folds.f1_at_25.mean()),
            smoothed_window_metrics=class_metrics(confusion),exact_temporal_metrics=class_metrics(temporal),
            jj_video_mae=float(counts.error.abs().mean()),jj_video_median_absolute_error=float(counts.error.abs().median()),jj_exact_videos=int(counts.error.eq(0).sum()),jj_within_one_videos=int(counts.error.abs().le(1).sum()),
            originally_failed_19_dominant_labels=pd.DataFrame(cases).dominant_label.value_counts().to_dict(),fold_diagnostics=diagnostics,
            feature_columns=X.columns.tolist())
        for filename,table in [('oof_predictions',stage),('cv_video_metrics',per_video),('cv_fold_metrics',folds.reset_index()),('jj_video_counts',counts),('original_19_windows',pd.DataFrame(cases))]: table.to_csv(directory/(filename+'.csv'),index=False)
        confusion.rename_axis('ground_truth').to_csv(directory/'smoothed_window_confusion.csv')
        temporal.rename_axis('ground_truth').to_csv(directory/'temporal_confusion_seconds.csv')
        (output/'progress.json').write_text(json.dumps(finite_json(summaries),indent=2,allow_nan=False))
        print(f"Completed {name}: Edit {summaries[name]['edit_mean']:.3f}, JJ recall {summaries[name]['exact_temporal_metrics']['jump_jack']['recall']:.3%}, fixed-counter JJ MAE {summaries[name]['jj_video_mae']:.3f}",flush=True)
    for p,h in {**protected,**source_hashes}.items():
        if sha256(p)!=h: raise ValueError('Protected input/code changed during experiment')
    report=dict(split='train',n_videos=120,n_windows=len(windows),ablations=summaries,inputs_and_code_verified_unchanged=True,baseline_oof_reproduced=True,
        limitations=['Four predeclared feature groups; RF/purity/smoothing and per-fold JJ counter fixed. No audit exclusions.','Grouped source CV prevents source leakage but model/family choices used development data; not unbiased nested CV.','Centered 8 s context uses future frames and may cross synthetic exercise boundaries inside the same held-out video.','2 s spectra have coarse frequency resolution; short-gap filling is the original preprocessing, missing features imputed within fitting fold only.','Dominant frequency is not guaranteed rep frequency; harmonics/camera motion/other exercises may be periodic.','JJ count metrics use all 120 videos including absent zeros; this is downstream JJ diagnostic, not all-exercise rep evaluation.','The error-selected 19-window cohort is descriptive only. No test inference, activation, canonical data writes, counter changes, commits or pushes.'])
    (output/'report.json').write_text(json.dumps(finite_json(report),indent=2,allow_nan=False))
    table=pd.DataFrame([dict(group=name,**{k:v for k,v in value.items() if k in ['n_features','edit_mean','edit_fold_sd','frame_accuracy_mean','f1_at_25_mean','jj_video_mae','jj_exact_videos','jj_within_one_videos']},jj_recall=value['exact_temporal_metrics']['jump_jack']['recall']) for name,value in summaries.items()])
    table.to_csv(output/'comparison.csv',index=False); print(table.to_string(index=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args(); evaluate(args.out_dir)
