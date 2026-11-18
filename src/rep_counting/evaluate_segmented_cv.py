"""Train OOF segmentation -> current versus development-selected rep counters."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.segmentation import workflow
from src.data_pipeline.v2_protocol import ROOT, SYNTH_DIR, FEATURES_DIR, sha256, feature_frame
from .rep_counter import EXERCISES, count_for_exercise
from .peak_improvements import ImprovementSettings, count_improved_peaks
from .coordinate_counter import CoordinatePeakSettings, count_coordinate_peaks
from .extract_coordinate_training import init_worker, add_signals, verify_case, signature, case_name
from . import extract_coordinate_training as extraction
from .extract_pushup_diagnostic import POINTS, extract_pass
from .evaluate_coordinate_frozen import paired_summary
from .analyze_peak_errors import assign_reps
from .validate_peak_counter import timing_metrics

EXPERIMENT=ROOT/'outputs/segmentation_v2/experiments/cv8_v2_seed42_p080_20261117'
PEAK_REPORT=ROOT/'outputs/peak_improvements/verified_cached_signal_comparison_20261118/report.json'
COORD_REPORT=ROOT/'outputs/pushup_coordinates/train_v1_validation_20261118/report.json'
CACHE=ROOT/'outputs/pushup_coordinates/train_v1_20261118'
RAW=['frame','time_s','detected','n_poses']+[f'{p}_{c}' for p in POINTS for c in ('x','y','vis')]


def even_frames(start,end,fps,n_frames,padding=.3):
    first=int(np.ceil(start*fps-1e-8)); first+=first%2
    stop=min(n_frames,int(np.ceil((end+padding)*fps-1e-8)))
    return list(range(first,stop,2))


def missing_runs(frames):
    frames=sorted(set(frames))
    if not frames: return []
    boundaries=np.flatnonzero(np.diff(frames)>2)+1
    return [(int(a[0]),int(a[-1])+1) for a in np.split(np.asarray(frames),boundaries)]


def extract_missing(task):
    if task['split']!='train': raise ValueError('No test frames in training CV')
    frame,_,_=extract_pass(extraction.LANDMARKER,Path(task['video_path']),task['start'],task['end'],task['fps'])
    if frame.frame.tolist()!=list(range(task['start'],task['end'],2)):
        raise ValueError('Incomplete missing-frame extraction')
    return task,frame[RAW]


def verify_oof(predictions,windows,videos):
    workflow.validate_windows(predictions,videos)
    if not predictions.split.eq('train').all() or not predictions.cv_fold.eq(predictions.fold).all():
        raise ValueError('Expected held-out training OOF rows only')
    key=['video_id','win_idx']
    if predictions.duplicated(key).any() or set(map(tuple,predictions[key].to_numpy()))!=set(map(tuple,windows[key].to_numpy())):
        raise ValueError('OOF coverage differs from training windows')


def evaluate(output,workers=2):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Use a new output directory under outputs')
    selection=json.loads((EXPERIMENT/'selection.json').read_text())
    windows,videos=workflow.read_training()
    if workflow.signature(videos,windows,FEATURES_DIR)!=selection['train_signature']:
        raise ValueError('Segmentation training inputs changed')
    oof_path=EXPERIMENT/'oof_predictions.csv'
    if sha256(oof_path)!=selection['oof_sha256']: raise ValueError('OOF predictions changed')
    predictions=pd.read_csv(oof_path); verify_oof(predictions,windows,videos)
    train={v:m for v,m in videos.items() if m['split']=='train'}
    if len(train)!=120: raise ValueError('Expected all 120 registered training videos')
    peak=json.loads(PEAK_REPORT.read_text()); coord=json.loads(COORD_REPORT.read_text())
    for report in (peak,coord):
        for name,digest in report['source_hashes'].items():
            if sha256(Path(__file__).with_name(name))!=digest: raise ValueError('Development implementation changed: '+name)
    settings={fold:{} for fold in range(5)}
    for row in peak['checks']:
        if row['exercise'] in ('jump_jack','squat'):
            settings[row['fold']][row['exercise']]=row['settings']
    for row in coord['by_family']['median']['checks']:
        settings[row['fold']]['push_up']=row['settings']
    if any(set(s)!=set(EXERCISES) for s in settings.values()): raise ValueError('Incomplete other-fold settings')
    protected={str(p):sha256(p) for p in (EXPERIMENT/'selection.json',oof_path,PEAK_REPORT,COORD_REPORT,CACHE/'plan.json')}
    model=ROOT/'models/pose_landmarker_full.task'; protected[str(model)]=sha256(model)
    cache_plan=json.loads((CACHE/'plan.json').read_text())['plan']
    if sha256(model)!=cache_plan['model_sha256']: raise ValueError('Pose model differs from cached extraction')
    for name,digest in cache_plan['source_hashes'].items():
        if sha256(Path(__file__).with_name(name))!=digest: raise ValueError('Pose extraction protocol changed')
    segments=[]; segment_scores=[]; requested={}; raw_cache={}; tasks=[]
    for vid,meta in sorted(train.items()):
        for p in (FEATURES_DIR/f'{vid}_angles.csv',SYNTH_DIR/'train'/f'{vid}.json',SYNTH_DIR/'train'/f'{vid}.mp4'):
            protected[str(p)]=sha256(p)
        rows=predictions[predictions.video_id.eq(vid)].sort_values('win_idx')
        truth,end=workflow.gt_from_json(meta)
        labels=workflow.smooth(rows.pred_label.tolist(),rows.max_prob.tolist(),selection['best']['tau'],selection['best']['min_len'])
        predicted=workflow.to_segments(labels,rows.t_center.tolist(),end)
        segment_scores.append(dict(video_id=vid,fold=meta['fold'],**workflow.all_metrics(truth,predicted,end,workflow.CLASSES)))
        needed=set()
        for idx,segment in enumerate(predicted):
            item=dict(video_id=vid,fold=meta['fold'],segment_index=idx,**segment)
            segments.append(item)
            if segment['exercise']=='push_up': needed.update(even_frames(segment['start_time'],segment['end_time'],meta['fps'],meta['n_frames']))
        requested[vid]=sorted(needed)
        cached=[]
        for case in cache_plan['cases']:
            if case['video_id']!=vid: continue
            verify_case(CACHE,case,signature(cache_plan))
            p=CACHE/(case_name(case)+'.csv'); protected[str(p)]=sha256(p)
            cached.append(pd.read_csv(p,usecols=RAW))
        combined=pd.concat(cached) if cached else pd.DataFrame(columns=RAW)
        if combined.frame.duplicated().any(): raise ValueError('Ambiguous cached frame observations')
        combined=combined[combined.frame.isin(needed)].copy()
        raw_cache[vid]=[combined] if len(combined) else []
        for start,stop in missing_runs(needed-set(combined.frame)):
            tasks.append(dict(video_id=vid,split='train',video_path=str(SYNTH_DIR/'train'/f'{vid}.mp4'),fps=meta['fps'],start=start,end=stop))
    output.mkdir(parents=True); (output/'coordinate_frames').mkdir()
    plan=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),split='train',n_videos=len(train),
              segmentation_selection=selection['best'],counter_settings_by_heldout_fold=settings,
              no_new_tuning=True,end_padding_s=.3,cache_policy='Reuse only raw pixel landmarks/visibility from hash-verified frozen IMAGE sidecars; extract every missing requested frame with same pose protocol; recompute torso scale/signals within each predicted window.',
              requested_coordinate_frames=sum(map(len,requested.values())),reused_coordinate_frames=sum(len(x) for parts in raw_cache.values() for x in parts),
              missing_frame_runs=tasks,protected_hashes=protected,source_sha256=sha256(__file__))
    (output/'evaluation_plan.json').write_text(json.dumps(plan,indent=2))
    pd.DataFrame(segments).to_csv(output/'predicted_segments.csv',index=False)
    print(f"120 train OOF videos; {sum(s['exercise']=='push_up' for s in segments)} predicted push-up windows; {plan['reused_coordinate_frames']}/{plan['requested_coordinate_frames']} raw frames reused; {len(tasks)} missing runs",flush=True)
    if tasks:
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(model),)) as pool:
            futures=[pool.submit(extract_missing,task) for task in tasks]
            for done,future in enumerate(as_completed(futures),1):
                task,frame=future.result(); raw_cache[task['video_id']].append(frame)
                frame.to_csv(output/'coordinate_frames'/f"new__{task['video_id']}__{task['start']}.csv",index=False)
                if done%10==0 or done==len(tasks): print(f'Extracted missing runs {done}/{len(tasks)}',flush=True)
    assembled={}
    for vid,frames in raw_cache.items():
        if not requested[vid]: continue
        frame=pd.concat(frames).sort_values('frame').reset_index(drop=True)
        if frame.frame.tolist()!=requested[vid]: raise ValueError('Incomplete/duplicated predicted-window coordinates')
        frame.to_csv(output/'coordinate_frames'/f'{vid}.csv',index=False); assembled[vid]=frame
    video_rows=[]; counted_rows=[]; timing_rows=[]
    for vid,meta in sorted(train.items()):
        frame=feature_frame(FEATURES_DIR/f'{vid}_angles.csv',meta)
        truth={e:[] for e in EXERCISES}
        for segment in meta['segments']:
            if segment.get('label') in truth:
                truth[segment['label']].extend((a/meta['fps'],b/meta['fps']) for a,b in segment['rep_pairs_out'])
        times={method:{e:[] for e in EXERCISES} for method in ('current','experimental')}
        for segment in (s for s in segments if s['video_id']==vid and s['exercise'] in EXERCISES):
            ex=segment['exercise']; start,end=segment['start_time'],segment['end_time']
            old=count_for_exercise(frame,ex,start,end,end_pad_s=.3)
            cfg=settings[meta['fold']][ex]
            if ex=='push_up':
                ids=even_frames(start,end,meta['fps'],meta['n_frames'])
                raw=assembled[vid][assembled[vid].frame.isin(ids)].copy()
                if len(raw)>=8:
                    derived,_=add_signals(raw)
                    new=count_coordinate_peaks(derived,CoordinatePeakSettings(**cfg))
                else: new=dict(reps=0,rep_times=[],selected_signal='too_short')
            else: new=count_improved_peaks(frame,ex,start,end+.3,settings=ImprovementSettings(**cfg))
            for method,result in [('current',old),('experimental',new)]:
                times[method][ex].extend(result['rep_times'])
                counted_rows.append(segment|dict(method=method,predicted=result['reps'],rep_times=result['rep_times'],selected_signal=result.get('selected_signal',result.get('signal',''))))
        for ex in EXERCISES:
            video_rows.append(dict(video_id=vid,fold=meta['fold'],exercise=ex,annotated=len(truth[ex]),old_predicted=len(times['current'][ex]),predicted=len(times['experimental'][ex])))
            for method in times:
                hits,_,_=assign_reps(times[method][ex],truth[ex])
                timing_rows.append(dict(video_id=vid,fold=meta['fold'],exercise=ex,method=method,annotated=len(truth[ex]),predicted=len(times[method][ex]),hits=len(hits)))
        print(f'Counted OOF {vid}',flush=True)
    rows=pd.DataFrame(video_rows); rows['error']=rows.predicted-rows.annotated
    scores=pd.DataFrame(segment_scores); timing=pd.DataFrame(timing_rows)
    totals=rows.groupby(['video_id','fold'])[['annotated','old_predicted','predicted']].sum().reset_index()
    rows.to_csv(output/'video_exercise_comparison.csv',index=False)
    totals.to_csv(output/'whole_video_comparison.csv',index=False)
    pd.DataFrame(counted_rows).to_csv(output/'counted_segments.csv',index=False)
    timing.to_csv(output/'timing.csv',index=False); scores.to_csv(output/'segmentation_metrics.csv',index=False)
    return finalize(output)


def finite_json(value):
    if isinstance(value,dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list): return [finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    return value


def finalize(output):
    """Finalize saved, complete count tables without any pose/count inference."""
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()) or (output/'report.json').exists():
        raise ValueError('Finalize only a new report under outputs')
    plan=json.loads((output/'evaluation_plan.json').read_text())
    protected=plan['protected_hashes']
    rows=pd.read_csv(output/'video_exercise_comparison.csv')
    scores=pd.read_csv(output/'segmentation_metrics.csv')
    timing=pd.read_csv(output/'timing.csv')
    if len(rows)!=360 or rows.video_id.nunique()!=120 or rows.duplicated(['video_id','exercise']).any() or set(rows.fold)!=set(range(5)):
        raise ValueError('Incomplete saved training comparison')
    if not rows.groupby('video_id').exercise.apply(lambda s:set(s)==set(EXERCISES)).all():
        raise ValueError('Missing exercise in saved comparison')
    if len(scores)!=120 or set(scores.video_id)!=set(rows.video_id) or len(timing)!=720:
        raise ValueError('Incomplete saved segmentation/timing tables')
    comparison=paired_summary(rows)
    by_exercise={ex:paired_summary(group) for ex,group in rows.groupby('exercise')}
    totals=rows.groupby(['video_id','fold'])[['annotated','old_predicted','predicted']].sum().reset_index()
    report=dict(split='train',scope='OOF segmentation + fold-selected rep counters; exploratory selection-CV',n_videos=120,
                overall_video_exercise=comparison,by_exercise=by_exercise,
                active_by_exercise={ex:paired_summary(group[group.annotated.gt(0)]) for ex,group in rows.groupby('exercise')},
                whole_video=paired_summary(totals),
                fold_mae={method:rows.assign(a=abs(rows[column]-rows.annotated)).groupby(['fold','exercise']).a.mean().unstack().to_dict() for method,column in [('current','old_predicted'),('experimental','predicted')]},
                timing={method:{ex:timing_metrics(group) for ex,group in timing[timing.method.eq(method)].groupby('exercise')} for method in ('current','experimental')},
                segmentation_fold_metrics=scores.groupby('fold').mean(numeric_only=True).to_dict(),
                undefined_segmentation_boundary_metrics=[dict(video_id=r.video_id,boundary_mean_s=None,boundary_median_s=None) for r in scores.itertuples() if not np.isfinite(r.boundary_mean_s) or not np.isfinite(r.boundary_median_s)],
                serialization_note='Non-finite segmentation boundary distances are null in JSON, never zero. Raw per-video CSV retains infinity; count scores are finite and unchanged.',
                finalizer_source_sha256=sha256(__file__),count_table_hashes={name:sha256(output/name) for name in ('video_exercise_comparison.csv','whole_video_comparison.csv','counted_segments.csv','timing.csv','segmentation_metrics.csv')},
                reused_coordinate_frames=plan['reused_coordinate_frames'],new_coordinate_frames=plan['requested_coordinate_frames']-plan['reused_coordinate_frames'],
                limitations=['All 120 train videos, GT used only for scores and existing extraction-cache availability; segmentation uses source-held-out OOF predictions.','Rep hyperparameters reuse prior other-fold development choices; no new tuning on segmentation results.','Median family and segmentation smoothing/model were selected after development inspection: not unbiased nested CV.','Different held-out folds can use different rep settings; this is not a new score for the single all-training frozen test config.','Video/exercise MAE averages 360 synthetic units, distinct from GT source-weighted window MAE.','Synthetic source reuse reduces independence; paired video bootstrap is descriptive, not source-independent inference.','Canonical upstream segmentation features use annotation-assisted person tracking; coordinate person selection uses largest pose per frame.','New coordinate frames use identical IMAGE/stride-2 protocol; only raw landmarks reused, signals renormalized per predicted window.','No test feature/video inference, canonical writes, default changes, commits or pushes.'])
    for p,h in protected.items():
        if sha256(p)!=h: raise ValueError('Protected input/result changed: '+p)
    report['inputs_and_previous_results_verified_unchanged']=True
    (output/'report.json').write_text(json.dumps(finite_json(report),indent=2,allow_nan=False))
    print(json.dumps(dict(overall=comparison,by_exercise=by_exercise),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--finalize-only',action='store_true',help='Finalize complete saved count tables without repeating inference')
    args=parser.parse_args()
    finalize(args.out_dir) if args.finalize_only else evaluate(args.out_dir,args.workers)
