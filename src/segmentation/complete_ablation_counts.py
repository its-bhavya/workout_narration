"""Complete all-class counts from saved ablation OOF; counter settings fixed."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.segmentation import workflow
from src.data_pipeline.v2_protocol import ROOT, SYNTH_DIR, FEATURES_DIR, sha256, feature_frame
from src.rep_counting.evaluate_segmented_cv import RAW, CACHE, even_frames, missing_runs, extract_missing, finite_json
from src.rep_counting.extract_coordinate_training import init_worker, add_signals, verify_case, signature, case_name
from src.rep_counting.coordinate_counter import CoordinatePeakSettings, count_coordinate_peaks
from src.rep_counting.peak_improvements import ImprovementSettings, count_improved_peaks
from src.rep_counting.evaluate_coordinate_frozen import paired_summary

ABLATION=ROOT/'outputs/segmentation_v2/experiments/periodicity_context_20261118'
BASELINE=ROOT/'outputs/pushup_coordinates/segmented_training_cv_20261118'
GROUPS=['periodicity','context','both']


def merge_raw_frames(primary,secondary,requested):
    """Prefer prior baseline observations; use GT caches only for absent frames."""
    result=primary[primary.frame.isin(requested)][RAW].copy()
    secondary=secondary[secondary.frame.isin(set(requested)-set(result.frame))][RAW]
    result=pd.concat([result,secondary],ignore_index=True)
    if result.frame.duplicated().any(): raise ValueError('Ambiguous coordinate frame reuse')
    return result.apply(pd.to_numeric,errors='raise').sort_values('frame').reset_index(drop=True)


def complete(output,workers=2,resume_counts=False):
    output=Path(output).resolve()
    execution_hash=sha256(__file__)
    if not output.is_relative_to((ROOT/'outputs').resolve()) or (output.exists() and not resume_counts) or (output/'report.json').exists(): raise ValueError('Use a new output directory under outputs, or explicitly resume incomplete saved counts')
    prior=json.loads((BASELINE/'evaluation_plan.json').read_text())
    ablation=json.loads((ABLATION/'plan.json').read_text())
    original_report=json.loads((BASELINE/'report.json').read_text())
    if not original_report['inputs_and_previous_results_verified_unchanged']: raise ValueError('Baseline incomplete')
    protected={str(p):sha256(p) for p in (BASELINE/'report.json',BASELINE/'evaluation_plan.json',ABLATION/'report.json',ABLATION/'plan.json',BASELINE/'video_exercise_comparison.csv',CACHE/'plan.json')}
    for path,digest in {**prior['protected_hashes'],**ablation['protected_inputs'],**ablation['source_hashes']}.items():
        if sha256(path)!=digest: raise ValueError('Saved experiment inputs/code changed: '+path)
        protected[path]=digest
    windows,videos=workflow.read_training(); videos={v:m for v,m in videos.items() if m['split']=='train'}
    if len(videos)!=120: raise ValueError('Expected 120 training videos')
    truth=pd.read_csv(BASELINE/'video_exercise_comparison.csv')
    truth=truth[['video_id','fold','exercise','annotated','predicted']].rename(columns={'predicted':'old_predicted'})
    selections=prior['counter_settings_by_heldout_fold']
    segments={}; jj_counts={}; requests={vid:set() for vid in videos}
    for group in GROUPS:
        paths=[ABLATION/group/n for n in ('oof_predictions.csv','jj_video_counts.csv')]
        for p in paths: protected[str(p)]=sha256(p)
        oof=pd.read_csv(paths[0]); jj_counts[group]=pd.read_csv(paths[1])
        if not oof.cv_fold.eq(oof.fold).all() or not oof.split.eq('train').all() or set(oof.video_id)!=set(videos): raise ValueError('Invalid ablation OOF')
        if oof.duplicated(['video_id','win_idx']).any() or set(map(tuple,oof[['video_id','win_idx']].to_numpy()))!=set(map(tuple,windows[['video_id','win_idx']].to_numpy())): raise ValueError('Incomplete OOF windows')
        segments[group]={}
        for vid,rows in oof.groupby('video_id',sort=True):
            meta=videos[vid]; rows=rows.sort_values('win_idx')
            expected=workflow.smooth(rows.pred_label.tolist(),rows.max_prob.tolist(),.5,3)
            if rows.smoothed_label.tolist()!=expected: raise ValueError('Ablation smoothing differs')
            predicted=workflow.to_segments(expected,rows.t_center.tolist(),meta['n_frames']/meta['fps'])
            segments[group][vid]=predicted
            for seg in predicted:
                if seg['exercise']=='push_up': requests[vid].update(even_frames(seg['start_time'],seg['end_time'],meta['fps'],meta['n_frames']))
    cache_plan=json.loads((CACHE/'plan.json').read_text())['plan']
    model=ROOT/'models/pose_landmarker_full.task'
    if sha256(model)!=cache_plan['model_sha256']: raise ValueError('Pose model changed')
    protected[str(model)]=sha256(model)
    for name,digest in cache_plan['source_hashes'].items():
        path=ROOT/'src/rep_counting'/name
        if sha256(path)!=digest: raise ValueError('Coordinate extraction code changed')
        protected[str(path)]=digest
    frames={}; tasks=[]; reused=0
    for vid,meta in sorted(videos.items()):
        compiled=BASELINE/'coordinate_frames'/f'{vid}.csv'
        if compiled.exists():
            protected[str(compiled)]=sha256(compiled); primary=pd.read_csv(compiled,usecols=RAW)
        else: primary=pd.DataFrame(columns=RAW)
        extra=[]
        for case in cache_plan['cases']:
            if case['video_id']!=vid: continue
            verify_case(CACHE,case,signature(cache_plan))
            path=CACHE/(case_name(case)+'.csv'); protected[str(path)]=sha256(path)
            extra.append(pd.read_csv(path,usecols=RAW))
        secondary=pd.concat(extra) if extra else pd.DataFrame(columns=RAW)
        combined=merge_raw_frames(primary,secondary,requests[vid]); reused+=len(combined)
        frames[vid]=[combined] if len(combined) else []
        for start,end in missing_runs(requests[vid]-set(combined.frame)):
            tasks.append(dict(video_id=vid,split='train',video_path=str(SYNTH_DIR/'train'/f'{vid}.mp4'),fps=meta['fps'],start=start,end=end))
        protected[str(SYNTH_DIR/'train'/f'{vid}.mp4')]=sha256(SYNTH_DIR/'train'/f'{vid}.mp4')
    plan=dict(groups=GROUPS,split='train',n_videos=120,counter_settings_by_fold=selections,requested_frames=sum(map(len,requests.values())),reused_frames=reused,
              missing_runs=tasks,protected_inputs=protected,source_sha256=sha256(__file__),
              policy='Reuse saved JJ counts and baseline all-class counts. Reuse frozen raw coordinate caches; fill every missing requested push-up frame with the identical pose protocol. Derive signals per predicted window. Same per-fold counter settings and 0.3 s padding for all groups; no retuning.')
    if resume_counts:
        saved=json.loads((output/'plan.json').read_text())
        for key in ('groups','split','n_videos','counter_settings_by_fold','requested_frames','reused_frames','missing_runs','protected_inputs'):
            if saved[key]!=plan[key]: raise ValueError('Incomplete-run plan differs: '+key)
        (output/'count_resume.json').write_text(json.dumps(dict(original_plan_sha256=sha256(output/'plan.json'),original_source_sha256=saved['source_sha256'],resume_source_sha256=execution_hash,reason='Numeric dtype correction after concatenating an empty raw cache; pose observations and counter settings unchanged.',pose_inference_repeated=False),indent=2))
    else:
        output.mkdir(parents=True); (output/'coordinate_frames').mkdir()
        (output/'plan.json').write_text(json.dumps(plan,indent=2))
    print(f"All-class completion: {len(GROUPS)} feature groups; {reused}/{plan['requested_frames']} raw frames reused; {len(tasks)} missing runs",flush=True)
    if tasks and not resume_counts:
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(model),)) as pool:
            futures=[pool.submit(extract_missing,task) for task in tasks]
            for done,future in enumerate(as_completed(futures),1):
                task,frame=future.result(); frames[task['video_id']].append(frame)
                frame.to_csv(output/'coordinate_frames'/f"new__{task['video_id']}__{task['start']}.csv",index=False)
                if done%10==0 or done==len(tasks): print(f'Extracted missing runs {done}/{len(tasks)}',flush=True)
    assembled={}
    for vid,parts in frames.items():
        if not requests[vid]: continue
        path=output/'coordinate_frames'/f'{vid}.csv'
        frame=pd.read_csv(path,usecols=RAW) if resume_counts else pd.concat(parts)
        frame=frame.apply(pd.to_numeric,errors='raise').sort_values('frame').reset_index(drop=True)
        if frame.frame.tolist()!=sorted(requests[vid]): raise ValueError('Incomplete/duplicated coordinates')
        assembled[vid]=frame
        if not resume_counts: frame.to_csv(path,index=False)
    tables={group:truth.copy() for group in GROUPS}; counted=[]
    for number,(vid,meta) in enumerate(sorted(videos.items()),1):
        frame=feature_frame(FEATURES_DIR/f'{vid}_angles.csv',meta)
        for group in GROUPS:
            totals={'push_up':0,'squat':0}
            for idx,seg in enumerate(segments[group][vid]):
                ex=seg['exercise']
                if ex not in totals: continue
                settings=selections[str(meta['fold'])][ex]
                if ex=='squat': result=count_improved_peaks(frame,ex,seg['start_time'],seg['end_time']+.3,settings=ImprovementSettings(**settings))
                else:
                    ids=even_frames(seg['start_time'],seg['end_time'],meta['fps'],meta['n_frames'])
                    local=assembled[vid][assembled[vid].frame.isin(ids)].copy()
                    if len(local)>=8:
                        local,_=add_signals(local); result=count_coordinate_peaks(local,CoordinatePeakSettings(**settings))
                    else: result=dict(reps=0,rep_times=[])
                totals[ex]+=result['reps']
                counted.append(dict(group=group,video_id=vid,fold=meta['fold'],segment_index=idx,**seg,predicted=result['reps'],rep_times=result['rep_times']))
            jj=jj_counts[group][jj_counts[group].video_id.eq(vid)]
            if len(jj)!=1: raise ValueError('Missing saved JJ count')
            totals['jump_jack']=int(jj.iloc[0].predicted)
            for ex,value in totals.items(): tables[group].loc[tables[group].video_id.eq(vid)&tables[group].exercise.eq(ex),'predicted']=value
        if number%10==0: print(f'Counted all groups {number}/120 videos',flush=True)
    comparison={}; fold_rows=[]
    for group,table in tables.items():
        table['predicted']=table.predicted.astype(int); table['error']=table.predicted-table.annotated
        comparison[group]=dict(overall=paired_summary(table),by_exercise={ex:paired_summary(rows) for ex,rows in table.groupby('exercise')},
            active_by_exercise={ex:paired_summary(rows[rows.annotated.gt(0)]) for ex,rows in table.groupby('exercise')})
        table.to_csv(output/(group+'_video_exercise_counts.csv'),index=False)
        for ex,rows in table.groupby('exercise'):
            for fold,part in rows.groupby('fold'):
                old=float(abs(part.old_predicted-part.annotated).mean()); new=float(abs(part.error).mean())
                fold_rows.append(dict(group=group,exercise=ex,fold=int(fold),baseline_mae=old,new_mae=new,delta=new-old))
    fold_table=pd.DataFrame(fold_rows); fold_table.to_csv(output/'per_fold_count_comparison.csv',index=False)
    pd.DataFrame(counted).to_csv(output/'counted_pushup_squat_segments.csv',index=False)
    for path,digest in protected.items():
        if sha256(path)!=digest: raise ValueError('Protected input changed')
    if sha256(__file__)!=execution_hash: raise ValueError('Completion code changed during run')
    report=dict(split='train',n_videos=120,n_video_exercise_units=360,comparisons=comparison,
        fold_consistency={group:{ex:dict(improved=int(rows.delta.lt(-1e-9).sum()),tied=int(rows.delta.abs().le(1e-9).sum()),worsened=int(rows.delta.gt(1e-9).sum())) for ex,rows in fold_table[fold_table.group.eq(group)].groupby('exercise')} for group in GROUPS},
        inputs_verified_unchanged=True,counter_parameters_unchanged=True,source_sha256=execution_hash,resumed_from_saved_coordinates=resume_counts,
        limitations=['All original training videos retained; no audit exclusions or new tuning.','Fold-specific experimental counters, same source-grouped segmentation ablations.','Baseline 3.625 is the previous experimental combo, not the state-machine pipeline.','Raw pose observations are reused where available and filled consistently elsewhere; per-window torso normalization is recomputed.','JJ counts replay saved ablation results; only other-exercise counts completed here.','Development reuse/synthetic source repetition limit bootstrap independence; not a new test evaluation.','No model activation, canonical writes, commits or pushes.'])
    (output/'report.json').write_text(json.dumps(finite_json(report),indent=2,allow_nan=False))
    print(json.dumps({group:dict(overall=value['overall']['frozen']['mae'],exercises={ex:v['frozen']['mae'] for ex,v in value['by_exercise'].items()}) for group,value in comparison.items()},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True); parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--resume-counts',action='store_true',help='Resume complete saved coordinates without pose inference; original plan must match')
    args=parser.parse_args(); complete(args.out_dir,args.workers,args.resume_counts)
