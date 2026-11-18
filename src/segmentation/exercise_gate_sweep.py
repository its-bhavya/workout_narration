"""Sweep exercise-rejection probability using baseline OOF and fixed counters."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.segmentation import workflow
from src.segmentation.complete_ablation_counts import merge_raw_frames
from src.segmentation.diagnose_oof_classes import duration_confusion, class_metrics, LABELS
from src.data_pipeline.v2_protocol import ROOT, SYNTH_DIR, FEATURES_DIR, sha256, feature_frame
from src.rep_counting.evaluate_segmented_cv import RAW, CACHE, even_frames, missing_runs, extract_missing, finite_json
from src.rep_counting.extract_coordinate_training import init_worker, add_signals, verify_case, signature, case_name
from src.rep_counting.coordinate_counter import CoordinatePeakSettings, count_coordinate_peaks
from src.rep_counting.peak_improvements import ImprovementSettings, count_improved_peaks
from src.rep_counting.evaluate_coordinate_frozen import paired_summary
from src.rep_counting.classify_jumping_jack_windows import classify_predicted_window

EXPERIMENT=ROOT/'outputs/segmentation_v2/experiments/cv8_v2_seed42_p080_20261117'
BASELINE=ROOT/'outputs/pushup_coordinates/segmented_training_cv_20261118'
NEW_CACHE=ROOT/'outputs/segmentation_v2/experiments/periodicity_context_all_counts_20261118/coordinate_frames'
EXERCISES=['jump_jack','push_up','squat']
THRESHOLDS=[round(i*.05,2) for i in range(21)]


def exercise_gate(probabilities,threshold):
    """Original RF exercise probabilities; no renormalization or transition argmax."""
    values=np.asarray(probabilities,dtype=float)
    if values.ndim!=2 or values.shape[1]!=3 or not np.isfinite(values).all() or (values<0).any() or (values>1).any() or not 0<=threshold<=1:
        raise ValueError('Expected finite three-exercise probabilities and threshold in [0,1]')
    best=values.argmax(axis=1)
    return [EXERCISES[index] if values[i,index]>=threshold else 'transition' for i,index in enumerate(best)]


def gate_segments(rows,threshold,end):
    labels=exercise_gate(rows[[f'prob_{ex}' for ex in EXERCISES]].to_numpy(),threshold)
    # The new gate replaces confidence forward-fill, leaving only fixed
    # structural smoothing. No second confidence threshold is introduced.
    labels=workflow.smooth(labels,[1.]*len(labels),tau=0.,min_len=3)
    return workflow.to_segments(labels,rows.t_center.tolist(),end)


def choose_threshold(scores,heldout=None):
    columns=[f'fold_{fold}_mae' for fold in range(5) if fold!=heldout]
    ranked=scores.assign(objective=scores[columns].mean(axis=1))
    # Ties prefer stronger rejection, a predeclared conservative rule.
    return float(ranked.sort_values(['objective','threshold'],ascending=[True,False]).iloc[0].threshold)


def evaluate(output,workers=2,resume_counts=False):
    output=Path(output).resolve(); execution_hash=sha256(__file__)
    if not output.is_relative_to((ROOT/'outputs').resolve()) or (output.exists() and not resume_counts) or (output/'report.json').exists():
        raise ValueError('Use new output directory or explicitly resume an incomplete count run')
    selection=json.loads((EXPERIMENT/'selection.json').read_text())
    windows,videos=workflow.read_training(); videos={v:m for v,m in videos.items() if m['split']=='train'}
    if len(videos)!=120 or workflow.signature(videos,windows,FEATURES_DIR)!=selection['train_signature']: raise ValueError('Training inputs changed')
    oof=pd.read_csv(EXPERIMENT/'oof_predictions.csv')
    if sha256(EXPERIMENT/'oof_predictions.csv')!=selection['oof_sha256'] or not oof.fold.eq(oof.cv_fold).all() or not oof.split.eq('train').all(): raise ValueError('Invalid baseline OOF')
    keys=['video_id','win_idx']
    if oof.duplicated(keys).any() or set(map(tuple,oof[keys].to_numpy()))!=set(map(tuple,windows[keys].to_numpy())): raise ValueError('Incomplete OOF')
    counter_plan=json.loads((BASELINE/'evaluation_plan.json').read_text()); settings=counter_plan['counter_settings_by_heldout_fold']
    baseline=pd.read_csv(BASELINE/'video_exercise_comparison.csv')[['video_id','fold','exercise','annotated','predicted']].rename(columns={'predicted':'old_predicted'})
    protected={str(p):sha256(p) for p in (EXPERIMENT/'selection.json',EXPERIMENT/'oof_predictions.csv',BASELINE/'report.json',BASELINE/'video_exercise_comparison.csv',BASELINE/'evaluation_plan.json',CACHE/'plan.json')}
    for p,h in counter_plan['protected_hashes'].items():
        if sha256(p)!=h: raise ValueError('Baseline input changed')
        protected[p]=h
    for p in [ROOT/'src/rep_counting'/n for n in ['coordinate_counter.py','peak_improvements.py','peak_counter.py','rep_counter.py','extract_coordinate_training.py','extract_pushup_diagnostic.py']]+[Path(workflow.__file__).with_name('smooth.py')]: protected[str(p)]=sha256(p)
    cache_plan=json.loads((CACHE/'plan.json').read_text())['plan']
    model=ROOT/'models/pose_landmarker_full.task'
    if sha256(model)!=cache_plan['model_sha256']: raise ValueError('Pose model changed')
    protected[str(model)]=sha256(model)
    for n,h in cache_plan['source_hashes'].items():
        if sha256(ROOT/'src/rep_counting'/n)!=h: raise ValueError('Extraction implementation changed')
    segments={threshold:{} for threshold in THRESHOLDS}; requests={vid:set() for vid in videos}; segment_rows=[]
    for vid,rows in oof.groupby('video_id',sort=True):
        meta=videos[vid]; rows=rows.sort_values('win_idx'); end=meta['n_frames']/meta['fps']
        for threshold in THRESHOLDS:
            predicted=gate_segments(rows,threshold,end); segments[threshold][vid]=predicted
            for idx,seg in enumerate(predicted):
                segment_rows.append(dict(threshold=threshold,video_id=vid,fold=meta['fold'],segment_index=idx,**seg))
                if seg['exercise']=='push_up': requests[vid].update(even_frames(seg['start_time'],seg['end_time'],meta['fps'],meta['n_frames']))
    parts={}; tasks=[]; reused=0
    for vid,meta in sorted(videos.items()):
        cached=pd.DataFrame(columns=RAW)
        for folder in (BASELINE/'coordinate_frames',NEW_CACHE):
            p=folder/f'{vid}.csv'
            if p.exists():
                protected[str(p)]=sha256(p)
                cached=merge_raw_frames(cached,pd.read_csv(p,usecols=RAW),requests[vid])
        for case in cache_plan['cases']:
            if case['video_id']!=vid: continue
            verify_case(CACHE,case,signature(cache_plan)); p=CACHE/(case_name(case)+'.csv'); protected[str(p)]=sha256(p)
            cached=merge_raw_frames(cached,pd.read_csv(p,usecols=RAW),requests[vid])
        reused+=len(cached); parts[vid]=[cached] if len(cached) else []
        for start,end in missing_runs(requests[vid]-set(cached.frame)):
            tasks.append(dict(video_id=vid,split='train',video_path=str(SYNTH_DIR/'train'/f'{vid}.mp4'),fps=meta['fps'],start=start,end=end))
        for p in (SYNTH_DIR/'train'/f'{vid}.mp4',SYNTH_DIR/'train'/f'{vid}.json',FEATURES_DIR/f'{vid}_angles.csv'): protected[str(p)]=sha256(p)
    plan=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),split='train',thresholds=THRESHOLDS,
        decision_rule='Choose largest original unnormalized exercise probability; transition only when that maximum < threshold. Transition probability is ignored for argmax, not redistributed.',
        smoothing='Gate replaces confidence forward-fill; fixed min_len 3 and sandwich merge retained, allowing structural relabeling after the gate.',
        objective='Equal-fold mean per-video/exercise counting MAE (360 units); ties choose larger threshold. No transition F1 selection.',
        counter_settings_by_fold=settings,end_padding_s=.3,requested_frames=sum(map(len,requests.values())),reused_frames=reused,missing_runs=tasks,protected_inputs=protected,source_sha256=execution_hash)
    if resume_counts:
        saved=json.loads((output/'plan.json').read_text())
        for key in ('thresholds','decision_rule','smoothing','objective','counter_settings_by_fold','requested_frames','reused_frames','missing_runs','protected_inputs'):
            if saved[key]!=plan[key]: raise ValueError('Resume differs: '+key)
        if saved['source_sha256']!=execution_hash: raise ValueError('Sweep implementation changed; do not silently resume')
    else:
        output.mkdir(parents=True); (output/'coordinate_frames').mkdir(); (output/'candidates').mkdir()
        (output/'plan.json').write_text(json.dumps(plan,indent=2))
        pd.DataFrame(segment_rows).to_csv(output/'all_candidate_segments.csv',index=False)
    print(f"21 thresholds; {reused}/{plan['requested_frames']} coordinate frames cached; {len(tasks)} missing runs",flush=True)
    if tasks and not resume_counts:
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(model),)) as pool:
            futures=[pool.submit(extract_missing,task) for task in tasks]
            for done,future in enumerate(as_completed(futures),1):
                task,frame=future.result(); parts[task['video_id']].append(frame)
                frame.to_csv(output/'coordinate_frames'/f"new__{task['video_id']}__{task['start']}.csv",index=False)
                if done%10==0 or done==len(tasks): print(f'Extracted missing runs {done}/{len(tasks)}',flush=True)
    coordinates={}
    for vid,frames in parts.items():
        if not requests[vid]: continue
        p=output/'coordinate_frames'/f'{vid}.csv'
        frame=pd.read_csv(p,usecols=RAW) if resume_counts else pd.concat(frames)
        frame=frame.apply(pd.to_numeric,errors='raise').sort_values('frame').reset_index(drop=True)
        if frame.frame.tolist()!=sorted(requests[vid]): raise ValueError('Incomplete/duplicated pose frames')
        coordinates[vid]=frame
        if not resume_counts: frame.to_csv(p,index=False)
    all_counts={t:[] for t in THRESHOLDS}; diagnostics={t:[] for t in THRESHOLDS}; segment_count_rows=[]
    for number,(vid,meta) in enumerate(sorted(videos.items()),1):
        frame=feature_frame(FEATURES_DIR/f'{vid}_angles.csv',meta)
        gt,end=workflow.gt_from_json(meta)
        truth=[dict(label=s['exercise'],start=s['start_time'],end=s['end_time'],index=i,clip_id=None) for i,s in enumerate(gt)]
        truth_counts=baseline[baseline.video_id.eq(vid)].set_index('exercise')
        cache={}
        for threshold in THRESHOLDS:
            predicted=segments[threshold][vid]; counts={e:0 for e in EXERCISES}; overlong=False
            for idx,seg in enumerate(predicted):
                ex=seg['exercise']
                if ex=='transition': continue
                key=(ex,seg['start_time'],seg['end_time'])
                if key not in cache:
                    cfg=settings[str(meta['fold'])][ex]
                    if ex=='push_up':
                        ids=even_frames(seg['start_time'],seg['end_time'],meta['fps'],meta['n_frames'])
                        local=coordinates[vid][coordinates[vid].frame.isin(ids)].copy()
                        if len(local)>=8:
                            local,_=add_signals(local); result=count_coordinate_peaks(local,CoordinatePeakSettings(**cfg))
                        else: result=dict(reps=0,rep_times=[])
                    else: result=count_improved_peaks(frame,ex,seg['start_time'],seg['end_time']+.3,settings=ImprovementSettings(**cfg))
                    cache[key]=result
                result=cache[key]; counts[ex]+=result['reps']
                segment_count_rows.append(dict(threshold=threshold,video_id=vid,segment_index=idx,**seg,predicted=result['reps'],rep_times=result['rep_times']))
                if ex=='jump_jack':
                    flags=classify_predicted_window(seg['start_time'],seg['end_time'],truth)
                    overlong|=flags['overlong_start'] or flags['overlong_end']
            for ex in EXERCISES:
                all_counts[threshold].append(dict(threshold=threshold,video_id=vid,fold=meta['fold'],exercise=ex,annotated=int(truth_counts.loc[ex,'annotated']),old_predicted=int(truth_counts.loc[ex,'old_predicted']),predicted=counts[ex],error=counts[ex]-int(truth_counts.loc[ex,'annotated'])))
            temporal=duration_confusion(gt,predicted)
            metrics=workflow.all_metrics(gt,predicted,end,workflow.CLASSES)
            diagnostics[threshold].append(dict(video_id=vid,fold=meta['fold'],**metrics,jj_gt_seconds=float(temporal.loc['jump_jack'].sum()),jj_correct_seconds=float(temporal.loc['jump_jack','jump_jack']),transition_gt_seconds=float(temporal.loc['transition'].sum()),transition_correct_seconds=float(temporal.loc['transition','transition']),jj_overlong=overlong,jj_error=counts['jump_jack']-int(truth_counts.loc['jump_jack','annotated'])))
        if number%10==0: print(f'Counted threshold sweep {number}/120 videos',flush=True)
    score_rows=[]; tables={}
    for threshold,records in all_counts.items():
        table=pd.DataFrame(records); tables[threshold]=table; diag=pd.DataFrame(diagnostics[threshold])
        folds=table.assign(a=abs(table.error)).groupby('fold').a.mean()
        score_rows.append(dict(threshold=threshold,mae=float(folds.mean()),**{f'fold_{f}_mae':float(value) for f,value in folds.items()},
            **{ex+'_mae':float(rows.error.abs().mean()) for ex,rows in table.groupby('exercise')},
            exact_units=int(table.error.eq(0).sum()),within_one_units=int(table.error.abs().le(1).sum()),overcount_units=int(table.error.gt(0).sum()),undercount_units=int(table.error.lt(0).sum()),bias=float(table.error.mean()),
            jj_recall=float(diag.jj_correct_seconds.sum()/diag.jj_gt_seconds.sum()),transition_recall=float(diag.transition_correct_seconds.sum()/diag.transition_gt_seconds.sum()),
            jj_overlong_videos=int(diag.jj_overlong.sum()),jj_overcount_videos=int(diag.jj_error.gt(0).sum()),jj_overcount_and_overlong=int((diag.jj_error.gt(0)&diag.jj_overlong).sum()),
            edit_mean=float(diag.groupby('fold').edit_score.mean().mean()),frame_accuracy_mean=float(diag.groupby('fold').frame_accuracy.mean().mean()),f1_at_25_mean=float(diag.groupby('fold').f1_at_25.mean().mean())))
        table.to_csv(output/'candidates'/f'counts_{threshold:.2f}.csv',index=False)
        diag.to_csv(output/'candidates'/f'segmentation_{threshold:.2f}.csv',index=False)
    scores=pd.DataFrame(score_rows); scores.to_csv(output/'sweep.csv',index=False)
    chosen=choose_threshold(scores); heldout=[]; checks=[]
    for fold in range(5):
        threshold=choose_threshold(scores,fold)
        heldout.append(tables[threshold][tables[threshold].fold.eq(fold)])
        checks.append(dict(fold=fold,threshold=threshold,heldout_mae=float(scores[scores.threshold.eq(threshold)][f'fold_{fold}_mae'].iloc[0])))
    heldout=pd.concat(heldout)
    def summary(table):
        return dict(overall=paired_summary(table),by_exercise={ex:paired_summary(rows) for ex,rows in table.groupby('exercise')})
    selected=tables[chosen]
    report=dict(split='train',n_videos=120,n_units=360,selected_threshold_all_oof=chosen,selected_all_oof=summary(selected),
        other_fold_threshold_selection=summary(heldout),heldout_checks=checks,baseline_overall_mae=float(abs(baseline.old_predicted-baseline.annotated).mean()),
        selected_sweep_row=scores[scores.threshold.eq(chosen)].iloc[0].to_dict(),source_sha256=execution_hash,
        limitations=['Threshold minimizes all-class end-to-end video/exercise MAE, not transition F1; all original training data retained.','Gate replaces original confidence forward-fill; fixed min-run and sandwich rules still modify gated labels.','Uses original four-class RF exercise probabilities without renormalization or new fitting.','All-OOF selected score is selection-biased; other-fold threshold diagnostic also reuses development model/smoothing/counter choices and is not strict nested CV.','OOF predictions on other folds came from models whose fitting sets include the target fold: threshold-selection diagnostic is not a fully nested outer-CV evaluation.','Coordinate extraction and all counter settings unchanged; cached raw data reused and missing requested frames filled consistently.','Counter safeguards need not reject every non-exercise interval; overcounts and overlong JJ windows are retained in sweep.','Synthetic source reuse limits video-bootstrap independence. No test inference, canonical writes, activation, commits or pushes.'])
    for p,h in protected.items():
        if sha256(p)!=h: raise ValueError('Protected input changed during sweep')
    if sha256(__file__)!=execution_hash: raise ValueError('Sweep implementation changed')
    report['inputs_verified_unchanged']=True
    selected.to_csv(output/'selected_all_oof_counts.csv',index=False); heldout.to_csv(output/'other_fold_selected_counts.csv',index=False)
    pd.DataFrame(segment_count_rows).to_csv(output/'counted_segments.csv',index=False)
    (output/'report.json').write_text(json.dumps(finite_json(report),indent=2,allow_nan=False))
    print(scores[['threshold','mae','jump_jack_mae','push_up_mae','squat_mae','jj_recall','jj_overcount_videos','jj_overcount_and_overlong']].to_string(index=False))
    print(json.dumps(dict(chosen=chosen,heldout_checks=checks,selected=report['selected_all_oof']['overall'],other_fold=report['other_fold_threshold_selection']['overall']),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True); parser.add_argument('--workers',type=int,default=2); parser.add_argument('--resume-counts',action='store_true')
    args=parser.parse_args(); evaluate(args.out_dir,args.workers,args.resume_counts)
