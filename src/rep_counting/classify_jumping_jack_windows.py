"""Read-only temporal failure classification of saved train OOF JJ predictions."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import ROOT, SYNTH_DIR, catalog, sha256
from .analyze_peak_errors import assign_reps

RUN=ROOT/'outputs/pushup_coordinates/segmented_training_cv_20261118'
GT_COUNTS=ROOT/'outputs/peak_improvements/verified_cached_signal_comparison_20261118/heldout_segments.csv'
LABEL='jump_jack'
TOLERANCE=.5


def overlap(a,b,c,d):
    return max(0.,min(b,d)-max(a,c))


def dominant_label(start,end,segments):
    durations={}
    for s in segments:
        seconds=overlap(start,end,s['start'],s['end'])
        durations[s['label']]=durations.get(s['label'],0.)+seconds
    return max(durations,key=durations.get) if durations and max(durations.values())>0 else 'none'


def classify_gt_window(start,end,predicted,tolerance=TOLERANCE):
    hits=[s for s in predicted if s['label']==LABEL and overlap(start,end,s['start'],s['end'])>0]
    coverage=sum(overlap(start,end,s['start'],s['end']) for s in hits)
    first=min((max(start,s['start']) for s in hits),default=end)
    last=max((min(end,s['end']) for s in hits),default=start)
    return dict(n_predicted_jj_overlaps=len(hits),jj_coverage_s=coverage,jj_coverage_fraction=coverage/(end-start),
        dominant_predicted_label=dominant_label(start,end,predicted),
        wholly_missed=not hits,clipped_start=bool(hits) and first-start>tolerance,
        clipped_end=bool(hits) and end-last>tolerance,
        fragmented=sum(overlap(start,end,s['start'],s['end'])>tolerance for s in hits)>=2,
        first_correct_label_s=first if hits else None,last_correct_label_s=last if hits else None,
        unlabelled_as_jj_s=end-start-coverage)


def classify_predicted_window(start,end,truth,tolerance=TOLERANCE):
    hits=[s for s in truth if s['label']==LABEL and overlap(start,end,s['start'],s['end'])>0]
    matched=max(hits,key=lambda s:overlap(start,end,s['start'],s['end'])) if hits else None
    inside=sum(overlap(start,end,s['start'],s['end']) for s in hits)
    result=dict(dominant_gt_label=dominant_label(start,end,truth),n_gt_jj_overlaps=len(hits),
        jj_purity_fraction=inside/(end-start),off_class_duration_s=end-start-inside,
        mostly_wrong_label=inside/(end-start)<.5,merged_gt_windows=sum(overlap(start,end,s['start'],s['end'])>tolerance for s in hits)>=2)
    result.update(dict(matched_gt_index=matched['index'],matched_gt_clip_id=matched['clip_id'],
        gt_start_s=matched['start'],gt_end_s=matched['end'],gt_length_s=matched['end']-matched['start'],
        start_delta_s=start-matched['start'],end_delta_s=end-matched['end'],length_delta_s=(end-start)-(matched['end']-matched['start']),
        overlong_start=matched['start']-start>tolerance,overlong_end=end-matched['end']>tolerance) if matched else
        dict(matched_gt_index=None,matched_gt_clip_id=None,gt_start_s=None,gt_end_s=None,gt_length_s=None,
             start_delta_s=None,end_delta_s=None,length_delta_s=None,overlong_start=False,overlong_end=False))
    return result


def analyze(output,run=RUN):
    output,run=Path(output).resolve(),Path(run).resolve()
    if output.exists() or not output.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Use a new output directory under outputs')
    paths=[run/n for n in ('evaluation_plan.json','report.json','predicted_segments.csv','counted_segments.csv','video_exercise_comparison.csv')]+[GT_COUNTS]
    protected={str(p):sha256(p) for p in paths}
    plan=json.loads((run/'evaluation_plan.json').read_text())
    for p,h in plan['protected_hashes'].items():
        if sha256(p)!=h: raise ValueError('Saved run provenance changed')
    predictions=pd.read_csv(run/'predicted_segments.csv')
    counts=pd.read_csv(run/'counted_segments.csv')
    counts=counts[counts.method.eq('experimental') & counts.exercise.eq(LABEL)].copy()
    counts.rep_times=counts.rep_times.map(ast.literal_eval)
    totals=pd.read_csv(run/'video_exercise_comparison.csv'); totals=totals[totals.exercise.eq(LABEL)]
    gt_counts=pd.read_csv(GT_COUNTS); gt_counts=gt_counts[gt_counts.exercise.eq(LABEL)]
    videos={v:m for v,m in catalog().items() if m['split']=='train'}
    if set(totals.video_id)!=set(videos) or len(videos)!=120: raise ValueError('Expected all training videos')
    gt_rows=[]; pred_rows=[]; intersections=[]; video_rows=[]; event_rows=[]; timelines={}
    for vid,meta in sorted(videos.items()):
        annotation=SYNTH_DIR/'train'/f'{vid}.json'; protected[str(annotation)]=sha256(annotation)
        truth=[dict(index=i,label=s['label'],start=s['start_frame']/meta['fps'],end=s['end_frame']/meta['fps'],clip_id=s.get('clip_id'),pairs=[(a/meta['fps'],b/meta['fps']) for a,b in s.get('rep_pairs_out',[])]) for i,s in enumerate(meta['segments'])]
        predicted=[dict(index=int(s.segment_index),label=s.exercise,start=float(s.start_time),end=float(s.end_time)) for s in predictions[predictions.video_id.eq(vid)].itertuples()]
        jj_truth=[s for s in truth if s['label']==LABEL]
        video_count=totals[totals.video_id.eq(vid)].iloc[0]
        times=[t for ts in counts[counts.video_id.eq(vid)].rep_times for t in ts]
        if len(times)!=video_count.predicted: raise ValueError('Saved timelines/counts disagree')
        base=dict(video_id=vid,fold=meta['fold'],video_signed_error=int(video_count.error))
        local_gt=[]; local_pred=[]
        for gt in jj_truth:
            flags=classify_gt_window(gt['start'],gt['end'],predicted)
            row=base|dict(gt_index=gt['index'],clip_id=gt['clip_id'],gt_label=LABEL,gt_start_s=gt['start'],gt_end_s=gt['end'],gt_length_s=gt['end']-gt['start'],annotated=len(gt['pairs']),predicted_events_inside_gt=sum(gt['start']<=t<gt['end'] for t in times))|flags
            row['event_allocated_signed_error']=row['predicted_events_inside_gt']-row['annotated']
            row['mostly_mislabeled']=flags['dominant_predicted_label']!=LABEL
            gt_rows.append(row); local_gt.append(row)
        for pred in predicted:
            if pred['label']!=LABEL and not any(overlap(pred['start'],pred['end'],g['start'],g['end'])>0 for g in jj_truth): continue
            matched=counts[(counts.video_id==vid)&(counts.segment_index==pred['index'])]
            number=int(matched.iloc[0].predicted) if len(matched) else 0
            row=base|dict(predicted_index=pred['index'],predicted_label=pred['label'],pred_start_s=pred['start'],pred_end_s=pred['end'],pred_length_s=pred['end']-pred['start'],effective_count_end_s=min(meta['n_frames']/meta['fps'],pred['end']+.3) if pred['label']==LABEL else None,predicted_jj_reps=number)
            row.update(classify_predicted_window(pred['start'],pred['end'],truth))
            pred_rows.append(row)
            if pred['label']==LABEL: local_pred.append(row)
            for gt in truth:
                seconds=overlap(pred['start'],pred['end'],gt['start'],gt['end'])
                if seconds<=0 or not (pred['label']==LABEL or gt['label']==LABEL): continue
                intersections.append(base|dict(predicted_index=pred['index'],predicted_label=pred['label'],gt_index=gt['index'],gt_label=gt['label'],clip_id=gt['clip_id'],pred_start_s=pred['start'],pred_end_s=pred['end'],pred_length_s=pred['end']-pred['start'],gt_start_s=gt['start'],gt_end_s=gt['end'],gt_length_s=gt['end']-gt['start'],overlap_s=seconds,start_delta_s=pred['start']-gt['start'],end_delta_s=pred['end']-gt['end']))
        pairs=[pair for gt in jj_truth for pair in gt['pairs']]
        owners=[gt for gt in jj_truth for _ in gt['pairs']]
        matches,missed,extra=assign_reps(times,pairs)
        for index in missed:
            a,b=pairs[index]
            cover=sum(overlap(a,b,s['start'],s['end']) for s in predicted if s['label']==LABEL)
            fraction=cover/(b-a) if b>a else float(any(s['start']<=b<s['end'] and s['label']==LABEL for s in predicted))
            category='no_jj_window' if fraction<1e-8 else 'partially_covered_cycle' if fraction<1-1e-8 else 'fully_covered_counter_miss'
            event_rows.append(base|dict(kind='missed',category=category,clip_id=owners[index]['clip_id'],start_s=a,end_s=b,jj_cycle_coverage_fraction=fraction,dominant_predicted_label=dominant_label(a,b,predicted)))
        for index in extra:
            t=times[index]; inside=any(g['start']<=t<g['end'] for g in jj_truth)
            event_rows.append(base|dict(kind='extra',category='inside_gt_jj' if inside else 'outside_gt_jj',clip_id=None,start_s=t,end_s=t,jj_cycle_coverage_fraction=None,dominant_predicted_label=dominant_label(t,t+1e-6,predicted)))
        gt_counter=int(gt_counts[gt_counts.video_id.eq(vid)].predicted.sum())
        video_rows.append(base|dict(annotated=int(video_count.annotated),predicted=int(video_count.predicted),error=int(video_count.error),absolute_error=abs(int(video_count.error)),gt_window_counter_predicted=gt_counter,gt_window_counter_error=gt_counter-int(video_count.annotated),n_gt_jj_windows=len(jj_truth),n_pred_jj_windows=len(local_pred),
            clipped=any(g['clipped_start'] or g['clipped_end'] for g in local_gt),overlong=any(g['overlong_start'] or g['overlong_end'] for g in local_pred),
            fragmented=any(g['fragmented'] for g in local_gt),merged=any(g['merged_gt_windows'] for g in local_pred),
            mislabeled_gt=any(g['mostly_mislabeled'] for g in local_gt),mislabeled_pred=any(g['mostly_wrong_label'] for g in local_pred),
            wholly_missed_gt=any(g['wholly_missed'] for g in local_gt),false_positive_absent=not jj_truth and bool(local_pred),
            missed_reps=len(missed),extra_reps=len(extra),matched_reps=len(matches),events_outside_gt_windows=sum(not any(g['start']<=t<g['end'] for g in jj_truth) for t in times)))
        timelines[vid]=(truth,predicted,times)
    video=pd.DataFrame(video_rows); gt=pd.DataFrame(gt_rows); pred=pd.DataFrame(pred_rows); events=pd.DataFrame(event_rows)
    flags=['clipped','overlong','fragmented','merged','mislabeled_gt','mislabeled_pred','wholly_missed_gt','false_positive_absent']
    tally={flag:dict(all=int(video[flag].sum()),undercount=int(video.loc[video.error<0,flag].sum()),overcount=int(video.loc[video.error>0,flag].sum()),exact=int(video.loc[video.error==0,flag].sum())) for flag in flags}
    wrong=video[video.error.ne(0)]
    report=dict(split='train',method='Saved experimental combo on OOF segmentation; no new inference/counting',n_videos=len(video),n_gt_jj_windows=len(gt),n_pred_jj_windows=int(pred.predicted_label.eq(LABEL).sum()),
        signed_error_video_tally=dict(under=int(video.error.lt(0).sum()),over=int(video.error.gt(0).sum()),exact=int(video.error.eq(0).sum())),
        overlapping_video_flags=tally,n_erroneous_videos_with_any_flag=int(wrong[flags].any(axis=1).sum()),n_erroneous_videos_without_flags=int((~wrong[flags].any(axis=1)).sum()),
        gt_window_flags={k:int(gt[k].sum()) for k in ['clipped_start','clipped_end','fragmented','mostly_mislabeled','wholly_missed']},
        predicted_jj_flags={k:int(pred.loc[pred.predicted_label.eq(LABEL),k].sum()) for k in ['overlong_start','overlong_end','mostly_wrong_label','merged_gt_windows']},
        event_tally=events.groupby(['kind','category']).size().unstack(fill_value=0).to_dict(),
        gt_jj_duration_by_predicted_label=pd.DataFrame(intersections).query('gt_label == @LABEL').groupby('predicted_label').overlap_s.sum().to_dict(),
        definitions=dict(boundary_tolerance_s=TOLERANCE,clipped='GT JJ starts >0.5 s before first or ends >0.5 s after last overlapping predicted JJ window; wholly missed windows separately flagged.',overlong='Predicted JJ starts >0.5 s before or ends >0.5 s after its maximum-overlap GT JJ window; merged windows also flagged.',fragmented='At least two separate predicted JJ windows each overlap one GT JJ window by >0.5 s.',merged='One predicted JJ window overlaps at least two GT JJ windows each by >0.5 s.',mislabeled_gt='Dominant predicted label by duration within GT JJ window is not JJ.',mislabeled_pred='Less than 50% of predicted JJ duration overlaps GT JJ; includes no overlap.',signed_error='Experimental video-total JJ count minus all annotated JJ reps, including false positives in absent videos.',boundary_delta='Predicted minus GT; positive start = late, negative end = early. Raw unpadded windows; effective +0.3 s counting end is separate.',event_allocated_signed_error='GT-window field assigns saved predicted JJ completion events by timestamp inside GT [start,end); events outside all GT windows remain separately counted at video level. Not a rerun or one-to-one rep-match count.'),
        limitations=['Flags overlap and are temporal associations, not causal attribution of count error.','Dominant/max-overlap matching is descriptive, not a one-to-one alignment; full intersection table retains all labels/overlaps.','Missed rep categories use interval coverage, with class-aware one-to-one matching tolerance 0.3 s; full coverage does not prove signal quality.','Boundary flags use nominal windows; padding may recover cycles. Raw coverage and deltas retained.','Synthetic source reuse and prior development selection remain; no parameter tuning, test inference, commits or pushes.'],
        source_sha256=sha256(__file__),protected_inputs=protected)
    for p,h in protected.items():
        if sha256(p)!=h: raise ValueError('Input changed during analysis')
    report['inputs_verified_unchanged']=True
    output.mkdir(parents=True)
    for name,frame in [('videos',video),('gt_windows',gt),('predicted_windows',pred),('window_intersections',pd.DataFrame(intersections)),('rep_event_classification',events)]: frame.to_csv(output/(name+'.csv'),index=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    plot_timelines(output,video,timelines)
    print(json.dumps({k:report[k] for k in ['signed_error_video_tally','overlapping_video_flags','gt_window_flags','event_tally','gt_jj_duration_by_predicted_label']},indent=2))
    return report


def plot_timelines(output,video,timelines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'jump_jack':'#277eb5','squat':'#de8b32','push_up':'#4d9d65','transition':'#c5c5c5'}
    worst=video.sort_values('absolute_error',ascending=False).head(6)
    fig,axes=plt.subplots(len(worst),1,figsize=(14,12),layout='constrained')
    for ax,row in zip(axes,worst.itertuples()):
        truth,predicted,times=timelines[row.video_id]
        for y,segments in [(1,truth),(0,predicted)]:
            for s in segments:
                ax.broken_barh([(s['start'],s['end']-s['start'])],(y-.25,.5),facecolors=colors[s['label']])
                if s['end']-s['start']>6: ax.text((s['start']+s['end'])/2,y,s['label'],ha='center',va='center',fontsize=7)
        ax.vlines(times,-.37,-.27,color='black',linewidth=.6)
        ax.set(yticks=[0,1],yticklabels=['Predicted','GT'],xlabel='Time (s)',title=f'{row.video_id}: JJ GT {row.annotated}, predicted {row.predicted}, signed error {row.error:+d}')
    fig.savefig(output/'largest_errors_timelines.png',dpi=150); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args(); analyze(args.out_dir)
