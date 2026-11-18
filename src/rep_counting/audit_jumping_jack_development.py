"""Objective rate/duration flags and concentration, without changing labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import ROOT, SYNTH_DIR, catalog, sha256

FAILURES=ROOT/'outputs/pushup_coordinates/jumping_jack_window_failures_20261118'


def count_metrics(rows):
    if rows.empty: return dict(n=0,mae=None,median_absolute_error=None)
    return dict(n=len(rows),mae=float(rows.absolute_error.mean()),median_absolute_error=float(rows.absolute_error.median()),
                exact=int(rows.error.eq(0).sum()),within_one=int(rows.error.abs().le(1).sum()))


def audit(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to((ROOT/'outputs').resolve()): raise ValueError('Use a new output directory under outputs')
    paths=[FAILURES/n for n in ('videos.csv','rep_event_classification.csv','report.json')]
    protected={str(p):sha256(p) for p in paths}
    rows=[]
    for vid,meta in sorted(catalog().items()):
        if meta['split']!='train': continue
        p=SYNTH_DIR/'train'/f'{vid}.json'; protected[str(p)]=sha256(p)
        for idx,s in enumerate(meta['segments']):
            if s.get('label')!='jump_jack': continue
            duration=(s['end_frame']-s['start_frame'])/meta['fps']
            rate=s['rep_count_annotated']/duration
            rows.append(dict(video_id=vid,fold=meta['fold'],segment_index=idx,clip_id=s['clip_id'],
                start_s=s['start_frame']/meta['fps'],end_s=s['end_frame']/meta['fps'],duration_s=duration,
                duration_key=round(duration,9),n_output_frames=s['end_frame']-s['start_frame'],annotated=s['rep_count_annotated'],rate_hz=rate,
                rate_flag=rate<1 or rate>4,rate_direction='below_1' if rate<1 else 'above_4' if rate>4 else 'within_1_4'))
    windows=pd.DataFrame(rows)
    groups=windows.groupby('duration_key').agg(occurrences=('video_id','size'),distinct_sources=('clip_id','nunique'))
    windows=windows.merge(groups,on='duration_key',validate='many_to_one')
    windows['duplicated_duration']=windows.occurrences.gt(1)
    windows['cross_source_duplicated_duration']=windows.distinct_sources.gt(1)
    windows['audit_flag']=windows.rate_flag | windows.cross_source_duplicated_duration
    video=pd.read_csv(FAILURES/'videos.csv')
    flags=windows.groupby('video_id')[['rate_flag','cross_source_duplicated_duration','audit_flag']].any()
    video=video.merge(flags,left_on='video_id',right_index=True,how='left').fillna({'rate_flag':False,'cross_source_duplicated_duration':False,'audit_flag':False})
    events=pd.read_csv(FAILURES/'rep_event_classification.csv')
    missed=events[events.kind.eq('missed')]
    video['missed_reps']=video.video_id.map(missed.groupby('video_id').size()).fillna(0).astype(int)
    # Existing saved GT-window counter, with exactly the same fold settings.
    video['gt_counter_absolute_error']=abs(video.gt_window_counter_error)
    sensitivity={}
    for name,keep in [('all',pd.Series(True,index=video.index)),('without_rate_flags',~video.rate_flag.astype(bool)),('without_cross_source_duplicate_lengths',~video.cross_source_duplicated_duration.astype(bool)),('without_union_audit_flags',~video.audit_flag.astype(bool))]:
        subset=video[keep]
        sensitivity[name]=dict(predicted_windows=count_metrics(subset),gt_windows=count_metrics(subset.assign(error=subset.gt_window_counter_error,absolute_error=subset.gt_counter_absolute_error)),
            active_predicted_windows=count_metrics(subset[subset.annotated.gt(0)]),excluded_videos=video.loc[~keep,'video_id'].tolist())
    concentration={}
    for ranking in ('absolute_error','missed_reps'):
        ranked=video.sort_values([ranking,'video_id'],ascending=[False,True])
        concentration[ranking]={}
        for n in (5,10):
            top=ranked.head(n)
            concentration[ranking][str(n)]=dict(videos=top.video_id.tolist(),absolute_error=int(top.absolute_error.sum()),absolute_error_fraction=float(top.absolute_error.sum()/video.absolute_error.sum()),
                missed_reps=int(top.missed_reps.sum()),missed_rep_fraction=float(top.missed_reps.sum()/len(missed)),audit_flagged_videos=top.loc[top.audit_flag.astype(bool),'video_id'].tolist(),
                remaining=count_metrics(video[~video.video_id.isin(top.video_id)]))
    covered=missed[missed.category.eq('fully_covered_counter_miss')].copy()
    # Each source has one rate per fixed synthetic source window; retain all occurrences in denominator.
    rate_map=windows.groupby(['video_id','clip_id']).agg(rate_hz=('rate_hz','first'),annotated=('annotated','sum'))
    covered=covered.merge(rate_map.reset_index(),on=['video_id','clip_id'],validate='many_to_one')
    covered['audit_flag']=covered.video_id.map(video.set_index('video_id').audit_flag).astype(bool)
    covered['tempo_band']=pd.cut(covered.rate_hz,[-np.inf,1,2,3,4,np.inf],right=False,labels=['below_1','1_to_2','2_to_3','3_to_4','4_plus'])
    clean_windows=windows[~windows.video_id.isin(video.loc[video.audit_flag.astype(bool),'video_id'])].copy()
    clean_windows['tempo_band']=pd.cut(clean_windows.rate_hz,[-np.inf,1,2,3,4,np.inf],right=False,labels=['below_1','1_to_2','2_to_3','3_to_4','4_plus'])
    tempo=[]
    for label in ['below_1','1_to_2','2_to_3','3_to_4','4_plus']:
        denominator=int(clean_windows.loc[clean_windows.tempo_band.eq(label),'annotated'].sum())
        num=int((covered.tempo_band.eq(label)&~covered.audit_flag).sum())
        tempo.append(dict(tempo_band=label,clean_annotated_reps=denominator,clean_fully_covered_misses=num,miss_fraction=num/denominator if denominator else None))
    report=dict(split='train',rate_rule='Flag GT count/window duration <1 or >4 Hz; inclusive 1-4 Hz retained.',
        duration_rule='Exact output duration to 9 decimal seconds. Same-source reuse is expected; cross-source equal durations are separately flagged, not proven errors.',
        n_windows=len(windows),n_sources=windows.clip_id.nunique(),rate_flag_windows=int(windows.rate_flag.sum()),
        rate_flag_sources=windows.loc[windows.rate_flag,'clip_id'].nunique(),rate_flag_videos=int(video.rate_flag.sum()),
        cross_source_duplicate_duration_windows=int(windows.cross_source_duplicated_duration.sum()),union_flag_videos=int(video.audit_flag.sum()),
        sensitivity=sensitivity,concentration=concentration,total_absolute_error=int(video.absolute_error.sum()),total_missed_reps=len(missed),
        all_video_metrics=count_metrics(video),active_video_metrics=count_metrics(video[video.annotated.gt(0)]),
        covered_miss_count=len(covered),covered_misses_excluded_by_audit=int(covered.audit_flag.sum()),covered_misses_clean=int((~covered.audit_flag).sum()),covered_misses_by_tempo_clean=tempo,
        limitations=['Flags are objectively reproducible but are not validated label errors; slow legitimate jacks, speed changes and rounded durations are possible.','No labels corrected or examples removed from primary scores, model fitting or evaluation.','Sensitivity exclusions remove every video containing any flagged window, including otherwise valid windows; rate-only, duplicate-only and union shown separately.','Synthetic source reuse is not independent evidence.','Tempo is annotated count divided by whole-window duration, not measured instantaneous cadence or proof of counter parameter causation.'],protected_inputs=protected,source_sha256=sha256(__file__))
    for p,h in protected.items():
        if sha256(p)!=h: raise ValueError('Input changed')
    output.mkdir(parents=True)
    windows.to_csv(output/'all_jj_window_audit.csv',index=False)
    windows[windows.audit_flag].to_csv(output/'flagged_windows.csv',index=False)
    video.sort_values('absolute_error',ascending=False).to_csv(output/'video_concentration_and_flags.csv',index=False)
    covered.to_csv(output/'covered_misses_tempo.csv',index=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ['rate_flag_windows','rate_flag_sources','rate_flag_videos','cross_source_duplicate_duration_windows','union_flag_videos','sensitivity','concentration','covered_misses_by_tempo_clean']},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args(); audit(args.out_dir)
