"""One exploratory evaluation of tagged coordinate voting; never tune here."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import ROOT, SYNTH_DIR, FEATURES_DIR, catalog, sha256
from .coordinate_counter import CoordinatePeakSettings, count_coordinate_peaks
from .extract_coordinate_training import init_worker, add_signals
from . import extract_coordinate_training as extraction
from .extract_pushup_diagnostic import extract_pass
from .analyze_peak_errors import assign_reps, count_statistics
from .validate_peak_counter import fold_mae

FREEZE = ROOT / 'src/rep_counting/coordinate_voting_frozen.json'
TAG = 'coordinate-voting-v1-20261118'
PREVIOUS = ROOT / 'outputs/peak_improvements/test_followup_20261118'
TRAIN = ROOT / 'outputs/pushup_coordinates/train_v1_20261118'
TRAIN_REFERENCE = ROOT / 'outputs/peak_improvements/error_analysis_with_phase_20261118/segments.csv'
METHOD = 'proposed_unguided_peaks'


def paired_summary(table):
    """Paired video-cluster percentile bootstrap of new minus old count MAE."""
    table = table.copy()
    table['old_abs'] = abs(table.old_predicted - table.annotated)
    table['new_abs'] = abs(table.predicted - table.annotated)
    clusters = table.groupby('video_id')[['old_abs', 'new_abs']].sum()
    sizes = table.groupby('video_id').size().reindex(clusters.index).to_numpy()
    rng = np.random.default_rng(20261118)
    draw = rng.integers(0, len(clusters), size=(10000, len(clusters)))
    sums = clusters.to_numpy()[draw].sum(axis=1)
    delta = (sums[:, 1] - sums[:, 0]) / sizes[draw].sum(axis=1)
    def stats(column):
        rows = table.assign(predicted=table[column], error=table[column]-table.annotated)
        return count_statistics(rows)
    return dict(old=stats('old_predicted'), frozen=stats('predicted'),
                mae_change=float(table.new_abs.mean()-table.old_abs.mean()),
                mae_change_ci95=np.quantile(delta, [.025, .975]).tolist(),
                ci_method='10000 paired video-cluster percentile bootstrap draws; seed 20261118; new minus old; negative favors frozen',
                n_video_clusters=len(clusters))


def sensitivity(table, clips, training=False):
    result = {}
    for name, rows in [('all', table), ('without_top_two', table[~table.clip_id.isin(clips)])]:
        value = paired_summary(rows)
        if training:
            value['old_source_weighted_fold_mae'] = float(fold_mae(rows.assign(absolute_error=abs(rows.old_predicted-rows.annotated))).mean())
            value['frozen_source_weighted_fold_mae'] = float(fold_mae(rows.assign(absolute_error=abs(rows.predicted-rows.annotated))).mean())
        result[name] = value
    return dict(excluded_clip_ids=list(clips), **result)


def train_diagnostic(settings, output):
    reference = pd.read_csv(TRAIN_REFERENCE)
    reference = reference[reference.exercise.eq('push_up')].copy()
    ranking = reference.groupby('clip_id').absolute_error.mean().sort_values(ascending=False)
    clips = ranking.head(2).index.tolist()
    plan = json.loads((TRAIN/'plan.json').read_text())['plan']
    predictions = []
    for case in plan['cases']:
        path = TRAIN / f"{case['video_id']}__segment_{case['segment_index']:03d}.csv"
        from .extract_coordinate_training import verify_case, signature
        verify_case(TRAIN, case, signature(plan))
        result = count_coordinate_peaks(pd.read_csv(path), settings)
        predictions.append(dict(video_id=case['video_id'], segment_index=case['segment_index'],
                                predicted=result['reps'], selected_signal=result['selected_signal'], rep_times=result['rep_times']))
    rows = reference.drop(columns=['rep_times']).rename(columns={'predicted':'old_predicted', 'error':'old_error', 'absolute_error':'old_absolute_error'})
    rows = rows.merge(pd.DataFrame(predictions), on=['video_id','segment_index'], validate='one_to_one')
    rows['error'] = rows.predicted-rows.annotated
    rows['absolute_error'] = abs(rows.error)
    rows.to_csv(output/'training_window_signed_errors.csv', index=False)
    return sensitivity(rows, clips, training=True)


def extract_and_count(case, parameters):
    frame, _, _ = extract_pass(extraction.LANDMARKER, Path(case['video_path']), case['start_frame'], case['end_frame'], case['fps'])
    expected = list(range(case['start_frame']+case['start_frame']%2, case['end_frame'], 2))
    if frame.frame.tolist() != expected:
        raise ValueError('Incomplete test sidecar')
    frame, _ = add_signals(frame)
    result = count_coordinate_peaks(frame, CoordinatePeakSettings(**parameters))
    return case, frame, result


def evaluate(output, workers=2):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()) or output.exists():
        raise ValueError('Use a new directory under outputs; preserve previous runs')
    frozen = json.loads(FREEZE.read_text())
    commit = subprocess.check_output(['git','rev-parse',TAG+'^{}'], cwd=ROOT, text=True).strip()
    committed = subprocess.check_output(['git','show',TAG+':src/rep_counting/coordinate_voting_frozen.json'], cwd=ROOT)
    if json.loads(committed) != frozen:
        raise ValueError('Working freeze differs from tagged commit')
    for name, digest in frozen['source_hashes'].items():
        if sha256(ROOT/'src/rep_counting'/name) != digest:
            raise ValueError('Frozen implementation changed: '+name)
    model = ROOT/'models/pose_landmarker_full.task'
    if sha256(model) != frozen['pose_model_sha256']:
        raise ValueError('Frozen pose model changed')
    settings = CoordinatePeakSettings(**frozen['settings'])
    videos = {v:m for v,m in catalog().items() if m['split']=='test'}
    if len(videos) != 21:
        raise ValueError('Expected exactly 21 registered test videos')
    protected = {str(FREEZE):sha256(FREEZE), str(model):sha256(model), str(TRAIN_REFERENCE):sha256(TRAIN_REFERENCE)}
    for path in PREVIOUS.iterdir():
        if path.is_file(): protected[str(path)] = sha256(path)
    prior_report = json.loads((PREVIOUS/'report.json').read_text())
    for name, digest in prior_report['frozen_settings']['source_hashes'].items():
        if sha256(ROOT/'src/rep_counting'/name) != digest:
            raise ValueError('Earlier test method source changed')
    predicted = pd.read_csv(PREVIOUS/'predicted_segments.csv')
    predicted = predicted[predicted.method.eq(METHOD)].copy()
    oracle = pd.read_csv(PREVIOUS/'oracle_segments.csv')
    oracle = oracle[oracle.method.eq(METHOD)].copy()
    counts = pd.read_csv(PREVIOUS/'video_exercise.csv')
    counts = counts[counts.method.eq(METHOD)].copy()
    prior_inputs = {x['video_id']:x for x in prior_report['inputs']}
    cases = []
    for vid, meta in sorted(videos.items()):
        video = SYNTH_DIR/'test'/f'{vid}.mp4'
        for path in [video,video.with_suffix('.json'),FEATURES_DIR/f'{vid}_angles.csv']:
            protected[str(path)] = sha256(path)
        if sha256(video.with_suffix('.json')) != prior_inputs[vid]['annotations_sha256'] or sha256(FEATURES_DIR/f'{vid}_angles.csv') != prior_inputs[vid]['features_sha256']:
            raise ValueError('Test inputs differ from first evaluation')
        for row in predicted[predicted.video_id.eq(vid) & predicted.exercise.eq('push_up')].to_dict('records'):
            start = int(np.ceil(row['start_s']*meta['fps']-1e-8))
            end = min(meta['n_frames'], int(np.ceil((row['end_s']+.3)*meta['fps']-1e-8)))
            cases.append(dict(video_id=vid, kind='predicted', segment_index=int(row['segment_index']), start_frame=start,end_frame=end,fps=meta['fps'],video_path=str(video)))
        for idx, seg in enumerate(meta['segments']):
            if seg.get('label') == 'push_up':
                cases.append(dict(video_id=vid, kind='gt', segment_index=idx,start_frame=seg['start_frame'],end_frame=seg['end_frame'],fps=meta['fps'],video_path=str(video),clip_id=seg['clip_id'],annotated=seg['rep_count_annotated'],pairs=[(a/meta['fps'],b/meta['fps']) for a,b in seg['rep_pairs_out']]))
    output.mkdir(parents=True)
    sidecars=output/'sidecars'; sidecars.mkdir()
    record=dict(created_at_utc=datetime.now(timezone.utc).isoformat(), tag=TAG,commit=commit,freeze=frozen,settings=asdict(settings),cases=cases,protected_hashes=protected,evaluator_sha256=sha256(__file__),evaluation_label='Second exploratory test evaluation; already-inspected test set')
    (output/'evaluation_plan.json').write_text(json.dumps(record,indent=2))
    print(f'Tag {TAG} ({commit[:8]}): {len(cases)} frozen test windows, {len(videos)} videos',flush=True)
    train_results=train_diagnostic(settings,output)
    gt_predictions=[]; pred_predictions=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(model),)) as pool:
        futures=[pool.submit(extract_and_count,case,asdict(settings)) for case in cases]
        for done,future in enumerate(as_completed(futures),1):
            case,frame,result=future.result()
            stem=f"{case['video_id']}__{case['kind']}__{case['segment_index']:03d}"
            path=sidecars/(stem+'.csv'); frame.to_csv(path,index=False)
            (sidecars/(stem+'.json')).write_text(json.dumps(dict(case=case,csv_sha256=sha256(path),result=result),indent=2))
            key=dict(video_id=case['video_id'],segment_index=case['segment_index'])
            if case['kind']=='gt':
                old=oracle[oracle.video_id.eq(case['video_id']) & oracle.segment_index.eq(case['segment_index'])].iloc[0]
                hits,_,_=assign_reps(result['rep_times'],case['pairs'])
                gt_predictions.append(key|dict(clip_id=case['clip_id'],exercise='push_up',annotated=case['annotated'],old_predicted=int(old.predicted),predicted=result['reps'],error=result['reps']-case['annotated'],hits=len(hits),selected_signal=result['selected_signal'],rep_times=result['rep_times']))
            else:
                pred_predictions.append(key|dict(predicted=result['reps'],selected_signal=result['selected_signal'],rep_times=result['rep_times']))
            print(f'[{done}/{len(cases)}] {stem}',flush=True)
    gt=pd.DataFrame(gt_predictions); pred=pd.DataFrame(pred_predictions)
    new_counts=counts.rename(columns={'predicted':'old_predicted'}).copy()
    new_counts['predicted']=new_counts.old_predicted
    for index,row in new_counts[new_counts.exercise.eq('push_up')].iterrows():
        times=sorted(t for ts in pred[pred.video_id.eq(row.video_id)].rep_times for t in ts)
        new_counts.at[index,'predicted']=len(times)
    new_counts['error']=new_counts.predicted-new_counts.annotated
    all_oracle=oracle.rename(columns={'predicted':'old_predicted'}).copy()
    all_oracle['predicted']=all_oracle.old_predicted
    for row in gt.to_dict('records'):
        mask=all_oracle.video_id.eq(row['video_id']) & all_oracle.segment_index.eq(row['segment_index'])
        all_oracle.loc[mask,'predicted']=row['predicted']
    all_oracle['error']=all_oracle.predicted-all_oracle.annotated
    report=dict(evaluation_label=record['evaluation_label'],tag=TAG,commit=commit,freeze_sha256=sha256(FREEZE),n_videos=21,
                training=train_results,end_to_end=paired_summary(new_counts),
                end_to_end_by_exercise={ex:paired_summary(rows) for ex,rows in new_counts.groupby('exercise')},
                end_to_end_video_total=paired_summary(new_counts.groupby('video_id')[['annotated','old_predicted','predicted']].sum().reset_index()),
                gt_windows=paired_summary(all_oracle),gt_by_exercise={ex:paired_summary(rows) for ex,rows in all_oracle.groupby('exercise')},
                test_gt_sensitivity=sensitivity(gt,['stu9_52.mp4','stu4_46.mp4']),
                previous_run_report=str(PREVIOUS/'report.json'),previous_run_metrics=prior_report['methods'],
                no_parameters_changed=True,limitations=['Already-inspected test set: exploratory, not untouched holdout.','Only push-ups changed; jumping jacks and squats replay the earlier frozen proposed method.','Saved frozen segmentation boundaries reused; canonical upstream pose features remain annotation-assisted.','Coordinate person selection is largest pose per frame and can switch people.','GT windows are diagnostic; predicted windows get 0.3 s padding, GT windows zero.','Per-window torso normalization uses offline lookahead.','Video-cluster bootstrap has only 21 clusters; synthetic reuse of source clips may further reduce independence.','Top-two exclusion is descriptive and never changes headline metrics.','No production adoption or test-driven retuning.'])
    for path,digest in protected.items():
        if sha256(path)!=digest: raise ValueError('Protected input/result changed: '+path)
    report['inputs_and_previous_results_verified_unchanged']=True
    for name,rows in [('gt_pushup_window_signed_errors',gt),('predicted_pushup_windows',pred),('video_exercise_comparison',new_counts),('all_gt_window_comparison',all_oracle)]:
        rows.to_csv(output/(name+'.csv'),index=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ['end_to_end','end_to_end_by_exercise','gt_by_exercise']},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    evaluate(args.out_dir,args.workers)
