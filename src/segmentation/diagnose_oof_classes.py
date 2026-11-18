"""Read-only per-class OOF diagnostics, including each smoothing stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.segmentation import workflow
from src.segmentation.smooth import _apply_low_confidence, _enforce_min_len, _merge_sandwiched_transitions, to_segments
from src.data_pipeline.v2_protocol import ROOT, SYNTH_DIR, FEATURES_DIR, sha256

EXPERIMENT=ROOT/'outputs/segmentation_v2/experiments/cv8_v2_seed42_p080_20261117'
FAILURES=ROOT/'outputs/pushup_coordinates/jumping_jack_window_failures_20261118/gt_windows.csv'
LABELS=['jump_jack','push_up','squat','transition']


def duration_confusion(truth,predicted):
    """Exact interval overlap in seconds; rows GT, columns predicted."""
    matrix=pd.DataFrame(0.,index=LABELS,columns=LABELS)
    for gt in truth:
        for pred in predicted:
            seconds=max(0.,min(gt['end_time'],pred['end_time'])-max(gt['start_time'],pred['start_time']))
            matrix.loc[gt['exercise'],pred['exercise']]+=seconds
    return matrix


def class_metrics(matrix):
    result={}
    for label in LABELS:
        tp=float(matrix.loc[label,label]); support=float(matrix.loc[label].sum()); predicted=float(matrix[label].sum())
        precision=tp/predicted if predicted else 0.; recall=tp/support if support else 0.
        result[label]=dict(support=support,predicted=predicted,precision=precision,recall=recall,
            f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    return result


def stages(labels,confidence,tau,min_len):
    raw=list(labels)
    filled=_apply_low_confidence(raw,list(confidence),tau)
    minimum=_enforce_min_len(filled,min_len)
    final=_merge_sandwiched_transitions(minimum)
    return dict(raw=raw,confidence=filled,minimum_run=minimum,final=final)


def diagnose(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to((ROOT/'outputs').resolve()): raise ValueError('Use a new output directory under outputs')
    selection_path=EXPERIMENT/'selection.json'; predictions_path=EXPERIMENT/'oof_predictions.csv'
    selection=json.loads(selection_path.read_text())
    windows,videos=workflow.read_training()
    if workflow.signature(videos,windows,FEATURES_DIR)!=selection['train_signature'] or sha256(predictions_path)!=selection['oof_sha256']:
        raise ValueError('Segmentation inputs differ from saved selection')
    oof=pd.read_csv(predictions_path)
    workflow.validate_windows(oof,videos)
    keys=['video_id','win_idx']
    if not oof.split.eq('train').all() or not oof.fold.eq(oof.cv_fold).all() or oof.duplicated(keys).any() or set(map(tuple,oof[keys].to_numpy()))!=set(map(tuple,windows[keys].to_numpy())):
        raise ValueError('Incomplete or invalid held-out OOF coverage')
    failed=pd.read_csv(FAILURES); failed=failed[failed.mostly_mislabeled].copy()
    if len(failed)!=19 or failed.video_id.nunique()!=19: raise ValueError('Expected the recorded 19 mislabeled JJ videos/windows')
    protected={str(p):sha256(p) for p in (selection_path,predictions_path,FAILURES,workflow.WINDOWS_CSV)}
    names=['raw','confidence','minimum_run','final']
    pooled={name:pd.DataFrame(0.,index=LABELS,columns=LABELS) for name in names}
    target={name:pd.DataFrame(0.,index=LABELS,columns=LABELS) for name in names}
    fold_matrices={fold:{name:pd.DataFrame(0.,index=LABELS,columns=LABELS) for name in names} for fold in range(5)}
    cases=[]; all_windows=[]; flow=[]; per_video=[]
    for vid,rows in oof.groupby('video_id',sort=True):
        meta=videos[vid]; protected[str(SYNTH_DIR/'train'/f'{vid}.json')]=sha256(SYNTH_DIR/'train'/f'{vid}.json')
        rows=rows.sort_values('win_idx').copy()
        labels=stages(rows.pred_label,rows.max_prob,selection['best']['tau'],selection['best']['min_len'])
        if labels['final']!=workflow.smooth(rows.pred_label.tolist(),rows.max_prob.tolist(),selection['best']['tau'],selection['best']['min_len']): raise ValueError('Stage diagnostic differs from production smoothing')
        truth,end=workflow.gt_from_json(meta)
        by_stage={}
        for name,values in labels.items():
            rows[name+'_label']=values
            segments=to_segments(values,rows.t_center.tolist(),end); by_stage[name]=segments
            matrix=duration_confusion(truth,segments); pooled[name]+=matrix; fold_matrices[int(meta['fold'])][name]+=matrix
            if vid in set(failed.video_id): target[name]+=matrix
            for gt_label in LABELS:
                per_video.append(dict(video_id=vid,fold=meta['fold'],stage=name,gt_label=gt_label,**{f'pred_{p}_s':float(matrix.loc[gt_label,p]) for p in LABELS}))
        all_windows.append(rows)
        for case in failed[failed.video_id.eq(vid)].to_dict('records'):
            gt=[dict(exercise='jump_jack',start_time=case['gt_start_s'],end_time=case['gt_end_s'])]
            record={key:case[key] for key in ('video_id','fold','gt_index','clip_id','gt_start_s','gt_end_s','gt_length_s')}
            for name in names:
                overlap=duration_confusion(gt,by_stage[name]).loc['jump_jack']
                record[name+'_dominant_label']=overlap.idxmax()
                for label in LABELS: record[f'{name}_{label}_fraction']=float(overlap[label]/case['gt_length_s'])
            if record['final_dominant_label']!=case['dominant_predicted_label']: raise ValueError('Failed-window diagnosis differs from saved result')
            cases.append(record)
        # Count which raw JJ-majority windows change labels at each smoothing stage.
        selected=rows[rows.label.eq('jump_jack')]
        for a,b in zip(names[:-1],names[1:]):
            changes=selected.groupby([a+'_label',b+'_label']).size()
            for (before,after),count in changes.items():
                flow.append(dict(video_id=vid,fold=meta['fold'],stage_from=a,stage_to=b,label_before=before,label_after=after,n_windows=int(count)))
    all_windows=pd.concat(all_windows); cases=pd.DataFrame(cases)
    window_reports={}
    for scope,rows in [('all',all_windows),('high_purity',all_windows[all_windows.purity.ge(.8)])]:
        window_reports[scope]={}
        for name in names:
            matrix=pd.crosstab(rows.label,rows[name+'_label']).reindex(index=LABELS,columns=LABELS,fill_value=0)
            window_reports[scope][name]=dict(confusion=matrix.to_dict(orient='index'),metrics=class_metrics(matrix))
    frame_reports={name:dict(confusion_seconds=matrix.to_dict(orient='index'),metrics=class_metrics(matrix)) for name,matrix in pooled.items()}
    cohort_reports={name:dict(confusion_seconds=matrix.to_dict(orient='index'),metrics=class_metrics(matrix)) for name,matrix in target.items()}
    case_tally={name:cases[name+'_dominant_label'].value_counts().to_dict() for name in names}
    stages_flow=pd.DataFrame(flow).groupby(['stage_from','stage_to','label_before','label_after']).n_windows.sum().reset_index()
    jj_stats={}
    for name in names:
        rows=all_windows[all_windows.label.eq('jump_jack')]
        jj_stats[name]={}
        for label,group in rows.groupby(name+'_label'):
            jj_stats[name][label]=dict(n_windows=len(group),low_confidence_fraction=float(group.max_prob.lt(.5).mean()),
                mean_raw_jj_probability=float(group.prob_jump_jack.mean()),mean_raw_transition_probability=float(group.prob_transition.mean()))
    report=dict(split='train',n_videos=all_windows.video_id.nunique(),n_oof_windows=len(all_windows),labels=LABELS,
        matrix_orientation='Rows=ground truth, columns=prediction. Window matrices are counts; temporal matrices are exact overlap seconds.',
        window_diagnostics=window_reports,temporal_diagnostics=frame_reports,mislabelled_19_video_temporal_diagnostics=cohort_reports,
        failed_window_dominant_label_tally=case_tally,jj_window_confidence=jj_stats,
        macro_fold_recall={name:{label:float(np.mean([class_metrics(fold_matrices[f][name])[label]['recall'] for f in range(5)])) for label in LABELS} for name in names},
        smoothing=selection['best'],smoothing_rules=['Low-confidence non-transition predictions inherit prior smoothed label; raw transition predictions are exempt.','Short non-transition runs (<3 windows) are absorbed into a neighbor; transition runs are exempt.','Transitions sandwiched between identical exercises are merged into that exercise.'],
        protected_inputs=protected,source_sha256=sha256(__file__),
        limitations=['Development selection-CV with source-held-out predictions, not independent nested CV.','Raw window labels use majority GT and include mixed boundaries; high-purity diagnostics are separate.','Exact temporal overlap uses continuous seconds, not the official 10 Hz accuracy grid, so scores need not be numerically identical.','The 19-video cohort was selected after observing errors; it is diagnostic, not a new evaluation set.','Stage changes and probabilities describe associations, not causal feature attribution.','No model/pose/counter inference, tuning, data writes, production changes, commits or pushes.'])
    for p,h in protected.items():
        if sha256(p)!=h: raise ValueError('Input changed during diagnostics')
    report['inputs_verified_unchanged']=True
    output.mkdir(parents=True)
    for name,matrix in pooled.items(): matrix.rename_axis('ground_truth').to_csv(output/f'{name}_temporal_confusion_seconds.csv')
    for name in ('raw','final'):
        matrix=pd.DataFrame(window_reports['all'][name]['confusion']).T
        matrix.rename_axis('ground_truth').to_csv(output/f'{name}_window_confusion.csv')
    cases.to_csv(output/'mislabelled_19_windows.csv',index=False)
    pd.DataFrame(per_video).to_csv(output/'per_video_temporal_confusion.csv',index=False)
    stages_flow.to_csv(output/'jj_smoothing_label_flow.csv',index=False)
    all_windows.to_csv(output/'oof_window_stage_labels.csv',index=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(window_raw=window_reports['all']['raw']['metrics'],window_final=window_reports['all']['final']['metrics'],temporal_recall={n:{c:v['recall'] for c,v in x['metrics'].items()} for n,x in frame_reports.items()},failed_window_dominant_label_tally=case_tally),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args(); diagnose(args.out_dir)
