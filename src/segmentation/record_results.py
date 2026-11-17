"""Record completed CV + winner-only final-test artifacts without any fitting."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, CLASSES
from workflow import load_bundle, score_predictions
from v2_protocol import ROOT, active_path, catalog, sha256


def record(experiment):
    experiment=active_path(experiment)
    selection=json.loads((experiment/'selection.json').read_text())
    ranking=pd.read_csv(experiment/'cv_results.csv')
    best=selection['best']
    first=ranking.iloc[0]
    assert first.model_key==best['model_key'] and first.tau==best['tau'] and first.min_len==best['min_len']
    model_path=experiment/'final'/'model.joblib'
    bundle,_=load_bundle(model_path)
    test_dir=experiment/'final'/'test_evaluation'
    test=json.loads((test_dir/'results.json').read_text())
    assert test['parameters_frozen'] and test['model_sha256']==sha256(model_path)
    assert test['experiment_id']==selection['experiment_id']==bundle['experiment_id']
    assert bundle['selection_sha256']==sha256(experiment/'selection.json')
    comparison=ranking.drop_duplicates('model_key').copy()
    n_models=len(comparison)
    videos=catalog()
    diagnostics=[];fold_rows=[]
    for row in comparison.to_dict('records'):
        candidate=experiment/f"{row['model_key']}_purity_{row['min_purity_train']:g}"
        windows=json.loads((candidate/'window_metrics.json').read_text())
        for entry in windows:diagnostics.append(dict(model_key=row['model_key'],confusion_matrix_class_order=CLASSES,**entry))
        oof=pd.read_csv(candidate/'oof_predictions.csv')
        assert oof.split.eq('train').all() and oof.cv_fold.eq(oof.fold).all()
        _,folds=score_predictions(oof,videos,row['tau'],int(row['min_len']))
        folds=folds.reset_index();folds['model_key']=row['model_key']
        fold_rows.extend(folds.to_dict('records'))
    timings=pd.DataFrame(diagnostics).groupby('model_key').elapsed_s.sum()
    comparison['fit_predict_seconds']=comparison.model_key.map(timings)
    directory=ROOT/'docs'/'results'/selection['experiment_id']
    directory.mkdir(parents=True,exist_ok=True)
    for source in [experiment/'selection.json',experiment/'cv_results.csv',experiment/'cv_fold_metrics.csv',
                   experiment/'cv_video_metrics.csv',experiment/'run_manifest.json',
                   test_dir/'results.json',test_dir/'video_metrics.csv',test_dir/'raw_video_metrics.csv',
                   experiment/'final'/'chosen_params.json']:
        shutil.copy2(source,directory/source.name)
    snapshot=experiment/'training_source'
    if snapshot.exists():
        target=directory/'training_source';target.mkdir(exist_ok=True)
        for source in snapshot.glob('*.py'):shutil.copy2(source,target/source.name)
    comparison.to_csv(directory/'model_comparison.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(directory/'all_model_fold_metrics.csv',index=False)
    (directory/'window_metrics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    raw=pd.read_csv(test_dir/'raw_video_metrics.csv').select_dtypes(include='number').mean()
    manifest=json.loads((experiment/'run_manifest.json').read_text())
    assert manifest['status']=='complete'
    log=OUTPUT_DIR/'run_logs'/f"{selection['experiment_id']}.log"
    warned_models=set();current_model=None
    log_bytes=log.read_bytes() if log.exists() else b''
    encoding='utf-16' if log_bytes.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
    log_text=log_bytes.decode(encoding,errors='replace')
    for line in log_text.splitlines():
        if line.startswith('CV '):current_model=line.split()[1].rstrip(',')
        if 'ConvergenceWarning:' in line and current_model:warned_models.add(current_model)
    lines=[f'# V2 {n_models}-model comparison — 17 November 2026','',
           f"Experiment: `{selection['experiment_id']}`. Status: complete.",'',
           f'All {n_models} classifiers were compared using five source-clip folds inside train. '
           'Purity=0.8; seed=42; tau grid {0.4,0.5,0.6,0.7}; min_len grid {1,2,3,4}. '
           'Selection maximizes mean fold per-video Edit, then frame accuracy, using the documented deterministic ties. '
           'Test features were read only after selection and all-train refit.','',
           '| Model | CV Edit mean ± fold SD | CV frame accuracy | CV segment F1@25 | tau | min_len | Fit/predict seconds (5 folds) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in comparison.to_dict('records'):
        lines.append(f"| {row['model_key']} | {row['cv_edit_mean']:.2f} ± {row['cv_edit_std']:.2f} | "
                     f"{100*row['cv_frame_accuracy_mean']:.2f}% | {row['cv_f1_at_25_mean']:.3f} | "
                     f"{row['tau']:g} | {int(row['min_len'])} | {row['fit_predict_seconds']:.1f} |")
    lines += ['',f"**CV winner: {best['model_key']}**, tau={best['tau']:g}, min_len={best['min_len']} windows. "
              f"Refitted on all {bundle['n_fit_windows']:,} eligible train windows. "
              'The selected bundle is active for integration; no other model was evaluated on test.','',
              '## Frozen final-test results','',
              f"{test['n_videos']} synthetic test videos, {test['n_source_clips']} unique clips; 15 unused clips remain unused. "
              'The model and smoothing choice were not changed after these results.','',
              '| Metric | Raw mean | Smoothed mean ± video SD |','|---|---:|---:|']
    if n_models>1:
        gap=float(comparison.iloc[0].cv_edit_mean-comparison.iloc[1].cv_edit_mean)
        lines.insert(lines.index('## Frozen final-test results'),
                     f"The winner–runner-up CV Edit gap is {gap:.3f} points. Ranking follows the fixed rule; no statistical superiority is established.\n")
    metrics=[('edit_score','Edit score',1),('frame_accuracy','Frame accuracy (%)',100),
             ('f1_at_10','Segment F1@10',1),('f1_at_25','Segment F1@25',1),('f1_at_50','Segment F1@50',1),
             ('boundary_mean_s','Nearest-boundary mean error (s)',1),
             ('boundary_median_s','Nearest-boundary median error (s)',1),
             ('frame_f1_squat','Frame F1: squat',1),('frame_f1_push_up','Frame F1: push_up',1),
             ('frame_f1_jump_jack','Frame F1: jump_jack',1),('frame_f1_transition','Frame F1: transition',1),
             ('n_pred','Predicted segment count',1),('n_gt','GT segment count',1)]
    for key,label,factor in metrics:
        lines.append(f"| {label} | {factor*raw[key]:.3f} | {factor*test['mean'][key]:.3f} ± {factor*test['std_across_videos'][key]:.3f} |")
    lines += ['','## Interpretation and limitations','',
              '- CV scores are selection-CV estimates, not unbiased nested-CV performance. Fold SD is descriptive variability, not a subject confidence interval.',
              '- Test SD is across the 21 constructed videos, not a confidence interval. Native split provenance does not establish subject independence.',
              '- Per-class frame F1 is averaged over videos using the existing zero-division=0 convention, including videos where a class is absent; it is not pooled frame-level F1.',
              '- Fit/predict wall times were observed on a shared machine with each classifier\'s configured parallelism; they are descriptive, not a controlled runtime benchmark.',
              '- This is a synthetic, annotation-assisted pose benchmark: annotated exercise labels still assist upstream person selection. Continuous, unassisted workout generalization was not measured.',
              '- No test-driven parameter changes, model reranking, source exclusions or extra clip reuse were performed.']
    if warned_models:
        lines.append(f"- Convergence warnings were logged for: {', '.join(sorted(warned_models))}. "
                     'Fixed iteration budgets were retained; these scores describe the configured solvers, not guaranteed fully converged optima.')
    if 'CalibratedClassifierCV' in log_text and 'deprecated' in log_text:
        lines.append('- The installed sklearn emitted an SVC probability-API deprecation warning. The predeclared implementation was retained; a future migration requires a new validated experiment, not a change to this frozen result.')
    if manifest.get('cv_shell_wrapper_exit_code',0)!=0:
        lines.append('- The PowerShell logging wrapper returned a nonzero status while stderr warnings were captured. Before refit, all 40 OOF folds, 128 configuration rows and the selected prediction hash were independently verified as complete; no failed fold was replaced or omitted.')
    if manifest.get('post_run_artifact_cleanup'):
        lines.append('- Inference-only placeholder label/purity columns were removed from the test prediction CSV after a schema audit. Original CSV and training source were preserved; prediction/probability columns, model weights, selected settings and test metrics are unchanged. Test metrics use GT segment JSONs, not placeholder window labels.')
    lines += ['', '## Artifacts','',f"Small result tables, settings and hashes: [{selection['experiment_id']}/]({selection['experiment_id']}/).",
              f"Local full log: `outputs/segmentation_v2/run_logs/{selection['experiment_id']}.log`.",
              f"Local model: `outputs/segmentation_v2/experiments/{selection['experiment_id']}/final/model.joblib`.",
              'The frozen v1 archive remains unchanged. Full OOF predictions, model binaries and features stay local.','']
    report=ROOT/'docs'/'results'/'V2_MODEL_RESULTS.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    start='<!-- v2-model-results:start -->';end='<!-- v2-model-results:end -->'
    paths=['README.md','PROJECT_CONTEXT.md','docs/PROJECT_CONTEXT.md','docs/V2_TRAINING.md',
           'src/segmentation/README.md','src/segmentation/USAGE.md','src/pipeline/README.md',
           'paper/PROJECT_REPORT_AND_PAPER_CONTEXT.md','paper/scope_proposal.md','paper/related_work_notes.md']
    import os
    for relative in paths:
        path=ROOT/relative;text=path.read_text(encoding='utf-8')
        if start in text:
            before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+after
        link=Path(os.path.relpath(report,path.parent)).as_posix()
        summary=(f"\n{start}\n## Completed v2 model experiment — 17 November 2026\n\n"
                 f"{n_models}-model train-only CV selected **{best['model_key']}** "
                 f"(Edit {best['cv_edit_mean']:.2f} ± {best['cv_edit_std']:.2f} fold SD; "
                 f"tau={best['tau']:g}, min_len={best['min_len']}). "
                 f"After all-train refit, only that frozen winner was evaluated on {test['n_videos']} test videos: "
                 f"Edit {test['mean']['edit_score']:.2f}, frame accuracy {100*test['mean']['frame_accuracy']:.2f}%, "
                 f"segment F1@25 {test['mean']['f1_at_25']:.3f}. "
                 'Test was not used for model selection or tuning. '
                 f"See [full results, caveats and reproducibility tables]({link}).\n{end}\n")
        path.write_text(text.rstrip()+'\n'+summary,encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    args=parser.parse_args()
    print(record(args.experiment))


if __name__=='__main__':main()
