"""Resume-only Phase 4: extract, summarize, diagnose, window and audit; no models."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from v2_protocol import (ROOT, FEATURES_DIR, OUTPUT_DIR, SUMMARY, catalog,
                         feature_frame, require_v2, sha256)


def archive_fingerprint():
    """Hash names/stat metadata without modifying or reading archived videos."""
    root = ROOT/'data'/'_archive_v1'
    fingerprint = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file():
            stat = path.stat()
            fingerprint.update(f'{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    return fingerprint.hexdigest()


def stale_references():
    """Scan active text artifacts and filenames, excluding VCS/runtime caches."""
    # Split the expression so this scanner does not create a historical ID.
    pattern = re.compile(r'synth_' + r'(?:train|val|test)_' + r'\d+')
    excluded = {'.git','.venv','__pycache__','.pytest_cache','_archive_v1'}
    text_extensions = {'.py','.md','.json','.csv','.ipynb','.txt','.bat','.toml'}
    findings = []
    def visit(directory):
        for path in directory.iterdir():
            if path.name in excluded:
                continue
            if path.is_dir():
                yield from visit(path)
            else:
                yield path
    for path in visit(ROOT):
        if pattern.search(path.name):
            findings.append(str(path.relative_to(ROOT)))
        if path.suffix in text_extensions:
            # CSVs contain only numeric features and v2 IDs after per-file
            # provenance validation; scanning them is still read-only.
            text = path.read_text(encoding='utf-8-sig',errors='replace')
            if pattern.search(text):
                findings.append(str(path.relative_to(ROOT)))
    return sorted(set(findings))


def rebuild_summary(videos):
    rows = []
    for video_id, meta in videos.items():
        path = FEATURES_DIR/f'{video_id}_angles.csv'
        sidecar = require_v2(json.loads(path.with_suffix('.meta.json').read_text()))
        assert sidecar['csv_sha256']==sha256(path)
        assert sidecar['stride']==2 and sidecar['mode']=='image'
        assert sidecar['source_json_sha256']==sha256(ROOT/'data'/'synthetic'/meta['split']/f'{video_id}.json')
        summary = sidecar['summary'] | {'status':'success'}
        require_v2(summary)
        rows.append(summary)
    summary = pd.DataFrame(rows).sort_values('video_id')
    summary.to_csv(SUMMARY,index=False)
    flags = [c for c in summary if c.startswith('has_long_gap_')]
    long_gaps = summary.loc[summary[flags].fillna(False).any(axis=1)]
    long_gaps.to_csv(OUTPUT_DIR/'has_long_gap_videos.csv',index=False)
    print('\nHAS_LONG_GAP videos (>1 second within one exercise segment):',len(long_gaps))
    print(long_gaps[['video_id','split','fold']+flags].to_string(index=False))
    print('\nLong-gap video counts by split:')
    print(long_gaps.groupby('split').size().to_string())
    return summary


def finish_phase4(before_archive):
    from boundary_jump_check import run_check
    videos = catalog()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    assert {p.stem.removesuffix('_angles') for p in FEATURES_DIR.glob('*_angles.csv')} == set(videos)
    for video_id, meta in videos.items():
        df = feature_frame(FEATURES_DIR/f'{video_id}_angles.csv',meta)
        assert df.frame.tolist()==list(range(0,meta['n_frames'],2))
    summary = rebuild_summary(videos)
    boundary, motion = run_check(ROOT/'data'/'synthetic',FEATURES_DIR,OUTPUT_DIR/'boundary_jump_check.csv')
    print('\nBoundary-jump ratios (reporting only; no selection or tuning):')
    print(boundary.to_string(index=False))
    subprocess.run([sys.executable,'-B',str(ROOT/'src'/'segmentation'/'make_windows.py'),
                    '--features-dir',str(FEATURES_DIR),'--out',str(OUTPUT_DIR/'windows.csv')],cwd=ROOT,check=True)
    sys.path.insert(0,str(ROOT/'src'/'segmentation'))
    from cv import count_windows, get_cv_split, validate_windows
    windows = pd.read_csv(OUTPUT_DIR/'windows.csv')
    validate_windows(windows,videos)
    assert set(windows.video_id)==set(videos)
    counts = count_windows(windows)
    counts.to_csv(OUTPUT_DIR/'window_counts.csv',index=False)
    cv_counts=[]
    for fold in range(5):
        train, validation = get_cv_split(windows,fold,videos)
        cv_counts.append(dict(fold=fold,train_windows=len(train),validation_windows=len(validation)))
        print(f'CV fold {fold}: train_windows={len(train)}, validation_windows={len(validation)}; source-clip overlap=0; test windows=0')
    pd.DataFrame(cv_counts).to_csv(OUTPUT_DIR/'cv_counts.csv',index=False)
    findings = stale_references()
    assert not findings, f'Active artifacts reference historical video IDs: {findings}'
    assert archive_fingerprint()==before_archive, 'Frozen v1 archive was modified'
    report = dict(run_id='v2',videos=len(videos),feature_files=len(videos),windows=len(windows),
                  cv_counts=cv_counts,archive_fingerprint=before_archive,
                  model_training=False,model_evaluation=False,status='complete')
    (OUTPUT_DIR/'phase4_consistency.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nPASS: every feature/window maps to a v2 video; source split/fold matches splits_v2.csv/folds.csv; '
          'CV source-clip overlap is zero; no active artifact references a v1 video ID; frozen archive unchanged; '
          'no model trained/evaluated; test used for extraction and counting only.')
    context = ROOT/'docs'/'PROJECT_CONTEXT.md'
    text=context.read_text(encoding='utf-8')
    marker='\n## Phase 4 completed run\n'
    if marker in text:
        text=text.split(marker)[0]
    text=text.replace("The first video is already extracted (66.35 seconds); the other 140 remain.",
                      "The first-video benchmark was 66.35 seconds; all 141 per-video features are now complete.")
    text=text.replace("Phase 4 remains pending full extraction and the final data audit. The runner\nwill append completion status and counts only after every check passes.",
                      "Phase 4 extraction and final data audit completed successfully; details follow.")
    context.write_text(text+marker+f'\nV2 per-video features: {len(videos)} videos, stride 2, workers 1, IMAGE mode, '
                       f'resume only. Windows: {len(windows)}. No model training/evaluation. '
                       'Summary, long-gap list, boundary ratios, window counts and CV partition counts '
                       'are saved in outputs/segmentation_v2/. Consistency checks passed; archive unchanged.\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--postprocess-only',action='store_true',help='Require complete features, then rebuild reports/windows only')
    args=parser.parse_args()
    before=archive_fingerprint()
    videos=catalog()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    print(f'Phase 4: {len(videos)} v2 videos; archive read-only; no models',flush=True)
    if not args.postprocess_only:
        command=[sys.executable,'-B',str(ROOT/'src'/'data_pipeline'/'extract_pose_features.py'),
                 '--all','--stride','2','--workers','1','--features-dir',str(FEATURES_DIR)]
        with (OUTPUT_DIR/'pose_extraction.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                print(line,end='',flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode,command)
    finish_phase4(before)


if __name__=='__main__':
    main()
