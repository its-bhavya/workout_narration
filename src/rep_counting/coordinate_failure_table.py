"""Describe frozen-run failures without rerunning or tuning the counter."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import ROOT, sha256


def missing_fraction(frame, signal):
    values=frame[signal].to_numpy(dtype=float)
    confidence=frame[signal+'_confidence'].to_numpy(dtype=float)
    return float(1-(np.isfinite(values)&np.isfinite(confidence)&(confidence>=.5)).mean())


def build(run_dir, diagnosis_dir):
    run_dir,diagnosis_dir=Path(run_dir).resolve(),Path(diagnosis_dir).resolve()
    if not run_dir.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Failure artifacts must stay under outputs')
    target=run_dir/'largest_signed_error_failures.csv'
    if target.exists(): raise FileExistsError('Preserve existing failure analysis')
    error_path=run_dir/'gt_pushup_window_signed_errors.csv'
    view_path=diagnosis_dir/'test_signed_errors_and_camera_view.csv'
    arm_path=diagnosis_dir/'arm_tracking_diagnostics.csv'
    rows=pd.read_csv(error_path)
    rows['absolute_error']=abs(rows.error)
    rows=rows.sort_values(['absolute_error','clip_id'],ascending=[False,True]).head(8)
    views=pd.read_csv(view_path)
    arms=pd.read_csv(arm_path)
    table=[]; protected={str(p):sha256(p) for p in (error_path,view_path,arm_path)}
    for row in rows.to_dict('records'):
        view=views[(views.video_id==row['video_id']) & (views.segment_index==row['segment_index'])]
        arm=arms[(arms.video_id==row['video_id']) & (arms.clip_id==row['clip_id'])]
        if len(view)!=1 or len(arm)!=1: raise ValueError('Ambiguous or absent camera/arm provenance')
        view,arm=view.iloc[0],arm.iloc[0]
        sidecar=run_dir/'sidecars'/f"{row['video_id']}__gt__{row['segment_index']:03d}.csv"
        protected[str(sidecar)]=sha256(sidecar)
        frame=pd.read_csv(sidecar)
        selected=row['selected_signal']
        table.append(dict(clip_id=row['clip_id'],video_id=row['video_id'],segment_index=row['segment_index'],
            camera_view=view.camera_view,view_evidence=view.view_evidence,
            review_notes='' if pd.isna(view.review_notes) else view.review_notes,
            annotated=row['annotated'],old_predicted=row['old_predicted'],frozen_predicted=row['predicted'],
            old_signed_error=row['old_predicted']-row['annotated'],frozen_signed_error=row['error'],
            old_selected_elbow=arm.selected,old_elbow_missing_fraction=1-float(arm['coverage_'+arm.selected[-1]]),
            frozen_selected_signal=selected,frozen_selected_missing_fraction=missing_fraction(frame,selected),
            fresh_left_elbow_missing_fraction=missing_fraction(frame,'elbow_L'),
            fresh_right_elbow_missing_fraction=missing_fraction(frame,'elbow_R')))
    pd.DataFrame(table).to_csv(target,index=False)
    metadata=dict(selection='Eight largest absolute signed errors of frozen GT-window push-up counts; ties by clip ID',
        view_scope='Previously saved manual inspection of three sampled frames; coarse, uncalibrated tags',
        missing_definition='1 - fraction of raw sampled signal values that are finite with finite model confidence >=0.5, before interpolation/smoothing. Old elbow coverage comes from saved canonical-arm diagnostic.',
        interpretation='Model-estimated observation availability, not verified landmark accuracy. High visibility can coexist with wrong identity, edited clips or unsuitable motion signals. Voting returns one actual candidate timeline; selected-signal missingness is not whole-ensemble missingness.',
        protected_inputs=protected,source_sha256=sha256(__file__),result_sha256=sha256(target),
        no_inference_or_counting_repeated=True)
    for p,h in protected.items():
        if sha256(p)!=h: raise ValueError('Input changed during analysis')
    (run_dir/'largest_signed_error_failures_provenance.json').write_text(json.dumps(metadata,indent=2))
    return pd.DataFrame(table)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--diagnosis-dir',type=Path,default=ROOT/'outputs/pushup_diagnosis/initial_20261118')
    args=parser.parse_args()
    print(build(args.run_dir,args.diagnosis_dir).to_string(index=False))
