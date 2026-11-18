"""Isolated pose/crop pilot; write new coordinate sidecars under outputs only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.data_pipeline.v2_protocol import ROOT, sha256, load_json
from .peak_improvements import standardized
from .peak_counter import PeakSettings, detect_peaks
from .analyze_peak_errors import assign_reps


POINTS = dict(nose=0, shoulder_L=11, shoulder_R=12, elbow_L=13, elbow_R=14,
              wrist_L=15, wrist_R=16, hip_L=23, hip_R=24)


def extract_pass(landmarker, video, start_frame, end_frame, fps, roi=None):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows, boxes = [], []
    try:
        for frame_id in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read {video}:{frame_id}")
            if frame_id % 2:
                continue
            height, width = frame.shape[:2]
            x0, y0, x1, y1 = roi if roi else (0, 0, width, height)
            image = frame[y0:y1, x0:x1]
            if roi:
                image = cv2.resize(image, None, fx=2., fy=2., interpolation=cv2.INTER_CUBIC)
            result = landmarker.detect(Image(image_format=ImageFormat.SRGB,
                                             data=np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
            poses = result.pose_landmarks or []
            def area(pose):
                xy=np.array([(p.x,p.y) for p in pose if p.visibility >= .5])
                return float(np.ptp(xy[:,0])*np.ptp(xy[:,1])) if len(xy) >= 3 else 0.
            pose = max(poses, key=area) if poses else None
            row = dict(frame=frame_id, time_s=frame_id / fps, detected=pose is not None, n_poses=len(poses))
            for name, index in POINTS.items():
                point = pose[index] if pose else None
                row[f"{name}_x"] = x0 + point.x * (x1-x0) if point else np.nan
                row[f"{name}_y"] = y0 + point.y * (y1-y0) if point else np.nan
                row[f"{name}_vis"] = point.visibility if point else 0.
            if pose:
                xy=np.array([(x0+p.x*(x1-x0),y0+p.y*(y1-y0)) for p in pose if p.visibility >= .5])
                if len(xy) >= 3:boxes.append([xy[:,0].min(),xy[:,1].min(),xy[:,0].max(),xy[:,1].max()])
            rows.append(row)
    finally:
        cap.release()
    return pd.DataFrame(rows), boxes, (width, height)


def coordinate_signals(frame):
    """Fixed median torso scale avoids denominator-induced oscillations."""
    def pair(name):
        values=[];weights=[]
        for side in ("L","R"):
            point=frame[[f"{name}_{side}_x",f"{name}_{side}_y"]].to_numpy(dtype=float)
            vis=frame[f"{name}_{side}_vis"].to_numpy(dtype=float)
            w=np.where(np.isfinite(point).all(axis=1)&(vis>=.5),vis,0.)
            values.append(np.where(w[:,None]>0,point,0.));weights.append(w)
        denom=weights[0]+weights[1]
        out=np.full_like(values[0],np.nan)
        np.divide(values[0]*weights[0][:,None]+values[1]*weights[1][:,None],denom[:,None],out=out,where=denom[:,None]>0)
        return out
    shoulder,hip,wrist=pair("shoulder"),pair("hip"),pair("wrist")
    torso=np.linalg.norm(shoulder-hip,axis=1)
    valid=torso[np.isfinite(torso)&(torso>1.)]
    scale=float(np.median(valid)) if len(valid) else np.nan
    signals={"shoulder_height":-shoulder[:,1]/scale,
             "shoulder_to_wrist_height":(wrist[:,1]-shoulder[:,1])/scale,
             "head_height":-frame.nose_y.to_numpy(dtype=float)/scale}
    signals["head_height"][frame.nose_vis.to_numpy()<.5]=np.nan
    for side in ("L","R"):
        a=frame[[f"shoulder_{side}_x",f"shoulder_{side}_y"]].to_numpy()
        b=frame[[f"elbow_{side}_x",f"elbow_{side}_y"]].to_numpy()
        c=frame[[f"wrist_{side}_x",f"wrist_{side}_y"]].to_numpy()
        u,v=a-b,c-b;denom=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        cosine=np.divide((u*v).sum(axis=1),denom,out=np.full(len(frame),np.nan),where=denom>1e-9)
        angles=np.degrees(np.arccos(np.clip(cosine,-1,1)))
        confidence=frame[[f"shoulder_{side}_vis",f"elbow_{side}_vis",f"wrist_{side}_vis"]].min(axis=1).to_numpy()
        angles[confidence<.5]=np.nan;signals[f"elbow_{side}"]=angles
    return signals,scale


def run(output, cases):
    output=Path(output)
    if output.exists():raise FileExistsError("Use a new output directory")
    if not output.resolve().is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError("Diagnostic sidecars must stay under outputs; never rewrite canonical data")
    model=ROOT/'models/pose_landmarker_full.task'
    options=vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,num_poses=2,output_segmentation_masks=False)
    protected={str(model):sha256(model)};records=[]
    output.mkdir(parents=True)
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for case in cases:
            split,vid,index=case['split'],case['video_id'],case['segment_index']
            video=ROOT/f'data/synthetic/{split}/{vid}.mp4';annotation=video.with_suffix('.json')
            protected[str(video)]=sha256(video);protected[str(annotation)]=sha256(annotation)
            meta=load_json(annotation);seg=meta['segments'][index]
            if seg.get('label')!='push_up':raise ValueError("Pilot supports push-up windows only")
            print(f"Full-frame pilot: {split} {vid}:{index} {seg['clip_id']}",flush=True)
            full,boxes,size=extract_pass(landmarker,video,seg['start_frame'],seg['end_frame'],meta['fps'])
            roi=None
            if boxes:
                b=np.array(boxes);x0,y0=np.quantile(b[:,:2],.02,axis=0);x1,y1=np.quantile(b[:,2:],.98,axis=0)
                dx,dy=(x1-x0)*.15,(y1-y0)*.15;w,h=size
                roi=(max(0,int(x0-dx)),max(0,int(y0-dy)),min(w,int(np.ceil(x1+dx))),min(h,int(np.ceil(y1+dy))))
                if roi[2]-roi[0]<32 or roi[3]-roi[1]<32:roi=None
            print(f"Crop pilot: {vid}:{index}, fixed ROI={roi}",flush=True)
            crop,_,_=extract_pass(landmarker,video,seg['start_frame'],seg['end_frame'],meta['fps'],roi=roi)
            pairs=[(a/meta['fps'],b/meta['fps']) for a,b in seg['rep_pairs_out']]
            for mode,frame in [('full_frame',full),('person_crop_2x',crop)]:
                signals,scale=coordinate_signals(frame)
                for name,raw in signals.items():
                    signal=standardized(raw,15.,.3,20. if name.startswith('elbow') else .1)
                    predicted=detect_peaks(signal,frame.time_s.to_numpy(),'down_up',.2,
                        PeakSettings(prominence_fraction=.45,min_prominence_deg=.1,min_width_s=1/15,period_distance_fraction=0.,recovery_fraction=.7))
                    matches,_,_=assign_reps(predicted['rep_times'],pairs)
                    records.append(dict(split=split,video_id=vid,segment_index=index,clip_id=seg['clip_id'],mode=mode,
                        signal=name,annotated=len(pairs),predicted=predicted['reps'],error=predicted['reps']-len(pairs),
                        coverage=float(np.isfinite(raw).mean()),hits=len(matches),torso_scale_pixels=scale,
                        roi=roi,rep_times=predicted['rep_times']))
                    frame[name]=raw
                frame.to_csv(output/f'{split}_{vid}_{index}_{mode}.csv',index=False)
    for path,digest in protected.items():
        if sha256(path)!=digest:raise ValueError("Existing input changed")
    pd.DataFrame(records).to_csv(output/'pilot_results.csv',index=False)
    report=dict(cases=cases,protected_hashes=protected,model_sha256=sha256(model),inputs_verified_unchanged=True,
        settings=dict(smoothing_s=.3,min_spacing_s=.2,prominence_fraction=.45,recovery_fraction=.7),
        limitations=['Descriptive pilot, not a fold-validated coordinate-signal method or new headline result.',
         'Counter settings fixed before extraction; no test-driven setting or signal selection.',
         'Largest pose bounding box used in both passes; differs from canonical annotation-assisted person selection.',
         'Crop can change subject identity; improvement alone does not prove tracking was the sole cause.',
         'Fixed crop uses full-frame detections from the same window and has offline lookahead.',
         'No existing features, manifests, models, video or annotation files rewritten.',
         'Spatial signals use a per-pass fixed median torso scale; camera motion and edit cuts remain confounds.'])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True);args=parser.parse_args()
    run(args.out_dir,json.loads(args.cases.read_text(encoding='utf-8')))
