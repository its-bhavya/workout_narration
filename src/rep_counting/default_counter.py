"""Integrated frozen peak/coordinate combo adopted on 18 November 2026."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import ROOT, sha256
from .coordinate_counter import CoordinatePeakSettings, count_coordinate_peaks
from .peak_improvements import ImprovementSettings, count_improved_peaks

FREEZE_PATH=Path(__file__).with_name('coordinate_voting_frozen.json')
COUNTER_ID='experimental_combo_v1'
PEAK_SETTINGS=MappingProxyType({
    'jump_jack':ImprovementSettings(signal='baseline',smooth_s=0.,prominence_fraction=.25,width_s=1/15,min_gap_s=.25),
    'squat':ImprovementSettings(signal='alternate',smooth_s=.6,prominence_fraction=.35,width_s=1/15,min_gap_s=.35),
})
RAW_POINTS=['nose','shoulder_L','shoulder_R','elbow_L','elbow_R','wrist_L','wrist_R','hip_L','hip_R']
RAW_COLUMNS=['frame','time_s']+[f'{point}_{axis}' for point in RAW_POINTS for axis in ('x','y','vis')]


def frozen_configuration():
    frozen=json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
    if frozen['selection_split']!='train' or frozen['settings']['signal']!='median':
        raise ValueError('Expected frozen training-selected coordinate median config')
    for name,digest in frozen['source_hashes'].items():
        data=Path(__file__).with_name(name).read_bytes()
        actual=sha256(Path(__file__).with_name(name))
        if actual!=digest and hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()!=digest and hashlib.sha256(data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()!=digest:
            raise ValueError(f'Frozen counter implementation changed: {name}')
    return frozen


class ExperimentalCounter:
    """One analysis session; lazy IMAGE-mode pose handle, closed on every exit.

    Only raw coordinates are accepted from callers. Derived channels are always
    recomputed inside the requested predicted window including end padding.
    No canonical data or hidden experiment-result cache is read or written.
    """

    def __init__(self,video_path,fps,n_frames,coordinate_features=None):
        self.video_path=Path(video_path); self.fps=float(fps); self.n_frames=int(n_frames)
        if not np.isfinite(self.fps) or self.fps<=0 or self.n_frames<=0: raise ValueError('Invalid video timing')
        self.frozen=frozen_configuration()
        self.pushup_settings=CoordinatePeakSettings(**self.frozen['settings'])
        self.coordinates=coordinate_features
        self.landmarker=None
        self.model_path=ROOT/'models/pose_landmarker_full.task'
        self.provenance=dict(id=COUNTER_ID,pushup_freeze_id=self.frozen['freeze_id'],freeze_sha256=sha256(FREEZE_PATH),
            settings={**{ex:asdict(cfg) for ex,cfg in PEAK_SETTINGS.items()},'push_up':asdict(self.pushup_settings)},
            coordinate_source='not_used',offline=True)

    def __enter__(self): return self

    def __exit__(self,*args):
        if self.landmarker is not None:
            self.landmarker.close(); self.landmarker=None

    def raw_window(self,start,end,padding):
        first=int(np.ceil(start*self.fps-1e-8)); stop=min(self.n_frames,int(np.ceil((end+padding)*self.fps-1e-8)))
        expected=list(range(first+first%2,stop,2))
        if len(expected)<8: return None
        if self.coordinates is not None:
            missing=[c for c in RAW_COLUMNS if c not in self.coordinates]
            if missing: raise ValueError(f'Push-up coordinates require raw pixel landmarks, not derived signals: {missing}')
            frame=self.coordinates[self.coordinates.frame.isin(expected)].copy()
            frame=frame.sort_values('frame').reset_index(drop=True)
            self.provenance['coordinate_source']='provided_raw_coordinates; extraction provenance not verified'
        else:
            if not self.video_path.is_file():
                raise FileNotFoundError('Default push-up voting requires the registered video or analyze(..., coordinate_features=raw_frame); no angle-only fallback: '+str(self.video_path))
            if self.landmarker is None:
                if not self.model_path.is_file() or sha256(self.model_path)!=self.frozen['pose_model_sha256']:
                    raise ValueError('Default coordinate extraction requires the frozen pose_landmarker_full.task model')
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision
                self.landmarker=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),running_mode=vision.RunningMode.IMAGE,num_poses=2,output_segmentation_masks=False))
                self.provenance.update(coordinate_source='registered_video_IMAGE_stride2_largest_visible_pose',
                    video_sha256=sha256(self.video_path),pose_model_sha256=self.frozen['pose_model_sha256'])
            from .extract_pushup_diagnostic import extract_pass
            frame,_,_=extract_pass(self.landmarker,self.video_path,first,stop,self.fps)
        if frame.frame.tolist()!=expected or not np.allclose(frame.time_s,frame.frame/self.fps):
            raise ValueError('Incomplete or misaligned raw coordinates; missing poses must retain their timestamped NaN rows')
        frame[RAW_COLUMNS]=frame[RAW_COLUMNS].apply(pd.to_numeric,errors='raise')
        return frame

    def count(self,frame,exercise,start,end,end_pad_s=.3):
        if not np.isfinite([start,end,end_pad_s]).all() or start<0 or end<=start or end_pad_s<0:
            raise ValueError('Invalid counting interval or padding')
        if exercise in PEAK_SETTINGS:
            cfg=PEAK_SETTINGS[exercise]
            result=count_improved_peaks(frame,exercise,start,end+end_pad_s,settings=cfg)
            return result|dict(method='unguided_peaks',counter_settings=asdict(cfg),
                signal='arm_raise_L+arm_raise_R' if exercise=='jump_jack' else 'confidence_weighted_hip_angles')
        if exercise!='push_up': raise ValueError('Unsupported exercise: '+exercise)
        raw=self.raw_window(start,end,end_pad_s)
        if raw is None:
            result=dict(reps=0,rep_times=[],flag='too_short',selected_signal='')
        else:
            from .extract_coordinate_training import add_signals
            derived,_=add_signals(raw.copy())
            result=count_coordinate_peaks(derived,self.pushup_settings)
            result['flag']='unusable' if result['selected_signal']=='unusable' else ''
        return result|dict(method='coordinate_median_peaks',signal=result['selected_signal'],counter_settings=asdict(self.pushup_settings))
