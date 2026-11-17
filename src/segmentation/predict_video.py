"""Predict one v2 video with the same frozen bundle used by integration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from config import FEATURES_DIR, SYNTH_DIR, WINDOWS_CSV
from workflow import load_bundle, bundle_smoothing, infer_frame, verify_bundle_training
from v2_protocol import catalog, feature_frame
from metrics import all_metrics, gt_from_json


def predict(video_id,features_dir=FEATURES_DIR,synth_dir=SYNTH_DIR,model_path=None,
            params_path=None,show_windows=False,show_gt=False,windows_path=WINDOWS_CSV):
    meta=catalog(synth_dir)[video_id]
    bundle,path=load_bundle(model_path)
    verify_bundle_training(bundle,windows_path,features_dir)
    bundle_smoothing(bundle,path,params_path)
    frame=feature_frame(features_dir/f'{video_id}_angles.csv',meta)
    windows,raw,segments,end=infer_frame(bundle,frame,meta)
    if show_windows:print(windows.to_string(index=False))
    result=dict(run_id='v2',experiment_id=bundle['experiment_id'],video_id=video_id,
                split=meta['split'],fold=meta['fold'],duration_s=end,segments=segments)
    if show_gt:
        truth,_=gt_from_json(meta)
        result['comparison_scope']='final_test' if meta['split']=='test' else 'in_sample_train_demo'
        result['metrics']=all_metrics(truth,segments,end,list(bundle['classes']))
    print(json.dumps(result,indent=2))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video_id')
    parser.add_argument('--model',type=Path,default=None)
    parser.add_argument('--params',type=Path,default=None)
    parser.add_argument('--features-dir',type=Path,default=FEATURES_DIR)
    parser.add_argument('--synth-dir',type=Path,default=SYNTH_DIR)
    parser.add_argument('--windows',type=Path,default=WINDOWS_CSV)
    parser.add_argument('--show-windows',action='store_true')
    parser.add_argument('--show-gt',action='store_true',help='Explicit comparison; train demos are in-sample')
    args=parser.parse_args(argv)
    predict(args.video_id,args.features_dir,args.synth_dir,args.model,args.params,args.show_windows,args.show_gt,args.windows)


if __name__=="__main__":main()
