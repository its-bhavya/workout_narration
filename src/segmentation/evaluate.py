"""Report train OOF metrics or explicitly score a frozen model on final test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from config import FEATURES_DIR, WINDOWS_CSV
from cv import validate_windows
from workflow import evaluate_final, read_training, score_predictions, signature
from v2_protocol import active_path, require_v2, sha256


def evaluate_cv(selection_path,windows_path=WINDOWS_CSV,features_dir=FEATURES_DIR):
    path=active_path(selection_path)
    selected=require_v2(json.loads(path.read_text()))
    windows,videos=read_training(windows_path,features_dir)
    if signature(videos,windows,features_dir)!=selected['train_signature']:
        raise ValueError("CV inputs changed since selection")
    predictions_path=path.parent/'oof_predictions.csv'
    if sha256(predictions_path)!=selected['oof_sha256']:raise ValueError('OOF predictions changed since selection')
    predictions=pd.read_csv(predictions_path)
    validate_windows(predictions,videos)
    if not predictions.split.eq('train').all():raise ValueError("OOF metrics cannot include test")
    if not predictions.cv_fold.eq(predictions.fold).all():raise ValueError("Wrong held-out OOF fold")
    if predictions.duplicated(['video_id','win_idx']).any() or len(predictions)!=len(windows):
        raise ValueError('Incomplete/duplicated OOF predictions')
    best=selected['best']
    _,folds=score_predictions(predictions,videos,best['tau'],best['min_len'])
    print("Selection-CV fold metrics (not an unbiased nested-CV estimate):")
    print(folds.to_string())
    return folds


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--selection",type=Path,help="Report the selected train-only OOF predictions")
    action.add_argument("--final-test",action="store_true",help="Evaluate a frozen all-train model; never tune on test")
    parser.add_argument("--model",type=Path)
    parser.add_argument("--windows",type=Path,default=WINDOWS_CSV)
    parser.add_argument("--features-dir",type=Path,default=FEATURES_DIR)
    parser.add_argument("--out-dir",type=Path,default=None)
    args=parser.parse_args(argv)
    if args.final_test:
        if args.model is None:parser.error("--final-test requires --model")
        report=evaluate_final(args.model,args.windows,args.features_dir,args.out_dir)
        print(json.dumps(report,indent=2))
    else:
        if args.model or args.out_dir:parser.error("CV reporting uses --selection only")
        evaluate_cv(args.selection,args.windows,args.features_dir)


if __name__=="__main__":main()
