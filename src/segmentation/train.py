"""Source-clip CV training, or explicit all-train refit of a frozen selection."""
from __future__ import annotations
import argparse
from pathlib import Path
from config import FEATURES_DIR, OUTPUT_DIR, WINDOWS_CSV, MIN_PURITY_TRAIN, SEED
from classifiers import _build_classifiers
from workflow import (build_feature_matrix, read_training, run_selection, refit_selection)
from cv import get_cv_split

AVAILABLE_MODELS=list(_build_classifiers(SEED))
DEFAULT_MODEL="rf"


def preflight(windows_path=WINDOWS_CSV,features_dir=FEATURES_DIR):
    windows,videos=read_training(windows_path,features_dir)
    X,channels=build_feature_matrix(windows,features_dir,videos)
    print(f"V2 train-only preflight: {len(windows)} windows, {windows.video_id.nunique()} videos, {len(X.columns)} features")
    for fold in range(5):
        train,validation=get_cv_split(windows,fold,videos)
        print(f"fold={fold}: train={len(train)}, validation={len(validation)}, source overlap=0, test windows=0")
    print("PASS: source-fold CV, feature schema and row slices; no model fitted; no test angle files read")
    return X,channels


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=AVAILABLE_MODELS,default=DEFAULT_MODEL)
    parser.add_argument("--windows",type=Path,default=WINDOWS_CSV)
    parser.add_argument("--features-dir",type=Path,default=FEATURES_DIR)
    parser.add_argument("--out-dir",type=Path,default=OUTPUT_DIR)
    parser.add_argument("--experiment-id",default=None)
    parser.add_argument("--min-purity-train",type=float,default=MIN_PURITY_TRAIN)
    parser.add_argument("--tau-grid",type=float,nargs="+",default=[.4,.5,.6,.7])
    parser.add_argument("--min-len-grid",type=int,nargs="+",default=[1,2,3,4])
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--selection",type=Path,help="Refit this existing train-only CV selection; do not rerun CV")
    parser.add_argument("--activate",action="store_true",help="Activate the refitted bundle for integration")
    parser.add_argument("--check-only",action="store_true",help="Read-only train-data/CV preflight; no fitting")
    args=parser.parse_args(argv)
    if args.check_only:
        if args.selection or args.activate:parser.error("--check-only cannot refit or activate")
        preflight(args.windows,args.features_dir)
    elif args.selection:
        path=refit_selection(args.selection,args.windows,args.features_dir,args.activate,args.out_dir)
        print(f"Frozen all-train model: {path}")
    else:
        if args.activate:parser.error("--activate requires --selection (all-train refit)")
        run_selection(args.windows,args.features_dir,args.out_dir,[args.model],[args.min_purity_train],
                      args.tau_grid,args.min_len_grid,args.seed,args.experiment_id)


if __name__=="__main__":main()
