"""Rank classifiers/purity/smoothing on five train source folds only."""
from __future__ import annotations
import argparse
from pathlib import Path
from classifiers import _build_classifiers
from config import FEATURES_DIR, OUTPUT_DIR, WINDOWS_CSV, SEED
from workflow import build_feature_matrix, run_selection


def run_sweep(**kwargs):
    return run_selection(**kwargs)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifiers",nargs="+",choices=list(_build_classifiers(SEED)),default=None)
    parser.add_argument("--quick",action="store_true",help="RF and LR only; five full source folds")
    parser.add_argument("--min-purity-grid",nargs="+",type=float,default=[.8])
    parser.add_argument("--tau-grid",nargs="+",type=float,default=[.4,.5,.6,.7])
    parser.add_argument("--min-len-grid",nargs="+",type=int,default=[1,2,3,4])
    parser.add_argument("--windows",type=Path,default=WINDOWS_CSV)
    parser.add_argument("--features-dir",type=Path,default=FEATURES_DIR)
    parser.add_argument("--out-dir",type=Path,default=OUTPUT_DIR)
    parser.add_argument("--experiment-id",default=None)
    parser.add_argument("--seed",type=int,default=SEED)
    args=parser.parse_args(argv)
    keys=args.classifiers or (["rf","lr"] if args.quick else list(_build_classifiers(args.seed)))
    run_selection(args.windows,args.features_dir,args.out_dir,keys,args.min_purity_grid,
                  args.tau_grid,args.min_len_grid,args.seed,args.experiment_id)


if __name__=="__main__":main()
