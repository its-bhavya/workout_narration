"""hparam_sweep.py -- classifier + smoothing hyperparameter sweep.

Loads the pre-built windows.csv and angle CSVs, builds the feature matrix
ONCE (fixed windowing / NaN-fill, same as train.py), then sweeps:

    * classifier  : RF, ExtraTrees, GradientBoosting, HistGBT,
                    LogisticRegression, SVM (RBF), KNN, MLP
    * min_purity_train : threshold for eligible training windows
    * smoothing   : tau × min_len grid-searched on val per experiment

Nothing upstream of the feature matrix (windowing, NaN fill, pose extraction)
is touched.  Results are ranked by test Edit Score and written to:

    outputs/segmentation/sweep_results.csv

Usage (run from repo root)::

    # Full sweep (all 8 classifiers × 2 purity thresholds × 4×4 smoothing)
    python src/segmentation/hparam_sweep.py

    # Quick sanity check: RF + LR only
    python src/segmentation/hparam_sweep.py --quick

    # Specific classifiers
    python src/segmentation/hparam_sweep.py --classifiers rf extra_trees hist_gbt

    # Wider smoothing search
    python src/segmentation/hparam_sweep.py --tau-grid 0.3 0.4 0.5 0.6 0.7 --min-len-grid 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in [str(_HERE), str(_ROOT / "src" / "data_pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (  # noqa: E402
    CLASSES,
    FEATURES_DIR,
    MIN_PURITY_TRAIN,
    OUTPUT_DIR,
    SEED,
    SYNTH_DIR,
    WINDOWS_CSV,
)
from features import feature_names, fill_video_nans, window_features  # noqa: E402
from config import feature_columns as resolve_feature_columns  # noqa: E402
from metrics import all_metrics, gt_from_json  # noqa: E402
from smooth import smooth, to_segments  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*max_iter.*")
warnings.filterwarnings("ignore", message=".*Convergence.*")
warnings.filterwarnings("ignore", message=".*lbfgs.*")

# ---------------------------------------------------------------------------
# Sweep grid definitions
# ---------------------------------------------------------------------------
MIN_PURITY_GRID = [0.7, 0.8]        # training window purity thresholds
TAU_GRID = [0.4, 0.5, 0.6, 0.7]    # smoothing confidence threshold
MIN_LEN_GRID = [1, 2, 3, 4]        # smoothing min run-length (windows)

QUICK_MIN_PURITY_GRID = [0.8]


# ---------------------------------------------------------------------------
# Classifier catalogue
# ---------------------------------------------------------------------------

def _build_classifiers(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    return {
        "rf": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
        "gbt": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            subsample=0.8, random_state=seed,
        ),
        "hist_gbt": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=5,
            class_weight="balanced", random_state=seed,
        ),
        "lr": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            solver="saga", random_state=seed, n_jobs=-1,
        ),
        "svm": SVC(
            kernel="rbf", class_weight="balanced", probability=True,
            random_state=seed,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=7, metric="euclidean", n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            random_state=seed,
        ),
    }

QUICK_CLASSIFIERS = {"rf", "lr"}


# ---------------------------------------------------------------------------
# Feature matrix (built once, shared across all experiments)
# ---------------------------------------------------------------------------

def build_feature_matrix(
    windows: pd.DataFrame,
    features_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the feature matrix from windows.csv + angle CSVs.

    Identical to train.py's build_feature_matrix — fixed windowing and
    NaN fill settings (MAX_FILL_GAP_S = 0.5 s from features.py defaults).
    Each video's angle CSV is read and filled once, then cached.
    """
    video_cache: dict[str, pd.DataFrame] = {}

    def get_filled(video_id: str) -> pd.DataFrame:
        if video_id not in video_cache:
            path = features_dir / f"{video_id}_angles.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing angle CSV: {path}")
            video_cache[video_id] = fill_video_nans(pd.read_csv(path))
        return video_cache[video_id]

    first_vid = str(windows.iloc[0]["video_id"])
    channels = resolve_feature_columns(get_filled(first_vid).columns)
    canon_cols = feature_names(channels)

    feat_rows: list[dict] = []
    for _, w in windows.iterrows():
        vdf = get_filled(str(w["video_id"]))
        rows = vdf.iloc[int(w["row_start"]):int(w["row_end"])]
        feat_rows.append(window_features(rows))

    X = pd.DataFrame(feat_rows).reindex(columns=canon_cols)
    return X, channels


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def _load_gt_json(synth_dir: Path, split: str, video_id: str) -> dict:
    p = synth_dir / split / f"{video_id}.json"
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _smoothed_segments(
    wdf: pd.DataFrame, video_end_s: float, tau: float, min_len: int
) -> list[dict]:
    labels = smooth(
        list(wdf["pred_label"]),
        list(wdf["max_prob"]),
        tau=tau,
        min_len=min_len,
    )
    return to_segments(labels, list(wdf["t_center"]), video_end_s)


def grid_search_smoothing(
    val_pred_df: pd.DataFrame,
    synth_dir: Path,
    tau_grid: list[float],
    min_len_grid: list[int],
) -> tuple[float, int, float, float]:
    """Exhaustive tau × min_len search on val. Returns (tau, min_len, edit, acc)."""
    val_preds = {
        vid: grp.sort_values("win_idx").reset_index(drop=True)
        for vid, grp in val_pred_df.groupby("video_id")
    }
    best: tuple[float, float, float, int] | None = None
    for tau in tau_grid:
        for ml in min_len_grid:
            edits, accs = [], []
            for vid_id, wdf in val_preds.items():
                meta = _load_gt_json(synth_dir, "val", vid_id)
                gt_segs, video_end_s = gt_from_json(meta)
                smo = _smoothed_segments(wdf, video_end_s, tau, ml)
                m = all_metrics(gt_segs, smo, video_end_s, CLASSES)
                edits.append(m["edit_score"])
                accs.append(m["frame_accuracy"])
            me = float(np.mean(edits))
            ma = float(np.mean(accs))
            if best is None or (me, ma) > (best[0], best[1]):
                best = (me, ma, tau, ml)
    assert best is not None
    return best[2], best[3], best[0], best[1]


def eval_test(
    test_pred_df: pd.DataFrame,
    synth_dir: Path,
    tau: float,
    min_len: int,
) -> dict[str, float]:
    """Mean metrics across all test videos with given smoothing params."""
    test_preds = {
        vid: grp.sort_values("win_idx").reset_index(drop=True)
        for vid, grp in test_pred_df.groupby("video_id")
    }
    rows: list[dict] = []
    for vid_id, wdf in sorted(test_preds.items()):
        meta = _load_gt_json(synth_dir, "test", vid_id)
        gt_segs, video_end_s = gt_from_json(meta)
        smo = _smoothed_segments(wdf, video_end_s, tau, min_len)
        rows.append(all_metrics(gt_segs, smo, video_end_s, CLASSES))
    df_m = pd.DataFrame(rows)
    return {col: float(df_m[col].mean()) for col in df_m.columns}


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(
    clf_name: str,
    clf,
    windows: pd.DataFrame,
    X_filled: pd.DataFrame,    # NaN-filled, train-median-imputed feature matrix
    min_purity_train: float,
    synth_dir: Path,
    tau_grid: list[float],
    min_len_grid: list[int],
) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    y = windows["label"].astype(str)
    purity = windows["purity"].astype(float).to_numpy()
    is_train = (windows["split"] == "train").to_numpy()
    is_val = (windows["split"] == "val").to_numpy()
    is_test = (windows["split"] == "test").to_numpy()

    train_elig = is_train & (purity >= min_purity_train)
    if int(train_elig.sum()) == 0:
        log.warning("No eligible train windows for purity=%.2f; skipping.", min_purity_train)
        return {}

    y_train = y.loc[train_elig]
    X_tr = X_filled.loc[train_elig].to_numpy()

    # Train.
    clf.fit(X_tr, y_train.to_numpy())

    # Val window-level metrics.
    X_val = X_filled.loc[is_val].to_numpy()
    y_val = y.loc[is_val].to_numpy()
    y_pred_val = clf.predict(X_val)
    val_acc = float(accuracy_score(y_val, y_pred_val))
    val_macro_f1 = float(f1_score(y_val, y_pred_val, average="macro",
                                   labels=CLASSES, zero_division=0))
    val_f1_per = f1_score(y_val, y_pred_val, average=None,
                          labels=CLASSES, zero_division=0)
    val_f1_cls = dict(zip(CLASSES, val_f1_per.tolist()))

    # Per-window prediction DataFrames for smoothing + test eval.
    def _pred_df(mask: np.ndarray) -> pd.DataFrame:
        proba = clf.predict_proba(X_filled.loc[mask].to_numpy())
        pred = clf.classes_[proba.argmax(axis=1)]
        w = windows.loc[mask].reset_index(drop=True).copy()
        w["pred_label"] = pred
        for i, cls in enumerate(clf.classes_):
            w[f"prob_{cls}"] = proba[:, i]
        w["max_prob"] = proba.max(axis=1)
        return w

    val_pred_df = _pred_df(is_val)
    test_pred_df = _pred_df(is_test) if int(is_test.sum()) > 0 else pd.DataFrame()

    # Grid-search smoothing on val.
    tau, min_len, val_edit, val_sacc = grid_search_smoothing(
        val_pred_df, synth_dir, tau_grid, min_len_grid
    )

    # Test evaluation.
    test_m: dict[str, float] = {}
    if not test_pred_df.empty:
        test_m = eval_test(test_pred_df, synth_dir, tau, min_len)

    result: dict = {
        "classifier": clf_name,
        "min_purity_train": min_purity_train,
        # smoothing
        "best_tau": tau,
        "best_min_len": min_len,
        # val smoothed
        "val_smooth_edit_score": round(val_edit, 4),
        "val_smooth_frame_acc": round(val_sacc, 6),
        # val window-level
        "val_win_accuracy": round(val_acc, 6),
        "val_win_macro_f1": round(val_macro_f1, 6),
        **{f"val_win_f1_{c}": round(val_f1_cls.get(c, 0.0), 6) for c in CLASSES},
        # counts
        "n_train_elig": int(train_elig.sum()),
        "n_val_wins": int(is_val.sum()),
        "n_test_wins": int(is_test.sum()),
    }
    for k, v in test_m.items():
        result[f"test_{k}"] = round(float(v), 6)

    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    windows_csv: Path,
    features_dir: Path,
    synth_dir: Path,
    out_dir: Path,
    classifier_names: list[str],
    min_purity_grid: list[float],
    tau_grid: list[float],
    min_len_grid: list[int],
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_clfs = _build_classifiers(seed)
    selected = {k: all_clfs[k] for k in classifier_names if k in all_clfs}
    if not selected:
        raise SystemExit(
            f"No valid classifiers in {classifier_names}. "
            f"Available: {list(all_clfs.keys())}"
        )

    log.info("Classifiers : %s", list(selected.keys()))
    log.info("Purity grid : %s", min_purity_grid)
    log.info("Tau grid    : %s", tau_grid)
    log.info("Min-len grid: %s", min_len_grid)

    # ---- Load windows -------------------------------------------------------
    if not windows_csv.exists():
        raise SystemExit(
            f"windows.csv not found: {windows_csv}\n"
            "Run `python src/segmentation/make_windows.py` first."
        )
    windows = pd.read_csv(windows_csv)
    log.info("Loaded %d windows from %s", len(windows), windows_csv)

    # ---- Build feature matrix ONCE -----------------------------------------
    log.info("Building feature matrix (this may take ~30s) ...")
    t0 = time.time()
    X, channels = build_feature_matrix(windows, features_dir)
    log.info("Feature matrix: %d × %d  (%.0fs)", len(X), X.shape[1], time.time() - t0)

    # NaN-fill medians fit on TRAIN eligible (highest purity threshold to be
    # conservative; the fill is the same for all purity variants since only
    # the eligible mask changes, not the raw feature values).
    max_purity = max(min_purity_grid)
    is_train = (windows["split"] == "train").to_numpy()
    purity_arr = windows["purity"].astype(float).to_numpy()
    train_elig_for_fill = is_train & (purity_arr >= max_purity)
    fill_values = X.loc[train_elig_for_fill].median(axis=0).fillna(0.0)
    X_filled = X.fillna(fill_values)
    log.info("NaN imputation done (medians from %d train windows, purity≥%.1f)",
             int(train_elig_for_fill.sum()), max_purity)

    # ---- Sweep --------------------------------------------------------------
    n_total = len(selected) * len(min_purity_grid)
    log.info("Total experiments: %d", n_total)

    results: list[dict] = []
    sweep_t0 = time.time()

    for exp_i, (clf_name, clf_proto) in enumerate(selected.items(), 1):
        for purity in min_purity_grid:
            exp_label = f"[{(exp_i-1)*len(min_purity_grid) + min_purity_grid.index(purity)+1}/{n_total}]"
            log.info("%s  %s  purity=%.2f", exp_label, clf_name, purity)
            t1 = time.time()
            try:
                result = run_experiment(
                    clf_name=clf_name,
                    clf=copy.deepcopy(clf_proto),
                    windows=windows,
                    X_filled=X_filled,
                    min_purity_train=purity,
                    synth_dir=synth_dir,
                    tau_grid=tau_grid,
                    min_len_grid=min_len_grid,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("  FAILED: %s", exc)
                result = {}

            elapsed = time.time() - t1
            if result:
                result["elapsed_s"] = round(elapsed, 1)
                results.append(result)
                log.info(
                    "  val_edit=%.1f  val_acc=%.4f  "
                    "test_edit=%.1f  test_acc=%.4f  tau=%.1f  min_len=%d  (%.0fs)",
                    result.get("val_smooth_edit_score", float("nan")),
                    result.get("val_smooth_frame_acc", float("nan")),
                    result.get("test_edit_score", float("nan")),
                    result.get("test_frame_accuracy", float("nan")),
                    result.get("best_tau", float("nan")),
                    result.get("best_min_len", 0),
                    elapsed,
                )
            else:
                log.warning("  No result (skipped).")

    total_elapsed = time.time() - sweep_t0
    log.info("Sweep finished in %.0fs", total_elapsed)

    if not results:
        log.error("No results — nothing to save.")
        return

    df = pd.DataFrame(results)

    # Sort by test_edit_score desc, then val_smooth_edit_score desc.
    sort_cols = [c for c in ["test_edit_score", "val_smooth_edit_score"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    # Preferred column order.
    leading = [
        "rank", "classifier", "min_purity_train",
        "best_tau", "best_min_len",
        # test (primary target)
        "test_edit_score", "test_frame_accuracy",
        "test_f1_at_10", "test_f1_at_25", "test_f1_at_50",
        "test_boundary_mean_s", "test_boundary_median_s",
        "test_frame_f1_squat", "test_frame_f1_push_up",
        "test_frame_f1_jump_jack", "test_frame_f1_transition",
        "test_n_pred", "test_n_gt",
        # val smoothed
        "val_smooth_edit_score", "val_smooth_frame_acc",
        # val window-level
        "val_win_accuracy", "val_win_macro_f1",
        "val_win_f1_squat", "val_win_f1_push_up",
        "val_win_f1_jump_jack", "val_win_f1_transition",
        # counts / timing
        "n_train_elig", "n_val_wins", "n_test_wins", "elapsed_s",
    ]
    extra = [c for c in df.columns if c not in leading]
    df = df[[c for c in leading if c in df.columns] + extra]

    out_path = out_dir / "sweep_results.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved %d results -> %s", len(df), out_path)

    # Print summary table.
    print("\n" + "=" * 100)
    print("SWEEP RESULTS — ranked by test Edit Score")
    print("=" * 100)
    show_cols = [
        "rank", "classifier", "min_purity_train",
        "best_tau", "best_min_len",
        "test_edit_score", "test_frame_accuracy",
        "test_f1_at_25", "test_f1_at_50",
        "val_smooth_edit_score",
    ]
    disp = df[[c for c in show_cols if c in df.columns]]
    print(disp.to_string(index=False))
    print(f"\nFull results: {out_path}")
    print(f"Sweep time  : {total_elapsed:.0f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    all_clf_names = list(_build_classifiers(SEED).keys())

    parser = argparse.ArgumentParser(
        description="Sweep classifiers and smoothing params over the pre-built feature matrix."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="RF + LR only, single purity threshold — fast sanity check.",
    )
    parser.add_argument(
        "--classifiers", nargs="+", default=None, metavar="CLF",
        help=f"Classifiers to run. Choices: {all_clf_names}. Default: all.",
    )
    parser.add_argument(
        "--min-purity-grid", nargs="+", type=float, default=None, metavar="P",
        help="Min purity thresholds for training windows. Default: 0.7 0.8",
    )
    parser.add_argument(
        "--tau-grid", nargs="+", type=float, default=None, metavar="T",
        help="Smoothing confidence thresholds. Default: 0.4 0.5 0.6 0.7",
    )
    parser.add_argument(
        "--min-len-grid", nargs="+", type=int, default=None, metavar="L",
        help="Smoothing min run-length values. Default: 1 2 3 4",
    )
    parser.add_argument(
        "--windows", type=Path, default=WINDOWS_CSV,
        help="Path to windows.csv.",
    )
    parser.add_argument(
        "--features-dir", type=Path, default=FEATURES_DIR,
        help="Directory of *_angles.csv files.",
    )
    parser.add_argument(
        "--synth-dir", type=Path, default=SYNTH_DIR,
        help="Root of synthetic video/JSON tree.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory (default: outputs/segmentation).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.quick:
        clf_names = sorted(QUICK_CLASSIFIERS)
        purity_grid = QUICK_MIN_PURITY_GRID
        log.info("Quick mode: %s", clf_names)
    else:
        clf_names = args.classifiers or all_clf_names
        purity_grid = args.min_purity_grid or MIN_PURITY_GRID

    tau_grid = args.tau_grid or TAU_GRID
    min_len_grid = args.min_len_grid or MIN_LEN_GRID

    run_sweep(
        windows_csv=args.windows,
        features_dir=args.features_dir,
        synth_dir=args.synth_dir,
        out_dir=args.out_dir,
        classifier_names=clf_names,
        min_purity_grid=purity_grid,
        tau_grid=tau_grid,
        min_len_grid=min_len_grid,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
