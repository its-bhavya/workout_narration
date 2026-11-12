"""train.py -- train the sliding-window classifier and evaluate on val.

End-to-end pipeline::

    1. Read the window index (outputs/segmentation/windows.csv).
    2. Per video: read its angle CSV and fill short NaN gaps (features.fill_video_nans).
    3. Per window: slice rows and compute window_features -> one feature row.
    4. Build the feature matrix; split by the window's ``split`` (never tuned on test).
    5. Fit per-feature NaN-fill medians on TRAIN windows only (eligible subset).
    6. Train the selected classifier on train windows with purity >= MIN_PURITY_TRAIN.
    7. Evaluate on VAL: accuracy, macro-F1, per-class F1, confusion matrix,
       majority-class baseline, top-10 feature importances (to spot leakage).

Available classifiers (``--model`` argument):
    rf           RandomForestClassifier       (default)
    extra_trees  ExtraTreesClassifier
    gbt          GradientBoostingClassifier
    hist_gbt     HistGradientBoostingClassifier
    lr           LogisticRegression
    svm          SVC (RBF kernel)
    knn          KNeighborsClassifier
    mlp          MLPClassifier

Versioning: every run is archived under ``outputs/segmentation/runs/<run_id>/``
with the full snapshot (report, model bundle, val+test predictions, and a
machine-readable ``run_meta.json``), and one row is appended to the master
``outputs/segmentation/runs_log.csv`` so runs can be compared when you change
models, data, or hyperparameters. The top-level files are overwritten as the
"latest" copy so existing references keep working.

Run from repo root::

    python src/segmentation/train.py [--model hist_gbt] [--tag "description"]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from config import (
    CLASSES,
    FEATURES_DIR,
    MIN_PURITY_TRAIN,
    MIN_ROWS_FRAC,
    OUTPUT_DIR,
    SEED,
    STRIDE_S,
    WINDOW_S,
    WINDOWS_CSV,
)
from config import feature_columns
from features import MAX_FILL_GAP_S, feature_names, fill_video_nans, window_features
from hparam_sweep import _build_classifiers  # reuse the shared classifier catalogue

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# All classifier keys available via --model (mirrors hparam_sweep.py catalogue).
AVAILABLE_MODELS = ["rf", "extra_trees", "gbt", "hist_gbt", "lr", "svm", "knn", "mlp"]
DEFAULT_MODEL = "rf"


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    windows: pd.DataFrame,
    features_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (X, channels) where X is the per-window feature matrix.

    Each video's angle CSV is read once, NaN-filled per video, and cached.
    Windows are sliced with ``df.iloc[row_start:row_end]`` (the row indices in
    ``windows.csv`` refer to the time-sorted, reset-index frame, which
    ``fill_video_nans`` reproduces).
    """
    video_cache: dict[str, pd.DataFrame] = {}

    def get_filled(video_id: str) -> pd.DataFrame:
        if video_id not in video_cache:
            path = features_dir / f"{video_id}_angles.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing angle CSV: {path}")
            video_cache[video_id] = fill_video_nans(pd.read_csv(path))
        return video_cache[video_id]

    # Resolve the canonical channel set + feature column order from the first
    # video so every window produces a consistent column set.
    first_vid = str(windows.iloc[0]["video_id"])
    channels = feature_columns(get_filled(first_vid).columns)
    canon_cols = feature_names(channels)

    feat_rows: list[dict] = []
    for _, w in windows.iterrows():
        vdf = get_filled(str(w["video_id"]))
        rows = vdf.iloc[int(w["row_start"]):int(w["row_end"])]
        feat_rows.append(window_features(rows))

    X = pd.DataFrame(feat_rows).reindex(columns=canon_cols)
    return X, channels


# ---------------------------------------------------------------------------
# Run id
# ---------------------------------------------------------------------------

def make_run_id(runs_root: Path) -> str:
    """A sortable timestamp run id; bumps a suffix if the folder already exists."""
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = base
    n = 2
    while (runs_root / run_id).exists():
        run_id = f"{base}_{n}"
        n += 1
    return run_id


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_val_report(
    metrics: dict,
    cm: np.ndarray,
    classes: list[str],
    importances: pd.Series | None,
    run_id: str,
    tag: str,
    model_name: str = "",
) -> str:
    lines: list[str] = []
    lines.append("=== Val window report ===")
    lines.append(f"run_id            : {run_id}")
    if model_name:
        lines.append(f"model             : {model_name}")
    if tag:
        lines.append(f"tag               : {tag}")
    lines.append("")
    lines.append(f"n_val_windows      : {metrics['n_val_windows']}")
    lines.append(f"accuracy           : {metrics['accuracy']:.4f}")
    lines.append(
        f"majority baseline  : {metrics['majority_baseline']:.4f}  "
        f"(predict '{metrics['majority_label']}' for all)"
    )
    lines.append(f"macro-F1           : {metrics['macro_f1']:.4f}")
    lines.append("")
    lines.append("per-class F1:")
    for c in classes:
        lines.append(f"  {c:11s} {metrics['f1_per_class'][c]:.4f}")
    lines.append("")
    lines.append("confusion matrix (rows=true, cols=pred):")
    header = "  " + " ".join(f"{c:>10s}" for c in classes)
    lines.append(header)
    for i, c in enumerate(classes):
        row = "  " + f"{c:>10s} " + " ".join(
            f"{int(cm[i, j]):>10d}" for j in range(len(classes))
        )
        lines.append(row)
    lines.append("")
    if importances is not None:
        lines.append("top-10 feature importances:")
        for name, imp in importances.head(10).items():
            lines.append(f"  {name:32s} {imp:.4f}")
    else:
        lines.append("(feature importances not available for this model type)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the window classifier and evaluate on val."
    )
    parser.add_argument("--windows", type=Path, default=WINDOWS_CSV,
                        help="Path to windows.csv (default: outputs/segmentation/windows.csv)")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR,
                        help="Directory of *_angles.csv (default: data/features)")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: outputs/segmentation)")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help=(
            f"Classifier to train (default: {DEFAULT_MODEL}). "
            "Options: rf, extra_trees, gbt, hist_gbt, lr, svm, knn, mlp"
        ),
    )
    parser.add_argument("--tag", type=str, default="",
                        help="Free-text label for this run, recorded in the log "
                             "(e.g. 'rf200_baseline', 'no_torso_tilt').")
    args = parser.parse_args()

    if not args.windows.exists():
        raise SystemExit(
            f"windows.csv not found: {args.windows}\n"
            "Run `python src/segmentation/make_windows.py` first."
        )

    out_dir = args.out_dir
    runs_root = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.windows)
    log.info("Loaded %d windows from %s", len(windows), args.windows)

    # ---- Select classifier -----------------------------------------------
    all_clfs = _build_classifiers(SEED)
    clf = all_clfs[args.model]
    clf_class_name = type(clf).__name__
    log.info("Classifier: %s (%s)", args.model, clf_class_name)

    # ---- Build feature matrix --------------------------------------------
    X, channels = build_feature_matrix(windows, args.features_dir)
    canon_cols = list(X.columns)
    log.info("Feature matrix: %d windows x %d features", len(X), X.shape[1])

    y = windows["label"].astype(str)
    bad = set(y.unique()) - set(CLASSES)
    assert not bad, f"unexpected labels (label leakage risk): {bad}"

    purity = windows["purity"].astype(float)
    is_train = (windows["split"] == "train").to_numpy()
    is_val = (windows["split"] == "val").to_numpy()
    is_test = (windows["split"] == "test").to_numpy()

    train_elig = is_train & (purity.to_numpy() >= MIN_PURITY_TRAIN)
    log.info(
        "train windows: %d total, %d eligible (purity >= %.2f)",
        int(is_train.sum()), int(train_elig.sum()), MIN_PURITY_TRAIN,
    )
    log.info("val windows: %d | test windows: %d", int(is_val.sum()), int(is_test.sum()))

    # ---- Fit NaN-fill medians on TRAIN only ------------------------------
    X_train = X.loc[train_elig]
    y_train = y.loc[train_elig]
    fill_values = X_train.median(axis=0).fillna(0.0)  # all-NaN train feature -> 0
    X_filled = X.fillna(fill_values)  # apply train medians to ALL splits

    # ---- Train -----------------------------------------------------------
    clf.fit(X_filled.loc[train_elig].to_numpy(), y_train.to_numpy())
    log.info("Trained %s on %d windows", clf_class_name, int(train_elig.sum()))

    # ---- Evaluate on val -------------------------------------------------
    if int(is_val.sum()) == 0:
        raise SystemExit("No val windows to evaluate.")
    X_val = X_filled.loc[is_val].to_numpy()
    y_val = y.loc[is_val].to_numpy()
    y_pred = clf.predict(X_val)

    majority_label = pd.Series(y_train).value_counts().idxmax()
    baseline_acc = accuracy_score(y_val, np.full(len(y_val), majority_label))
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", labels=CLASSES, zero_division=0)
    per_class = f1_score(y_val, y_pred, average=None, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=CLASSES)
    f1_per_class = {c: float(f) for c, f in zip(CLASSES, per_class)}

    # Feature importances: available for tree ensembles, None otherwise.
    importances: pd.Series | None = None
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=canon_cols).sort_values(
            ascending=False
        )

    metrics = {
        "n_val_windows": int(len(y_val)),
        "accuracy": float(acc),
        "majority_baseline": float(baseline_acc),
        "majority_label": str(majority_label),
        "macro_f1": float(macro_f1),
        "f1_per_class": f1_per_class,
    }

    # ---- Archive this run ------------------------------------------------
    run_id = make_run_id(runs_root)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now().isoformat(timespec="seconds")

    report = build_val_report(metrics, cm, CLASSES, importances, run_id, args.tag,
                              model_name=f"{args.model} ({clf_class_name})")
    (run_dir / "val_window_report.txt").write_text(report + "\n", encoding="utf-8")

    bundle = {
        "model": clf,
        "feature_cols": canon_cols,
        "fill_values": fill_values.to_dict(),
        "classes": list(clf.classes_),
    }
    joblib.dump(bundle, run_dir / "model.joblib")

    for split, mask in (("val", is_val), ("test", is_test)):
        if int(mask.sum()) == 0:
            log.warning("No %s windows; skipping predictions", split)
            continue
        Xs = X_filled.loc[mask].to_numpy()
        proba = clf.predict_proba(Xs)
        pred = clf.classes_[proba.argmax(axis=1)]
        wsub = windows.loc[mask].reset_index(drop=True).copy()
        wsub["pred_label"] = pred
        for i, cls in enumerate(clf.classes_):
            wsub[f"prob_{cls}"] = proba[:, i]
        wsub["max_prob"] = proba.max(axis=1)
        out_cols = (
            ["video_id", "split", "win_idx", "t_start", "t_end", "t_center",
             "label", "purity", "pred_label"]
            + [f"prob_{c}" for c in clf.classes_]
            + ["max_prob"]
        )
        wsub[out_cols].to_csv(run_dir / f"window_predictions_{split}.csv", index=False)

    # Machine-readable record of this run.
    run_meta = {
        "run_id": run_id,
        "timestamp": now_iso,
        "tag": args.tag,
        "config": {
            "model_key": args.model,
            "model_class": clf_class_name,
            "seed": SEED,
            "window_s": WINDOW_S,
            "stride_s": STRIDE_S,
            "min_purity_train": MIN_PURITY_TRAIN,
            "min_rows_frac": MIN_ROWS_FRAC,
            "max_fill_gap_s": MAX_FILL_GAP_S,
            "classes": CLASSES,
            "feature_channels": channels,
        },
        "data": {
            "n_windows_total": int(len(windows)),
            "n_windows_train_total": int(is_train.sum()),
            "n_windows_train_eligible": int(train_elig.sum()),
            "n_windows_val": int(is_val.sum()),
            "n_windows_test": int(is_test.sum()),
            "n_features": int(X.shape[1]),
        },
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "top10_importances": importances.head(10).to_dict() if importances is not None else {},
        "files": {
            "report": "val_window_report.txt",
            "model": "model.joblib",
            "predictions_val": "window_predictions_val.csv",
            "predictions_test": "window_predictions_test.csv",
        },
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )

    # ---- Master comparison log (append one row) -------------------------
    log_row = {
        "run_id": run_id,
        "timestamp": now_iso,
        "tag": args.tag,
        "model_key": args.model,
        "model_class": clf_class_name,
        "seed": SEED,
        "window_s": WINDOW_S,
        "stride_s": STRIDE_S,
        "min_purity_train": MIN_PURITY_TRAIN,
        "min_rows_frac": MIN_ROWS_FRAC,
        "max_fill_gap_s": MAX_FILL_GAP_S,
        "n_windows_total": int(len(windows)),
        "n_windows_train_total": int(is_train.sum()),
        "n_windows_train_eligible": int(train_elig.sum()),
        "n_windows_val": int(is_val.sum()),
        "n_windows_test": int(is_test.sum()),
        "n_features": int(X.shape[1]),
        "val_accuracy": float(acc),
        "val_majority_baseline": float(baseline_acc),
        "val_macro_f1": float(macro_f1),
        "val_f1_squat": f1_per_class["squat"],
        "val_f1_push_up": f1_per_class["push_up"],
        "val_f1_jump_jack": f1_per_class["jump_jack"],
        "val_f1_transition": f1_per_class["transition"],
        "top1_feature": str(importances.index[0]) if importances is not None else "",
    }
    log_path = out_dir / "runs_log.csv"
    pd.DataFrame([log_row]).to_csv(
        log_path, mode="a", header=not log_path.exists(), index=False
    )

    # ---- Latest copies at top level (overwrite) -------------------------
    for fname in [
        "val_window_report.txt",
        "model.joblib",
        "window_predictions_val.csv",
        "window_predictions_test.csv",
    ]:
        shutil.copy2(run_dir / fname, out_dir / fname)

    log.info("Archived run %s -> %s", run_id, run_dir)
    log.info("Appended row to %s", log_path)

    # ---- Print the val report --------------------------------------------
    print()
    print(report)
    print(f"Archived as run_id {run_id} in {run_dir}")


if __name__ == "__main__":
    main()
