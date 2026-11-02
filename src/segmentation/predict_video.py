"""predict_video.py -- run the trained classifier on one synthetic test video
and print the predicted segments with a ground-truth comparison.

The angles CSV must already exist (produced by extract_pose_features.py).
No re-extraction or MediaPipe needed.

Usage (run from repo root)::

    python src/segmentation/predict_video.py synth_test_0001
    python src/segmentation/predict_video.py synth_test_0003 --no-gt
    python src/segmentation/predict_video.py synth_test_0001 --show-windows
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import (
    FEATURES_DIR,
    OUTPUT_DIR,
    SYNTH_DIR,
    WINDOW_S,
    STRIDE_S,
    MIN_ROWS_FRAC,
)
from features import fill_video_nans, window_features, feature_names
from config import feature_columns
from make_windows import make_windows_for_video
from smooth import smooth, to_segments

# Defaults
MODEL_PATH = OUTPUT_DIR / "rf_model.joblib"
PARAMS_PATH = OUTPUT_DIR / "chosen_params.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: Path) -> dict:
    bundle = joblib.load(model_path)
    return bundle  # keys: model, feature_cols, fill_values, classes


def _load_params(params_path: Path) -> tuple[float, int]:
    with open(params_path) as fh:
        p = json.load(fh)
    return float(p["tau"]), int(p["min_len"])


def _load_gt(video_id: str, synth_dir: Path) -> tuple[list[dict], float] | tuple[None, None]:
    """Return (gt_segments, video_end_s) from the boundary JSON, or (None, None)."""
    for split in ("test", "val", "train"):
        j = synth_dir / split / f"{video_id}.json"
        if j.exists():
            with open(j) as fh:
                meta = json.load(fh)
            fps = float(meta.get("fps", 30))
            n_frames = int(meta.get("n_frames", 0))
            raw = meta.get("segments", [])
            segs = []
            for s in raw:
                # JSONs store frame indices; convert to seconds
                start_s = s.get("start_s", s["start_frame"] / fps)
                end_s   = s.get("end_s",   s["end_frame"]   / fps)
                segs.append({
                    "exercise":   s["label"],
                    "start_time": start_s,
                    "end_time":   end_s,
                })
            video_end_s = float(meta.get("duration_s", n_frames / fps if n_frames else
                                         (segs[-1]["end_time"] if segs else 0.0)))
            return segs, video_end_s
    return None, None


def _fmt_seg(seg: dict) -> str:
    return (
        f"  {seg['start_time']:6.1f}s – {seg['end_time']:6.1f}s  "
        f"({seg['end_time'] - seg['start_time']:.1f}s)  "
        f"{seg['exercise']}"
    )


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def predict(
    video_id: str,
    features_dir: Path = FEATURES_DIR,
    synth_dir: Path = SYNTH_DIR,
    model_path: Path = MODEL_PATH,
    params_path: Path = PARAMS_PATH,
    show_windows: bool = False,
    show_gt: bool = True,
) -> None:
    # ---- 1. Load angles CSV ----------------------------------------------
    csv_path = features_dir / f"{video_id}_angles.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"Angles CSV not found: {csv_path}\n"
            "Run extract_pose_features.py on this video first."
        )
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    # ---- 2. Load model ---------------------------------------------------
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            "Run train.py first."
        )
    bundle = _load_model(model_path)
    rf = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]
    fill_values: dict = bundle["fill_values"]
    print(f"Model: RandomForest ({len(rf.estimators_)} trees), "
          f"{len(feature_cols)} features")

    # ---- 3. Load smoothing params ----------------------------------------
    if not params_path.exists():
        raise SystemExit(
            f"chosen_params.json not found: {params_path}\n"
            "Run evaluate.py first."
        )
    tau, min_len = _load_params(params_path)
    print(f"Smoothing: tau={tau}  min_len={min_len}")

    # ---- 4. Slide windows ------------------------------------------------
    # split="" is fine here — we only need the windows, not the split label
    df_filled = fill_video_nans(df)
    windows = make_windows_for_video(df_filled, video_id, split="test")

    if not windows:
        raise SystemExit("No windows produced — video may be too short.")
    print(f"Windows: {len(windows)}")

    # ---- 5. Build feature matrix -----------------------------------------
    feat_rows = []
    for w in windows:
        rows_slice = df_filled.iloc[w["row_start"]:w["row_end"]]
        feat_rows.append(window_features(rows_slice))

    X = pd.DataFrame(feat_rows).reindex(columns=feature_cols)
    X = X.fillna(pd.Series(fill_values))

    # ---- 6. Predict -------------------------------------------------------
    proba = rf.predict_proba(X.to_numpy())          # (n_windows, n_classes)
    pred_labels = rf.classes_[proba.argmax(axis=1)]
    max_probs = proba.max(axis=1)
    t_centers = [w["t_center"] for w in windows]
    video_end_s = float(df["time_s"].max()) + WINDOW_S / 2

    # ---- 7. Smooth & segment ---------------------------------------------
    raw_labels = list(pred_labels)
    smo_labels = smooth(raw_labels, list(max_probs), tau=tau, min_len=min_len)

    raw_segs = to_segments(raw_labels, t_centers, video_end_s)
    smo_segs = to_segments(smo_labels, t_centers, video_end_s)

    # ---- 8. Print window table (optional) --------------------------------
    if show_windows:
        print(f"\n{'win':>4}  {'t_start':>7}  {'t_end':>7}  "
              f"{'label_gt':>12}  {'pred':>12}  {'conf':>6}  {'smooth':>12}")
        print("-" * 72)
        for i, w in enumerate(windows):
            print(
                f"{w['win_idx']:>4}  {w['t_start']:>7.2f}  {w['t_end']:>7.2f}  "
                f"{w['label']:>12}  {pred_labels[i]:>12}  "
                f"{max_probs[i]:>6.3f}  {smo_labels[i]:>12}"
            )

    # ---- 9. Print segment tables -----------------------------------------
    print(f"\n{'='*55}")
    print(f"  Predicted segments  ({video_id})")
    print(f"{'='*55}")

    print("\nRAW (no smoothing):")
    for s in raw_segs:
        print(_fmt_seg(s))

    print("\nSMOOTHED:")
    for s in smo_segs:
        print(_fmt_seg(s))

    if show_gt:
        gt_segs, _ = _load_gt(video_id, synth_dir)
        if gt_segs:
            print("\nGROUND TRUTH:")
            for s in gt_segs:
                print(_fmt_seg(s))
        else:
            print("\n(ground-truth JSON not found — skipping GT)")

    # ---- 10. Quick accuracy summary --------------------------------------
    # Frame-level: compare predicted label at each row's time to gt label
    if show_gt:
        gt_segs, video_end_s_gt = _load_gt(video_id, synth_dir)
        if gt_segs and "seg_label" in df.columns:
            gt_col = df["seg_label"].astype(str).to_numpy()
            # Map each row's time to a smoothed label
            pred_at_row = []
            tc = np.array(t_centers)
            for _, row in df.iterrows():
                t = float(row["time_s"])
                # find closest window center
                idx = int(np.argmin(np.abs(tc - t)))
                pred_at_row.append(smo_labels[idx])
            match = np.array(pred_at_row) == gt_col
            acc = match.mean()
            print(f"\nFrame-level accuracy (smoothed vs seg_label): {acc:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict exercise segments for a synthetic test video."
    )
    parser.add_argument(
        "video_id",
        help="Synthetic video id, e.g. synth_test_0001",
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Skip ground-truth comparison",
    )
    parser.add_argument(
        "--show-windows",
        action="store_true",
        help="Print per-window prediction table",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--params", type=Path, default=PARAMS_PATH)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--synth-dir", type=Path, default=SYNTH_DIR)
    args = parser.parse_args()

    predict(
        video_id=args.video_id,
        features_dir=args.features_dir,
        synth_dir=args.synth_dir,
        model_path=args.model,
        params_path=args.params,
        show_windows=args.show_windows,
        show_gt=not args.no_gt,
    )


if __name__ == "__main__":
    main()
