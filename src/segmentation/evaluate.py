"""evaluate.py -- val grid-search + test evaluation for segmentation.

Run from repo root::

    python src/segmentation/evaluate.py

Steps
-----
1. Grid-search tau x min_len on VAL predictions; pick best by Edit score
   (tie-break: frame accuracy).  Save outputs/segmentation/chosen_params.json.
2. Run ONCE on TEST with those params.  Compare raw (no smoothing; only merge
   identical neighbours) vs smoothed.
   Save:
     outputs/segmentation/results.csv         -- mean/std/n across test videos
     outputs/segmentation/test_segments.json  -- smoothed segments per video
     outputs/segmentation/timeline_example.png -- GT / raw / smoothed for one video
3. Print test video count and source-subject prefixes in test and val clips.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap (bare imports, same convention as the rest of the package)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import CLASSES, OUTPUT_DIR, SYNTH_DIR  # noqa: E402
from metrics import all_metrics, gt_from_json        # noqa: E402
from smooth import smooth, to_segments               # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TAU_GRID = [0.4, 0.5, 0.6, 0.7]
MIN_LEN_GRID = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_predictions(pred_csv: Path, split: str) -> dict[str, pd.DataFrame]:
    """Return {video_id: per-window DataFrame} for the given split."""
    df = pd.read_csv(pred_csv)
    df = df[df["split"] == split].copy()
    return {vid: grp.sort_values("win_idx").reset_index(drop=True)
            for vid, grp in df.groupby("video_id")}


def _merge_identical(segs: list[dict]) -> list[dict]:
    """Merge consecutive segments with the same label (raw baseline)."""
    out: list[dict] = []
    for s in segs:
        if out and out[-1]["exercise"] == s["exercise"]:
            out[-1]["end_time"] = s["end_time"]
        else:
            out.append(dict(s))
    return out


def _raw_segments(wdf: pd.DataFrame, video_end_s: float) -> list[dict]:
    """Segments from raw predictions (no confidence filter, no min-len,
    only merge identical neighbours)."""
    segs = to_segments(
        list(wdf["pred_label"]),
        list(wdf["t_center"]),
        video_end_s,
    )
    return _merge_identical(segs)


def _smoothed_segments(wdf: pd.DataFrame, video_end_s: float,
                       tau: float, min_len: int) -> list[dict]:
    labels = smooth(
        list(wdf["pred_label"]),
        list(wdf["max_prob"]),
        tau=tau,
        min_len=min_len,
    )
    return to_segments(labels, list(wdf["t_center"]), video_end_s)


def _find_json(video_id: str) -> Path:
    """Locate the synthetic JSON for a video_id under SYNTH_DIR."""
    split = video_id.split("_")[1]  # synth_val_0001 -> val
    p = SYNTH_DIR / split / f"{video_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"Cannot find JSON for {video_id}: {p}")
    return p


def _eval_video(wdf: pd.DataFrame, meta: dict, tau: float, min_len: int) -> tuple[dict, dict]:
    """Return (raw_metrics, smoothed_metrics) for one video."""
    gt_segs, video_end_s = gt_from_json(meta)
    raw = _raw_segments(wdf, video_end_s)
    smo = _smoothed_segments(wdf, video_end_s, tau, min_len)
    rm = all_metrics(gt_segs, raw, video_end_s, CLASSES)
    sm = all_metrics(gt_segs, smo, video_end_s, CLASSES)
    return rm, sm


def _subject_prefixes(clip_ids: list[str]) -> set[str]:
    """Extract stuN or other prefix from clip filenames."""
    prefixes: set[str] = set()
    for clip in clip_ids:
        clip = clip.replace(".mp4", "")
        m = re.match(r"^(stu\d+|train|val|test)_?\d*", clip)
        if m:
            # For stuN, keep stuN; for train/val/test IDs keep the word
            word = re.match(r"^(stu\d+|[a-z]+)", clip)
            if word:
                prefixes.add(word.group(1))
        else:
            prefixes.add(clip)
    return prefixes


# ---------------------------------------------------------------------------
# Grid search on VAL
# ---------------------------------------------------------------------------

def grid_search_val(val_preds: dict[str, pd.DataFrame],
                    synth_dir: Path) -> tuple[float, int]:
    """Return (best_tau, best_min_len) maximising Edit score, tie-break frame accuracy."""
    log.info("Grid-searching tau=%s x min_len=%s on %d val videos",
             TAU_GRID, MIN_LEN_GRID, len(val_preds))
    best: tuple[float, float, float, int] | None = None  # (edit, acc, tau, min_len)
    for tau in TAU_GRID:
        for min_len in MIN_LEN_GRID:
            edits, accs = [], []
            for vid_id, wdf in val_preds.items():
                meta = _load_json(synth_dir / "val" / f"{vid_id}.json")
                gt_segs, video_end_s = gt_from_json(meta)
                smo = _smoothed_segments(wdf, video_end_s, tau, min_len)
                m = all_metrics(gt_segs, smo, video_end_s, CLASSES)
                edits.append(m["edit_score"])
                accs.append(m["frame_accuracy"])
            mean_edit = float(np.mean(edits))
            mean_acc = float(np.mean(accs))
            log.info("  tau=%.1f  min_len=%d  edit=%.2f  acc=%.4f",
                     tau, min_len, mean_edit, mean_acc)
            if best is None or (mean_edit, mean_acc) > (best[0], best[1]):
                best = (mean_edit, mean_acc, tau, min_len)
    assert best is not None
    log.info("Best: tau=%.1f  min_len=%d  -> edit=%.2f  acc=%.4f",
             best[2], best[3], best[0], best[1])
    return best[2], best[3]


# ---------------------------------------------------------------------------
# Timeline plot
# ---------------------------------------------------------------------------

def _timeline_plot(video_id: str, gt_segs: list[dict],
                   raw_segs: list[dict], smo_segs: list[dict],
                   video_end_s: float, out_path: Path) -> None:
    colors = {
        "squat": "#4C72B0",
        "push_up": "#DD8452",
        "jump_jack": "#55A868",
        "transition": "#C44E52",
    }
    fig, axes = plt.subplots(3, 1, figsize=(14, 3), sharex=True)
    rows = [("Ground truth", gt_segs), ("Raw", raw_segs), ("Smoothed", smo_segs)]
    for ax, (title, segs) in zip(axes, rows):
        for s in segs:
            w = s["end_time"] - s["start_time"]
            c = colors.get(s["exercise"], "#888888")
            ax.barh(0, w, left=s["start_time"], height=0.8, color=c, edgecolor="white",
                    linewidth=0.4)
            if w > video_end_s * 0.03:
                ax.text(s["start_time"] + w / 2, 0, s["exercise"],
                        ha="center", va="center", fontsize=6, color="white", fontweight="bold")
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=8, rotation=0, ha="right", labelpad=4)
        ax.set_xlim(0, video_end_s)
    axes[-1].set_xlabel("Time (s)", fontsize=8)
    fig.suptitle(f"Segmentation: {video_id}", fontsize=9)
    legend_patches = [mpatches.Patch(color=c, label=lbl) for lbl, c in colors.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=7, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved timeline plot -> %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation on val and test.")
    parser.add_argument("--pred-val", type=Path,
                        default=OUTPUT_DIR / "window_predictions_val.csv")
    parser.add_argument("--pred-test", type=Path,
                        default=OUTPUT_DIR / "window_predictions_test.csv")
    parser.add_argument("--synth-dir", type=Path, default=SYNTH_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not args.pred_val.exists():
        raise SystemExit(f"Missing val predictions: {args.pred_val}\n"
                         "Run train.py first.")
    if not args.pred_test.exists():
        raise SystemExit(f"Missing test predictions: {args.pred_test}\n"
                         "Run train.py first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. VAL grid search ------------------------------------------------
    val_preds = _load_predictions(args.pred_val, "val")
    tau, min_len = grid_search_val(val_preds, args.synth_dir)

    params = {"tau": tau, "min_len": min_len}
    params_path = args.out_dir / "chosen_params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    log.info("Saved chosen params -> %s", params_path)

    # ---- 2. TEST evaluation ------------------------------------------------
    test_preds = _load_predictions(args.pred_test, "test")
    n_test = len(test_preds)

    raw_rows: list[dict] = []
    smo_rows: list[dict] = []
    test_segments_out: dict[str, list[dict]] = {}

    for vid_id, wdf in sorted(test_preds.items()):
        meta = _load_json(args.synth_dir / "test" / f"{vid_id}.json")
        gt_segs, video_end_s = gt_from_json(meta)
        raw = _raw_segments(wdf, video_end_s)
        smo = _smoothed_segments(wdf, video_end_s, tau, min_len)
        rm = all_metrics(gt_segs, raw, video_end_s, CLASSES)
        sm = all_metrics(gt_segs, smo, video_end_s, CLASSES)
        rm["video_id"] = vid_id
        sm["video_id"] = vid_id
        raw_rows.append(rm)
        smo_rows.append(sm)
        test_segments_out[vid_id] = smo
        log.info("%s  raw_edit=%.1f  smo_edit=%.1f",
                 vid_id, rm["edit_score"], sm["edit_score"])

    # Mean / std across videos.
    raw_df = pd.DataFrame(raw_rows).set_index("video_id")
    smo_df = pd.DataFrame(smo_rows).set_index("video_id")
    metric_cols = [c for c in raw_df.columns if c != "video_id"]

    summary_rows = []
    for col in metric_cols:
        summary_rows.append({
            "metric": col,
            "raw_mean": float(raw_df[col].mean()),
            "raw_std": float(raw_df[col].std(ddof=0)),
            "smoothed_mean": float(smo_df[col].mean()),
            "smoothed_std": float(smo_df[col].std(ddof=0)),
            "n_videos": n_test,
        })
    results_df = pd.DataFrame(summary_rows)
    results_path = args.out_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Saved results -> %s", results_path)

    # Test segments JSON.
    segs_path = args.out_dir / "test_segments.json"
    segs_path.write_text(json.dumps(test_segments_out, indent=2), encoding="utf-8")
    log.info("Saved test segments -> %s", segs_path)

    # ---- Timeline plot (first test video) ----------------------------------
    first_vid = sorted(test_preds.keys())[0]
    wdf_ex = test_preds[first_vid]
    meta_ex = _load_json(args.synth_dir / "test" / f"{first_vid}.json")
    gt_ex, ve_ex = gt_from_json(meta_ex)
    raw_ex = _raw_segments(wdf_ex, ve_ex)
    smo_ex = _smoothed_segments(wdf_ex, ve_ex, tau, min_len)
    plot_path = args.out_dir / "timeline_example.png"
    _timeline_plot(first_vid, gt_ex, raw_ex, smo_ex, ve_ex, plot_path)

    # ---- 3. Subject prefix summary ----------------------------------------
    print(f"\nTest videos: {n_test}")

    def _collect_prefixes(split: str) -> set[str]:
        prefixes: set[str] = set()
        for jpath in sorted((args.synth_dir / split).glob("*.json")):
            meta = _load_json(jpath)
            prefixes |= _subject_prefixes(meta.get("clip_ids_in_order", []))
        return prefixes

    val_prefixes = _collect_prefixes("val")
    test_prefixes = _collect_prefixes("test")
    print(f"Val  clip prefixes  ({len(val_prefixes)}):  {sorted(val_prefixes)}")
    print(f"Test clip prefixes  ({len(test_prefixes)}):  {sorted(test_prefixes)}")

    # ---- Print summary table -----------------------------------------------
    print("\n=== Test results (mean ± std across videos) ===")
    key_metrics = ["frame_accuracy", "edit_score",
                   "f1_at_10", "f1_at_25", "f1_at_50",
                   "boundary_mean_s", "boundary_median_s"]
    print(f"{'metric':<24} {'raw':>14}  {'smoothed':>14}")
    print("-" * 56)
    for _, row in results_df[results_df["metric"].isin(key_metrics)].iterrows():
        raw_str = f"{row['raw_mean']:.4f} ± {row['raw_std']:.4f}"
        smo_str = f"{row['smoothed_mean']:.4f} ± {row['smoothed_std']:.4f}"
        print(f"  {row['metric']:<22} {raw_str:>14}  {smo_str:>14}")
    print(f"\nChosen params: tau={tau}  min_len={min_len}")
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
