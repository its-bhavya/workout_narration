"""Training-fold comparison after unguided-peak error analysis; no data rewriting."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import ROOT, FEATURES_DIR, SYNTH_DIR, sha256
from . import rep_counter as counter
from .analyze_peak_errors import load_training, count_statistics
from .peak_improvements import ImprovementSettings, build_signals, count_built, count_improved_peaks
from .validate_peak_counter import make_row, fold_mae, timing_metrics


BASELINE_DIR = ROOT / "outputs/peak_improvements/error_analysis_20261118"


def metrics(rows, all_videos):
    video = rows.groupby("video_id")[["annotated", "predicted", "error"]].sum().reindex(all_videos, fill_value=0).reset_index()
    return dict(source_weighted_fold_mae=float(fold_mae(rows).mean()), timing=timing_metrics(rows),
                segment_counts=count_statistics(rows), video_counts=count_statistics(video),
                active_video_counts=count_statistics(video[video.annotated > 0]))


def validate(output, plots=True, baseline_dir=BASELINE_DIR):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new experiment output directory")
    baseline_dir = Path(baseline_dir)
    baseline = pd.read_csv(baseline_dir / "segments.csv")
    baseline["rep_times"] = baseline.rep_times.map(ast.literal_eval)
    baseline_report = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    expected_inputs = {r["video_id"]: r for r in baseline_report["inputs"]}
    original_config = json.dumps(counter.EXERCISES, sort_keys=True)
    samples, inputs, videos = [], [], []
    for vid, meta, frame, hashes, _ in load_training():
        if any(hashes[k] != expected_inputs[vid][k] for k in hashes):
            raise ValueError("Inputs changed since baseline error analysis")
        videos.append(vid)
        inputs.append(dict(video_id=vid, **hashes))
        for index, segment in enumerate(meta["segments"]):
            if segment["type"] != "exercise":
                continue
            exercise = segment["label"]
            start, end = (segment[k] / meta["fps"] for k in ("start_frame", "end_frame"))
            sliced = frame[(frame.time_s >= start) & (frame.time_s < end)]
            sample = dict(video_id=vid, fold=meta["fold"], exercise=exercise, clip_id=segment["clip_id"],
                          segment_index=index, annotated=segment["rep_count_annotated"],
                          pairs=[(a / meta["fps"], b / meta["fps"]) for a, b in segment["rep_pairs_out"]],
                          frame=sliced, start=start, end=end, t=sliced.time_s.to_numpy(), signals={})
            for smoothing in (0., .3, .6):
                sample["signals"][smoothing] = build_signals(sliced, exercise, smoothing)
            samples.append(sample)
    print(f"Prepared {len(samples)} training segments; baseline/data hashes match", flush=True)
    reports, scores, selected_frames, heldout_frames, checks = {}, [], [], [], []
    best_settings = {}
    for exercise in sorted(counter.EXERCISES):
        print(f"Evaluating cached-signal and smoothing alternatives: {exercise}", flush=True)
        source = [s for s in samples if s["exercise"] == exercise]
        families = ("baseline", "fused", "median") if exercise == "push_up" else ("baseline", "alternate", "fused", "median")
        gaps = {"jump_jack": (.25, .5), "push_up": (.2, .35), "squat": (.35, .5)}[exercise]
        settings = [ImprovementSettings(signal=family, smooth_s=smooth, prominence_fraction=prominence,
                                         min_gap_s=gap) for family, smooth, prominence, gap in itertools.product(
                                             families, (0., .3, .6), (.15, .25, .35, .45), gaps)]
        predictions, rows = [], []
        for candidate_id, setting in enumerate(settings):
            frame = pd.DataFrame([make_row(s, count_built(s["signals"][setting.smooth_s], s["t"], exercise, setting)) for s in source])
            folds = fold_mae(frame)
            rows.append(dict(candidate_id=candidate_id, **asdict(setting), macro_fold_mae=float(folds.mean()),
                             **{f"fold_{f}_mae": float(v) for f, v in folds.items()}))
            predictions.append(frame)
            scores.append(dict(exercise=exercise, **rows[-1]))
        candidates = pd.DataFrame(rows)
        best = int(candidates.macro_fold_mae.idxmin())
        best_settings[exercise] = settings[best]
        selected_frames.append(predictions[best])
        heldout, exercise_checks = [], []
        for fold in range(5):
            columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
            choice = int(candidates[columns].mean(axis=1).idxmin())
            heldout.append(predictions[choice][predictions[choice].fold == fold])
            check = dict(exercise=exercise, fold=fold, settings=asdict(settings[choice]),
                         mae=rows[choice][f"fold_{fold}_mae"])
            checks.append(check); exercise_checks.append(check)
        heldout = pd.concat(heldout)
        heldout_frames.append(heldout)
        by_family = {}
        for family, candidate_family in candidates.groupby("signal"):
            family_predictions = []
            for fold in range(5):
                columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
                choice = int(candidate_family[columns].mean(axis=1).idxmin())
                family_predictions.append(predictions[choice][predictions[choice].fold == fold])
            by_family[family] = metrics(pd.concat(family_predictions), videos)
        # Hold one setting at its old effective value to distinguish spacing
        # from smoothing within the same normalized-signal search family.
        baseline_family = candidates[candidates.signal == "baseline"]
        native_gap = counter.EXERCISES[exercise]["min_gap_s"]
        native_smoothing = .6 if exercise == "squat" else .3
        ablations = {}
        subsets = {"old_gap": baseline_family[baseline_family.min_gap_s == native_gap],
                   "old_effective_smoothing": baseline_family[baseline_family.smooth_s == native_smoothing],
                   "old_gap_and_smoothing": baseline_family[(baseline_family.min_gap_s == native_gap) & (baseline_family.smooth_s == native_smoothing)]}
        for label, subset in subsets.items():
            values = []
            for fold in range(5):
                columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
                choice = int(subset[columns].mean(axis=1).idxmin())
                values.append(rows[choice][f"fold_{fold}_mae"])
            ablations[label] = float(np.mean(values))
        reports[exercise] = dict(baseline=metrics(baseline[baseline.exercise == exercise], videos),
                                  selected_all_train=metrics(predictions[best], videos),
                                  selected_other_folds=metrics(heldout, videos),
                                  selected_settings=asdict(settings[best]), by_signal_family=by_family,
                                  baseline_signal_ablations=ablations,
                                  candidate_count=len(settings))
    selected = pd.concat(selected_frames)
    for sample in samples:
        actual = count_improved_peaks(sample["frame"], sample["exercise"], sample["start"], sample["end"],
                                     settings=best_settings[sample["exercise"]])
        expected = selected[(selected.video_id == sample["video_id"]) & (selected.segment_index == sample["segment_index"])].iloc[0]
        if actual["rep_times"] != expected.rep_times:
            raise ValueError("Runtime differs from cached selected experiment")
    for item in inputs:
        vid = item["video_id"]
        if sha256(FEATURES_DIR / f"{vid}_angles.csv") != item["features_sha256"] or sha256(SYNTH_DIR / "train" / f"{vid}.json") != item["annotations_sha256"]:
            raise ValueError("Existing training data changed during experiment")
    if json.dumps(counter.EXERCISES, sort_keys=True) != original_config:
        raise ValueError("Production defaults mutated")
    heldout = pd.concat(heldout_frames)
    overall_videos = heldout.groupby("video_id")[["annotated", "predicted", "error"]].sum().reset_index()
    result = dict(run_id="v2", split="train", n_segments=len(samples), n_videos=len(videos),
                  exercises=reports, overall_video_counts=count_statistics(overall_videos), checks=checks, inputs=inputs,
                  inputs_verified_unchanged=True, runtime_verified_segments=len(samples),
                  source_hashes={name: sha256(Path(__file__).with_name(name)) for name in (
                      "peak_improvements.py", "validate_peak_improvements.py", "peak_counter.py", "rep_counter.py")},
                  limitations=["GT exercise windows; not end-to-end segmentation/counting or new test results.",
                               "Exploratory training-fold development, not unbiased nested CV.",
                               "No raw wrist/hip-height coordinates are cached; no extraction or data rewriting performed.",
                               "Normalized ankle spacing is divided by shoulder width in the existing cache.",
                               "Fusion requires all component signals at a frame; it does not reconstruct occluded limbs.",
                               "Median voting uses three usable signal timelines or falls back to baseline.",
                               "Synthetic source reuse is weighted equally within folds for MAE only.",
                               "No automatic adoption of candidate settings; production defaults are unchanged."])
    output.mkdir(parents=True)
    pd.DataFrame(scores).to_csv(output / "candidates.csv", index=False)
    selected.to_csv(output / "selected_segments.csv", index=False)
    heldout.to_csv(output / "heldout_segments.csv", index=False)
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if plots:
        plot_results(output, result)
    return result


def plot_results(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exercises = sorted(result["exercises"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(3)
    for offset, key, label in [(-.18, "baseline", "Frozen unguided-peak baseline"),
                               (.18, "selected_other_folds", "Alternatives selected on other folds")]:
        axes[0].bar(x + offset, [result["exercises"][e][key]["source_weighted_fold_mae"] for e in exercises], .36, label=label)
        axes[1].bar(x + offset, [100 * result["exercises"][e][key]["active_video_counts"]["within_one_fraction"] for e in exercises], .36, label=label)
    axes[0].set(ylabel="Source-weighted fold MAE", title="Count error")
    axes[1].set(ylabel="Active videos within +/-1 rep (%)", title="Per-video exercise totals")
    for ax in axes:
        ax.set_xticks(x, exercises); ax.legend(fontsize=7)
    fig.savefig(output / "comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    args = parser.parse_args()
    result = validate(args.out_dir, baseline_dir=args.baseline_dir)
    print(json.dumps({e: dict(baseline=v["baseline"]["source_weighted_fold_mae"],
                              selected_other_folds=v["selected_other_folds"]["source_weighted_fold_mae"],
                              selected_settings=v["selected_settings"]) for e, v in result["exercises"].items()}, indent=2))
