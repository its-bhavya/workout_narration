"""Controlled, training-only 2D/world-3D angle comparison for both rep counters."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import catalog, feature_frame, FEATURES_DIR, SYNTH_DIR, sha256
from . import rep_counter as counter
from .peak_counter import PeakSettings, detect_peaks
from .validate_peak_counter import fold_mae, make_row, timing_metrics


def require_columns(frame, exercise, dimensions):
    if dimensions not in ("2d", "3d"):
        raise ValueError("dimensions must be 2d or 3d")
    names = [name + ("_3d" if dimensions == "3d" else "")
             for name in counter.EXERCISES[exercise]["columns"]]
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"Missing {dimensions} columns (fallback forbidden): {missing}")
    return names


def fixed_parameters(exercise, method):
    cfg = counter.EXERCISES[exercise]
    if method == "state_machine":
        return dict(fracs=list(cfg["fracs"]), confirm_s=cfg["confirm_s"])
    if method == "peaks":
        # Previously selected all-training guided-peak settings, unchanged in
        # both dimensions for the controlled fixed-settings comparison.
        return asdict(PeakSettings(prominence_fraction=.45 if exercise == "push_up" else .25,
                                   min_width_s=1 / 15, period_distance_fraction=.35))
    raise ValueError("Unknown counting method")


def count_prepared(sample, dimensions, method, parameters):
    prepared = sample["prepared"][dimensions]
    chosen = prepared["chosen"]
    if chosen is None:
        return []
    cfg = counter.EXERCISES[sample["exercise"]]
    if method == "peaks":
        return detect_peaks(chosen["values"], sample["t"], cfg["pattern"], cfg["min_gap_s"],
                            PeakSettings(**parameters))["rep_times"]
    sig = prepared["normalized"] if cfg.get("adaptive_window_s") else chosen["values"]
    thresholds = parameters["fracs"] if cfg.get("adaptive_window_s") else counter.adaptive_thresholds(sig, parameters["fracs"])
    indices = counter.run_state_machine(sig, *thresholds, cfg["pattern"], sample["fs"], parameters["confirm_s"])
    times = counter.drop_close([float(sample["t"][i]) for i in indices], cfg["min_gap_s"])
    return [round(t, 3) for t in times]


def prepare_sample(frame, exercise, start, end):
    segment = frame[(frame.time_s >= start) & (frame.time_s < end)]
    cfg = counter.EXERCISES[exercise]
    sample = dict(exercise=exercise, t=segment.time_s.to_numpy(), fs=counter.sampling_rate(segment.time_s.to_numpy()), prepared={})
    for dimensions in ("2d", "3d"):
        names = require_columns(segment, exercise, dimensions)
        candidates = counter.prepare_candidates(segment, cfg, use_3d=dimensions == "3d") if len(segment) >= 8 else []
        chosen = next((c for c in candidates if c["usable"]), None)
        sample["prepared"][dimensions] = dict(chosen=chosen,
            normalized=counter.local_motion_signal(chosen["values"], sample["fs"], cfg["adaptive_window_s"], cfg["min_range_deg"])
                if chosen is not None and cfg.get("adaptive_window_s") else None,
            raw_coverage=float(np.isfinite(segment[names].to_numpy(dtype=float)).mean()) if len(segment) else 0.)
    return sample


def count_variant(frame, exercise, start_s, end_s, *, dimensions="3d", method="state_machine", parameters=None):
    """Opt-in public experiment API; strict columns, no global configuration mutation."""
    require_columns(frame, exercise, dimensions)
    if method not in ("state_machine", "peaks"):
        raise ValueError("Unknown counting method")
    # Prepare both modalities to match the experiment's paired comparison.
    sample = prepare_sample(frame, exercise, start_s, end_s)
    times = count_prepared(sample, dimensions, method, parameters or fixed_parameters(exercise, method))
    chosen = sample["prepared"][dimensions]["chosen"]
    return dict(reps=len(times), rep_times=times, dimensions=dimensions, method=method,
                signal=chosen["signal"] if chosen else "", usable=chosen is not None)


def summarize(rows):
    folds = fold_mae(rows)
    return dict(macro_fold_mae=float(folds.mean()), fold_mae={str(f): float(v) for f, v in folds.items()},
                timing=timing_metrics(rows), exact_match=float(rows.error.eq(0).mean()),
                within_one=float(rows.error.abs().le(1).mean()),
                wape=float(rows.absolute_error.sum() / rows.annotated.sum()) if rows.annotated.sum() else None)


def compare(output, plots=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new comparison output directory")
    original_config = deepcopy(counter.EXERCISES)
    samples, inputs, coverage = [], [], []
    for vid, meta in sorted(catalog().items()):
        if meta["split"] != "train":
            continue
        path = FEATURES_DIR / f"{vid}_angles.csv"
        frame = feature_frame(path, meta)
        inputs.append(dict(video_id=vid, features_sha256=sha256(path),
                           annotations_sha256=sha256(SYNTH_DIR / "train" / f"{vid}.json")))
        for index, seg in enumerate(meta["segments"]):
            if seg["type"] != "exercise":
                continue
            start, end = (seg[key] / meta["fps"] for key in ("start_frame", "end_frame"))
            sample = prepare_sample(frame, seg["label"], start, end)
            sample.update(video_id=vid, fold=meta["fold"], clip_id=seg["clip_id"], segment_index=index,
                          annotated=int(seg["rep_count_annotated"]),
                          pairs=[(a / meta["fps"], b / meta["fps"]) for a, b in seg["rep_pairs_out"]],
                          frame=frame[(frame.time_s >= start) & (frame.time_s < end)], start=start, end=end)
            if sample["annotated"] != len(sample["pairs"]):
                raise ValueError("Inconsistent rep annotations")
            samples.append(sample)
            for dimensions, prepared in sample["prepared"].items():
                chosen = prepared["chosen"]
                coverage.append({k: sample[k] for k in ("video_id", "fold", "exercise", "clip_id", "segment_index")}
                    | dict(dimensions=dimensions, raw_coverage=prepared["raw_coverage"], usable=chosen is not None,
                           selected_signal=chosen["signal"] if chosen else ""))
            # Independent public baseline sanity check, including actual 3D selection.
            for dimensions in ("2d", "3d"):
                actual = counter.count_for_exercise(frame, seg["label"], start, end, use_3d=dimensions == "3d")
                expected = count_prepared(sample, dimensions, "state_machine", fixed_parameters(seg["label"], "state_machine"))
                if actual["rep_times"] != expected:
                    raise ValueError(f"Baseline mismatch: {vid}:{index}:{dimensions}")
    print(f"Loaded {len(samples)} paired training segments; both dimensions verified", flush=True)
    fixed_rows, tuned_rows, heldout_rows, candidate_rows, checks = [], [], [], [], []
    report = {}
    for exercise in sorted(counter.EXERCISES):
        subset = [s for s in samples if s["exercise"] == exercise]
        report[exercise] = {}
        for method in ("state_machine", "peaks"):
            if method == "state_machine":
                candidates = [dict(fracs=list(f), confirm_s=c) for f, c in itertools.product(
                    ((.2, .55), (.35, .65), (.45, .75)), (0., .1, .15))]
            else:
                candidates = [asdict(PeakSettings(prominence_fraction=p, min_width_s=w, period_distance_fraction=d))
                    for p, w, d in itertools.product((.15, .25, .35, .45), (1 / 15, 2 / 15, .2), (0., .35, .5, .65))]
            report[exercise][method] = {}
            for dimensions in ("2d", "3d"):
                print(f"Evaluating {exercise}: {method}, {dimensions}", flush=True)
                tag = dict(exercise=exercise, method=method, dimensions=dimensions)
                fixed = pd.DataFrame([make_row(s, count_prepared(s, dimensions, method, fixed_parameters(exercise, method))) | tag for s in subset])
                fixed_rows.append(fixed)
                predictions, scores = [], []
                for candidate_id, parameters in enumerate(candidates):
                    table = pd.DataFrame([make_row(s, count_prepared(s, dimensions, method, parameters)) | tag for s in subset])
                    folds = fold_mae(table)
                    scores.append(dict(candidate_id=candidate_id, parameters=parameters, macro_fold_mae=float(folds.mean()),
                                       **{f"fold_{f}_mae": float(v) for f, v in folds.items()}))
                    predictions.append(table)
                    candidate_rows.append(tag | scores[-1])
                scores_frame = pd.DataFrame(scores)
                best = int(scores_frame.macro_fold_mae.idxmin())
                tuned_rows.append(predictions[best])
                heldout = []
                for fold in range(5):
                    columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
                    selected = int(scores_frame[columns].mean(axis=1).idxmin())
                    heldout.append(predictions[selected][predictions[selected].fold == fold])
                    checks.append(tag | dict(fold=fold, parameters=candidates[selected],
                                              mae=scores[selected][f"fold_{fold}_mae"]))
                combined = pd.concat(heldout)
                heldout_rows.append(combined)
                report[exercise][method][dimensions] = dict(fixed=summarize(fixed),
                    tuned_all_train=summarize(predictions[best]), tuned_other_folds=summarize(combined),
                    selected_parameters=candidates[best], fixed_parameters=fixed_parameters(exercise, method),
                    candidate_count=len(candidates))
                # The opt-in public API must reproduce selected settings on all segments.
                for sample, row in zip(subset, predictions[best].to_dict("records")):
                    actual = count_variant(sample["frame"], exercise, sample["start"], sample["end"],
                                           dimensions=dimensions, method=method, parameters=candidates[best])
                    if actual["rep_times"] != row["rep_times"]:
                        raise ValueError("Selected public API disagrees with validation")
    if counter.EXERCISES != original_config:
        raise ValueError("Experiment mutated production defaults")
    result = dict(run_id="v2", split="train", n_segments=len(samples), exercises=report,
                  inputs=inputs, fold_checks=checks, counter_config=counter.EXERCISES,
                  hashes={name: sha256(Path(__file__).with_name(name)) for name in (
                      "compare_signal_dimensions.py", "rep_counter.py", "peak_counter.py", "arm_quality.py")},
                  timing_tolerance_s=.3,
                  limitations=["World-3D angles estimated by MediaPipe; not motion-capture ground truth.",
                               "Ground-truth exercise windows; upstream pose person selection is annotation-assisted.",
                               "Fixed peak settings and baseline choices previously selected on all training data.",
                               "Exploratory training-fold comparison, not an unbiased nested-CV estimate.",
                               "No test feature frames loaded; no production defaults changed.",
                               "MAE averages source clips within folds; timing and exact/within-one/WAPE pool segment occurrences.",
                               "Preprocessing, arm-selection rules, and motion guards held fixed; tuning changes counter parameters only."])
    output.mkdir(parents=True)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output / "candidates.csv", index=False)
    pd.concat(fixed_rows).to_csv(output / "fixed_predictions.csv", index=False)
    pd.concat(tuned_rows).to_csv(output / "tuned_predictions.csv", index=False)
    pd.concat(heldout_rows).to_csv(output / "heldout_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if plots:
        plot_comparison(output, result)
    return result


def plot_comparison(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exercises = sorted(report["exercises"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharey=True)
    x = np.arange(len(exercises))
    for ax, stage, title in ((axes[0], "fixed", "Same settings: isolate signal change"),
                             (axes[1], "tuned_other_folds", "Settings selected on other four folds")):
        for i, (method, dimensions) in enumerate(itertools.product(("state_machine", "peaks"), ("2d", "3d"))):
            values = [report["exercises"][e][method][dimensions][stage]["macro_fold_mae"] for e in exercises]
            ax.bar(x + (i - 1.5) * .2, values, .2, label=f"{method.replace('_', ' ')} {dimensions.upper()}")
        ax.set(xticks=x, xticklabels=exercises, title=title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("MAE (reps / source clip, averaged across folds)")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.out_dir)
    print(json.dumps({e: {m: {d: round(v["tuned_other_folds"]["macro_fold_mae"], 3)
                              for d, v in ms.items()} for m, ms in methods.items()}
                      for e, methods in result["exercises"].items()}, indent=2))
