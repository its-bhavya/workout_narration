"""Compare experimental peaks with the current counter on training folds only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import catalog, feature_frame, FEATURES_DIR, SYNTH_DIR, sha256
from . import rep_counter as counter
from .peak_counter import PeakSettings, count_peaks_for_exercise, detect_peaks


def match_intervals(times, intervals, tolerance=.3):
    """One-to-one maximum interval/point matching; earliest deadlines first."""
    points = sorted(times)
    used = set()
    for start, end in sorted(intervals, key=lambda pair: pair[1]):
        for i, point in enumerate(points):
            if i not in used and start - tolerance <= point <= end + tolerance:
                used.add(i)
                break
    return len(used)


def fold_mae(rows):
    return rows.groupby(["fold", "clip_id"]).absolute_error.mean().groupby("fold").mean()


def timing_metrics(rows):
    predicted, annotated, hits = (int(rows[key].sum()) for key in ("predicted", "annotated", "hits"))
    precision = hits / predicted if predicted else 0.
    recall = hits / annotated if annotated else 0.
    return dict(predicted=predicted, annotated=annotated, hits=hits, precision=precision,
                recall=recall, f1=2 * precision * recall / (precision + recall) if precision + recall else 0.)


def validate(output, plots=6):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    settings = [PeakSettings(prominence_fraction=p, min_width_s=w, period_distance_fraction=d)
                for p, w, d in itertools.product((.15, .25, .35, .45), (1 / 15, 2 / 15, .2), (0., .35, .5, .65))]
    samples, inputs, baseline_rows = [], [], []
    for vid, meta in sorted(catalog().items()):
        if meta["split"] != "train":
            continue
        path = FEATURES_DIR / f"{vid}_angles.csv"
        frame = feature_frame(path, meta)
        annotation = SYNTH_DIR / "train" / f"{vid}.json"
        inputs.append(dict(video_id=vid, features_sha256=sha256(path), annotations_sha256=sha256(annotation)))
        for index, seg in enumerate(meta["segments"]):
            if seg["type"] != "exercise":
                continue
            exercise = seg["label"]
            start, end = (seg[key] / meta["fps"] for key in ("start_frame", "end_frame"))
            sliced = frame[(frame.time_s >= start) & (frame.time_s < end)]
            cfg = counter.EXERCISES[exercise]
            candidates = counter.prepare_candidates(sliced, cfg) if len(sliced) >= 8 else []
            chosen = next((c for c in candidates if c["usable"]), None)
            sample = dict(video_id=vid, fold=meta["fold"], clip_id=seg["clip_id"], segment_index=index,
                          exercise=exercise, annotated=int(seg["rep_count_annotated"]),
                          pairs=[(a / meta["fps"], b / meta["fps"]) for a, b in seg["rep_pairs_out"]],
                          t=sliced.time_s.to_numpy(), chosen=chosen, frame=sliced, start=start, end=end)
            if sample["annotated"] != len(sample["pairs"]):
                raise ValueError("Annotated counts and rep intervals disagree")
            baseline = counter.count_for_exercise(frame, exercise, start, end)
            sample["baseline_times"] = baseline["rep_times"]
            samples.append(sample)
            baseline_rows.append(make_row(sample, baseline["rep_times"]))
    baseline = pd.DataFrame(baseline_rows)
    results, predictions = [], []
    for candidate_id, setting in enumerate(settings):
        rows = []
        for s in samples:
            times = []
            if s["chosen"] is not None:
                cfg = counter.EXERCISES[s["exercise"]]
                detected = detect_peaks(s["chosen"]["values"], s["t"], cfg["pattern"], cfg["min_gap_s"], setting)
                times = detected["rep_times"]
            rows.append(make_row(s, times))
        table = pd.DataFrame(rows)
        predictions.append(table)
        for exercise, group in table.groupby("exercise"):
            folds = fold_mae(group)
            results.append(dict(candidate_id=candidate_id, exercise=exercise, **asdict(setting),
                                macro_fold_mae=float(folds.mean()),
                                **{f"fold_{f}_mae": float(e) for f, e in folds.items()}))
    scores = pd.DataFrame(results)
    reports, checks, selected_tables, heldout_tables = {}, [], [], []
    selected_settings = {}
    guided_settings, guided_tables, guided_heldout_tables = {}, [], []
    for exercise, group in scores.groupby("exercise"):
        base = baseline[baseline.exercise == exercise]
        base_folds = fold_mae(base)
        best = group.loc[group.macro_fold_mae.idxmin()]
        candidate_id = int(best.candidate_id)
        selected_settings[exercise] = settings[candidate_id]
        selected = predictions[candidate_id]
        selected = selected[selected.exercise == exercise]
        selected_tables.append(selected)
        held_errors = []
        for fold in range(5):
            columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
            chosen = group.loc[group[columns].mean(axis=1).idxmin()]
            check = dict(exercise=exercise, fold=fold, candidate_id=int(chosen.candidate_id),
                         baseline_mae=float(base_folds.loc[fold]), peak_mae=float(chosen[f"fold_{fold}_mae"]))
            checks.append(check)
            held_errors.append(check["peak_mae"])
            prediction = predictions[int(chosen.candidate_id)]
            heldout_tables.append(prediction[(prediction.exercise == exercise) & (prediction.fold == fold)])
        reports[exercise] = dict(n_segments=len(base), n_source_clips=base.clip_id.nunique(),
                                 baseline_mae=float(base_folds.mean()),
                                 selected_peak_mae=float(best.macro_fold_mae),
                                 heldout_peak_mae=float(np.mean(held_errors)),
                                 selected_settings=asdict(settings[candidate_id]),
                                 baseline_timing=timing_metrics(base), selected_peak_timing=timing_metrics(selected))
        families = {}
        for family, subset in (("guided", group[group.period_distance_fraction > 0]),
                               ("unguided", group[group.period_distance_fraction == 0])):
            family_best = subset.loc[subset.macro_fold_mae.idxmin()]
            family_id = int(family_best.candidate_id)
            family_selected = predictions[family_id]
            family_selected = family_selected[family_selected.exercise == exercise]
            family_checks, family_heldout = [], []
            for fold in range(5):
                columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
                choice = subset.loc[subset[columns].mean(axis=1).idxmin()]
                family_checks.append(dict(fold=fold, candidate_id=int(choice.candidate_id),
                                          mae=float(choice[f"fold_{fold}_mae"])))
                table = predictions[int(choice.candidate_id)]
                family_heldout.append(table[(table.exercise == exercise) & (table.fold == fold)])
            families[family] = dict(selected_mae=float(family_best.macro_fold_mae),
                                     heldout_mae=float(np.mean([c["mae"] for c in family_checks])),
                                     settings=asdict(settings[family_id]), checks=family_checks,
                                     heldout_timing=timing_metrics(pd.concat(family_heldout)))
            if family == "guided":
                guided_settings[exercise] = settings[family_id]
                guided_tables.append(family_selected)
                guided_heldout_tables.extend(family_heldout)
        reports[exercise]["families"] = families
    # Ensure the public experimental API and cached validation agree on every segment.
    selected = pd.concat(selected_tables)
    guided = pd.concat(guided_tables)
    spacing = {e: dict(runs=0, period_estimated=0, spacing_increased=0) for e in reports}
    for s in samples:
        for table, configuration in ((selected, selected_settings), (guided, guided_settings)):
            actual = count_peaks_for_exercise(s["frame"], s["exercise"], s["start"], s["end"],
                                              settings=configuration[s["exercise"]])
            row = table[(table.video_id == s["video_id"]) & (table.segment_index == s["segment_index"])].iloc[0]
            if actual["rep_times"] != row.rep_times:
                raise ValueError(f"Runtime disagrees with experiment: {s['video_id']}:{s['segment_index']}")
            if table is guided:
                for run in actual["runs"]:
                    stats = spacing[s["exercise"]]
                    stats["runs"] += 1
                    stats["period_estimated"] += int(run["period_s"] is not None)
                    stats["spacing_increased"] += int(run["distance_s"] > counter.EXERCISES[s["exercise"]]["min_gap_s"] + 1e-9)
    report = dict(run_id="v2", split="train", exercises=reports, heldout_checks=checks,
                  candidate_count_per_exercise=len(settings), runtime_verified_segments=len(samples),
                  counter_source_sha256=sha256(Path(counter.__file__)),
                  peak_source_sha256=sha256(Path(__file__).with_name("peak_counter.py")),
                  arm_quality_sha256=sha256(Path(__file__).with_name("arm_quality.py")), inputs=inputs,
                  validation_source_sha256=sha256(Path(__file__)), guided_spacing_diagnostics=spacing,
                  timing_tolerance_s=.3, selection="Equal source-clip weight within fold; equal fold weight; MAE only",
                  timing_aggregation="Pooled annotated training segment occurrences, including source reuse",
                  limitations=["GT exercise windows isolate counting; no test feature frames loaded.",
                               "Exploratory development, not an unbiased nested-CV estimate.",
                               "The existing baseline arm-selection settings were developed on training data.",
                               "No automatic replacement of the current counter."])
    output.mkdir(parents=True)
    scores.to_csv(output / "candidates.csv", index=False)
    baseline.to_csv(output / "baseline_segments.csv", index=False)
    selected.to_csv(output / "selected_peak_segments.csv", index=False)
    guided.to_csv(output / "selected_guided_segments.csv", index=False)
    pd.concat(guided_heldout_tables).to_csv(output / "heldout_guided_segments.csv", index=False)
    pd.concat(heldout_tables).to_csv(output / "heldout_peak_segments.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_results(output, samples, guided, reports, guided_settings, plots)
    return report


def make_row(sample, times):
    return {k: sample[k] for k in ("video_id", "fold", "clip_id", "segment_index", "exercise", "annotated")} | dict(
        predicted=len(times), error=len(times) - sample["annotated"],
        absolute_error=abs(len(times) - sample["annotated"]),
        hits=match_intervals(times, sample["pairs"]), rep_times=times)


def plot_results(output, samples, selected, reports, settings, limit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exercises = sorted(reports)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    x = np.arange(len(exercises))
    ax.bar(x - .25, [reports[e]["baseline_mae"] for e in exercises], .25, label="Current state machine")
    ax.bar(x, [reports[e]["families"]["guided"]["heldout_mae"] for e in exercises], .25, label="Guided peaks, other-fold selection")
    ax.bar(x + .25, [reports[e]["families"]["unguided"]["heldout_mae"] for e in exercises], .25, label="Unguided peaks, other-fold selection")
    ax.set(xticks=x, xticklabels=exercises, ylabel="MAE (reps / source clip)", title="Training-fold alternative counter comparison")
    ax.legend(); fig.savefig(output / "comparison.png", dpi=150); plt.close(fig)
    # Select one largest gain and one largest regression per exercise.
    example_rows = []
    for exercise in exercises:
        choices = []
        for s in samples:
            if s["exercise"] != exercise:
                continue
            row = selected[(selected.video_id == s["video_id"]) & (selected.segment_index == s["segment_index"])].iloc[0]
            gain = abs(len(s["baseline_times"]) - s["annotated"]) - row.absolute_error
            choices.append((gain, s, row))
        choices.sort(key=lambda item: item[0])
        example_rows.extend([choices[-1], choices[0]])
    for gain, s, row in example_rows[:limit]:
        if s["chosen"] is None:
            continue
        cfg = counter.EXERCISES[s["exercise"]]
        detected = detect_peaks(s["chosen"]["values"], s["t"], cfg["pattern"], cfg["min_gap_s"], settings[s["exercise"]])
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout="constrained")
        axes[0].plot(s["t"], s["chosen"]["values"], label=s["chosen"]["signal"])
        for peak in detected["peak_times"]: axes[0].axvline(peak, alpha=.25, color="orange")
        axes[0].set(ylabel="Angle (degrees)", title=f"{s['exercise']} / {s['video_id']} / {s['clip_id']}\nAnnotated {s['annotated']}, current {len(s['baseline_times'])}, peaks {row.predicted}")
        for i, (a, b) in enumerate(s["pairs"]):
            axes[1].axvspan(a, b, alpha=.15, color="green", label="Annotated rep interval" if i == 0 else None)
        for times, y, label in [(s["baseline_times"], .7, "Current completions"), (detected["rep_times"], .3, "Peak recovery completions")]:
            axes[1].scatter(times, np.full(len(times), y), marker="|", s=160, label=label)
        axes[1].set(xlabel="Absolute video time (seconds)", yticks=[], ylim=(0, 1))
        for ax in axes: ax.legend(loc="upper right", fontsize=8)
        fig.savefig(output / f"{s['exercise']}_{s['video_id']}_{s['segment_index']:02d}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plots", type=int, default=6)
    args = parser.parse_args()
    result = validate(args.out_dir, args.plots)
    print(json.dumps(result["exercises"], indent=2))
