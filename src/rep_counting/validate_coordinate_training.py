"""Fold-validate a complete training-only coordinate sidecar set."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import itertools
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import ROOT, load_json, sha256, feature_frame
from . import rep_counter as counter
from .extract_coordinate_training import PROTOCOL, signature, verify_case, case_name
from .coordinate_counter import SIGNALS, FAMILIES, QUALITY_WEIGHTS, CoordinatePeakSettings, prepare, detect_all, choose, count_coordinate_peaks
from .peak_improvements import ImprovementSettings, count_improved_peaks
from .validate_peak_counter import make_row, fold_mae, timing_metrics
from .analyze_peak_errors import count_statistics


REFERENCE = ROOT / "outputs/peak_improvements/error_analysis_with_phase_20261118"


def summarize(rows, videos):
    video = rows.groupby("video_id")[["annotated", "predicted", "error"]].sum().reindex(videos, fill_value=0).reset_index()
    return dict(mae=float(fold_mae(rows).mean()), timing=timing_metrics(rows),
                segment_counts=count_statistics(rows), video_counts=count_statistics(video),
                active_video_counts=count_statistics(video[video.annotated > 0]))


def validate(sidecars, output, plots=True):
    sidecars, output = Path(sidecars).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous results; choose a new validation directory")
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Validation artifacts must stay under outputs")
    saved = json.loads((sidecars / "plan.json").read_text(encoding="utf-8"))
    extraction = json.loads((sidecars / "extraction_report.json").read_text(encoding="utf-8"))
    plan, plan_sig = saved["plan"], saved["plan_signature"]
    if plan_sig != signature(plan) or extraction["plan_signature"] != plan_sig or plan["protocol"] != PROTOCOL or plan["split"] != "train":
        raise ValueError("Incompatible or incomplete training sidecar protocol")
    if extraction["n_windows"] != len(plan["cases"]):
        raise ValueError("Incomplete training sidecar set")
    for name, digest in plan["source_hashes"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("Extractor code changed after sidecars were produced")
    for path, digest in plan["protected_hashes"].items():
        if sha256(path) != digest:
            raise ValueError("Extraction source inputs changed")
    previous = pd.read_csv(REFERENCE / "segments.csv")
    previous = previous[previous.exercise == "push_up"].copy()
    previous["rep_times"] = previous.rep_times.map(ast.literal_eval)
    previous_report = json.loads((REFERENCE / "report.json").read_text(encoding="utf-8"))
    previous_inputs = {r["video_id"]: r for r in previous_report["inputs"]}
    protected_results = {str(path): sha256(path) for path in (REFERENCE / "segments.csv", REFERENCE / "report.json")}
    original_defaults = json.dumps(counter.EXERCISES, sort_keys=True)
    samples, coverage, fixed_old, fixed_reextracted, provenance = [], [], [], [], []
    frames = {}
    for case in plan["cases"]:
        if case["split"] != "train":
            raise ValueError("Test sidecars are forbidden in training validation")
        if previous_inputs[case["video_id"]]["features_sha256"] != case["canonical_features_sha256"]:
            raise ValueError("Canonical comparison inputs differ from reference")
        metadata = verify_case(sidecars, case, plan_sig)
        if metadata is None:
            raise ValueError("A required sidecar is missing")
        path = sidecars / f"{case_name(case)}.csv"
        frame = pd.read_csv(path)
        expected = list(range(case["start_frame"] + case["start_frame"] % 2, case["end_frame"], 2))
        if frame.frame.tolist() != expected or not np.allclose(frame.time_s, frame.frame / case["fps"]):
            raise ValueError("Sidecar timestamp/frame alignment differs")
        meta = load_json(case["annotation_path"])
        if meta["split"] != "train" or meta["fold"] != case["fold"]:
            raise ValueError("Sidecar fold metadata mismatch")
        segment = meta["segments"][case["segment_index"]]
        if segment["clip_id"] != case["clip_id"] or segment["label"] != "push_up":
            raise ValueError("Sidecar window identity mismatch")
        pairs = [(a / case["fps"], b / case["fps"]) for a, b in segment["rep_pairs_out"]]
        if len(pairs) != segment["rep_count_annotated"]:
            raise ValueError("Rep intervals and annotated count disagree")
        sample = {k: case[k] for k in ("video_id", "fold", "segment_index", "clip_id")}
        sample.update(exercise="push_up", annotated=len(pairs), pairs=pairs, frame=frame,
                      t=frame.time_s.to_numpy(), prepared={s: prepare(frame, s) for s in (.3, .5)})
        samples.append(sample)
        for name in SIGNALS:
            signal = sample["prepared"][.3][name]
            coverage.append({k: case[k] for k in ("video_id", "fold", "segment_index", "clip_id")} | dict(
                signal=name, raw_coverage=signal["raw_coverage"], cleaned_coverage=signal["cleaned_coverage"],
                usable=signal["usable"], quality_score=signal["score"], n_multi_pose_frames=int(frame.n_poses.gt(1).sum())))
        if case["video_id"] not in frames:
            frames[case["video_id"]] = feature_frame(case["canonical_features_path"], meta)
        canonical = count_improved_peaks(frames[case["video_id"]], "push_up", case["start_frame"] / case["fps"],
            case["end_frame"] / case["fps"], settings=ImprovementSettings(signal="baseline", smooth_s=.3,
                prominence_fraction=.45, min_gap_s=.2))
        fixed_old.append(make_row(sample, canonical["rep_times"]))
        # Hold the complete old arm-ranking/cleaning/counting path fixed while
        # replacing only its angle/visibility inputs with the fresh extraction.
        fresh_input = frame[["time_s", "elbow_L", "elbow_R"]].copy()
        for side in ("L", "R"):
            fresh_input[f"elbow_{side}_vis"] = frame[f"elbow_{side}_confidence"]
        fresh = count_improved_peaks(fresh_input, "push_up", case["start_frame"] / case["fps"],
            case["end_frame"] / case["fps"], settings=ImprovementSettings(signal="baseline", smooth_s=.3,
                prominence_fraction=.45, min_gap_s=.2))
        fixed_reextracted.append(make_row(sample, fresh["rep_times"]))
        provenance.append(dict(sidecar=str(path), csv_sha256=sha256(path)))
    keys = {(s["video_id"], s["segment_index"]) for s in samples}
    if keys != set(zip(previous.video_id, previous.segment_index)):
        raise ValueError("Sidecar set differs from complete reference training push-up windows")
    if pd.DataFrame([{k: s[k] for k in ("clip_id", "fold")} for s in samples]).groupby("clip_id").fold.nunique().gt(1).any():
        raise ValueError("Source overlap between training folds")
    print(f"Validated all {len(samples)} training coordinate windows and source-disjoint folds", flush=True)
    # Compute six signal detections once for each setting; quality ranking never
    # sees annotation counts. Families share the same hyperparameter search.
    hyperparameters = list(itertools.product((.3, .5), (.2, .35, .5), (.25, .35, .45), (.5, .7)))
    records, score_rows, settings_list, fixed_fresh = [], [], [], None
    for smooth, gap, prominence, recovery in hyperparameters:
        per_family = {family: [] for family in FAMILIES}
        settings = CoordinatePeakSettings(smooth_s=smooth, min_gap_s=gap, prominence_fraction=prominence, recovery_fraction=recovery)
        for sample in samples:
            prepared = sample["prepared"][smooth]
            detections = detect_all(prepared, sample["t"], settings)
            for family in FAMILIES:
                times, chosen = choose(prepared, detections, family)
                per_family[family].append(make_row(sample, times) | dict(selected_signal=chosen, family=family))
        for family in FAMILIES:
            table = pd.DataFrame(per_family[family])
            folds = fold_mae(table)
            candidate_id = len(records)
            configuration = CoordinatePeakSettings(signal=family, smooth_s=smooth, min_gap_s=gap,
                                                    prominence_fraction=prominence, recovery_fraction=recovery)
            settings_list.append(configuration)
            score_rows.append(dict(candidate_id=candidate_id, **asdict(configuration), mae=float(folds.mean()),
                                   **{f"fold_{f}_mae": float(v) for f, v in folds.items()}))
            records.append(table)
            if family == "fresh_elbow_quality" and smooth == .3 and gap == .2 and prominence == .45 and recovery == .7:
                fixed_fresh = table
    scores = pd.DataFrame(score_rows)
    videos = sorted({s["video_id"] for s in samples})
    # Reindex against all registered training videos for comparable video rates.
    all_videos = sorted(previous_report["inputs"][i]["video_id"] for i in range(len(previous_report["inputs"])))
    def select(subset):
        heldout, checks = [], []
        for fold in range(5):
            columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
            chosen = int(subset[columns].mean(axis=1).idxmin())
            heldout.append(records[chosen][records[chosen].fold == fold])
            checks.append(dict(fold=fold, settings=asdict(settings_list[chosen]), mae=score_rows[chosen][f"fold_{fold}_mae"]))
        heldout = pd.concat(heldout)
        return summarize(heldout, all_videos), checks, heldout
    overall, checks, heldout = select(scores)
    by_family, family_predictions = {}, []
    for family, subset in scores.groupby("signal"):
        metrics, family_checks, predictions = select(subset)
        by_family[family] = dict(metrics=metrics, checks=family_checks)
        family_predictions.append(predictions)
    best = int(scores.mae.idxmin())
    # Verify the actual opt-in API, rather than only a cached sweep, at the
    # chosen all-training configuration for every sample.
    for sample, expected in zip(samples, records[best].to_dict("records")):
        actual = count_coordinate_peaks(sample["frame"], settings_list[best])
        if actual["rep_times"] != expected["rep_times"] or actual["selected_signal"] != expected["selected_signal"]:
            raise ValueError("Public coordinate API disagrees with validation")
    for path, digest in plan["protected_hashes"].items():
        if sha256(path) != digest:
            raise ValueError("Canonical input changed during validation")
    for item in provenance:
        if sha256(item["sidecar"]) != item["csv_sha256"]:
            raise ValueError("Sidecar changed during validation")
    for path, digest in protected_results.items():
        if sha256(path) != digest:
            raise ValueError("Existing reference result changed")
    if json.dumps(counter.EXERCISES, sort_keys=True) != original_defaults:
        raise ValueError("Production defaults changed")
    report = dict(protocol=PROTOCOL, split="train", n_windows=len(samples), n_source_clips=len({s['clip_id'] for s in samples}),
                  n_active_videos=len(videos), sidecar_plan_signature=plan_sig,
                  previous_other_fold_peak_baseline=summarize(previous, all_videos),
                  canonical_fixed_normalized_elbow=summarize(pd.DataFrame(fixed_old), all_videos),
                  fresh_same_counter_elbow=summarize(pd.DataFrame(fixed_reextracted), all_videos),
                  fresh_fixed_elbow_quality=summarize(fixed_fresh, all_videos),
                  selected_other_folds=overall, selected_all_training=summarize(records[best], all_videos),
                  selected_all_training_settings=asdict(settings_list[best]), checks=checks, by_family=by_family,
                  quality_weights=QUALITY_WEIGHTS, candidate_count=len(records),
                  inputs_verified_unchanged=True, sidecars_verified_unchanged=True, runtime_verified_windows=len(samples),
                  source_hashes={name: sha256(Path(__file__).with_name(name)) for name in (
                      "validate_coordinate_training.py", "coordinate_counter.py", "peak_counter.py", "peak_improvements.py", "arm_quality.py", "rep_counter.py")},
                  environment=dict(python=sys.version, packages={name: version(name) for name in (
                      "numpy", "pandas", "scipy", "mediapipe", "opencv-python")}),
                  sidecar_provenance=provenance,
                  limitations=["All registered training push-up GT windows; test frames/features not loaded.",
                               "New full-frame largest-pose extraction differs from canonical annotation-assisted person tracking.",
                               "Fresh same-counter elbows hold the old arm-ranking/cleaning/counting path fixed; fresh quality elbows additionally change signal-quality ranking.",
                               "Signal quality uses no rep counts or annotations; global family/hyperparameters selected on other folds.",
                               "Median voting uses shoulder, shoulder-to-wrist, and weighted elbow counts, with quality fallback if unusable.",
                               "All processing is offline; fixed median torso scale uses a whole window.",
                               "Exploratory training development after inspected pilots, not unbiased nested CV or final-test performance.",
                               "MAE weights source clips within folds; timing and video metrics pool synthetic occurrences.",
                               "No production defaults, canonical inputs, previous results, or test scores updated."])
    output.mkdir(parents=True)
    scores.to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "signal_coverage.csv", index=False)
    heldout.to_csv(output / "heldout_predictions.csv", index=False)
    pd.concat(family_predictions).to_csv(output / "family_heldout_predictions.csv", index=False)
    records[best].to_csv(output / "selected_all_training_predictions.csv", index=False)
    pd.DataFrame(fixed_old).to_csv(output / "canonical_fixed_elbow_predictions.csv", index=False)
    pd.DataFrame(fixed_reextracted).to_csv(output / "fresh_same_counter_elbow_predictions.csv", index=False)
    fixed_fresh.to_csv(output / "fresh_fixed_elbow_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if plots:
        plot_results(output, report)
    return report


def plot_results(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    entries = [("Previous peaks", report["previous_other_fold_peak_baseline"]),
               ("Canonical fixed elbow", report["canonical_fixed_normalized_elbow"]),
               ("Fresh same counter", report["fresh_same_counter_elbow"]),
               ("Coordinate selection", report["selected_other_folds"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    x = np.arange(len(entries))
    axes[0].bar(x, [value["mae"] for _, value in entries], color=["#8a99aa", "#6a8bad", "#278994", "#c77a27"])
    axes[0].set(ylabel="Source-weighted fold MAE", title="Training push-ups: count error")
    axes[1].bar(x, [100 * value["active_video_counts"]["within_one_fraction"] for _, value in entries])
    axes[1].set(ylabel="Active videos within +/-1 rep (%)", title="Synthetic video/exercise totals")
    for ax in axes:
        ax.set_xticks(x, [name for name, _ in entries], rotation=15, ha="right")
    fig.savefig(output / "comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.sidecars, args.out_dir)
    print(json.dumps({key: result[key] for key in ("previous_other_fold_peak_baseline", "fresh_fixed_elbow_quality", "selected_other_folds", "selected_all_training_settings")}, indent=2))
