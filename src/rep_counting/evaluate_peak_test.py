"""Freeze training-selected peak variants and report existing synthetic test data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from src.pipeline import WorkoutPipeline
from src.data_pipeline.v2_protocol import ROOT, feature_frame, sha256
from . import rep_counter as counter
from .peak_counter import PeakSettings, count_peaks_for_exercise
from .peak_improvements import ImprovementSettings, count_improved_peaks
from .evaluate_final import metrics, grouped
from .analyze_peak_errors import assign_reps


TRAIN_SELECTION = ROOT / "outputs/peak_improvements/verified_cached_signal_comparison_20261118/report.json"
OLD_SELECTION = ROOT / "docs/results/peak_counter_20261117/report.json"


def freeze_settings(training_path, baseline_path):
    training = json.loads(Path(training_path).read_text(encoding="utf-8"))
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    if training.get("split") != "train" or baseline.get("split") != "train":
        raise ValueError("Peak settings must originate from training-only records")
    for name, digest in training["source_hashes"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Implementation differs from training selection: {name}")
    improved = {ex: ImprovementSettings(**training["exercises"][ex]["selected_settings"]) for ex in counter.EXERCISES}
    old = {ex: PeakSettings(**baseline["exercises"][ex]["families"]["unguided"]["settings"]) for ex in counter.EXERCISES}
    return old, improved


def timing_summary(rows):
    predicted = sum(r["predicted"] for r in rows)
    annotated = sum(r["annotated"] for r in rows)
    hits = sum(r["hits"] for r in rows)
    precision, recall = hits / predicted if predicted else 0., hits / annotated if annotated else 0.
    return dict(predicted=predicted, annotated=annotated, hits=hits, precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.)


def evaluate(output, training_path=TRAIN_SELECTION, baseline_path=OLD_SELECTION, **pipeline_options):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory; preserve previous evaluations")
    old, improved = freeze_settings(training_path, baseline_path)
    pipeline = WorkoutPipeline(**pipeline_options)
    test = {v: m for v, m in pipeline.videos.items() if m["split"] == "test"}
    if not test:
        raise ValueError("No indexed test videos")
    frozen = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), run_id="v2",
                  selection_split="train", parameters_tuned_on_test=False,
                  training_report_sha256=sha256(training_path), baseline_report_sha256=sha256(baseline_path),
                  old_unguided={e: asdict(s) for e, s in old.items()}, proposed={e: asdict(s) for e, s in improved.items()},
                  model_sha256=sha256(pipeline.model_path), experiment_id=pipeline.bundle["experiment_id"],
                  predicted_segment_end_padding_s=pipeline.end_pad_s,
                  oracle_segment_end_padding_s=0.,
                  source_hashes={name: sha256(Path(__file__).with_name(name)) for name in (
                      "evaluate_peak_test.py", "peak_improvements.py", "peak_counter.py", "rep_counter.py", "arm_quality.py")})
    output.mkdir(parents=True)
    # Persist before the first test inference/count: no settings are selected below.
    (output / "frozen_settings.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    counts, totals, oracle, predicted_segments, inputs = [], [], [], [], []
    methods = ("previous_unguided_peaks", "proposed_unguided_peaks")
    for vid, meta in sorted(test.items()):
        print(f"Evaluating {vid}", flush=True)
        path = pipeline.features_dir / f"{vid}_angles.csv"
        annotation = pipeline.synth_dir / "test" / f"{vid}.json"
        inputs.append(dict(video_id=vid, features_sha256=sha256(path), annotations_sha256=sha256(annotation)))
        frame = feature_frame(path, meta)
        result = pipeline.analyze(vid, show_gt=False)
        truth = {e: [] for e in counter.EXERCISES}
        for segment in meta["segments"]:
            if segment["type"] == "exercise":
                pairs = segment.get("rep_pairs_out", [])
                if len(pairs) != segment["rep_count_annotated"]:
                    raise ValueError("Annotation count and rep intervals disagree")
                truth[segment["label"]].extend((a / meta["fps"], b / meta["fps"]) for a, b in pairs)
        for method in methods:
            def count(exercise, start, end, padding=0.):
                if method == "previous_unguided_peaks":
                    return count_peaks_for_exercise(frame, exercise, start, end, settings=old[exercise], end_pad_s=padding)
                return count_improved_peaks(frame, exercise, start, end + padding, settings=improved[exercise])

            times = {e: [] for e in counter.EXERCISES}
            for seg in result.segments:
                if not seg.is_exercise:
                    continue
                counted = count(seg.exercise, seg.start_time, seg.end_time, pipeline.end_pad_s)
                times[seg.exercise].extend(counted["rep_times"])
                predicted_segments.append(dict(video_id=vid, method=method, segment_index=seg.segment_index,
                    exercise=seg.exercise, start_s=seg.start_time, end_s=seg.end_time,
                    predicted=counted["reps"], rep_times=counted["rep_times"], flag=counted["flag"]))
            for exercise in counter.EXERCISES:
                predictions = sorted(times[exercise])
                matches, missed, extra = assign_reps(predictions, truth[exercise])
                counts.append(dict(video_id=vid, method=method, exercise=exercise,
                    annotated=len(truth[exercise]), predicted=len(predictions),
                    error=len(predictions) - len(truth[exercise]), hits=len(matches),
                    missed=len(missed), extra=len(extra), rep_times=predictions))
            totals.append(dict(video_id=vid, method=method, annotated=sum(len(t) for t in truth.values()),
                               predicted=sum(len(t) for t in times.values())))
            for index, segment in enumerate(meta["segments"]):
                if segment["type"] != "exercise":
                    continue
                exercise = segment["label"]
                start, end = (segment[k] / meta["fps"] for k in ("start_frame", "end_frame"))
                counted = count(exercise, start, end)
                pairs = [(a / meta["fps"], b / meta["fps"]) for a, b in segment.get("rep_pairs_out", [])]
                matches, _, _ = assign_reps(counted["rep_times"], pairs)
                oracle.append(dict(video_id=vid, method=method, segment_index=index, exercise=exercise,
                    annotated=len(pairs), predicted=counted["reps"], hits=len(matches),
                    error=counted["reps"] - len(pairs), flag=counted["flag"], rep_times=counted["rep_times"]))
    reports = {}
    for method in methods:
        rows = [r for r in counts if r["method"] == method]
        gt_rows = [r for r in oracle if r["method"] == method]
        active = {e: metrics([r for r in rows if r["exercise"] == e and r["annotated"] > 0])
                  for e in counter.EXERCISES if any(r["exercise"] == e and r["annotated"] > 0 for r in rows)}
        reports[method] = dict(end_to_end_video_exercise=grouped(rows),
                              end_to_end_active_video_exercise=active,
                              end_to_end_video_total=metrics([r for r in totals if r["method"] == method]),
                              end_to_end_timing=timing_summary(rows),
                              timing_by_exercise={e: timing_summary([r for r in rows if r["exercise"] == e]) for e in counter.EXERCISES},
                              oracle_gt_segments=grouped(gt_rows), oracle_timing=timing_summary(gt_rows))
    for item in inputs:
        vid = item["video_id"]
        if sha256(pipeline.features_dir / f"{vid}_angles.csv") != item["features_sha256"] or sha256(pipeline.synth_dir / "test" / f"{vid}.json") != item["annotations_sha256"]:
            raise ValueError("Test data changed during evaluation")
    report = dict(run_id="v2", split="test", n_videos=len(test), frozen_settings=frozen, methods=reports,
                  inputs=inputs, test_inputs_verified_unchanged=True,
                  limitations=["Follow-up on an already inspected test set, not an untouched holdout.",
                               "Both configurations frozen from training records before the first test inference.",
                               "Same frozen segmentation predictions and 0.3 s end padding for both counters.",
                               "No test-based parameter changes or automatic production adoption.",
                               "Video/exercise counts include absent classes; active-only metrics are separate.",
                               "Timing is class-aware one-to-one interval matching with 0.3 s tolerance.",
                               "GT-segment diagnostics are oracle counters, separate from end-to-end results.",
                               "Pose extraction remains annotation-assisted upstream."])
    for name, rows in (("video_exercise", counts), ("video_totals", totals), ("oracle_segments", oracle), ("predicted_segments", predicted_segments)):
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, default=TRAIN_SELECTION)
    parser.add_argument("--baseline-report", type=Path, default=OLD_SELECTION)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    result = evaluate(args.out_dir, args.training_report, args.baseline_report, model_path=args.model)
    print(json.dumps({method: metrics["end_to_end_video_exercise"] for method, metrics in result["methods"].items()}, indent=2))
