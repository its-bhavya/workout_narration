"""Evaluate fixed rep counting on every indexed v2 test video; never tune here."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.pipeline import WorkoutPipeline
from src.rep_counting import rep_counter
from v2_protocol import feature_frame, sha256


def metrics(rows):
    """Count metrics over explicit evaluation units, including zeros."""
    if not rows:
        raise ValueError("No evaluation units")
    errors = [r["predicted"] - r["annotated"] for r in rows]
    total = sum(r["annotated"] for r in rows)
    absolute = sum(abs(e) for e in errors)
    return {"n": len(rows), "annotated_total": total,
            "predicted_total": sum(r["predicted"] for r in rows),
            "mae": absolute / len(rows), "bias": sum(errors) / len(rows),
            "exact_match": sum(e == 0 for e in errors) / len(rows),
            "within_one": sum(abs(e) <= 1 for e in errors) / len(rows),
            "normalized_mae": sum(abs(e) / max(r["annotated"], 1)
                                  for e, r in zip(errors, rows)) / len(rows),
            "wape": absolute / total if total else None}


def grouped(rows):
    return {"overall": metrics(rows), "by_exercise": {
        exercise: metrics([r for r in rows if r["exercise"] == exercise])
        for exercise in sorted({r["exercise"] for r in rows})}}


def evaluate(output_dir, **pipeline_options):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Choose a new output directory to preserve previous results")
    pipeline = WorkoutPipeline(**pipeline_options)
    videos = {vid: meta for vid, meta in pipeline.videos.items() if meta["split"] == "test"}
    if not videos:
        raise ValueError("No indexed test videos")
    oracle, end_to_end, totals, predictions, inputs = [], [], [], [], []
    for vid, meta in sorted(videos.items()):
        print(f"Evaluating {vid}", flush=True)
        result = pipeline.analyze(vid)
        predictions.append(result.to_dict())
        feature_path = pipeline.features_dir / f"{vid}_angles.csv"
        json_path = pipeline.synth_dir / "test" / f"{vid}.json"
        frame = feature_frame(feature_path, meta)
        inputs.append({"video_id": vid, "features_sha256": sha256(feature_path),
                       "annotations_sha256": sha256(json_path)})
        truth = {exercise: 0 for exercise in rep_counter.EXERCISES}
        for idx, segment in enumerate(meta["segments"]):
            if segment["type"] != "exercise":
                continue
            exercise = segment["label"]
            annotated = int(segment["rep_count_annotated"])
            if "rep_pairs_out" in segment and annotated != len(segment["rep_pairs_out"]):
                raise ValueError(f"Inconsistent annotated reps: {vid}, segment {idx}")
            truth[exercise] += annotated
            start, end = (segment[key] / meta["fps"] for key in ("start_frame", "end_frame"))
            counted = rep_counter.count_for_exercise(frame, exercise, start, end, end_pad_s=0.)
            oracle.append({"video_id": vid, "segment_index": idx, "exercise": exercise,
                           "clip_id": segment.get("clip_id"), "start_s": start, "end_s": end,
                           "annotated": annotated, "predicted": counted["reps"],
                           "flag": counted["flag"], "rep_times": counted["rep_times"]})
        for exercise, annotated in truth.items():
            end_to_end.append({"video_id": vid, "exercise": exercise, "annotated": annotated,
                               "predicted": result.summary.exercise_counts.get(exercise, 0)})
        totals.append({"video_id": vid, "annotated": sum(truth.values()),
                       "predicted": result.summary.total_reps})
    report = {"run_id": "v2", "split": "test", "n_videos": len(videos),
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "experiment_id": pipeline.bundle["experiment_id"],
              "model_path": str(pipeline.model_path), "model_sha256": sha256(pipeline.model_path),
              "counter_source_sha256": sha256(Path(rep_counter.__file__)),
              "arm_quality_source_sha256": sha256(Path(rep_counter.__file__).with_name("arm_quality.py")),
              "counter_config": rep_counter.EXERCISES,
              "settings": {"predicted_end_padding_s": pipeline.end_pad_s, "oracle_end_padding_s": 0.,
                           "max_gap_s": rep_counter.MAX_GAP_S,
                           "min_valid_fraction": rep_counter.MIN_VALID_FRACTION,
                           "tau": pipeline.tau, "min_len": pipeline.min_len},
              "parameters_tuned": False,
              "end_to_end_video_exercise": grouped(end_to_end),
              "end_to_end_video_total": metrics(totals),
              "oracle_gt_segment": grouped(oracle), "inputs": inputs,
              "limitations": ["Pose extraction is annotation-assisted upstream.",
                              "Oracle metrics use ground-truth windows, not predicted boundaries.",
                              "Video totals can hide errors cancelling across exercises.",
                              "Test results are reporting only; further tuning requires training data."]}
    output.mkdir(parents=True)
    for name, rows in [("video_exercise", end_to_end), ("video_totals", totals), ("oracle_segments", oracle)]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    report = evaluate(args.out_dir, model_path=args.model)
    print(json.dumps({k: report[k] for k in ("n_videos", "end_to_end_video_exercise",
                                            "end_to_end_video_total", "oracle_gt_segment")}, indent=2))


if __name__ == "__main__":
    main()
