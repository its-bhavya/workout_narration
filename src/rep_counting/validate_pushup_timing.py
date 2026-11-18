"""Training-only checks of conservative push-up spacing, smoothing and recovery."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import FEATURES_DIR, SYNTH_DIR, sha256
from . import rep_counter as counter
from .analyze_peak_errors import load_training
from .peak_improvements import build_signals
from .peak_counter import PeakSettings, detect_peaks
from .validate_peak_counter import make_row, fold_mae, timing_metrics


def validate(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve previous results; use a new output directory")
    samples, inputs, cadence = [], [], []
    for vid, meta, frame, hashes, _ in load_training():
        inputs.append(dict(video_id=vid, **hashes))
        for index, segment in enumerate(meta["segments"]):
            if segment.get("label") != "push_up":
                continue
            start, end = (segment[k] / meta["fps"] for k in ("start_frame", "end_frame"))
            sliced = frame[(frame.time_s >= start) & (frame.time_s < end)]
            pairs = [(a / meta["fps"], b / meta["fps"]) for a, b in segment["rep_pairs_out"]]
            times = [b for _, b in pairs]
            diffs = np.diff(times)
            sample = dict(video_id=vid, fold=meta["fold"], segment_index=index,
                          exercise="push_up", clip_id=segment["clip_id"], annotated=len(pairs), pairs=pairs,
                          t=sliced.time_s.to_numpy(), signals={})
            for smoothing in (.3, .4, .5, .6):
                sample["signals"][smoothing] = build_signals(sliced, "push_up", smoothing)["baseline"]
            samples.append(sample)
            cadence.append(dict(video_id=vid, fold=meta["fold"], clip_id=segment["clip_id"], annotated=len(times),
                                 intervals=len(diffs), under_0p7=int((diffs < .7).sum()), under_1p0=int((diffs < 1.).sum()),
                                 suppressed_0p7=len(times) - len(counter.drop_close(times, .7)),
                                 suppressed_1p0=len(times) - len(counter.drop_close(times, 1.))))
    print(f"Prepared {len(samples)} training push-up segments; test features not loaded", flush=True)
    configurations = [dict(min_gap_s=g, smooth_s=s, prominence_fraction=p, recovery_fraction=r)
                      for g, s, p, r in itertools.product((.2, .35, .5, .7, 1.), (.3, .4, .5, .6), (.25, .35, .45), (.5, .7))]
    predictions, scores = [], []
    for index, config in enumerate(configurations):
        settings = PeakSettings(prominence_fraction=config["prominence_fraction"], min_prominence_deg=.1,
                                 min_width_s=1 / 15, period_distance_fraction=0., recovery_fraction=config["recovery_fraction"])
        rows = []
        for sample in samples:
            detected = detect_peaks(sample["signals"][config["smooth_s"]], sample["t"], "down_up", config["min_gap_s"], settings)
            rows.append(make_row(sample, detected["rep_times"]))
        table = pd.DataFrame(rows)
        folds = fold_mae(table)
        scores.append(dict(candidate_id=index, **config, mae=float(folds.mean()),
                           **{f"fold_{f}_mae": float(e) for f, e in folds.items()}))
        predictions.append(table)
    score_frame = pd.DataFrame(scores)
    baseline_config = dict(min_gap_s=.2, smooth_s=.3, prominence_fraction=.45, recovery_fraction=.7)
    baseline_index = configurations.index(baseline_config)
    baseline = predictions[baseline_index]

    def select(subset):
        heldout, checks = [], []
        for fold in range(5):
            columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
            index = int(subset[columns].mean(axis=1).idxmin())
            heldout.append(predictions[index][predictions[index].fold == fold])
            checks.append(dict(fold=fold, settings=configurations[index], mae=scores[index][f"fold_{fold}_mae"]))
        heldout = pd.concat(heldout)
        return dict(mae=float(fold_mae(heldout).mean()), timing=timing_metrics(heldout), checks=checks), heldout

    selected, heldout = select(score_frame)
    spacing = {str(gap): select(group)[0] for gap, group in score_frame.groupby("min_gap_s")}
    grouped_cadence = pd.DataFrame(cadence).groupby(["fold", "clip_id"])[["annotated", "intervals", "under_0p7", "under_1p0", "suppressed_0p7", "suppressed_1p0"]].mean().sum()
    report = dict(run_id="v2", split="train", n_segments=len(samples), n_source_clips=len({s["clip_id"] for s in samples}),
                  frozen_baseline=dict(settings=baseline_config, mae=float(fold_mae(baseline).mean()), timing=timing_metrics(baseline)),
                  selected_other_folds=selected, by_spacing=spacing, annotation_cadence=grouped_cadence.to_dict(),
                  source_hashes={name: sha256(Path(__file__).with_name(name)) for name in (
                      "validate_pushup_timing.py", "peak_improvements.py", "peak_counter.py", "rep_counter.py")}, inputs=inputs,
                  limitations=["No test feature frames loaded; no parameters promoted or retested on test.",
                               "Training annotations describe video timing, not a universal physiological rate.",
                               "Exploratory development on existing training folds, not unbiased nested CV.",
                               "Input signal remains the motion-aware selected elbow; new coordinate signals are not included.",
                               "Saved baseline parameters were previously selected on training data; test diagnosis is not used for fitting."])
    for item in inputs:
        vid = item["video_id"]
        if sha256(FEATURES_DIR / f"{vid}_angles.csv") != item["features_sha256"] or sha256(SYNTH_DIR / "train" / f"{vid}.json") != item["annotations_sha256"]:
            raise ValueError("Training data changed")
    report["inputs_verified_unchanged"] = True
    output.mkdir(parents=True)
    score_frame.to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(cadence).to_csv(output / "training_cadence.csv", index=False)
    heldout.to_csv(output / "heldout_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.out_dir)
    print(json.dumps(dict(baseline=result["frozen_baseline"], selected=result["selected_other_folds"],
                         by_spacing={k: v["mae"] for k, v in result["by_spacing"].items()}), indent=2))
