"""Read-only error analysis of other-fold-selected 2D peaks without autocorrelation."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import ROOT, FEATURES_DIR, SYNTH_DIR, catalog, feature_frame, sha256
from . import rep_counter as counter
from .peak_counter import PeakSettings, detect_peaks
from .validate_peak_counter import fold_mae


REFERENCE = ROOT / "docs/results/peak_counter_20261117"


def assign_reps(times, intervals, tolerance=.3):
    """Earliest-deadline one-to-one interval matching, retaining original indices."""
    matches, used = [], set()
    for gi, (start, end) in sorted(enumerate(intervals), key=lambda x: x[1][1]):
        for pi, point in sorted(enumerate(times), key=lambda x: x[1]):
            if pi not in used and start - tolerance <= point <= end + tolerance:
                used.add(pi)
                matches.append((gi, pi))
                break
    matched_gt = {g for g, _ in matches}
    return matches, [i for i in range(len(intervals)) if i not in matched_gt], [i for i in range(len(times)) if i not in used]


def position_bin(time, start, end):
    position = (time - start) / (end - start)
    return "first_10_percent" if position < .1 else "last_10_percent" if position >= .9 else "middle_80_percent"


def count_statistics(rows):
    error = rows["error"]
    return dict(n=len(rows), annotated=int(rows.annotated.sum()), predicted=int(rows.predicted.sum()),
                mae=float(error.abs().mean()), bias=float(error.mean()),
                overcount_fraction=float(error.gt(0).mean()), undercount_fraction=float(error.lt(0).mean()),
                exact_fraction=float(error.eq(0).mean()), within_one_fraction=float(error.abs().le(1).mean()))


def load_training():
    """Validate reference inputs and yield train frames only; never write to data."""
    report = json.loads((REFERENCE / "report.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(REFERENCE / "candidates.csv")
    expected = {r["video_id"]: r for r in report["inputs"]}
    settings = {}
    for exercise, metrics in report["exercises"].items():
        for check in metrics["families"]["unguided"]["checks"]:
            row = candidates[(candidates.exercise == exercise) & (candidates.candidate_id == check["candidate_id"])].iloc[0]
            settings[exercise, check["fold"]] = PeakSettings(**{f.name: float(row[f.name]) for f in fields(PeakSettings)})
    for video_id, meta in sorted(catalog().items()):
        if meta["split"] != "train":
            continue
        path = FEATURES_DIR / f"{video_id}_angles.csv"
        annotation = SYNTH_DIR / "train" / f"{video_id}.json"
        hashes = dict(features_sha256=sha256(path), annotations_sha256=sha256(annotation))
        if video_id not in expected or any(hashes[k] != expected[video_id][k] for k in hashes):
            raise ValueError(f"Reference training inputs changed: {video_id}")
        yield video_id, meta, feature_frame(path, meta), hashes, settings


def analyze(output, plots=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    segments, events, annotated_events, provenance, metadata = [], [], [], [], {}
    for vid, meta, frame, hashes, settings in load_training():
        metadata[vid] = meta
        provenance.append(dict(video_id=vid, **hashes))
        for index, segment in enumerate(meta["segments"]):
            if segment["type"] != "exercise":
                continue
            exercise = segment["label"]
            start, end = (segment[k] / meta["fps"] for k in ("start_frame", "end_frame"))
            sliced = frame[(frame.time_s >= start) & (frame.time_s < end)]
            cfg = counter.EXERCISES[exercise]
            candidates = counter.prepare_candidates(sliced, cfg) if len(sliced) >= 8 else []
            chosen = next((c for c in candidates if c["usable"]), None)
            t = sliced.time_s.to_numpy()
            detected = detect_peaks(chosen["values"], t, cfg["pattern"], cfg["min_gap_s"], settings[exercise, meta["fold"]]) if chosen else dict(rep_times=[], peak_times=[], runs=[])
            times = detected["rep_times"]
            opposite = detect_peaks(chosen["values"], t, "low_high_low" if cfg["pattern"] == "down_up" else "down_up",
                                    cfg["min_gap_s"], settings[exercise, meta["fold"]]) if chosen else dict(rep_times=[])
            pairs = [(a / meta["fps"], b / meta["fps"]) for a, b in segment["rep_pairs_out"]]
            if len(pairs) != segment["rep_count_annotated"]:
                raise ValueError("Annotated count and intervals disagree")
            matches, missed, extra = assign_reps(times, pairs)
            base = dict(video_id=vid, fold=meta["fold"], exercise=exercise, segment_index=index, clip_id=segment["clip_id"],
                        start_s=start, end_s=end)
            row = base | dict(annotated=len(pairs), predicted=len(times), error=len(times) - len(pairs),
                               absolute_error=abs(len(times) - len(pairs)), hits=len(matches),
                               missed=len(missed), extra=len(extra), rep_times=times,
                               signal=chosen["signal"] if chosen else "", usable_signal=chosen is not None,
                               valid_fraction=chosen["valid_fraction"] if chosen else 0.,
                               opposite_phase_reps=len(opposite["rep_times"]),
                               phase_count_disagreement=abs(len(times) - len(opposite["rep_times"])))
            segments.append(row)
            matched_gt = {g for g, _ in matches}
            for i, (a, b) in enumerate(pairs):
                selected = (t >= a) & (t <= b)
                coverage = float(np.isfinite(chosen["values"][selected]).mean()) if chosen and selected.any() else 0.
                event = base | dict(kind="missed", time_s=b, position=(b - start) / (end - start),
                                    position_bin=position_bin(b, start, end), first_rep=i == 0, last_rep=i == len(pairs) - 1,
                                    edge_1s=min(a - start, end - b) <= 1., signal_coverage=coverage,
                                    signal_unusable=chosen is None, overlaps_missing=coverage < 1.,
                                    interval_start_s=a, interval_end_s=b, missed=i not in matched_gt)
                annotated_events.append(event)
                if i in missed:
                    events.append(event)
            for i in extra:
                point = times[i]
                events.append(base | dict(kind="extra", time_s=point, position=(point - start) / (end - start),
                    position_bin=position_bin(point, start, end), edge_1s=min(point - start, end - point) <= 1.,
                    signal_unusable=chosen is None))
    rows = pd.DataFrame(segments)
    temporal = pd.DataFrame(events)
    denominators = pd.DataFrame(annotated_events)
    # Every registered training video has an explicit row for all three classes.
    totals = []
    for vid in metadata:
        for exercise in counter.EXERCISES:
            subset = rows[(rows.video_id == vid) & (rows.exercise == exercise)]
            annotated, predicted = int(subset.annotated.sum()), int(subset.predicted.sum())
            totals.append(dict(video_id=vid, exercise=exercise, annotated=annotated,
                               predicted=predicted, error=predicted - annotated))
    by_video_exercise = pd.DataFrame(totals)
    video_totals = by_video_exercise.groupby("video_id")[["annotated", "predicted", "error"]].sum().reset_index()
    report = dict(run_id="v2", split="train", method="2D peaks without autocorrelation; other-fold settings",
                  n_videos=len(metadata), n_segments=len(rows), timing_tolerance_s=.3,
                  video_totals=count_statistics(video_totals), exercises={}, inputs=provenance,
                  source_hashes={name: sha256(Path(__file__).with_name(name)) for name in (
                      "analyze_peak_errors.py", "peak_counter.py", "rep_counter.py", "arm_quality.py")},
                  limitations=["GT exercise windows; video totals do not include segmentation errors.",
                               "Absent exercise classes add exact zero rows; active-only rates are also reported.",
                               "Video/class or total aggregation can hide cancellation of temporal errors.",
                               "Error position and missingness are associations, not proven causal attributions.",
                               "Edge analysis describes existing errors; annotations and counting conventions are unchanged."])
    for exercise, group in rows.groupby("exercise"):
        video = by_video_exercise[by_video_exercise.exercise == exercise]
        error_events = temporal[temporal.exercise == exercise] if len(temporal) else temporal
        reps = denominators[denominators.exercise == exercise]
        bins = {}
        for label, rep_group in reps.groupby("position_bin"):
            bins[label] = dict(annotated=len(rep_group), missed=int(rep_group.missed.sum()),
                               miss_rate=float(rep_group.missed.mean()),
                               extra=int(((error_events.kind == "extra") & (error_events.position_bin == label)).sum()) if len(error_events) else 0)
        missed = reps[reps.missed]
        flagged = group[group.phase_count_disagreement > 1]
        report["exercises"][exercise] = dict(segments=count_statistics(group), source_weighted_fold_mae=float(fold_mae(group).mean()),
            videos=count_statistics(video), active_videos=count_statistics(video[video.annotated > 0]),
            false_negatives=int(group.missed.sum()), false_positives=int(group.extra.sum()),
            positional_errors=bins, unusable_segments=int((~group.usable_signal).sum()),
            first_rep_miss_rate=float(reps[reps.first_rep].missed.mean()),
            last_rep_miss_rate=float(reps[reps.last_rep].missed.mean()),
            interior_rep_miss_rate=float(reps[~reps.first_rep & ~reps.last_rep].missed.mean()),
            missed_with_unusable_signal=int(missed.signal_unusable.sum()),
            missed_overlapping_missing=int(missed.overlaps_missing.sum()),
            phase_disagreement_flagged_segments=len(flagged),
            phase_flag_count_error_gt1_fraction=float(flagged.error.abs().gt(1).mean()) if len(flagged) else None)
    # Reproduce the frozen reference's unguided family before drawing conclusions.
    reference_report = json.loads((REFERENCE / "report.json").read_text(encoding="utf-8"))
    for exercise, metrics in report["exercises"].items():
        expected = reference_report["exercises"][exercise]["families"]["unguided"]
        if not np.isclose(metrics["source_weighted_fold_mae"], expected["heldout_mae"]):
            raise ValueError("Reconstructed unguided baseline differs from reference")
        timing = expected["heldout_timing"]
        if metrics["false_negatives"] != timing["annotated"] - timing["hits"] or metrics["false_positives"] != timing["predicted"] - timing["hits"]:
            raise ValueError("Reconstructed temporal errors differ from reference")
    for item in provenance:
        vid = item["video_id"]
        if sha256(FEATURES_DIR / f"{vid}_angles.csv") != item["features_sha256"] or sha256(SYNTH_DIR / "train" / f"{vid}.json") != item["annotations_sha256"]:
            raise ValueError("Training data changed during analysis")
    report["inputs_verified_unchanged"] = True
    output.mkdir(parents=True)
    rows.to_csv(output / "segments.csv", index=False)
    temporal.to_csv(output / "temporal_errors.csv", index=False)
    denominators.to_csv(output / "annotated_rep_diagnostics.csv", index=False)
    by_video_exercise.to_csv(output / "video_exercise.csv", index=False)
    video_totals.to_csv(output / "video_totals.csv", index=False)
    rows.groupby(["exercise", "fold", "clip_id"])[["error", "absolute_error", "missed", "extra"]].mean().to_csv(output / "source_clips.csv")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if plots:
        plot_analysis(output, report)
    return report


def plot_analysis(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exercises = sorted(report["exercises"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    x = np.arange(3)
    axes[0].bar(x - .18, [report["exercises"][e]["false_negatives"] for e in exercises], .36, label="Missed annotated reps")
    axes[0].bar(x + .18, [report["exercises"][e]["false_positives"] for e in exercises], .36, label="Unmatched detections")
    axes[0].set(xticks=x, xticklabels=exercises, ylabel="Events", title="Temporal errors cannot cancel")
    axes[0].legend(fontsize=8)
    for i, label in enumerate(("first_10_percent", "middle_80_percent", "last_10_percent")):
        values = [100 * report["exercises"][e]["positional_errors"].get(label, {}).get("miss_rate", 0.) for e in exercises]
        axes[1].bar(x + (i - 1) * .25, values, .25, label=label.replace("_", " "))
    axes[1].set(xticks=x, xticklabels=exercises, ylabel="Missed / annotated reps (%)", title="Miss rate by normalized segment position")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "error_analysis.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.out_dir)
    print(json.dumps({"video_totals": result["video_totals"], "exercises": result["exercises"]}, indent=2))
