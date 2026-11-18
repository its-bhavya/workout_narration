"""Read-only training-data evaluation of segment-local PCA plus peak detection."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import catalog, feature_frame, FEATURES_DIR, SYNTH_DIR, MANIFESTS, ROOT, sha256
from . import rep_counter as counter
from .pca_counter import PCASettings, project_motion, count_projection, count_pca_reps
from .validate_peak_counter import fold_mae, make_row, timing_metrics


REFERENCE_DIR = ROOT / "docs/results/signal_dimensions_20261117"


def summary(rows):
    return dict(mae=float(fold_mae(rows).mean()), timing=timing_metrics(rows),
                exact_match=float(rows.error.eq(0).mean()), within_one=float(rows.error.abs().le(1).mean()))


def validate(output, plots=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory; previous results are preserved")
    original = deepcopy(counter.EXERCISES)
    reference_report_path = REFERENCE_DIR / "report.json"
    reference_predictions_path = REFERENCE_DIR / "heldout_predictions.csv"
    reference_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
    reference = pd.read_csv(reference_predictions_path)
    prior_inputs = {r["video_id"]: r for r in reference_report["inputs"]}
    protected = {str(p): sha256(p) for p in (reference_report_path, reference_predictions_path)}
    for name in ("synthetic_index.csv", "splits_v2.csv", "folds.csv"):
        path = MANIFESTS / name
        protected[str(path)] = sha256(path)
    samples, baseline, reference_rows, projections, inputs = [], [], [], [], []
    for vid, meta in sorted(catalog().items()):
        if meta["split"] != "train":
            continue
        path = FEATURES_DIR / f"{vid}_angles.csv"
        annotation = SYNTH_DIR / "train" / f"{vid}.json"
        feature_hash, annotation_hash = sha256(path), sha256(annotation)
        if vid not in prior_inputs or prior_inputs[vid]["features_sha256"] != feature_hash or prior_inputs[vid]["annotations_sha256"] != annotation_hash:
            raise ValueError("Training inputs differ from the reference experiment")
        protected[str(path)], protected[str(annotation)] = feature_hash, annotation_hash
        inputs.append(dict(video_id=vid, features_sha256=feature_hash, annotations_sha256=annotation_hash))
        frame = feature_frame(path, meta)
        for index, seg in enumerate(meta["segments"]):
            if seg["type"] != "exercise":
                continue
            exercise = seg["label"]
            start, end = (seg[k] / meta["fps"] for k in ("start_frame", "end_frame"))
            sliced = frame[(frame.time_s >= start) & (frame.time_s < end)]
            sample = dict(video_id=vid, fold=meta["fold"], clip_id=seg["clip_id"], segment_index=index,
                          exercise=exercise, annotated=int(seg["rep_count_annotated"]),
                          pairs=[(a / meta["fps"], b / meta["fps"]) for a, b in seg["rep_pairs_out"]],
                          frame=sliced, start=start, end=end, t=sliced.time_s.to_numpy(), projection={})
            if sample["annotated"] != len(sample["pairs"]):
                raise ValueError("Annotated rep count and intervals disagree")
            for dimensions in ("2d", "3d"):
                projection = project_motion(sliced, PCASettings(dimensions=dimensions))
                sample["projection"][dimensions] = projection
                projections.append({k: sample[k] for k in ("video_id", "fold", "segment_index", "exercise", "clip_id")}
                    | {k: v for k, v in projection.items() if k != "signal"} | dict(dimensions=dimensions))
            samples.append(sample)
            current = counter.count_for_exercise(sliced, exercise, start, end)
            baseline.append(make_row(sample, current["rep_times"]))
            method, dimensions = {"squat": ("state_machine", "3d"), "push_up": ("state_machine", "2d"),
                                  "jump_jack": ("peaks", "2d")}[exercise]
            prior = reference[(reference.video_id == vid) & (reference.segment_index == index)
                              & (reference.method == method) & (reference.dimensions == dimensions)]
            if len(prior) != 1 or int(prior.iloc[0].annotated) != sample["annotated"]:
                raise ValueError("Reference predictions are incomplete or incompatible")
            reference_rows.append(prior.iloc[0].to_dict())
    print(f"Prepared {len(samples)} training segments; existing input hashes match", flush=True)
    candidates = [dict(prominence=p, width_s=w, spacing_fraction=d, polarity=sign)
                  for p, w, d, sign in itertools.product((.5, 1., 1.5), (1 / 15, 2 / 15), (0., .35), (1, -1))]
    baseline = pd.DataFrame(baseline)
    reference_rows = pd.DataFrame(reference_rows)
    scores, report, selected_rows, heldout_rows, checks = [], {}, [], [], []
    selected_parameters = {}
    for dimensions in ("2d", "3d"):
        print(f"Evaluating PCA {dimensions}: {len(candidates)} peak settings", flush=True)
        predictions = []
        for candidate_id, parameters in enumerate(candidates):
            rows = []
            for sample in samples:
                projection = sample["projection"][dimensions]
                result = count_projection(projection, sample["t"], sample["exercise"], **parameters)
                rows.append(make_row(sample, result["rep_times"]) | dict(dimensions=dimensions))
            table = pd.DataFrame(rows)
            predictions.append(table)
            for exercise, group in table.groupby("exercise"):
                folds = fold_mae(group)
                scores.append(dict(exercise=exercise, dimensions=dimensions, candidate_id=candidate_id,
                                   parameters=parameters, mae=float(folds.mean()),
                                   **{f"fold_{f}_mae": float(e) for f, e in folds.items()}))
        for exercise in counter.EXERCISES:
            group = pd.DataFrame([r for r in scores if r["exercise"] == exercise and r["dimensions"] == dimensions])
            best = group.loc[group.mae.idxmin()]
            choice = int(best.candidate_id)
            selected_parameters[exercise, dimensions] = candidates[choice]
            selected = predictions[choice]
            selected = selected[selected.exercise == exercise]
            selected_rows.append(selected)
            held = []
            for fold in range(5):
                columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
                chosen = group.loc[group[columns].mean(axis=1).idxmin()]
                prediction = predictions[int(chosen.candidate_id)]
                held.append(prediction[(prediction.exercise == exercise) & (prediction.fold == fold)])
                checks.append(dict(exercise=exercise, dimensions=dimensions, fold=fold,
                                   parameters=candidates[int(chosen.candidate_id)], mae=float(chosen[f"fold_{fold}_mae"])))
            held = pd.concat(held)
            heldout_rows.append(held)
            report.setdefault(exercise, dict(current=summary(baseline[baseline.exercise == exercise]),
                previous_best_development=summary(reference_rows[reference_rows.exercise == exercise])))
            report[exercise][dimensions] = dict(selected_all_train=summary(selected), heldout=summary(held),
                                                selected_peak_parameters=candidates[choice])
    for sample in samples:
        for dimensions in ("2d", "3d"):
            actual = count_pca_reps(sample["frame"], sample["exercise"], sample["start"], sample["end"],
                                   pca_settings=PCASettings(dimensions=dimensions),
                                   **selected_parameters[sample["exercise"], dimensions])
            expected = count_projection(sample["projection"][dimensions], sample["t"], sample["exercise"],
                                       **selected_parameters[sample["exercise"], dimensions])
            if actual["rep_times"] != expected["rep_times"]:
                raise ValueError("Public PCA API disagrees with cached validation")
    for path, digest in protected.items():
        if sha256(path) != digest:
            raise ValueError(f"Existing input changed during experiment: {path}")
    if counter.EXERCISES != original:
        raise ValueError("Production defaults changed during experiment")
    result = dict(run_id="v2", split="train", n_segments=len(samples), exercises=report,
                  settings={d: asdict(PCASettings(dimensions=d)) for d in ("2d", "3d")},
                  candidate_count_per_dimension=len(candidates), checks=checks, inputs=inputs,
                  protected_file_hashes=protected, existing_inputs_verified_unchanged=True,
                  runtime_verified_projections=2 * len(samples),
                  hashes={name: sha256(Path(__file__).with_name(name)) for name in (
                      "pca_counter.py", "validate_pca_counter.py", "rep_counter.py", "peak_counter.py")},
                  limitations=["Per-segment scaling/PCA is offline and fitted without annotation fields.",
                               "GT exercise windows and annotation-assisted upstream pose selection remain.",
                               "Canonical component sign is not anatomical phase; polarity is selected on other folds.",
                               "Mean imputation of remaining cells can introduce missingness-related motion artifacts.",
                               "All-channel PCA may follow posture/camera changes rather than repetitions.",
                               "Exploratory training development, not an unbiased nested-CV or final-test estimate.",
                               "Previous best comparator was chosen from inspected development results.",
                               "MAE source-weights within folds; timing/exact/within-one pool segment occurrences."])
    output.mkdir(parents=True)
    pd.DataFrame(scores).to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(projections).to_csv(output / "projection_diagnostics.csv", index=False)
    baseline.to_csv(output / "current_counter.csv", index=False)
    reference_rows.to_csv(output / "previous_best_development.csv", index=False)
    pd.concat(selected_rows).to_csv(output / "selected_predictions.csv", index=False)
    pd.concat(heldout_rows).to_csv(output / "heldout_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if plots:
        plot_results(output, samples, result, selected_parameters)
    return result


def plot_results(output, samples, report, parameters):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exercises = sorted(report["exercises"])
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    series = [("current", "Current 2D counter"), ("previous_best_development", "Previous best development"),
              ("2d", "PCA 2D + peaks"), ("3d", "PCA 3D + peaks")]
    for i, (key, label) in enumerate(series):
        values = [(report["exercises"][e][key]["heldout"] if key in ("2d", "3d") else report["exercises"][e][key])["mae"] for e in exercises]
        ax.bar(x + (i - 1.5) * .2, values, .2, label=label)
    ax.set(xticks=x, xticklabels=exercises, ylabel="MAE (reps / source clip)", title="PCA + peaks: training-fold comparison")
    ax.legend(fontsize=8); fig.savefig(output / "comparison.png", dpi=150); plt.close(fig)
    # A fixed, non-outcome-selected example per exercise; diagnostics are descriptive.
    for exercise in exercises:
        sample = next(s for s in samples if s["exercise"] == exercise)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained", sharex=True)
        for dimensions, color in (("2d", "tab:blue"), ("3d", "tab:orange")):
            projection = sample["projection"][dimensions]
            axes[0].plot(sample["t"], projection["signal"], color=color,
                         label=f"{dimensions}: PC1 variance {projection['explained_variance']:.1%}")
            detected = count_projection(projection, sample["t"], exercise, **parameters[exercise, dimensions])
            axes[1].scatter(detected["rep_times"], np.full(len(detected["rep_times"]), .3 if dimensions == "2d" else .7),
                            marker="|", s=150, label=f"{dimensions}: {detected['reps']} reps")
            if projection["loadings"]:
                # Show channel motion behind PC1 for the first dimension only.
                if dimensions == "2d":
                    for name, center, scale in zip(projection["channels"], projection["centers"], projection["scales"]):
                        axes[2].plot(sample["t"], (sample["frame"][name] - center) / scale, label=name, alpha=.6)
        for i, (a, b) in enumerate(sample["pairs"]):
            axes[1].axvspan(a, b, color="green", alpha=.12, label="Annotated rep interval" if i == 0 else None)
        axes[0].set(title=f"{exercise} / {sample['video_id']} / annotated {sample['annotated']}", ylabel="Standardized PC1")
        axes[1].set(ylabel="Recovery events", yticks=[], ylim=(0, 1))
        axes[2].set(ylabel="2D standardized channels", xlabel="Absolute video time (seconds)")
        for ax in axes: ax.legend(fontsize=7, loc="upper right")
        fig.savefig(output / f"{exercise}_example.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.out_dir)
    print(json.dumps({e: {"current": v["current"]["mae"], "previous_best": v["previous_best_development"]["mae"],
                         "pca_2d": v["2d"]["heldout"]["mae"], "pca_3d": v["3d"]["heldout"]["mae"]}
                      for e, v in result["exercises"].items()}, indent=2))
