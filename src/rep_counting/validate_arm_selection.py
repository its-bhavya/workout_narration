"""Training-fold validation and diagnostic plots for push-up arm selection."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import catalog, feature_frame, FEATURES_DIR, sha256
from . import rep_counter as counter
from .arm_quality import quality_score


def choose(candidates, weights=None, margin=0.):
    usable = [c for c in candidates if c["usable"]]
    baseline = usable[0] if usable else candidates[0]
    if not weights or not usable:
        return baseline
    best = max(usable, key=lambda c: quality_score(c["quality"], weights))
    return best if quality_score(best["quality"], weights) > quality_score(baseline["quality"], weights) + margin else baseline


def validation(output, plots=8):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new validation output directory")
    cfg = counter.EXERCISES["push_up"]
    samples, inputs = [], []
    for vid, meta in sorted(catalog().items()):
        if meta["split"] != "train":
            continue
        path = FEATURES_DIR / f"{vid}_angles.csv"
        frame = feature_frame(path, meta)
        inputs.append({"video_id": vid, "feature_sha256": sha256(path)})
        for index, segment in enumerate(meta["segments"]):
            if segment.get("label") != "push_up":
                continue
            start, end = (segment[k] / meta["fps"] for k in ("start_frame", "end_frame"))
            sliced = frame[(frame.time_s >= start) & (frame.time_s < end)]
            if len(sliced) < 8:
                raise ValueError(f"Too-short training segment: {vid}:{index}")
            candidates = counter.prepare_candidates(sliced, cfg, arm_selection="visibility")
            t = sliced.time_s.to_numpy()
            fs = counter.sampling_rate(t)
            for c in candidates:
                c["normalized"] = counter.local_motion_signal(c["values"], fs, cfg["adaptive_window_s"], cfg["min_range_deg"])
                indices = counter.run_state_machine(c["normalized"], *cfg["fracs"], cfg["pattern"], fs, cfg["confirm_s"])
                c["rep_times"] = counter.drop_close([float(t[i]) for i in indices], cfg["min_gap_s"]) if c["usable"] else []
            samples.append(dict(video_id=vid, fold=meta["fold"], segment_index=index,
                                clip_id=segment["clip_id"], annotated=int(segment["rep_count_annotated"]),
                                pairs=[(a / meta["fps"], b / meta["fps"]) for a, b in segment.get("rep_pairs_out", [])],
                                t=t, frame=sliced, start=start, end=end, candidates=candidates))
    policies = [(None, 0.)] + [(dict(visibility=2., continuity=1., periodicity=p), margin)
                               for p, margin in itertools.product((.5, 1., 2., 4.), (0., .1, .25, .5, 1., 2.))]
    policies_table, details = [], []
    for weights, margin in policies:
        rows = []
        for s in samples:
            c = choose(s["candidates"], weights, margin)
            predicted = len(c["rep_times"])
            rows.append({k: s[k] for k in ("video_id", "fold", "segment_index", "clip_id", "annotated")}
                        | dict(signal=c["signal"], predicted=predicted, error=predicted - s["annotated"],
                               absolute_error=abs(predicted - s["annotated"])))
        table = pd.DataFrame(rows)
        folds = table.groupby(["fold", "clip_id"]).absolute_error.mean().groupby("fold").mean()
        policies_table.append(dict(weights=weights, margin=margin, macro_fold_mae=float(folds.mean()),
                                   **{f"fold_{f}_mae": float(e) for f, e in folds.items()}))
        details.append(table)
    scores = pd.DataFrame(policies_table)
    best = int(scores.macro_fold_mae.idxmin())
    checks = []
    heldout_tables = []
    for fold in range(5):
        columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
        chosen = int(scores[columns].mean(axis=1).idxmin())
        checks.append(dict(fold=fold, selected_policy=policies_table[chosen],
                           baseline_mae=policies_table[0][f"fold_{fold}_mae"],
                           heldout_mae=policies_table[chosen][f"fold_{fold}_mae"]))
        heldout_tables.append(details[chosen][details[chosen].fold == fold])
    selected = details[best]
    comparison = details[0].merge(selected, on=["video_id", "fold", "segment_index", "clip_id", "annotated"],
                                  suffixes=("_baseline", "_selected"))
    # Verify the deployed runtime agrees with the selected policy before reporting it.
    if policies_table[best]["weights"] != cfg.get("arm_quality_weights") or policies_table[best]["margin"] != 0.:
        raise ValueError(f"Selected policy differs from runtime configuration: {policies_table[best]}")
    for s in samples:
        for mode, weights in (("visibility", None), ("motion", policies_table[best]["weights"])):
            expected = choose(s["candidates"], weights)
            actual = counter.count_for_exercise(s["frame"], "push_up", s["start"], s["end"], arm_selection=mode)
            if actual["reps"] != len(expected["rep_times"]) or (expected["usable"] and actual["signal"] != expected["signal"]):
                raise ValueError(f"Runtime disagrees with validation: {s['video_id']}:{s['segment_index']}:{mode}")
    report = dict(run_id="v2", split="train", n_segments=len(samples),
                  n_source_clips=selected.clip_id.nunique(), baseline=policies_table[0],
                  selected=policies_table[best], heldout_checks=checks,
                  heldout_baseline_mae=float(np.mean([c["baseline_mae"] for c in checks])),
                  heldout_selected_mae=float(np.mean([c["heldout_mae"] for c in checks])),
                  inputs=inputs, counter_source_sha256=sha256(Path(counter.__file__)),
                  quality_source_sha256=sha256(Path(__file__).with_name("arm_quality.py")),
                  runtime_verified_segments=len(samples),
                  limitations=["Training-only exploratory algorithm development, not an unbiased nested-CV estimate.",
                               "GT exercise windows; segmentation is not evaluated here.",
                               "The previous test set was not rerun or used to select this policy."])
    output.mkdir(parents=True)
    scores.to_csv(output / "policies.csv", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    pd.concat(heldout_tables).to_csv(output / "heldout_predictions.csv", index=False)
    quality_rows = [{k: s[k] for k in ("video_id", "fold", "segment_index", "clip_id", "annotated")}
                    | {k: c[k] for k in ("signal", "usable", "quality_score")}
                    | c["quality"] | {"predicted": len(c["rep_times"])}
                    for s in samples for c in s["candidates"]]
    pd.DataFrame(quality_rows).to_csv(output / "arm_candidates.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_diagnostics(output, samples, comparison, checks, policies[best], plots)
    return report


def plot_diagnostics(output, samples, comparison, checks, policy, limit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    x = np.arange(5)
    ax.bar(x - .18, [c["baseline_mae"] for c in checks], .36, label="Visibility baseline", color="#8a99aa")
    ax.bar(x + .18, [c["heldout_mae"] for c in checks], .36, label="Selected on other four folds", color="#167d9a")
    ax.set(xticks=x, xlabel="Held-out source fold", ylabel="MAE (reps / source clip)",
           title="Push-up arm selection: training-fold validation")
    ax.legend(); fig.savefig(output / "fold_validation.png", dpi=150); plt.close(fig)
    # Show improvements, regressions, and the largest remaining errors.
    frame = comparison.copy()
    frame["gain"] = frame.absolute_error_baseline - frame.absolute_error_selected
    selected_rows = pd.concat([frame.sort_values("gain", ascending=False).head(max(1, limit // 2)),
                               frame.sort_values("gain").head(max(1, limit // 4)),
                               frame.sort_values("absolute_error_selected", ascending=False).head(limit)]).drop_duplicates(
                                   ["video_id", "segment_index"]).head(limit)
    lookup = {(s["video_id"], s["segment_index"]): s for s in samples}
    index = []
    for row in selected_rows.to_dict("records"):
        s = lookup[row["video_id"], row["segment_index"]]
        chosen = choose(s["candidates"], *policy)
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout="constrained")
        for c in s["candidates"]:
            label = f"{c['signal']}: {len(c['rep_times'])} reps, quality={c['quality_score']:.2f}"
            axes[0].plot(s["t"], c["values"], label=label, alpha=.8)
            axes[1].plot(s["t"], c["normalized"], label=c["signal"], alpha=.8)
        for threshold in counter.EXERCISES["push_up"]["fracs"]:
            axes[1].axhline(threshold, color="black", linestyle="--", linewidth=.8)
        for i, (a, b) in enumerate(s["pairs"]):
            axes[2].axvspan(a, b, color="#167d9a", alpha=.15, label="Annotated rep interval" if i == 0 else None)
            axes[2].axvline(b, color="#167d9a", alpha=.6, linewidth=.7)
        base = choose(s["candidates"])
        axes[2].scatter(base["rep_times"], np.full(len(base["rep_times"]), .65), marker="|", s=180, label="Visibility completions")
        axes[2].scatter(chosen["rep_times"], np.full(len(chosen["rep_times"]), .35), marker="|", s=180, label="Motion-selected completions")
        axes[0].set(ylabel="Elbow angle (degrees)", title=f"{s['video_id']} / {s['clip_id']} / fold {s['fold']}\nAnnotated {s['annotated']}; baseline {row['predicted_baseline']}; selected {row['predicted_selected']} ({chosen['signal']})")
        axes[1].set(ylabel="Local normalized angle")
        axes[2].set(ylabel="Rep alignment", xlabel="Absolute video time (seconds)", yticks=[], ylim=(0, 1))
        for ax in axes: ax.legend(loc="upper right", fontsize=8)
        filename = f"{s['video_id']}_segment_{s['segment_index']:02d}.png"
        fig.savefig(output / filename, dpi=130); plt.close(fig)
        index.append(dict(file=filename, video_id=s["video_id"], clip_id=s["clip_id"],
                          gain=row["gain"], annotated=s["annotated"], selected_signal=chosen["signal"]))
    (output / "plots.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plots", type=int, default=8)
    args = parser.parse_args()
    report = validation(args.out_dir, args.plots)
    print(json.dumps({k: report[k] for k in ("baseline", "selected", "heldout_baseline_mae", "heldout_selected_mae")}, indent=2))
