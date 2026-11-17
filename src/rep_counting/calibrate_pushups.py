"""Select push-up settings from source-disjoint v2 training folds only."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import catalog, feature_frame, FEATURES_DIR, sha256
from . import rep_counter as counter


def calibrate(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new calibration output directory")
    cfg = counter.EXERCISES["push_up"]
    segments, provenance = [], []
    for vid, meta in sorted(catalog().items()):
        if meta["split"] != "train":
            continue
        path = FEATURES_DIR / f"{vid}_angles.csv"
        frame = feature_frame(path, meta)
        provenance.append({"video_id": vid, "features_sha256": sha256(path)})
        for i, seg in enumerate(meta["segments"]):
            if seg.get("label") != "push_up":
                continue
            start, end = (seg[k] / meta["fps"] for k in ("start_frame", "end_frame"))
            sliced = frame[(frame.time_s >= start) & (frame.time_s < end)]
            t = sliced.time_s.to_numpy()
            fs = counter.sampling_rate(t)
            signal = None
            if len(sliced) >= 8:
                for _, raw in counter.candidate_signals(sliced, cfg):
                    cleaned = counter.clean_signal(raw, fs, counter.MAX_GAP_S, cfg["smooth_s"])
                    if np.isfinite(cleaned).mean() < counter.MIN_VALID_FRACTION:
                        continue
                    if np.nanpercentile(cleaned, 90) - np.nanpercentile(cleaned, 10) >= cfg["min_range_deg"]:
                        signal = cleaned
                        break
            normalized = {w: counter.local_motion_signal(signal, fs, w, cfg["min_range_deg"])
                          if signal is not None else None for w in (2., 4., 6., 8.)}
            segments.append({"video_id": vid, "fold": meta["fold"], "clip_id": seg["clip_id"],
                             "segment_index": i, "annotated": seg["rep_count_annotated"],
                             "time": t, "fs": fs, "signal": signal, "normalized": normalized})
    candidates = [(0., .25, .55, .1)] + list(itertools.product(
        (2., 4., 6., 8.), (.2, .35, .45), (.55, .65, .75), (0., .1)))
    rows, details = [], []
    for window, low, high, confirm in candidates:
        errors = []
        for s in segments:
            sig = s["normalized"][window] if window else s["signal"]
            n = 0
            if sig is not None:
                a, b = (low, high) if window else counter.adaptive_thresholds(sig, (low, high))
                indices = counter.run_state_machine(sig, a, b, "down_up", s["fs"], confirm)
                n = len(counter.drop_close([float(s["time"][i]) for i in indices], cfg["min_gap_s"]))
            errors.append({k: s[k] for k in ("video_id", "fold", "clip_id", "segment_index", "annotated")}
                          | {"predicted": n, "error": n - s["annotated"], "absolute_error": abs(n - s["annotated"])})
        # Reused source clips get one weight, regardless of synthetic reuse.
        grouped = pd.DataFrame(errors).groupby(["fold", "clip_id"])[["absolute_error", "error"]].mean()
        folds = grouped.groupby("fold").mean()
        row = {"window_s": window, "low_fraction": low, "high_fraction": high, "confirm_s": confirm,
               "macro_fold_mae": float(folds.absolute_error.mean())}
        row.update({f"fold_{f}_mae": float(g.absolute_error) for f, g in folds.iterrows()})
        rows.append(row)
        details.append(errors)
    table = pd.DataFrame(rows)
    # Deterministic tie-breaking follows the candidate order.
    best = int(table.macro_fold_mae.idxmin())
    heldout = []
    for fold in range(5):
        columns = [f"fold_{f}_mae" for f in range(5) if f != fold]
        choice = int(table[columns].mean(axis=1).idxmin())
        heldout.append({"fold": fold, "selected": rows[choice],
                        "baseline_mae": rows[0][f"fold_{fold}_mae"],
                        "selected_mae": rows[choice][f"fold_{fold}_mae"]})
    report = {"run_id": "v2", "split": "train", "n_pushup_segments": len(segments),
              "n_source_clips": len({s["clip_id"] for s in segments}),
              "selection_metric": "MAE averaged within source clip, then within fold, then across folds",
              "baseline": rows[0], "selected": rows[best], "heldout_fold_checks": heldout,
              "heldout_baseline_mae": float(np.mean([r["baseline_mae"] for r in heldout])),
              "heldout_selected_mae": float(np.mean([r["selected_mae"] for r in heldout])),
              "counter_source_sha256": sha256(Path(counter.__file__)), "inputs": provenance,
              "limitations": ["Exploratory training-data algorithm development; not an unbiased nested-CV estimate.",
                              "Ground-truth exercise windows; no test feature frames are loaded."]}
    output.mkdir(parents=True)
    table.to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(details[best]).to_csv(output / "selected_segments.csv", index=False)
    (output / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.out_dir)
    print(json.dumps({k: result[k] for k in ("baseline", "selected", "heldout_baseline_mae",
                                            "heldout_selected_mae")}, indent=2))
