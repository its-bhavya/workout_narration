"""
Evaluate rep counting against RepCount's annotated reps, on the synthetic videos.

For every EXERCISE segment of every synthetic video, count reps inside the
ground-truth segment window and compare with the annotation.

This measures the counter ALONE (ground-truth segments). Errors caused by
imperfect segmentation are measured later by feeding predicted segments to
count_for_exercise (use end_pad_s ~0.3 there).

Metrics per exercise
    exact %        counted == annotated
    within-1 %     |counted - annotated| <= 1
    MAE, bias      mean absolute / signed error (counted - annotated)
    precision, recall
                   one-to-one match of counted rep times to annotated reps:
                   a counted rep is a hit if it falls inside an annotated rep
                   interval [start - tol, end + tol]. Catches cases where the
                   count is right by accident (one miss + one false alarm).
                   Does NOT depend on what body position the boundary marks.

Also prints calibration tables (annotated rep durations, angle range of the
real segments) to help set min_gap_s and min_range_deg.

By default only train and val are evaluated; test stays untouched until the
method is final.

Usage
    uv run python src/rep_counting/evaluate_rep_counts.py
    uv run python src/rep_counting/evaluate_rep_counts.py --mode adaptive --fracs 0.3 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rep_counter import EXERCISES, count_for_exercise, load_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def match_reps(
    counted: list[float],
    annotated: list[tuple[float, float]],
    tol: float,
) -> int:
    """Greedy one-to-one matching; returns the number of hits."""

    used: set[int] = set()
    hits = 0

    for start, end in annotated:
        candidates = [
            (abs(t - end), j)
            for j, t in enumerate(counted)
            if j not in used and start - tol <= t <= end + tol
        ]

        if candidates:
            used.add(min(candidates)[1])
            hits += 1

    return hits


def evaluate_video(
    json_path: Path,
    features_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:

    meta = json.loads(json_path.read_text())
    video_id = meta["video_id"]
    fps = float(meta["fps"])

    features_path = features_dir / f"{video_id}_angles.csv"

    if not features_path.exists():
        logger.warning("No features for %s, skipping", video_id)
        return []

    df = load_features(features_path)
    rows = []

    for seg in meta["segments"]:
        if seg.get("type") != "exercise":
            continue

        exercise = seg["label"]

        if "rep_pairs_out" not in seg:
            raise KeyError(
                f"{json_path.name}: segment has no 'rep_pairs_out'. "
                f"Keys: {list(seg)}"
            )

        start_s = seg["start_frame"] / fps
        end_s = seg["end_frame"] / fps

        pairs_s = [(a / fps, b / fps) for a, b in seg["rep_pairs_out"]]
        annotated = len(pairs_s)

        out = count_for_exercise(
            df,
            exercise,
            start_s,
            end_s,
            use_3d=args.use_3d,
            mode=args.mode,
            fracs=tuple(args.fracs) if args.fracs else None,
            end_pad_s=args.pad,
        )

        hits = match_reps(out["rep_times"], pairs_s, args.tol)

        rows.append({
            "video_id": video_id,
            "split": meta.get("split", ""),
            "exercise": exercise,
            "clip_id": seg.get("clip_id", ""),
            "annotated": annotated,
            "counted": out["reps"],
            "error": out["reps"] - annotated,
            "hits": hits,
            "signal": out["signal"],
            "range_deg": out["range_deg"],
            "valid_fraction": out["valid_fraction"],
            "fs": out["fs"],
            "flag": out["flag"],
            "median_rep_s": (
                float(np.median([b - a for a, b in pairs_s]))
                if pairs_s else float("nan")
            ),
        })

    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    def stats(g: pd.DataFrame) -> pd.Series:
        counted = g["counted"].sum()
        annotated = g["annotated"].sum()

        return pd.Series({
            "segments": len(g),
            "exact_%": 100 * (g["error"] == 0).mean(),
            "within1_%": 100 * (g["error"].abs() <= 1).mean(),
            "MAE": g["error"].abs().mean(),
            "bias": g["error"].mean(),
            "precision": g["hits"].sum() / counted if counted else np.nan,
            "recall": g["hits"].sum() / annotated if annotated else np.nan,
            "flagged_%": 100 * (g["flag"] != "").mean(),
        })

    return df.groupby("exercise").apply(stats, include_groups=False).round(2)


def print_calibration(rows: pd.DataFrame, synthetic_dir: Path, splits: list[str]):
    logger.info("\n--- Calibration helpers (ground-truth segments) ---")

    rep_durations: dict[str, list[float]] = {e: [] for e in EXERCISES}

    for split in splits:
        for jp in sorted((synthetic_dir / split).glob("*.json")):
            meta = json.loads(jp.read_text())
            fps = float(meta["fps"])

            for seg in meta["segments"]:
                if seg.get("type") == "exercise":
                    for a, b in seg.get("rep_pairs_out", []):
                        rep_durations[seg["label"]].append((b - a) / fps)

    table = {
        e: pd.Series(d).quantile([0.05, 0.25, 0.5, 0.75]).round(2)
        for e, d in rep_durations.items() if d
    }

    if table:
        logger.info(
            "Annotated rep duration in seconds (set min_gap_s to about half "
            "of the p5):\n%s",
            pd.DataFrame(table).T.rename(
                columns=lambda q: f"p{int(q * 100):02d}"
            ).to_string(),
        )

    ranges = (
        rows[rows["range_deg"].notna()]
        .groupby("exercise")["range_deg"]
        .quantile([0.05, 0.25, 0.5]).unstack().round(1)
    )

    logger.info(
        "Angle range p10..p90 inside real segments (set min_range_deg a "
        "little BELOW the p5):\n%s",
        ranges.rename(columns=lambda q: f"p{int(q * 100):02d}").to_string(),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic"))
    p.add_argument("--features-dir", type=Path, default=Path("data/features"))
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--out", type=Path, default=Path("outputs/rep_eval"))
    p.add_argument("--mode", choices=["adaptive", "fixed"], default=None)
    p.add_argument("--fracs", nargs=2, type=float, default=None)
    p.add_argument("--use-3d", action="store_true")
    p.add_argument("--pad", type=float, default=0.0,
                   help="seconds added to the end of each segment window")
    p.add_argument("--tol", type=float, default=0.3,
                   help="matching tolerance in seconds")
    args = p.parse_args()

    if "test" in args.splits:
        logger.warning("Evaluating on TEST. Only do this for the final numbers.")

    rows: list[dict] = []

    for split in args.splits:
        for jp in sorted((args.synthetic_dir / split).glob("*.json")):
            rows.extend(evaluate_video(jp, args.features_dir, args))

    if not rows:
        raise SystemExit("No segments evaluated. Check the paths.")

    df = pd.DataFrame(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "segments.csv", index=False)

    logger.info("\n=== Rep counting: %s ===", ", ".join(args.splits))
    logger.info("Segments evaluated: %d\n%s",
                len(df), summarise(df).to_string())

    for split, g in df.groupby("split"):
        logger.info("\n[%s]\n%s", split, summarise(g).to_string())

    flagged = df[df["flag"] != ""]
    if len(flagged):
        logger.info("\nFlagged segments by reason:\n%s",
                    flagged.groupby(["exercise", "flag"]).size().to_string())

    worst = df.reindex(df["error"].abs().sort_values(ascending=False).index).head(10)
    logger.info("\nLargest errors (check these in sanity_plots):\n%s",
                worst[["video_id", "exercise", "clip_id", "annotated",
                       "counted", "flag"]].to_string(index=False))

    print_calibration(df, args.synthetic_dir, args.splits)
    logger.info("\nSaved: %s", args.out / "segments.csv")


if __name__ == "__main__":
    main()