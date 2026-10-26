"""
boundary_jump_check.py – Shortcut diagnostic for hard-cut boundary leakage.

WHAT IT DOES
------------
For each frame t in a synthetic video, compute:

    jump(t) = L2 norm of the per-component angle change between frames t-1 and t
              over the vector [knee_best, elbow_best, arm_raise_L, arm_raise_R,
              leg_spread, torso_tilt], skipping NaN components at either frame.
              If fewer than 3 components remain, the frame is skipped.

Then classify every frame into one of three groups:

    (a) boundary  – frame t is at an exercise<->transition or
                    exercise<->exercise boundary (i.e., the segment type/label
                    changes between t-1 and t), OR t is a clip_join_frame.
    (b) clip_join – frame t is a clip_join_frame (subset of (a) for reporting).
    (c) random    – 10× as many frames as (a)+(b), drawn uniformly from inside
                    exercise segments, seeded at 42.

Report median, p90, and sample count per group, per split.  Also report
    ratio_boundary_vs_random  = median(a) / median(c)
    ratio_clipjoin_vs_random  = median(b) / median(c)
A ratio near 1 means no shortcut; clearly above 1 means the cuts leak.

Additionally, report mean absolute per-frame angle-vector change INSIDE
exercise segments vs INSIDE transition segments (>= 15 frames), per exercise.
This tells you how separable the transition class is from exercise by motion.

Output: data/manifests/boundary_jump_check.csv

INTERPRETATION
--------------
If ratio_boundary_vs_random >> 1, the hard cuts produce large angle jumps at
boundaries that a classifier could exploit, undermining the argument that
pose-only features are leakage-free.  Investigate by checking whether the
source clips were normalised to the same FPS and resolution (they should be),
and whether any pre/post-boundary frames are missing or corrupted.

A ratio near 1 confirms that the stitching does NOT give the classifier an
easy "frame looks like a boundary" cue.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pose_utils import angle_vector_jump

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_PATH = PROJECT_ROOT / "data" / "manifests" / "boundary_jump_check.csv"

ANGLE_COLS = [
    "knee_best", "elbow_best",
    "arm_raise_L", "arm_raise_R",
    "leg_spread", "torso_tilt",
]

SEED = 42
RANDOM_MULTIPLIER = 10
MIN_TRANSITION_FRAMES = 15

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_video_data(video_id: str, features_dir: Path) -> pd.DataFrame | None:
    """Load the angles CSV for a video; return None if missing."""
    csv_path = features_dir / f"{video_id}_angles.csv"
    if not csv_path.exists():
        log.warning("Missing angles CSV for %s", video_id)
        return None
    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    missing = [c for c in ANGLE_COLS if c not in df.columns]
    if missing:
        log.warning("Angles CSV for %s missing columns: %s", video_id, missing)
        return None
    return df


def _segment_boundaries(segments: list[dict]) -> set[int]:
    """Return the set of frame indices at which the segment TYPE or LABEL changes.

    Frame t is a boundary frame when the segment containing t differs in type
    or label from the segment containing t-1.  This naturally captures both
    exercise<->transition and exercise<->exercise transitions.
    """
    if not segments:
        return set()

    # Build frame-to-segment-index map.
    # Segments are contiguous and cover [0, n_frames).
    seg_idx_at: dict[int, int] = {}
    for si, seg in enumerate(segments):
        for fr in range(seg["start_frame"], seg["end_frame"]):
            seg_idx_at[fr] = si

    boundaries: set[int] = set()
    all_frames = sorted(seg_idx_at.keys())
    for fr in all_frames:
        if fr == 0:
            continue
        si_prev = seg_idx_at.get(fr - 1)
        si_curr = seg_idx_at.get(fr)
        if si_prev is None or si_curr is None:
            continue
        if si_prev != si_curr:
            boundaries.add(fr)
    return boundaries


def _frames_inside_exercise(segments: list[dict]) -> list[int]:
    """All frame indices that are inside exercise segments."""
    frames = []
    for seg in segments:
        if seg["type"] == "exercise":
            frames.extend(range(seg["start_frame"], seg["end_frame"]))
    return frames


def _compute_jumps(df: pd.DataFrame) -> pd.Series:
    """Compute angle-vector jump for every frame (NaN where not computable)."""
    n = len(df)
    jumps = np.full(n, np.nan, dtype=float)
    vals = df[ANGLE_COLS].to_numpy(dtype=float)

    for t in range(1, n):
        jumps[t] = angle_vector_jump(vals[t - 1], vals[t], min_components=3)

    return pd.Series(jumps, index=df.index, name="jump")


def _near_boundary(frame: int, boundary_set: set[int], radius: int = 2) -> bool:
    return any(abs(frame - b) <= radius for b in boundary_set)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_check(
    synth_dir: Path,
    features_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Run the full boundary jump check and write output_path.

    Returns the summary DataFrame.
    """
    rng = np.random.default_rng(SEED)

    # Collect all (video_id, split, meta, df) tuples
    records: list[dict] = []  # one per group (boundary / clip_join / random), per split

    # For exercise-motion vs transition-motion comparison
    motion_rows: list[dict] = []

    for json_path in sorted(synth_dir.rglob("*.json")):
        with open(json_path) as f:
            meta = json.load(f)

        video_id = meta["video_id"]
        split = meta.get("split", "unknown")
        fps = float(meta["fps"])
        segments = meta.get("segments", [])
        clip_join_set = set(meta.get("clip_join_frames", []))

        df = _load_video_data(video_id, features_dir)
        if df is None:
            continue

        jumps = _compute_jumps(df)
        boundary_set = _segment_boundaries(segments)

        # ----------------------------------------------------------------
        # Exclude ±2 frames around boundaries and clip joins from the
        # subject_jump diagnostic (per spec), but here we use all frames
        # for the jump distribution and just label each frame.
        # ----------------------------------------------------------------

        # (a) boundary frames (exercise<->transition or clip joins)
        all_boundary_frames = boundary_set | clip_join_set

        # (b) clip_join frames
        clip_join_frames_list = sorted(clip_join_set)

        # (c) random frames inside exercise segments
        #     excluding ±2 around all boundaries
        ex_frames = _frames_inside_exercise(segments)
        ex_frames_eligible = [
            f for f in ex_frames
            if not _near_boundary(f, all_boundary_frames, 2)
            and f > 0  # need t-1
            and not np.isnan(jumps.iloc[f])
        ]

        n_target_random = RANDOM_MULTIPLIER * len(all_boundary_frames)
        if ex_frames_eligible:
            n_sample = min(n_target_random, len(ex_frames_eligible))
            random_frames = rng.choice(ex_frames_eligible, size=n_sample, replace=False).tolist()
        else:
            random_frames = []

        def _j(frames):
            return [jumps.iloc[f] for f in frames
                    if 0 < f < len(jumps) and not np.isnan(jumps.iloc[f])]

        boundary_jumps = _j(sorted(all_boundary_frames))
        clipjoin_jumps = _j(clip_join_frames_list)
        random_jumps = _j(random_frames)

        def _stats(vals, group, video_id, split):
            if not vals:
                return
            arr = np.array(vals)
            records.append({
                "video_id": video_id,
                "split": split,
                "group": group,
                "n": len(arr),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
            })

        _stats(boundary_jumps, "boundary", video_id, split)
        _stats(clipjoin_jumps, "clip_join", video_id, split)
        _stats(random_jumps, "random", video_id, split)

        # ----------------------------------------------------------------
        # Motion inside exercise vs transition
        # ----------------------------------------------------------------
        for seg in segments:
            seg_frames = list(range(seg["start_frame"], seg["end_frame"]))
            seg_jumps_vals = [
                jumps.iloc[f] for f in seg_frames
                if 0 < f < len(jumps) and not np.isnan(jumps.iloc[f])
            ]
            if not seg_jumps_vals:
                continue

            seg_type = seg["type"]
            seg_label = seg.get("label", "")

            # Only include transition segments that are long enough
            if seg_type == "transition" and len(seg_frames) < MIN_TRANSITION_FRAMES:
                continue

            motion_rows.append({
                "video_id": video_id,
                "split": split,
                "seg_type": seg_type,
                "exercise": seg_label,
                "n_frames": len(seg_frames),
                "mean_abs_jump": float(np.mean(np.abs(seg_jumps_vals))),
            })

    if not records:
        log.warning("No data collected – are angles CSVs present?")
        return pd.DataFrame()

    rec_df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Aggregate per split
    # -----------------------------------------------------------------------
    agg_rows = []
    for split in rec_df["split"].unique():
        split_df = rec_df[rec_df["split"] == split]

        def _group_stats(group_name):
            sub = split_df[split_df["group"] == group_name]
            if sub.empty:
                return {"n": 0, "median": float("nan"), "p90": float("nan")}
            # weighted by n (each video contributes its own pool)
            all_vals_approx_median = np.average(
                sub["median"].values, weights=sub["n"].values
            )
            # Use simple average of p90s as approximation
            return {
                "n": int(sub["n"].sum()),
                "median": float(all_vals_approx_median),
                "p90": float(np.average(sub["p90"].values, weights=sub["n"].values)),
            }

        b = _group_stats("boundary")
        cj = _group_stats("clip_join")
        r = _group_stats("random")

        ratio_b = b["median"] / r["median"] if r["median"] > 0 else float("nan")
        ratio_cj = cj["median"] / r["median"] if r["median"] > 0 else float("nan")

        agg_rows.append({
            "split": split,
            "boundary_n": b["n"],
            "boundary_median": round(b["median"], 3),
            "boundary_p90": round(b["p90"], 3),
            "clip_join_n": cj["n"],
            "clip_join_median": round(cj["median"], 3),
            "clip_join_p90": round(cj["p90"], 3),
            "random_n": r["n"],
            "random_median": round(r["median"], 3),
            "random_p90": round(r["p90"], 3),
            "ratio_boundary_vs_random": round(ratio_b, 3),
            "ratio_clipjoin_vs_random": round(ratio_cj, 3),
        })

    agg_df = pd.DataFrame(agg_rows)

    # -----------------------------------------------------------------------
    # Motion by class table
    # -----------------------------------------------------------------------
    motion_df = pd.DataFrame(motion_rows)
    motion_summary_rows = []
    if not motion_df.empty:
        for (seg_type, exercise), grp in motion_df.groupby(["seg_type", "exercise"]):
            motion_summary_rows.append({
                "seg_type": seg_type,
                "exercise": exercise,
                "n_segments": len(grp),
                "n_frames_total": int(grp["n_frames"].sum()),
                "mean_abs_jump": round(grp["mean_abs_jump"].mean(), 3),
            })
    motion_summary_df = pd.DataFrame(motion_summary_rows)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write two sections in the same CSV: boundary stats, then motion stats
    with open(output_path, "w", newline="") as f:
        f.write("# Section 1: boundary jump stats per split\n")
    agg_df.to_csv(output_path, mode="a", index=False)

    motion_path = output_path.parent / "boundary_jump_motion.csv"
    if not motion_summary_df.empty:
        motion_summary_df.to_csv(motion_path, index=False)
        log.info("Motion-by-class table saved to %s", motion_path)

    log.info("Boundary jump check saved to %s", output_path)
    return agg_df, motion_summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Boundary jump check – shortcut diagnostic."
    )
    parser.add_argument("--synth-dir", type=Path, default=SYNTH_DIR)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    result = run_check(args.synth_dir, args.features_dir, args.output)
    if isinstance(result, tuple):
        agg_df, motion_df = result
    else:
        agg_df = result
        motion_df = pd.DataFrame()

    print("\n=== Boundary jump ratios per split ===")
    print(agg_df.to_string(index=False))

    if not motion_df.empty:
        print("\n=== Mean abs jump: exercise vs transition ===")
        print(motion_df.to_string(index=False))

    print(f"\nInterpretation: ratio near 1 = no shortcut; >> 1 = boundary cue leaks.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
