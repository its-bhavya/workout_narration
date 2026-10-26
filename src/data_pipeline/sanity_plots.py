"""
sanity_plots.py – Diagnostic angle plots for a synthetic video.

Usage
-----
    python src/data_pipeline/sanity_plots.py --video-id synth_train_0001

Produces a stacked subplot figure saved to outputs/<video_id>_angles.png.

For each exercise segment the relevant angle is plotted; the figure is shaded
by segment label (green = squat, blue = push_up, orange = jump_jack, grey =
transition).  Each rep pair is drawn as a thin vertical span.  Clip-join
frames are dashed vertical lines.

Peak counts (scipy.signal.find_peaks, prominence-based) are printed next to
the annotated rep count for a quick sanity check.  Per-rep range (max-min) of
the relevant angle is also reported, and median range per segment.

This script is a DIAGNOSTIC for Bhavya to verify the pipeline – it is not
the rep counter (Deepti owns that).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

log = logging.getLogger(__name__)

# Segment-type colours
SEG_COLOURS = {
    "squat": "#b7e4b7",       # light green
    "push_up": "#b7d4f0",     # light blue
    "jump_jack": "#ffd9a0",   # light orange
    "transition": "#e0e0e0",  # light grey
}

# Relevant angle per exercise (for display and peak counting)
EXERCISE_ANGLE = {
    "squat": "knee_best",
    "push_up": "elbow_best",
    "jump_jack": "arm_raise_L",   # arm raise is the primary oscillation
}

# A secondary angle for jump_jack
JJ_SECONDARY = "leg_spread"

# Prominence for peak detection (degrees) – tune per exercise
PEAK_PROMINENCE = {
    "squat": 10.0,
    "push_up": 10.0,
    "jump_jack": 15.0,
}


def _find_json(video_id: str, synth_dir: Path) -> Path | None:
    for p in synth_dir.rglob(f"{video_id}.json"):
        return p
    return None


def _find_csv(video_id: str, features_dir: Path) -> Path | None:
    for p in features_dir.rglob(f"{video_id}_angles.csv"):
        return p
    return None


def _shade_segment(ax, seg: dict, fps: float) -> None:
    label = seg.get("label", "transition")
    colour = SEG_COLOURS.get(label, "#eeeeee")
    t_start = seg["start_frame"] / fps
    t_end = seg["end_frame"] / fps
    ax.axvspan(t_start, t_end, alpha=0.3, color=colour, linewidth=0)


def _draw_rep_spans(ax, seg: dict, fps: float) -> None:
    pairs = seg.get("rep_pairs_out", [])
    for start_f, end_f in pairs:
        t0 = start_f / fps
        t1 = end_f / fps
        ax.axvspan(t0, t1, alpha=0.08, color="black", linewidth=0)


def _draw_clip_joins(ax, clip_join_frames: list[int], fps: float) -> None:
    for f in clip_join_frames:
        ax.axvline(f / fps, color="red", linestyle="--", linewidth=0.8, alpha=0.7)


def plot_video(video_id: str, features_dir: Path, synth_dir: Path, out_dir: Path) -> None:
    json_path = _find_json(video_id, synth_dir)
    if json_path is None:
        raise FileNotFoundError(f"No JSON found for video_id={video_id!r} under {synth_dir}")

    csv_path = _find_csv(video_id, features_dir)
    if csv_path is None:
        raise FileNotFoundError(
            f"No angles CSV for {video_id!r} under {features_dir}.\n"
            "Run extract_pose_features.py first."
        )

    with open(json_path) as f:
        meta = json.load(f)

    fps = float(meta["fps"])
    segments = meta["segments"]
    clip_join_frames = meta.get("clip_join_frames", [])
    split = meta.get("split", "")

    df = pd.read_csv(csv_path)
    time_s = df["time_s"].to_numpy(float)

    # Decide which angles to plot
    # We plot one row per unique angle that appears in the video's exercise segments.
    exercise_segments = [s for s in segments if s["type"] == "exercise"]
    exercises_present = list(dict.fromkeys(s["label"] for s in exercise_segments))

    plot_angles = []
    for ex in exercises_present:
        angle_col = EXERCISE_ANGLE.get(ex)
        if angle_col and angle_col not in plot_angles:
            plot_angles.append(angle_col)
    if "jump_jack" in exercises_present and JJ_SECONDARY not in plot_angles:
        plot_angles.append(JJ_SECONDARY)

    # Always add torso_tilt for context
    if "torso_tilt" not in plot_angles:
        plot_angles.append("torso_tilt")

    n_rows = len(plot_angles)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(16, 3 * n_rows),
        sharex=True,
    )
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(
        f"{video_id}  [{split}]  |  red dashed = clip join",
        fontsize=12,
    )

    print(f"\n{'='*70}")
    print(f"  Video: {video_id}  ({split})")
    print(f"{'='*70}")

    for ax, angle_col in zip(axes, plot_angles):
        if angle_col not in df.columns:
            ax.set_ylabel(angle_col)
            ax.text(0.5, 0.5, f"Column '{angle_col}' not found",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        vals = df[angle_col].to_numpy(float)

        # Shade all segments
        for seg in segments:
            _shade_segment(ax, seg, fps)

        # Draw rep spans for exercise segments whose primary angle matches
        for seg in exercise_segments:
            if EXERCISE_ANGLE.get(seg["label"]) == angle_col or (
                seg["label"] == "jump_jack" and angle_col == JJ_SECONDARY
            ):
                _draw_rep_spans(ax, seg, fps)

        # Clip-join lines
        _draw_clip_joins(ax, clip_join_frames, fps)

        # Plot angle
        ax.plot(time_s, vals, linewidth=0.8, color="steelblue", label=angle_col)
        ax.set_ylabel(f"{angle_col}\n(°)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Legend patches
        patches = [
            mpatches.Patch(color=c, alpha=0.5, label=lab)
            for lab, c in SEG_COLOURS.items()
            if lab in exercises_present or lab == "transition"
        ]
        ax.legend(handles=patches, fontsize=7, loc="upper right", ncol=4)

    # Peak analysis per exercise segment
    print(f"\n  {'Exercise':12s} {'Seg frames':>12s} {'Angle':>16s} "
          f"{'Annot reps':>12s} {'Peaks found':>12s}")
    print(f"  {'-'*70}")

    for seg in exercise_segments:
        ex = seg["label"]
        angle_col = EXERCISE_ANGLE.get(ex)
        if angle_col is None or angle_col not in df.columns:
            continue

        mask = (df["frame"] >= seg["start_frame"]) & (df["frame"] < seg["end_frame"])
        seg_vals = df.loc[mask, angle_col].to_numpy(float)
        seg_time = df.loc[mask, "time_s"].to_numpy(float)

        # Impute NaN with linear interpolation for peak finding only
        seg_vals_filled = _interpolate_for_peaks(seg_vals)

        prom = PEAK_PROMINENCE.get(ex, 10.0)
        peaks, props = find_peaks(seg_vals_filled, prominence=prom)
        n_peaks = len(peaks)
        n_annotated = seg.get("rep_count_annotated", "?")
        n_frames = seg["end_frame"] - seg["start_frame"]

        print(f"  {ex:12s} [{seg['start_frame']:6d}..{seg['end_frame']:6d}]"
              f" {angle_col:>16s}  annotated={n_annotated:>4}  peaks={n_peaks:>4}")

        # Per-rep range
        pairs = seg.get("rep_pairs_out", [])
        rep_ranges = []
        for start_f, end_f in pairs:
            rep_mask = (df["frame"] >= start_f) & (df["frame"] < end_f)
            rep_vals = df.loc[rep_mask, angle_col].dropna().to_numpy(float)
            if len(rep_vals) >= 3:
                rep_ranges.append(rep_vals.max() - rep_vals.min())

        if rep_ranges:
            print(f"    -> per-rep range (max-min):  "
                  f"min={min(rep_ranges):.1f}°  "
                  f"max={max(rep_ranges):.1f}°  "
                  f"median={np.median(rep_ranges):.1f}°")

    print()

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}_angles.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def _interpolate_for_peaks(vals: np.ndarray) -> np.ndarray:
    """Fill NaN linearly for peak detection only (does not modify original)."""
    out = vals.copy().astype(float)
    nans = np.isnan(out)
    if not nans.any():
        return out
    idx = np.arange(len(out))
    out[nans] = np.interp(idx[nans], idx[~nans], out[~nans]) if (~nans).sum() > 1 else 0.0
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Sanity plots for a synthetic video.")
    parser.add_argument("--video-id", required=True,
                        help="Video ID, e.g. synth_train_0001")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--synth-dir", type=Path, default=SYNTH_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    plot_video(args.video_id, args.features_dir, args.synth_dir, args.out_dir)


if __name__ == "__main__":
    main()
