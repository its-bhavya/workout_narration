"""make_windows.py -- sliding-window label builder for segmentation.

For every ``data/features/<video_id>_angles.csv`` this slides fixed-length
windows over ``time_s`` and assigns each window the majority ``seg_label``
among its rows. It writes one row per window to
``outputs/segmentation/windows.csv``.

This script does NOT compute feature vectors or train a classifier -- it only
produces window metadata (label, purity, row slice). The classifier step
(consumed later) will read ``windows.csv`` and slice the angle columns itself
using the FEATURE_CHANNELS allow-list from ``config.py``.

Windows are built from ``time_s`` (NOT row counts) because pose extraction used
``--stride 2``, so consecutive CSV rows are 2 frames apart and ``time_s`` is the
only reliable timeline.

Usage (run from repo root)::

    python src/segmentation/make_windows.py
    python src/segmentation/make_windows.py --features-dir data/features

Output columns: video_id, split, win_idx, t_start, t_end, t_center, label,
purity, row_start, row_end.
``row_end`` is exclusive (slice-style): ``df.iloc[row_start:row_end]`` recovers
the window's rows.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CLASSES,
    FEATURES_DIR,
    MIN_ROWS_FRAC,
    OUTPUT_DIR,
    STRIDE_S,
    SYNTH_DIR,
    WINDOW_S,
    WINDOWS_CSV,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Stable class index for deterministic tie-breaking.
CLASS_ORDER = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Split lookup (from each video's JSON "split" field)
# ---------------------------------------------------------------------------

def load_split_map(synth_dir: Path) -> dict[str, str]:
    """Build ``video_id -> split`` from every synthetic boundary JSON.

    The split is read from the JSON ``"split"`` field (never from the CSV and
    never tuned on test).
    """
    split_map: dict[str, str] = {}
    for j in sorted(synth_dir.rglob("*.json")):
        try:
            with open(j) as fh:
                meta = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read %s: %s", j, exc)
            continue
        vid = meta.get("video_id") or j.stem
        split_map[vid] = meta.get("split", "")
    return split_map


# ---------------------------------------------------------------------------
# Per-video windowing
# ---------------------------------------------------------------------------

def make_windows_for_video(
    df: pd.DataFrame,
    video_id: str,
    split: str,
    window_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
    min_rows_frac: float = MIN_ROWS_FRAC,
) -> list[dict]:
    """Slide ``[t, t+window_s)`` windows over ``time_s`` and label each.

    Parameters
    ----------
    df          : angle CSV loaded into a DataFrame (needs ``time_s`` and
                  ``seg_label`` columns).
    video_id    : synthetic video id (e.g. ``synth_train_0001``).
    split       : split label for this video.
    window_s    : window length in seconds.
    stride_s    : step between window starts in seconds.
    min_rows_frac : drop windows whose row count is below this fraction of the
                  expected row count for a full window.

    Returns
    -------
    list of window dicts with keys: video_id, split, win_idx, t_start, t_end,
    t_center, label, purity, row_start, row_end.
    """
    df = df.sort_values("time_s").reset_index(drop=True)
    times = df["time_s"].to_numpy(dtype=float)
    labels = df["seg_label"].astype(str).to_numpy()
    n = len(times)
    if n == 0:
        return []

    # Median sampling interval -> expected rows per full window.
    dt = float(np.median(np.diff(times))) if n > 1 else window_s
    expected_rows = max(1, int(round(window_s / dt))) if dt > 0 else 1
    min_rows = min_rows_frac * expected_rows

    t0 = float(times[0])
    t_last = float(times[-1])

    windows: list[dict] = []
    win_idx = 0
    i = 0
    while True:
        t_start = t0 + i * stride_s
        t_end = t_start + window_s
        # Drop windows that do not fully fit inside the timeline (the final
        # partial window and everything after it).
        if t_end > t_last + 1e-9:
            break

        start_idx = int(np.searchsorted(times, t_start, side="left"))
        end_idx = int(np.searchsorted(times, t_end, side="left"))
        n_rows = end_idx - start_idx
        if n_rows < min_rows:
            i += 1
            continue

        win_labels = labels[start_idx:end_idx]
        vals, counts = np.unique(win_labels, return_counts=True)
        # Majority label; ties broken by CLASSES order for determinism.
        order = sorted(
            range(len(vals)),
            key=lambda k: (-int(counts[k]), CLASS_ORDER.get(str(vals[k]), 99)),
        )
        maj_label = str(vals[order[0]])
        purity = float(counts[order[0]]) / n_rows

        windows.append({
            "video_id": video_id,
            "split": split,
            "win_idx": win_idx,
            "t_start": round(t_start, 6),
            "t_end": round(t_end, 6),
            "t_center": round((t_start + t_end) / 2.0, 6),
            "label": maj_label,
            "purity": round(purity, 6),
            "row_start": start_idx,
            "row_end": end_idx,  # exclusive (slice-style)
        })
        win_idx += 1
        i += 1

    return windows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sliding-window labels for the segmentation classifier."
    )
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR,
                        help="Directory of *_angles.csv files (default: data/features)")
    parser.add_argument("--synth-dir", type=Path, default=SYNTH_DIR,
                        help="Root of synthetic video/JSON tree (default: data/synthetic)")
    parser.add_argument("--out", type=Path, default=WINDOWS_CSV,
                        help="Output windows CSV (default: outputs/segmentation/windows.csv)")
    args = parser.parse_args()

    split_map = load_split_map(args.synth_dir)
    log.info("Loaded %d synthetic video JSONs from %s", len(split_map), args.synth_dir)

    # Only top-level *_angles.csv: this excludes the stray rep-counting eval
    # output that lives in data/features/repcounts/ (different columns, same
    # naming pattern). The split comes from the JSON, so any file without a
    # matching JSON is skipped.
    csvs = sorted(args.features_dir.glob("*_angles.csv"))
    log.info("Found %d top-level *_angles.csv files", len(csvs))

    all_windows: list[dict] = []
    seen_splits: dict[str, str] = {}

    for csv_path in csvs:
        video_id = csv_path.stem.removesuffix("_angles")
        split = split_map.get(video_id, "")
        if not split:
            log.warning("No JSON split for %s -- skipping", video_id)
            continue

        # Assert one split per video: a video_id must never map to two splits.
        if video_id in seen_splits and seen_splits[video_id] != split:
            raise AssertionError(
                f"{video_id} assigned conflicting splits: "
                f"{seen_splits[video_id]!r} vs {split!r}"
            )
        seen_splits[video_id] = split

        df = pd.read_csv(csv_path)
        wins = make_windows_for_video(df, video_id, split)
        all_windows.extend(wins)
        log.info("%-18s (%-5s): %d windows", video_id, split, len(wins))

    out_df = pd.DataFrame(
        all_windows,
        columns=["video_id", "split", "win_idx", "t_start", "t_end", "t_center",
                 "label", "purity", "row_start", "row_end"],
    )

    # Assert one split per video across the whole run.
    if len(out_df) > 0:
        n_splits_per_video = out_df.groupby("video_id")["split"].nunique()
        assert n_splits_per_video.max() == 1, (
            "a video has windows in more than one split"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    log.info("Wrote %d windows to %s", len(out_df), args.out)

    # Print window counts per split x class.
    print("\nWindow counts per split x class:")
    if len(out_df) > 0:
        pivot = out_df.pivot_table(
            index="split", columns="label", aggfunc="size", fill_value=0
        )
        for c in CLASSES:
            if c not in pivot.columns:
                pivot[c] = 0
        pivot = pivot.reindex(columns=CLASSES, fill_value=0)
        print(pivot.to_string())
    print(f"\nTotal windows: {len(out_df)}")


if __name__ == "__main__":
    main()
