"""metrics.py -- pure evaluation functions for segmentation predictions.

All functions operate on plain Python / numpy data; no file I/O.

Evaluation grid: 10 Hz (dt = 0.1 s). Every metric is computed over the full
video duration, not just annotated windows.

GT segment format (same as JSON after conversion)::

    [{"exercise": str, "start_time": float, "end_time": float}, ...]

Predicted segment format: same shape (output of smooth.to_segments).
"""
from __future__ import annotations

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# 10 Hz grid helpers
# ---------------------------------------------------------------------------

def _make_grid(video_end_s: float, hz: float = 10.0) -> list[float]:
    """Equally-spaced time points at ``hz`` Hz from 0 to video_end_s (inclusive)."""
    dt = 1.0 / hz
    n = int(round(video_end_s * hz)) + 1
    return [i * dt for i in range(n)]


def _label_at(segs: list[dict], t: float) -> str:
    """Label of the segment containing time ``t``."""
    for s in segs:
        if s["start_time"] <= t < s["end_time"]:
            return s["exercise"]
    # Exactly at end_time of the last segment.
    if segs and math.isclose(t, segs[-1]["end_time"], abs_tol=1e-6):
        return segs[-1]["exercise"]
    return "transition"


def _grid_labels(segs: list[dict], grid: list[float]) -> list[str]:
    return [_label_at(segs, t) for t in grid]


# ---------------------------------------------------------------------------
# Levenshtein (pure Python, no library needed)
# ---------------------------------------------------------------------------

def _levenshtein(a: Sequence, b: Sequence) -> int:
    """Standard Levenshtein edit distance between two sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def frame_accuracy(gt_segs: list[dict], pred_segs: list[dict],
                   video_end_s: float) -> float:
    """Fraction of 10 Hz grid points where pred label == GT label."""
    grid = _make_grid(video_end_s)
    gt_labs = _grid_labels(gt_segs, grid)
    pr_labs = _grid_labels(pred_segs, grid)
    if not grid:
        return 0.0
    return sum(g == p for g, p in zip(gt_labs, pr_labs)) / len(grid)


def frame_f1_per_class(gt_segs: list[dict], pred_segs: list[dict],
                       video_end_s: float,
                       classes: list[str] | None = None) -> dict[str, float]:
    """Per-class F1 on the 10 Hz evaluation grid."""
    grid = _make_grid(video_end_s)
    gt_labs = _grid_labels(gt_segs, grid)
    pr_labs = _grid_labels(pred_segs, grid)
    all_labels = classes or sorted(set(gt_labs) | set(pr_labs))
    result: dict[str, float] = {}
    for cls in all_labels:
        tp = sum(g == cls and p == cls for g, p in zip(gt_labs, pr_labs))
        fp = sum(g != cls and p == cls for g, p in zip(gt_labs, pr_labs))
        fn = sum(g == cls and p != cls for g, p in zip(gt_labs, pr_labs))
        denom = 2 * tp + fp + fn
        result[cls] = (2 * tp / denom) if denom > 0 else 0.0
    return result


def edit_score(gt_segs: list[dict], pred_segs: list[dict]) -> float:
    """Edit score = (1 - lev(label_seqs) / max(len)) * 100.

    Label sequences are the run-length-encoded label lists (one entry per
    segment), so a perfectly clean sequence scores 100.
    """
    gt_labs = [s["exercise"] for s in gt_segs]
    pr_labs = [s["exercise"] for s in pred_segs]
    m = max(len(gt_labs), len(pr_labs))
    if m == 0:
        return 100.0
    dist = _levenshtein(gt_labs, pr_labs)
    return max(0.0, (1.0 - dist / m) * 100.0)


def _seg_iou(gt: dict, pred: dict) -> float:
    inter = max(0.0, min(gt["end_time"], pred["end_time"])
                - max(gt["start_time"], pred["start_time"]))
    union = (gt["end_time"] - gt["start_time"]
             + pred["end_time"] - pred["start_time"] - inter)
    return inter / union if union > 0 else 0.0


def f1_at_iou(gt_segs: list[dict], pred_segs: list[dict],
              iou_threshold: float) -> float:
    """Segment-level F1@k (k = iou_threshold).

    A predicted segment matches a GT segment if they share the same label and
    their IoU >= iou_threshold. Each GT segment is matched at most once
    (greedy, highest-IoU first).
    """
    matched_gt: set[int] = set()
    matched_pr: set[int] = set()
    # Collect all valid (iou, gt_idx, pr_idx) triples.
    candidates = []
    for gi, g in enumerate(gt_segs):
        for pi, p in enumerate(pred_segs):
            if g["exercise"] == p["exercise"]:
                iou = _seg_iou(g, p)
                if iou >= iou_threshold:
                    candidates.append((iou, gi, pi))
    candidates.sort(key=lambda x: -x[0])
    for iou, gi, pi in candidates:
        if gi not in matched_gt and pi not in matched_pr:
            matched_gt.add(gi)
            matched_pr.add(pi)
    tp = len(matched_gt)
    fp = len(pred_segs) - len(matched_pr)
    fn = len(gt_segs) - len(matched_gt)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def boundary_errors(gt_segs: list[dict], pred_segs: list[dict]) -> dict[str, float]:
    """For each GT boundary, the abs time diff to the nearest predicted boundary.

    Boundaries are the internal edges only (not 0 and video_end_s).
    Returns {"mean_s": float, "median_s": float}.  If no GT boundaries exist,
    both values are 0.0.
    """
    def _boundaries(segs: list[dict]) -> list[float]:
        # Internal boundaries: end_time of every segment except the last.
        return [s["end_time"] for s in segs[:-1]]

    gt_bounds = _boundaries(gt_segs)
    pr_bounds = _boundaries(pred_segs)
    if not gt_bounds:
        return {"mean_s": 0.0, "median_s": 0.0}
    if not pr_bounds:
        # No predictions -> each GT boundary is maximally wrong; use inf
        # but fall back to the video end to keep it finite.
        errors = [float("inf")] * len(gt_bounds)
    else:
        errors = [min(abs(gb - pb) for pb in pr_bounds) for gb in gt_bounds]
    mean_e = sum(errors) / len(errors)
    sorted_e = sorted(errors)
    n = len(sorted_e)
    if n % 2 == 1:
        median_e = sorted_e[n // 2]
    else:
        median_e = (sorted_e[n // 2 - 1] + sorted_e[n // 2]) / 2.0
    return {"mean_s": mean_e, "median_s": median_e}


def segment_counts(gt_segs: list[dict], pred_segs: list[dict]) -> dict[str, int]:
    """Number of predicted and GT segments."""
    return {"n_pred": len(pred_segs), "n_gt": len(gt_segs)}


def all_metrics(gt_segs: list[dict], pred_segs: list[dict],
                video_end_s: float,
                classes: list[str] | None = None) -> dict:
    """Compute all metrics and return a flat dict."""
    acc = frame_accuracy(gt_segs, pred_segs, video_end_s)
    f1_cls = frame_f1_per_class(gt_segs, pred_segs, video_end_s, classes)
    es = edit_score(gt_segs, pred_segs)
    be = boundary_errors(gt_segs, pred_segs)
    counts = segment_counts(gt_segs, pred_segs)
    result: dict = {
        "frame_accuracy": acc,
        "edit_score": es,
        "f1_at_10": f1_at_iou(gt_segs, pred_segs, 0.10),
        "f1_at_25": f1_at_iou(gt_segs, pred_segs, 0.25),
        "f1_at_50": f1_at_iou(gt_segs, pred_segs, 0.50),
        "boundary_mean_s": be["mean_s"],
        "boundary_median_s": be["median_s"],
        "n_pred": counts["n_pred"],
        "n_gt": counts["n_gt"],
    }
    for cls, f1 in f1_cls.items():
        result[f"frame_f1_{cls}"] = f1
    return result


# ---------------------------------------------------------------------------
# GT loader
# ---------------------------------------------------------------------------

def gt_from_json(meta: dict) -> tuple[list[dict], float]:
    """Extract GT segments and video_end_s from a synthetic video JSON dict.

    GT segments use start_frame / end_frame + fps -> seconds.
    end_frame is exclusive (as documented in concatenate_videos.py), so
    video_end_s = n_frames / fps.
    """
    fps = float(meta["fps"])
    video_end_s = meta["n_frames"] / fps
    segs = [
        {
            "exercise": s["label"],
            "start_time": s["start_frame"] / fps,
            "end_time": s["end_frame"] / fps,
        }
        for s in meta["segments"]
    ]
    return segs, video_end_s
