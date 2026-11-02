"""smooth.py -- pure post-processing for per-window segmentation predictions.

Two pure functions (no file I/O, no model, no randomness):

* ``smooth(labels, confs, tau, min_len)`` -- turn noisy per-window predicted
  labels into a clean per-window label sequence by (a) replacing low-confidence
  windows with the previous window's label (the first window keeps its own)
  and (b) absorbing runs shorter than ``min_len`` windows into the previous run
  (the first run, if short, is absorbed into the next run).
* ``to_segments(labels, t_centers, video_end_s)`` -- collapse a per-window label
  sequence into a list of ``{"exercise", "start_time", "end_time"}`` segments.
  Boundaries are midpoints between consecutive window centers; the first segment
  starts at 0 and the last ends at ``video_end_s``; consecutive identical labels
  are merged. ``"transition"`` is treated as an ordinary label.

These operate on plain Python sequences so they are easy to unit-test and to
reuse from any caller (e.g. applied to the per-window predictions written by
``train.py``). Import locally the same way the rest of ``src/segmentation`` does::

    from smooth import smooth, to_segments

No tuning or evaluation lives here -- this module only transforms labels.
"""
from __future__ import annotations

from typing import Sequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _runs(labels: Sequence[str]) -> list[tuple[str, int, int]]:
    """Return maximal runs as ``(label, start, end)`` with ``end`` exclusive."""
    n = len(labels)
    runs: list[tuple[str, int, int]] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append((labels[i], i, j))
        i = j
    return runs


def _apply_low_confidence(labels: list[str], confs: list[float], tau: float) -> list[str]:
    """Forward-fill labels for windows below the confidence threshold.

    Window ``i`` (``i >= 1``) whose confidence is ``< tau`` takes the label of
    the previous window -- already smoothed at that point, so a streak of
    low-confidence windows all inherit the last confident label. The first
    window always keeps its own label (its confidence is never consulted).
    """
    out = list(labels)
    for i in range(1, len(out)):
        if confs[i] < tau:
            out[i] = out[i - 1]
    return out


def _enforce_min_len(labels: list[str], min_len: float) -> list[str]:
    """Absorb runs shorter than ``min_len`` windows into a neighbour.

    A short run (length ``< min_len``) is absorbed into the previous run; if it
    is the first run it is absorbed into the next run instead. Absorption is
    applied iteratively: after each absorption the runs are recomputed, so a
    short run is always absorbed into the *current* previous run (which may have
    grown from an earlier absorption). This is deterministic and terminates
    because every absorption strictly reduces the number of runs.

    Exception: ``transition`` runs are never absorbed regardless of length.
    Transitions are genuinely short by design (a few seconds of rest between
    exercises), so forcing them to meet ``min_len`` would erase real rest
    periods and merge adjacent exercise segments incorrectly.
    """
    if not labels or min_len <= 1:
        return list(labels)
    out = list(labels)
    while True:
        runs = _runs(out)
        if len(runs) <= 1:
            break
        # Find the first short run that is NOT transition, and whose absorption
        # target neighbour is also NOT transition (absorbing into a transition
        # run would grow it and then cause the far-side exercise run to be
        # swallowed by the enlarged transition in the next iteration).
        idx = None
        for k, (lab, s, e) in enumerate(runs):
            if (e - s) < min_len and lab != "transition":
                neighbour_label = runs[1][0] if k == 0 else runs[k - 1][0]
                if neighbour_label != "transition":
                    idx = k
                    break
        if idx is None:
            break
        _lab, s, e = runs[idx]
        if idx == 0:
            new_label = runs[1][0]        # absorb the first run into the next run
        else:
            new_label = runs[idx - 1][0]  # absorb into the previous run
        for k in range(s, e):
            out[k] = new_label
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smooth(
    labels: Sequence[str],
    confs: Sequence[float],
    tau: float,
    min_len: float,
) -> list[str]:
    """Smooth per-window predicted labels.

    Parameters
    ----------
    labels : per-window predicted labels (any sequence; e.g. ``pred_label``).
    confs  : per-window confidence in the predicted label (any sequence; e.g.
             ``max_prob``). Must be the same length as ``labels``.
    tau    : confidence threshold. A window with ``conf < tau`` is replaced by
             the previous window's label (the first window keeps its own).
    min_len: minimum run length in windows. Any run shorter than this is
             absorbed into the previous run (the first run, if short, is
             absorbed into the next run).

    Returns
    -------
    list[str]
        Smoothed per-window labels (same length as the input).
    """
    labels = list(labels)
    confs = [float(c) for c in confs]
    assert len(labels) == len(confs), "labels and confs must have the same length"
    labels = _apply_low_confidence(labels, confs, tau)
    labels = _enforce_min_len(labels, min_len)
    return labels


def to_segments(
    labels: Sequence[str],
    t_centers: Sequence[float],
    video_end_s: float,
) -> list[dict]:
    """Collapse per-window labels into time segments.

    Boundaries between consecutive runs are placed at the midpoint of the two
    neighbouring window centers. The first segment starts at ``0`` and the last
    ends at ``video_end_s``; consecutive identical labels are merged, so the
    returned segments tile ``[0, video_end_s]`` with no gaps or overlaps.
    ``"transition"`` is treated as an ordinary label (it can be its own segment).

    Parameters
    ----------
    labels     : per-window labels (e.g. the output of :func:`smooth`).
    t_centers  : per-window center times in seconds (same length as ``labels``).
    video_end_s: total video duration in seconds.

    Returns
    -------
    list[dict]
        ``[{"exercise": str, "start_time": float, "end_time": float}, ...]``.
    """
    n = len(labels)
    if n == 0:
        return []
    assert len(t_centers) == n, "labels and t_centers must have the same length"
    labels = list(labels)
    t_centers = [float(t) for t in t_centers]
    video_end_s = float(video_end_s)

    segments: list[dict] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        # Left boundary of this run: 0 for the first run, else the midpoint of
        # the two adjacent window centers (t_centers[i-1], t_centers[i]).
        start = 0.0 if i == 0 else (t_centers[i - 1] + t_centers[i]) / 2.0
        # Right boundary: video_end_s for the last run, else the midpoint of
        # (t_centers[j-1], t_centers[j]) -- shared with the next run's start.
        end = video_end_s if j == n else (t_centers[j - 1] + t_centers[j]) / 2.0
        segments.append({"exercise": labels[i], "start_time": start, "end_time": end})
        i = j
    return segments
