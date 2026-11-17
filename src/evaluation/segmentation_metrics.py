"""Segmentation metrics. All functions take segments as (label, start_frame, end_frame),
end EXCLUSIVE, contiguous, covering [0, n_frames)  -- the format in data/README.md.

All scores are in [0, 1] (boundary_error is in seconds).
Every label counts, including "transition". To score exercises only, filter first.
"""
import numpy as np


def to_frame_labels(segs, n_frames):
    labels = np.full(n_frames, "", dtype=object)
    for lab, s, e in segs:
        labels[s:e] = lab
    return labels


def frame_accuracy(gt, pred, n_frames=None):
    n = n_frames or max(max(e for _, _, e in gt), max(e for _, _, e in pred))
    return float((to_frame_labels(gt, n) == to_frame_labels(pred, n)).mean())


def _collapse(labels):
    out = []
    for l in labels:
        if not out or out[-1] != l:
            out.append(l)
    return out


def edit_score(gt, pred):
    """1 - Levenshtein(label sequences) / max length. Penalises over-segmentation."""
    a, b = _collapse([s[0] for s in gt]), _collapse([s[0] for s in pred])
    d = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(len(a) + 1), np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return 1.0 - d[-1, -1] / max(len(a), len(b), 1)


def _iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[1], b[1]))
    union = (a[2] - a[1]) + (b[2] - b[1]) - inter
    return inter / union if union else 0.0


def f1_at_k(gt, pred, k):
    """Segment F1 at IoU threshold k/100 (k = 10, 25, 50). A predicted segment is a hit
    if it overlaps an unused ground-truth segment of the SAME label with IoU >= k/100."""
    thr, used, tp = k / 100.0, set(), 0
    for p in pred:
        best, best_j = 0.0, None
        for j, g in enumerate(gt):
            if j in used or g[0] != p[0]:
                continue
            v = _iou(g, p)
            if v > best:
                best, best_j = v, j
        if best_j is not None and best >= thr:
            tp += 1
            used.add(best_j)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(pred), tp / len(gt)
    return 2 * prec * rec / (prec + rec)


def _boundaries(segs):
    """(label_before, label_after, frame) for every internal boundary."""
    return [(segs[i - 1][0], segs[i][0], segs[i][1]) for i in range(1, len(segs))]


def boundary_error(gt, pred, fps):
    """For each ground-truth boundary, distance (s) to the nearest predicted boundary of the
    SAME TYPE (same label before and after). Unmatched ground-truth boundaries are counted
    in n_missed and left out of the mean."""
    gb, pb = _boundaries(gt), _boundaries(pred)
    errs, missed = [], 0
    for a, b, f in gb:
        cands = [pf for pa, pb_, pf in pb if (pa, pb_) == (a, b)]
        if cands:
            errs.append(min(abs(c - f) for c in cands) / fps)
        else:
            missed += 1
    return {
        "n_gt": len(gb), "n_missed": missed,
        "mean_abs_s": float(np.mean(errs)) if errs else float("nan"),
        "median_abs_s": float(np.median(errs)) if errs else float("nan"),
    }


def evaluate_segmentation(gt, pred, fps, n_frames=None):
    return {
        "frame_acc": frame_accuracy(gt, pred, n_frames),
        "edit": edit_score(gt, pred),
        "f1@10": f1_at_k(gt, pred, 10),
        "f1@25": f1_at_k(gt, pred, 25),
        "f1@50": f1_at_k(gt, pred, 50),
        **{f"boundary_{k}": v for k, v in boundary_error(gt, pred, fps).items()},
    }
