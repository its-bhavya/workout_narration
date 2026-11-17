"""Rep-count metrics. pred/gt: dict clip_id -> count."""
import numpy as np


def rep_count_metrics(pred, gt):
    keys = [k for k in gt if k in pred]
    if not keys:
        return {"n": 0}
    err = np.array([pred[k] - gt[k] for k in keys], dtype=float)
    g = np.array([gt[k] for k in keys], dtype=float)
    return {
        "n": len(keys),
        "exact_match": float((err == 0).mean()),
        "off_by_one": float((np.abs(err) <= 1).mean()),   # OBO accuracy
        "mae": float(np.abs(err).mean()),
        "norm_mae": float((np.abs(err) / np.maximum(g, 1)).mean()),
        "bias": float(err.mean()),                         # >0 = over-counting
    }
