"""Turn RepCount ground truth into the segment format narration consumes.

Segment = {"exercise", "start_time", "end_time", "rep_times", "clip_id"}  (all times in seconds)
`pairs` in repcount_rep_boundaries.json are [start_frame, end_frame] per rep at the clip's own fps,
so a rep is COMPLETE at end_frame / fps. That is the moment we may say "that's 5".
"""
import json
from pathlib import Path

import pandas as pd

MANIFESTS = Path("data/manifests")


def load_manifests(manifests=MANIFESTS):
    bounds = json.loads((Path(manifests) / "repcount_rep_boundaries.json").read_text())
    splits = pd.read_csv(Path(manifests) / "splits.csv").set_index("clip_id")
    return bounds, splits


def clip_to_segment(clip_id, bounds, splits, offset_s=0.0):
    """One clip -> (segment, offset where the NEXT clip would start)."""
    info = bounds[clip_id]
    fps, pairs = info["fps"], info["pairs"]
    seg = {
        "clip_id": clip_id,
        "exercise": splits.loc[clip_id, "exercise"],
        "start_time": offset_s + pairs[0][0] / fps,
        "end_time": offset_s + pairs[-1][1] / fps,
        "rep_times": [offset_s + end / fps for _, end in pairs],
    }
    return seg, offset_s + info["total_frames"] / fps


def clips_to_segments(clip_ids, bounds, splits):
    """Concatenate clips back-to-back (mirrors concatenate_videos.py) -> (segments, total_duration_s)."""
    segments, offset = [], 0.0
    for cid in clip_ids:
        seg, offset = clip_to_segment(cid, bounds, splits, offset)
        segments.append(seg)
    return segments, offset
