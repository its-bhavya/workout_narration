"""How fast do reps actually come? Decides how much we can say.

    uv run python -m src.narration.rep_stats
Writes outputs/rep_period_stats.csv and prints a table + suspicious clips.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import load_manifests


def main():
    bounds, splits = load_manifests()
    rows, per_clip = [], []
    for clip_id, r in splits.iterrows():
        info = bounds.get(clip_id)
        if not info or len(info["pairs"]) < 2:
            continue
        ends = np.array([e for _, e in info["pairs"]], dtype=float)
        periods = np.diff(ends) / info["fps"]          # seconds between consecutive rep completions
        per_clip.append((clip_id, r["exercise"], float(np.median(periods)), info["fps"], len(periods) + 1))
        rows += [(r["exercise"], p) for p in periods]

    df = pd.DataFrame(rows, columns=["exercise", "period_s"])
    out = df.groupby("exercise")["period_s"].describe(percentiles=[.05, .25, .5, .75, .95]).round(2)
    out["share_lt_1.0s"] = df.groupby("exercise")["period_s"].apply(lambda s: (s < 1.0).mean()).round(2)
    out["share_lt_1.5s"] = df.groupby("exercise")["period_s"].apply(lambda s: (s < 1.5).mean()).round(2)
    print(out.to_string())

    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/rep_period_stats.csv")

    clips = pd.DataFrame(per_clip, columns=["clip_id", "exercise", "median_period_s", "fps", "reps"])
    print("\nClips with median rep period < 0.3 s (check annotations before trusting):")
    print(clips[clips.median_period_s < 0.3].to_string(index=False))


if __name__ == "__main__":
    main()
