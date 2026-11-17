"""Simulate imperfect rep predictions from ground-truth segments."""
import random


def perturb_segments(segments, jitter_s=0.15, drop_p=0.10, dup_p=0.03, seed=0):
    """Return copies of segments with jittered, dropped and duplicated rep times."""
    rng = random.Random(seed)
    out = []
    for seg in segments:
        times = []
        for t in seg["rep_times"]:
            if rng.random() < drop_p:
                continue                                   # missed rep
            times.append(t + rng.gauss(0, jitter_s))       # late/early detection
            if rng.random() < dup_p:
                times.append(t + rng.uniform(0.05, 0.3))   # double detection
        if not times:                                      # keep at least one rep
            times = [seg["rep_times"][-1]]
        times = sorted(max(seg["start_time"], min(t, seg["end_time"])) for t in times)
        new = dict(seg)
        new["rep_times"] = times
        out.append(new)
    return out