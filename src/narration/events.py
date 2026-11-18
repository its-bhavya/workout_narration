"""Segments -> timed narration events."""
import math
from pathlib import Path
import statistics
from functools import lru_cache

START_LEAD_S = 1.0
TRANSITION_DELAY_S = 0.2
MILESTONE_EVERY = 5           # for exercises where every rep is announced
FAST_MILESTONE_EVERY = 10     # for fast exercises (cadence > 1)


@lru_cache(maxsize=256)
def _cached_clip_duration(path, modified_ns, size):
    from .tts import duration_s
    return duration_s(path)


def get_p90_phrase_duration(cache_dir: str | Path = "outputs/tts_cache") -> float:
    """Compute p90 duration of cached speech clips, with a safe fallback."""
    durations = []
    for path in Path(cache_dir).glob("*.mp3"):
        try:
            stat = path.stat()
            duration = _cached_clip_duration(path.resolve(), stat.st_mtime_ns, stat.st_size)
        except (OSError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            durations.append(duration)
    if not durations:
        return 1.9
    durations.sort()
    return durations[math.ceil(len(durations) * 0.9) - 1]


def choose_cadence(rep_times, min_gap=None, p90_duration=None):
    """Smallest cadence where cadence * median_gap >= p90_phrase_duration + 0.3."""
    periods = [b - a for a, b in zip(rep_times, rep_times[1:]) if b > a]
    if not periods:
        return 1
    med = statistics.median(periods)
    if min_gap is None:
        p90 = p90_duration if p90_duration is not None else get_p90_phrase_duration()
        min_gap = p90 + 0.3
    return max(1, math.ceil(min_gap / med))


def build_events(segments, min_gap=None, *, align_starts=False, workout_end=None, p90_duration=None):
    events = []
    if min_gap is None and p90_duration is None:
        p90_duration = get_p90_phrase_duration()
    for idx, seg in enumerate(segments):
        if idx == 0:
            events.append({"time": max(0.0, seg["start_time"] - START_LEAD_S),
                           "type": "segment_start", "exercise": seg["exercise"]})
        else:
            prev = segments[idx - 1]
            prev_last_rep = prev["rep_times"][-1] if prev.get("rep_times") else prev["start_time"]
            if align_starts:
                trans_time = max(prev_last_rep + 0.3, seg["start_time"] - 1.2)
            else:
                trans_time = max(prev_last_rep + 0.3, prev["end_time"] + TRANSITION_DELAY_S)
            events.append({"time": trans_time, "type": "transition",
                           "prev": prev["exercise"], "next": seg["exercise"],
                           "n": len(prev["rep_times"]), "segment": idx - 1,
                           "not_before": prev_last_rep + 0.3,
                           "boundary": seg["start_time"] if align_starts else None})

        k = choose_cadence(seg["rep_times"], min_gap, p90_duration=p90_duration)
        seg["cadence"] = k
        milestone_every = MILESTONE_EVERY if k in (1, 5) else FAST_MILESTONE_EVERY
        n_reps = len(seg["rep_times"])
        # Milestones land only on multiples of cadence, snapping to nearest spoken count
        milestone_reps = {
            max(k, round(m / k) * k)
            for m in range(milestone_every, n_reps + 1, milestone_every)
        }

        for i, t in enumerate(seg["rep_times"], start=1):
            if i % k != 0:
                continue
            if t > seg["end_time"] + 0.3:
                continue
            is_milestone = i in milestone_reps
            events.append({"time": t, "type": "milestone" if is_milestone else "rep",
                           "n": i, "cadence": k, "segment": idx})

    if segments:
        last = segments[-1]
        if workout_end is not None:
            end_time = workout_end
        else:
            end_time = last["end_time"] + TRANSITION_DELAY_S
        events.append({"time": end_time, "type": "workout_end", "exercise": last["exercise"],
                       "n": len(last["rep_times"]), "segment": len(segments) - 1,
                       "deadline": workout_end})
    return sorted(events, key=lambda e: e["time"])
