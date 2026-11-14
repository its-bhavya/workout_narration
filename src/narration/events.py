"""Segments -> timed narration events."""
import statistics

START_LEAD_S = 1.0
TRANSITION_DELAY_S = 0.2
MIN_SPOKEN_GAP_S = 2.2        # raised from 1.5: spoken phrases take ~1.5-2.6 s
MILESTONE_EVERY = 5           # for exercises where every rep is announced
FAST_MILESTONE_EVERY = 10     # for fast exercises (cadence > 1)
CADENCES = (1, 2, 3, 5, 10)


def choose_cadence(rep_times, min_gap=MIN_SPOKEN_GAP_S):
    periods = [b - a for a, b in zip(rep_times, rep_times[1:]) if b > a]
    if not periods:
        return 1
    med = statistics.median(periods)
    for k in CADENCES:
        if k * med >= min_gap:
            return k
    return CADENCES[-1]


def build_events(segments, min_gap=MIN_SPOKEN_GAP_S):
    events = []
    for idx, seg in enumerate(segments):
        if idx == 0:
            events.append({"time": max(0.0, seg["start_time"] - START_LEAD_S),
                           "type": "segment_start", "exercise": seg["exercise"]})
        else:
            prev = segments[idx - 1]
            events.append({"time": prev["end_time"] + TRANSITION_DELAY_S, "type": "transition",
                           "prev": prev["exercise"], "next": seg["exercise"]})

        k = choose_cadence(seg["rep_times"], min_gap)
        seg["cadence"] = k
        milestone_every = MILESTONE_EVERY if k == 1 else FAST_MILESTONE_EVERY
        n_reps = len(seg["rep_times"])

        for i, t in enumerate(seg["rep_times"], start=1):
            milestone = i % milestone_every == 0
            if i % k and not milestone:
                continue
            if i == n_reps:   # the transition / workout_end line covers the last rep
                continue
            events.append({"time": t, "type": "milestone" if milestone else "rep", "n": i})

    if segments:
        last = segments[-1]
        events.append({"time": last["end_time"] + TRANSITION_DELAY_S,
                       "type": "workout_end", "exercise": last["exercise"]})
    return sorted(events, key=lambda e: e["time"])