"""rep_counter output -> narration segment."""
import statistics


def prediction_to_segment(exercise, rep_times, clip_id=None, offset_s=0.0):
    """rep_times: completion times in seconds, relative to the clip start.
    offset_s: where that clip begins in the concatenated video."""
    times = sorted(offset_s + t for t in rep_times)
    if not times:
        raise ValueError(f"No predicted reps for {clip_id or exercise}")
    periods = [b - a for a, b in zip(times, times[1:]) if b > a]
    period = statistics.median(periods) if periods else 2.0
    return {
        "clip_id": clip_id,
        "exercise": exercise,
        "start_time": max(0.0, times[0] - period),   # estimated: first rep began one period earlier
        "end_time": times[-1],
        "rep_times": times,
    }