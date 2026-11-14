from src.narration.events import build_events
from src.narration.schedule import build_timeline


def fake_synth(text):
    return f"{abs(hash(text))}.mp3"


def fake_dur(path):
    return 1.5


def seg(exercise, period, n, start=0.0):
    times = [start + period * (i + 1) for i in range(n)]
    return {"exercise": exercise, "start_time": start + 0.5,
            "end_time": times[-1], "rep_times": times}


def ns(events, etype):
    return {e["n"] for e in events if e["type"] == etype}


def test_milestones_slow_exercise():
    events = build_events([seg("squat", 3.0, 12)])
    assert ns(events, "milestone") == {5, 10}


def test_milestones_fast_exercise():
    events = build_events([seg("jump_jack", 0.9, 32)])
    assert ns(events, "milestone") == {10, 20, 30}


def test_last_rep_not_announced():
    events = build_events([seg("squat", 3.0, 12)])
    assert 12 not in {e.get("n") for e in events}


def test_workout_end_is_spoken():
    events = build_events([seg("squat", 3.0, 12), seg("push_up", 2.5, 10, start=40)])
    timeline, _ = build_timeline(events, synth=fake_synth, dur=fake_dur)
    assert timeline[-1]["event"] == "workout_end"


def test_no_overlap():
    events = build_events([seg("jump_jack", 0.9, 32)])
    timeline, _ = build_timeline(events, synth=fake_synth, dur=fake_dur)
    for a, b in zip(timeline, timeline[1:]):
        assert b["time"] >= a["end"]


def test_important_events_never_dropped():
    events = build_events([seg("squat", 0.5, 20), seg("push_up", 0.5, 20, start=15)])
    timeline, _ = build_timeline(events, synth=fake_synth, dur=fake_dur)
    kinds = [t["event"] for t in timeline]
    assert "transition" in kinds and "segment_start" in kinds