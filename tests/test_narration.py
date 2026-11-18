from src.narration.events import build_events, choose_cadence
from src.narration.schedule import build_timeline

# Pin p90 so tests are independent of whatever is in the TTS cache on disk.
_P90 = 1.9  # yields min_gap = 2.2 s — same as the old MIN_SPOKEN_GAP_S constant


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
    # period=3.0s, p90=1.9s → min_gap=2.2 → cadence=1 → milestones every 5 reps
    events = build_events([seg("squat", 3.0, 12)], p90_duration=_P90)
    assert ns(events, "milestone") == {5, 10}


def test_milestones_fast_exercise():
    # period=0.9s, p90=1.9s → min_gap=2.2 → cadence=3 (3×0.9=2.7≥2.2)
    # FAST_MILESTONE_EVERY=10, snapped to cadence multiples: 10→9, 20→21, 30→30
    events = build_events([seg("jump_jack", 0.9, 32)], p90_duration=_P90)
    assert ns(events, "milestone") == {9, 21, 30}
    # Verify no plain rep event is emitted when a milestone is emitted
    assert not (ns(events, "milestone") & ns(events, "rep"))
    # Read cadence from the first rep/milestone event rather than hard-coding 3
    cadence = next(e["cadence"] for e in events if e["type"] in ("rep", "milestone"))
    for ev in events:
        if ev["type"] in ("rep", "milestone"):
            assert ev["n"] % cadence == 0


def test_last_rep_allowed_through():
    # Item 7: rep at exactly end_time (not > end_time + 0.3) should be emitted
    s = seg("squat", 3.0, 12, start=0.0)  # rep 12 at t=36.0, end_time=36.0
    events = build_events([s], p90_duration=_P90)
    # With cadence 1, rep 12 passes the guard t > 36.3 → False → allowed
    # (cadence 1 so no milestone override — rep 12 is a plain rep or milestone)
    rep_or_milestone_ns = {e["n"] for e in events if e["type"] in ("rep", "milestone")}
    assert 12 in rep_or_milestone_ns


def test_last_rep_beyond_guard_is_dropped():
    # A rep clearly past end_time + 0.3 must not appear as a rep or milestone event
    s = seg("squat", 3.0, 12, start=0.0)
    s["rep_times"].append(37.0)  # 37.0 > 36.0 + 0.3 → should be dropped as a spoken rep
    events = build_events([s], p90_duration=_P90)
    counted_ns = {e["n"] for e in events if e["type"] in ("rep", "milestone")}
    assert 13 not in counted_ns


def test_transition_before_boundary():
    # Item 6: with align_starts=True the transition should fire before the next segment start
    segs = [seg("squat", 3.0, 5, start=0.0), seg("push_up", 2.0, 5, start=20.0)]
    events = build_events(segs, align_starts=True, p90_duration=_P90)
    trans = next(e for e in events if e["type"] == "transition")
    next_start = segs[1]["start_time"]
    assert trans["time"] < next_start


def test_workout_end_within_duration():
    # Item 8: workout_end must not overshoot the video duration
    s = seg("squat", 3.0, 12, start=0.0)
    duration = s["end_time"]  # 36.0 s
    events = build_events([s], workout_end=duration, p90_duration=_P90)
    we = next(e for e in events if e["type"] == "workout_end")
    assert we["time"] <= duration


def test_workout_end_is_spoken():
    events = build_events([seg("squat", 3.0, 12), seg("push_up", 2.5, 10, start=40)],
                          p90_duration=_P90)
    timeline, _ = build_timeline(events, synth=fake_synth, dur=fake_dur)
    assert timeline[-1]["event"] == "workout_end"


def test_no_overlap():
    events = build_events([seg("jump_jack", 0.9, 32)], p90_duration=_P90)
    timeline, _ = build_timeline(events, synth=fake_synth, dur=fake_dur)
    for a, b in zip(timeline, timeline[1:]):
        assert b["time"] >= a["end"]


def test_important_events_never_dropped():
    events = build_events([seg("squat", 0.5, 20), seg("push_up", 0.5, 20, start=15)],
                          p90_duration=_P90)
    timeline, _ = build_timeline(events, synth=fake_synth, dur=fake_dur)
    kinds = [t["event"] for t in timeline]
    assert "transition" in kinds and "segment_start" in kinds


def test_fast_cadence_rep_phrasing_plain():
    # Item 9: fast-cadence reps (cadence > 1) should use plain numbers, no praise prefix
    from src.narration.templates import Narrator
    narrator = Narrator(seed=42)
    for _ in range(10):
        text = narrator.say("rep", n=6, cadence=3)
        # Plain phrasing must NOT contain praise words
        assert not any(w in text.lower() for w in ("nice", "good", "keep")), \
            f"Fast cadence rep should use plain number only, got: {text!r}"


def test_slow_cadence_rep_phrasing_varied():
    from src.narration.templates import Narrator
    narrator = Narrator(seed=42, use_bank=True)
    texts = {narrator.say("rep", n=6, cadence=1) for _ in range(10)}
    assert texts == {"6.", "6 down."}


def test_cadence_can_use_any_interval():
    assert choose_cadence([0, 0.6, 1.2], p90_duration=1.9) == 4
    assert choose_cadence([0, 0.1, 0.2], p90_duration=1.9) == 22


def test_transition_finishes_before_first_count_and_says_guarded_rep():
    previous = seg("push_up", 3.0, 3)
    previous["rep_times"].append(9.5)
    following = seg("squat", 2.5, 3, start=12.0)
    events = build_events([previous, following], align_starts=True, p90_duration=_P90)
    timeline, _ = build_timeline(
        events, synth=lambda text: text,
        dur=lambda text: 2.5 if "squats" in text else 0.5,
    )
    transition = next(cue for cue in timeline if cue["event"] == "transition")
    first_count = next(cue for cue in timeline if cue["scheduled"] == 14.5)
    assert "4 reps" in transition["text"]
    assert transition["time"] >= 9.5
    assert transition["end"] == following["start_time"]
    assert first_count["time"] == 14.5


def test_outro_counts_rep_that_cannot_fit_at_video_end():
    segment = seg("squat", 3.0, 4)
    events = build_events([segment], workout_end=12.0, p90_duration=_P90)
    timeline, skipped = build_timeline(events, synth=fake_synth, dur=fake_dur)
    assert any(event["type"] == "rep" and event["n"] == 4 for event in skipped)
    assert timeline[-1]["text"] == "4 reps. Done."
    assert timeline[-1]["end"] == 12.0
    assert timeline[-2]["end"] <= timeline[-1]["time"]
