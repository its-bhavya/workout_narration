"""Events -> non-overlapping audio timeline [{time, end, text, audio, ...}]."""
from .templates import Narrator
from .tts import synthesize, duration_s

IMPORTANT = {"segment_start", "transition", "milestone", "workout_end"}
GAP_S = 0.2
MAX_MILESTONE_DELAY_S = 2.0   # a milestone delayed longer than this is stale, so drop it


def build_timeline(events, seed=42, synth=synthesize, dur=duration_s, use_bank=False):
    nar = Narrator(seed, use_bank=use_bank)
    timeline, skipped, busy_until = [], [], float("-inf")
    for ev in events:
        if ev["time"] < busy_until + GAP_S and ev["type"] not in IMPORTANT:
            skipped.append(ev)            # plain rep callout while audio is still playing
            continue
        start = max(ev["time"], busy_until + GAP_S)
        if ev["type"] == "milestone" and start - ev["time"] > MAX_MILESTONE_DELAY_S:
            skipped.append(ev)            # too late to be worth saying
            continue
        slots = {k: v for k, v in ev.items() if k not in ("time", "type")}
        text = nar.say(ev["type"], **slots)
        path = synth(text)
        d = dur(path)
        timeline.append({
            "event": ev["type"], "text": text, "audio": str(path),
            "time": round(start, 3), "end": round(start + d, 3),
            "scheduled": round(ev["time"], 3), "delay": round(start - ev["time"], 3),
        })
        busy_until = start + d
    return timeline, skipped


def summarize(timeline, skipped, n_events):
    delays = [t["delay"] for t in timeline]
    return {
        "events": n_events, "spoken": len(timeline), "skipped": len(skipped),
        "skip_rate": round(len(skipped) / n_events, 3) if n_events else 0.0,
        "max_delay_s": max(delays) if delays else 0.0,
        "mean_delay_s": round(sum(delays) / len(delays), 3) if delays else 0.0,
    }