"""Events -> non-overlapping audio timeline [{time, end, text, audio, ...}]."""
from .templates import Narrator
from .tts import synthesize, duration_s

IMPORTANT = {"segment_start", "transition", "milestone", "workout_end"}
GAP_S = 0.2
MAX_MILESTONE_DELAY_S = 2.0   # a milestone delayed longer than this is stale, so drop it


def build_timeline(events, seed=42, synth=synthesize, dur=duration_s, use_bank=False):
    nar = Narrator(seed, use_bank=use_bank)
    timeline, skipped, placed = [], [], []
    spoken_counts = {}
    first_counts = {}
    for ev in events:
        if ev["type"] in {"rep", "milestone"} and "segment" in ev:
            first_counts.setdefault(ev["segment"], ev["time"])
    closing = next((ev for ev in events if ev["type"] == "workout_end"), None)
    deadline = closing.get("deadline") if closing else None

    def clip(text):
        path = synth(text)
        return path, dur(path)

    def available_at():
        return placed[-1][1] + GAP_S if placed else 0.0

    closing_text = "Workout complete."
    if deadline is not None:
        closing_path, closing_duration = clip(closing_text)

    for ev in events:
        kind = ev["type"]
        segment = ev.get("segment")
        target = ev["time"]
        start = max(target, available_at())
        if target < available_at() and kind not in IMPORTANT:
            skipped.append(ev)            # plain rep callout while audio is still playing
            continue
        if kind == "milestone" and start - target > MAX_MILESTONE_DELAY_S:
            skipped.append(ev)            # too late to be worth saying
            continue

        count = ev.get("n", 0)
        missed_final = count > spoken_counts.get(segment, 0)
        if kind == "workout_end" and deadline is not None:
            text, path, d = closing_text, closing_path, closing_duration
        else:
            text = nar.say(kind, **ev)
            path, d = clip(text)

        if kind == "transition":
            boundary = ev.get("boundary")
            if missed_final:
                count_word = "rep" if count == 1 else "reps"
                counted_text = f"{count} {count_word}. {text}"
                path, d = clip(counted_text)
                text = counted_text
            if boundary is not None:
                target = max(ev["not_before"], boundary - d)
                start = max(target, available_at())
            next_count = first_counts.get(segment + 1) if segment is not None else None
            if next_count is not None and start + d + GAP_S > next_count:
                skipped.append(ev)
                continue
        elif kind == "workout_end":
            if missed_final:
                count_word = "rep" if count == 1 else "reps"
                text = f"{count} {count_word}. Done."
                path, d = clip(text)
            if deadline is not None:
                if d > deadline and missed_final:
                    text = f"{count} {count_word}."
                    path, d = clip(text)
                if d > deadline:
                    skipped.append(ev)
                    continue
                target = start = deadline - d
                while placed and available_at() > start:
                    skipped.append(placed.pop()[0])
                    timeline.pop()
        if kind != "workout_end" and deadline is not None and start + d + GAP_S > deadline - closing_duration:
            skipped.append(ev)
            continue

        timeline.append({
            "event": kind, "text": text, "audio": str(path),
            "time": round(start, 3),
            "end": min(round(start + d, 3), deadline) if deadline is not None else round(start + d, 3),
            "scheduled": round(target, 3), "delay": round(start - target, 3),
        })
        placed.append((ev, start + d))
        if kind in {"rep", "milestone"}:
            spoken_counts[segment] = count
    return timeline, skipped


def summarize(timeline, skipped, n_events):
    delays = [t["delay"] for t in timeline]
    return {
        "events": n_events, "spoken": len(timeline), "skipped": len(skipped),
        "skip_rate": round(len(skipped) / n_events, 3) if n_events else 0.0,
        "max_delay_s": max(delays) if delays else 0.0,
        "mean_delay_s": round(sum(delays) / len(delays), 3) if delays else 0.0,
    }
