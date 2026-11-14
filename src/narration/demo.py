"""End-to-end narration demo from ground truth (no video, no pose model needed).

    uv run python -m src.narration.demo
    uv run python -m src.narration.demo --noisy
    uv run python -m src.narration.demo --llm
    uv run python -m src.narration.demo --noisy --llm
"""
import argparse
import json
from pathlib import Path

from .loader import load_manifests, clips_to_segments
from .events import build_events
from .schedule import build_timeline, summarize
from .render import render_track
from .noise import perturb_segments

# squat -> push_up -> jump_jack, all from the test split
DEMO_CLIPS = ["stu2_72.mp4", "stu2_48.mp4", "stu4_19.mp4"]
OUT = Path("outputs")


def main(clips=DEMO_CLIPS, render=True, noisy=False, llm=False):
    OUT.mkdir(exist_ok=True)
    bounds, splits = load_manifests()
    segments, total = clips_to_segments(clips, bounds, splits)
    if noisy:
        segments = perturb_segments(segments)
    tag = ("_noisy" if noisy else "") + ("_llm" if llm else "")

    events = build_events(segments)
    timeline, skipped = build_timeline(events, use_bank=llm)

    for s in segments:
        print(f"{s['exercise']:10s} {s['start_time']:6.1f}-{s['end_time']:6.1f}s  "
              f"{len(s['rep_times'])} reps  cadence=every {s['cadence']}")
    for t in timeline:
        print(f"{t['time']:7.2f}-{t['end']:7.2f}  [{t['event']:13s}] {t['text']}")
    stats = summarize(timeline, skipped, len(events))
    print(stats)

    json_path = OUT / f"narration_demo_timeline{tag}.json"
    json_path.write_text(
        json.dumps({"clips": clips, "total_s": total, "noisy": noisy, "llm": llm,
                    "stats": stats, "timeline": timeline}, indent=2))
    if render:
        wav_path = OUT / f"narration_demo{tag}.wav"
        render_track(timeline, total, wav_path)
        print(f"wrote {wav_path} and {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy", action="store_true", help="jitter, drop and duplicate reps")
    ap.add_argument("--llm", action="store_true",
                    help="use data/phrase_bank.json (falls back to templates if missing)")
    ap.add_argument("--no-render", action="store_true", help="skip writing the wav")
    a = ap.parse_args()
    main(render=not a.no_render, noisy=a.noisy, llm=a.llm)