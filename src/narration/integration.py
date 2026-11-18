"""Narrate predicted pipeline results, without loading annotations or a model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .burn_in import generate_srt, burn_in_video
from .events import build_events
from .schedule import build_timeline, summarize


def analysis_to_segments(analysis):
    """Accept a WorkoutAnalysisResult or its JSON mapping; retain absolute times."""
    data = analysis.to_dict() if hasattr(analysis, "to_dict") else analysis
    if data.get("run_id") != "v2":
        raise ValueError("Expected a v2 workout analysis")
    duration = float(data["duration_s"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Expected a positive finite workout duration")
    segments = []
    previous_end = 0.0
    for source in data["segments"]:
        start, end = float(source["start_time"]), float(source["end_time"])
        if not (math.isfinite(start) and math.isfinite(end) and previous_end <= start < end <= duration):
            raise ValueError("Invalid or overlapping segment boundaries")
        previous_end = end
        if not source["is_exercise"]:
            continue
        times = [float(t) for t in source["rep_times"]]
        if source["rep_count"] != len(times):
            raise ValueError("Rep count does not match completion timestamps")
        # The counter deliberately pads segment ends so a boundary-cut rep can
        # complete just after the predicted boundary. Preserve those timestamps.
        if any(not math.isfinite(t) or not start <= t <= duration for t in times) or any(
            b <= a for a, b in zip(times, times[1:])
        ):
            raise ValueError("Rep timestamps must be ordered between segment start and video end")
        segments.append({"exercise": source["exercise"], "start_time": start,
                         "end_time": end, "rep_times": times})
    return data, segments


def narrate_analysis(analysis, output_dir, *, seed=42, use_bank=False,
                     synth=None, dur=None, render=None):
    """Write scheduled speech and a provenance report. Existing outputs are preserved.

    Synthesis and rendering are injectable for offline integration checks.
    """
    data, segments = analysis_to_segments(analysis)
    if not segments:
        raise ValueError("No predicted exercise segments to narrate")
    output = Path(output_dir)
    audio, report = output / "narration.wav", output / "narration.json"
    if audio.exists() or report.exists():
        raise FileExistsError("Narration outputs already exist; choose a new output directory")
    duration = float(data["duration_s"])
    events = build_events(segments, align_starts=True, workout_end=duration)
    options = {"seed": seed, "use_bank": use_bank}
    if synth is not None:
        options["synth"] = synth
    if dur is not None:
        options["dur"] = dur
    timeline, skipped = build_timeline(events, **options)
    if render is None:
        from .render import render_track
        render = render_track
    output.mkdir(parents=True, exist_ok=True)
    render(timeline, duration, audio)
    srt_file = output / "narration.srt"
    generate_srt(timeline, srt_file)
    result = {"video_id": data["video_id"], "run_id": data["run_id"],
              "rep_counter":data.get('rep_counter'), "rep_counter_config":data.get('rep_counter_config',{}),
              "experiment_id": data.get("experiment_id"), "duration_s": data["duration_s"],
              "phrase_source_requested": "bank" if use_bank else "templates", "seed": seed,
              "segments": segments, "timeline": timeline, "skipped": skipped,
              "summary": summarize(timeline, skipped, len(events)), "audio": str(audio),
              "subtitles": str(srt_file)}
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm", action="store_true", help="Use cached phrase bank with template fallback")
    parser.add_argument("--burn-in", action="store_true", help="Burn subtitles and audio into check.mp4")
    parser.add_argument("--video", help="Video path or video ID for burn-in (defaults to video_id from analysis)")
    args = parser.parse_args()
    result = narrate_analysis(json.loads(args.analysis.read_text(encoding="utf-8")),
                              args.output_dir, seed=args.seed, use_bank=args.llm)
    print(json.dumps(result["summary"], indent=2))
    if args.burn_in:
        vid_ref = args.video or result.get("video_id")
        if not vid_ref:
            raise ValueError("Cannot burn-in without a video path or video_id")
        check_path = burn_in_video(vid_ref, args.output_dir)
        print(f"Burned-in video written to {check_path}")


if __name__ == "__main__":
    main()
