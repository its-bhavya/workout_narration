"""display.py -- Pretty printing and presentation formatters for workout results."""
from __future__ import annotations

from .models import WorkoutAnalysisResult


def format_seconds(s: float) -> str:
    """Format seconds into MM:SS.S or SS.Ss."""
    mins = int(s // 60)
    secs = s % 60
    if mins > 0:
        return f"{mins:02d}:{secs:04.1f}"
    return f"{secs:5.1f}s"


def print_workout_report(
    result: WorkoutAnalysisResult,
    show_narration: bool = True,
    show_gt: bool = True,
    max_cues: int | None = 25,
) -> None:
    """Print an exhaustive, beautifully formatted workout analysis report to stdout."""
    res = result
    sum_data = res.summary

    # Header
    print("\n" + "=" * 78)
    print(f" WORKOUT ANALYSIS & NARRATION PIPELINE REPORT: {res.video_id}")
    print("=" * 78)

    # High-level summary
    print(f" Total Duration      : {res.duration_s:.1f}s ({format_seconds(res.duration_s)})")
    print(f" Exercise Time       : {sum_data.exercise_duration_s:.1f}s ({sum_data.exercise_duration_s/max(0.01, res.duration_s)*100:.1f}%)")
    print(f" Transition/Rest Time: {sum_data.transition_duration_s:.1f}s ({sum_data.transition_duration_s/max(0.01, res.duration_s)*100:.1f}%)")
    print(f" Total Reps Counted  : {sum_data.total_reps}")
    print(f" Exercise Breakdown  : " + ", ".join(f"{k}: {v} reps" for k, v in sum_data.exercise_counts.items()))

    # Ground-truth comparison if present
    if show_gt and res.ground_truth:
        gt = res.ground_truth
        print("-" * 78)
        print(" GROUND-TRUTH BENCHMARK COMPARISON:")
        if gt.frame_accuracy is not None:
            print(f" Frame-level Segmentation Accuracy: {gt.frame_accuracy * 100:.1f}%")
        print(" Rep Count Accuracy (Predicted vs Annotated):")
        all_exercises = sorted(set(list(sum_data.exercise_counts.keys()) + list(gt.annotated_reps.keys())))
        for ex in all_exercises:
            pred = sum_data.exercise_counts.get(ex, 0)
            anno = gt.annotated_reps.get(ex, 0)
            diff = pred - anno
            sign = f"+{diff}" if diff > 0 else str(diff)
            status = "MATCH" if diff == 0 else ("OFF-BY-1" if abs(diff) <= 1 else f"DIFF ({sign})")
            print(f"   • {ex:<12}: Counted = {pred:>2} reps | Annotated = {anno:>2} reps  [{status}]")

    # Segment Table
    print("\n" + "-" * 78)
    print(" SEGMENT BREAKDOWN & REP COUNTS:")
    print("-" * 78)
    print(
        f"{'#':<3} {'Type':<12} {'Time Range':<18} {'Duration':<9} {'Reps':<6} {'Pace':<9} {'Signal/Flag':<16}"
    )
    print("-" * 78)

    for seg in res.segments:
        time_range = f"{seg.start_time:5.1f}s - {seg.end_time:5.1f}s"
        dur_str = f"{seg.duration_s:5.1f}s"
        if not seg.is_exercise:
            reps_str = "-"
            pace_str = "-"
            detail = "[rest/idle]"
        else:
            reps_str = str(seg.rep_count)
            pace_str = f"{seg.pace_s_per_rep:.2f}s" if seg.pace_s_per_rep is not None else "-"
            detail = seg.flag if seg.flag else (seg.signal_used or "tracked")

        print(
            f"{seg.segment_index:<3} {seg.exercise:<12} {time_range:<18} {dur_str:<9} {reps_str:<6} {pace_str:<9} {detail:<16}"
        )

    # Detailed Rep Timestamps
    ex_segs = [s for s in res.segments if s.is_exercise and s.rep_times]
    if ex_segs:
        print("\n" + "-" * 78)
        print(" PER-REP TIMESTAMPS (SECONDS FROM VIDEO START):")
        print("-" * 78)
        for s in ex_segs:
            times_fmt = ", ".join(f"{t:.1f}s" for t in s.rep_times)
            print(f" Segment {s.segment_index} ({s.exercise}, {s.rep_count} reps):")
            print(f"   {times_fmt}")

    # Narration Timeline Cue Sheet
    if show_narration and res.timeline_cues:
        print("\n" + "=" * 78)
        print(" NARRATION ENGINE CUE SHEET (ALIGNED TO VIDEO TIMELINE):")
        print("=" * 78)
        print(f"{'Time':<8} {'Cue Type':<16} {'Narration Script / Prompt Text'}")
        print("-" * 78)

        cues_to_show = res.timeline_cues
        truncated = False
        if max_cues is not None and len(cues_to_show) > max_cues:
            cues_to_show = cues_to_show[:max_cues]
            truncated = True

        for cue in cues_to_show:
            t_str = f"{cue.time_s:6.1f}s"
            print(f"{t_str:<8} {cue.cue_type:<16} \"{cue.text}\"")

        if truncated:
            print(f" ... and {len(res.timeline_cues) - max_cues} more narration cues.")

    print("=" * 78 + "\n")
