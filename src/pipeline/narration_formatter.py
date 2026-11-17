"""narration_formatter.py -- Generate timed narration cues from segments and rep counts."""
from __future__ import annotations

from typing import Sequence
from .models import NarrationCue, SegmentResult, WorkoutSummary

EXERCISE_DISPLAY_NAMES = {
    "squat": "squats",
    "push_up": "push ups",
    "jump_jack": "jumping jacks",
    "transition": "rest",
}

EXERCISE_COACHING_TIPS = {
    "squat": "Keep your chest up and drive through your heels.",
    "push_up": "Maintain a straight back and steady core.",
    "jump_jack": "Stay light on your toes with full arm extension.",
}


def format_duration(seconds: float) -> str:
    """Format duration into friendly English (e.g. '1 minute 15 seconds' or '25 seconds')."""
    s = int(round(seconds))
    mins = s // 60
    secs = s % 60
    if mins > 0 and secs > 0:
        return f"{mins} minute{'s' if mins > 1 else ''} and {secs} second{'s' if secs > 1 else ''}"
    elif mins > 0:
        return f"{mins} minute{'s' if mins > 1 else ''}"
    else:
        return f"{secs} seconds"


def generate_segment_cues(
    segment: SegmentResult,
    next_segment: SegmentResult | None = None,
) -> list[NarrationCue]:
    """Generate narration cues for an individual segment."""
    cues: list[NarrationCue] = []

    if not segment.is_exercise:
        # Transition / rest segment
        rest_duration = round(segment.duration_s)
        next_hint = ""
        if next_segment and next_segment.is_exercise:
            next_name = EXERCISE_DISPLAY_NAMES.get(next_segment.exercise, next_segment.exercise)
            next_hint = f" Next up: {next_name}."

        if rest_duration >= 3.0:
            text = f"Rest for {format_duration(rest_duration)}.{next_hint}"
        else:
            text = f"Quick transition.{next_hint}"

        cues.append(
            NarrationCue(
                time_s=segment.start_time,
                cue_type="transition",
                text=text.strip(),
                exercise="transition",
                metadata={"duration_s": segment.duration_s},
            )
        )
        return cues

    # Exercise segment
    name = EXERCISE_DISPLAY_NAMES.get(segment.exercise, segment.exercise)
    tip = EXERCISE_COACHING_TIPS.get(segment.exercise, "")

    # 1. Start cue
    start_text = f"Let's go with {name} for {format_duration(segment.duration_s)}."
    if tip and segment.duration_s >= 8.0:
        start_text += f" {tip}"

    cues.append(
        NarrationCue(
            time_s=segment.start_time,
            cue_type="exercise_start",
            text=start_text,
            exercise=segment.exercise,
            metadata={"duration_s": segment.duration_s},
        )
    )

    # 2. Rep cues & milestones
    total_reps = len(segment.rep_times)
    for i, t in enumerate(segment.rep_times, start=1):
        # Determine milestone flavor
        if i == 5 and total_reps >= 8:
            rep_text = f"5, good pace!"
            cue_type = "milestone"
        elif i == 10 and total_reps >= 12:
            rep_text = f"10, keep strong!"
            cue_type = "milestone"
        elif i == total_reps and total_reps >= 3:
            rep_text = f"{i}, last one!"
            cue_type = "milestone"
        else:
            rep_text = str(i)
            cue_type = "rep"

        cues.append(
            NarrationCue(
                time_s=t,
                cue_type=cue_type,
                text=rep_text,
                exercise=segment.exercise,
                rep_number=i,
                metadata={"total_reps": total_reps},
            )
        )

    # 3. Exercise completion cue
    if total_reps > 0:
        summary_text = f"Completed {total_reps} {name}."
        if segment.pace_s_per_rep is not None and total_reps >= 2:
            summary_text += f" Average pace: {segment.pace_s_per_rep:.1f} seconds per rep."
    elif segment.flag:
        summary_text = f"{name.capitalize()} segment completed (flagged: {segment.flag})."
    else:
        summary_text = f"Completed {name}."

    # End cue placed at end of segment
    cues.append(
        NarrationCue(
            time_s=segment.end_time,
            cue_type="exercise_end",
            text=summary_text,
            exercise=segment.exercise,
            metadata={"rep_count": total_reps, "duration_s": segment.duration_s},
        )
    )

    return cues


def generate_workout_timeline(
    segments: Sequence[SegmentResult],
    summary: WorkoutSummary,
) -> list[NarrationCue]:
    """Generate the full chronological timeline of narration cues for the entire workout."""
    timeline: list[NarrationCue] = []

    # Opening workout announcement
    n_exercises = summary.exercise_segments
    timeline.append(
        NarrationCue(
            time_s=0.0,
            cue_type="workout_start",
            text=f"Welcome to your workout. {n_exercises} exercise sets lined up. Let's get started!",
            metadata={"total_duration_s": summary.total_duration_s, "exercise_sets": n_exercises},
        )
    )

    # Add per-segment cues
    for idx, seg in enumerate(segments):
        next_seg = segments[idx + 1] if idx + 1 < len(segments) else None
        seg_cues = generate_segment_cues(seg, next_seg)
        seg.cues = seg_cues
        timeline.extend(seg_cues)

    # Closing workout announcement
    exercise_breakdown_parts = []
    for ex, count in summary.exercise_counts.items():
        if count > 0:
            ex_name = EXERCISE_DISPLAY_NAMES.get(ex, ex)
            exercise_breakdown_parts.append(f"{count} {ex_name}")

    breakdown_str = ", ".join(exercise_breakdown_parts) if exercise_breakdown_parts else "all exercises"

    closing_text = (
        f"Workout complete! Total time: {format_duration(summary.total_duration_s)}. "
        f"You crushed {summary.total_reps} total reps including {breakdown_str}. Great job!"
    )

    timeline.append(
        NarrationCue(
            time_s=summary.total_duration_s,
            cue_type="workout_end",
            text=closing_text,
            metadata={"total_reps": summary.total_reps, "total_duration_s": summary.total_duration_s},
        )
    )

    # Sort strictly by time_s, then cue order
    timeline.sort(key=lambda c: (c.time_s, 0 if c.cue_type == "workout_start" else 1))
    return timeline
