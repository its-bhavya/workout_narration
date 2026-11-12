"""Data models for workout analysis, segmentation, rep counting, and narration."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class NarrationCue:
    """A single narration cue aligned to video time for a TTS/narration engine."""

    time_s: float
    cue_type: Literal[
        "workout_start",
        "exercise_start",
        "rep",
        "milestone",
        "exercise_end",
        "transition",
        "workout_end",
    ]
    text: str
    exercise: str | None = None
    rep_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": round(self.time_s, 2),
            "cue_type": self.cue_type,
            "text": self.text,
            "exercise": self.exercise,
            "rep_number": self.rep_number,
            "metadata": self.metadata,
        }


@dataclass
class SegmentResult:
    """Detailed result for an individual exercise or transition segment."""

    segment_index: int
    exercise: str  # "squat", "push_up", "jump_jack", "transition"
    start_time: float
    end_time: float
    duration_s: float
    is_exercise: bool
    rep_count: int = 0
    rep_times: list[float] = field(default_factory=list)
    pace_s_per_rep: float | None = None
    signal_used: str = ""
    thresholds: tuple[float, float] | None = None
    range_deg: float | None = None
    flag: str = ""
    valid_fraction: float | None = None
    cues: list[NarrationCue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "exercise": self.exercise,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration_s": round(self.duration_s, 2),
            "is_exercise": self.is_exercise,
            "rep_count": self.rep_count,
            "rep_times": [round(t, 2) for t in self.rep_times],
            "pace_s_per_rep": round(self.pace_s_per_rep, 2) if self.pace_s_per_rep is not None else None,
            "signal_used": self.signal_used,
            "thresholds": self.thresholds,
            "range_deg": round(self.range_deg, 1) if self.range_deg is not None else None,
            "flag": self.flag,
            "valid_fraction": self.valid_fraction,
            "cues": [c.to_dict() for c in self.cues],
        }


@dataclass
class GroundTruthComparison:
    """Optional comparison metrics if ground truth is available for the video."""

    gt_segments: list[dict[str, Any]]
    frame_accuracy: float | None = None
    annotated_reps: dict[str, int] = field(default_factory=dict)
    predicted_reps: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gt_segments": self.gt_segments,
            "frame_accuracy": round(self.frame_accuracy, 3) if self.frame_accuracy is not None else None,
            "annotated_reps": self.annotated_reps,
            "predicted_reps": self.predicted_reps,
        }


@dataclass
class WorkoutSummary:
    """High-level summary of the whole workout video."""

    video_id: str
    total_duration_s: float
    exercise_duration_s: float
    transition_duration_s: float
    total_reps: int
    exercise_counts: dict[str, int] = field(default_factory=dict)
    exercise_durations: dict[str, float] = field(default_factory=dict)
    total_segments: int = 0
    exercise_segments: int = 0
    transition_segments: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "total_duration_s": round(self.total_duration_s, 2),
            "exercise_duration_s": round(self.exercise_duration_s, 2),
            "transition_duration_s": round(self.transition_duration_s, 2),
            "total_reps": self.total_reps,
            "exercise_counts": self.exercise_counts,
            "exercise_durations": {k: round(v, 2) for k, v in self.exercise_durations.items()},
            "total_segments": self.total_segments,
            "exercise_segments": self.exercise_segments,
            "transition_segments": self.transition_segments,
        }


@dataclass
class WorkoutAnalysisResult:
    """Complete result from the integrated segmentation and rep counting pipeline."""

    video_id: str
    duration_s: float
    summary: WorkoutSummary
    segments: list[SegmentResult]
    timeline_cues: list[NarrationCue] = field(default_factory=list)
    ground_truth: GroundTruthComparison | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "duration_s": round(self.duration_s, 2),
            "summary": self.summary.to_dict(),
            "segments": [s.to_dict() for s in self.segments],
            "narration_timeline": [c.to_dict() for c in self.timeline_cues],
            "ground_truth": self.ground_truth.to_dict() if self.ground_truth else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Save results to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent), encoding="utf-8")

    def to_narration_script(self) -> list[dict[str, Any]]:
        """Return the flat chronological timeline of narration cues ready for TTS."""
        return [c.to_dict() for c in self.timeline_cues]
