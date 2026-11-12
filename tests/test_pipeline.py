"""tests/test_pipeline.py -- Unit and integration tests for the workout analysis pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    NarrationCue,
    SegmentResult,
    WorkoutAnalysisResult,
    WorkoutPipeline,
    WorkoutSummary,
    analyze_workout,
    resolve_features_input,
)
from src.pipeline.narration_formatter import (
    format_duration,
    generate_segment_cues,
    generate_workout_timeline,
)


def test_format_duration():
    assert format_duration(45) == "45 seconds"
    assert format_duration(60) == "1 minute"
    assert format_duration(75) == "1 minute and 15 seconds"
    assert format_duration(130) == "2 minutes and 10 seconds"


def test_narration_cue_generation():
    # Test transition segment
    t_seg = SegmentResult(
        segment_index=1,
        exercise="transition",
        start_time=20.0,
        end_time=28.0,
        duration_s=8.0,
        is_exercise=False,
    )
    next_ex = SegmentResult(
        segment_index=2,
        exercise="push_up",
        start_time=28.0,
        end_time=50.0,
        duration_s=22.0,
        is_exercise=True,
    )
    cues = generate_segment_cues(t_seg, next_ex)
    assert len(cues) == 1
    assert cues[0].cue_type == "transition"
    assert "Rest for 8 seconds" in cues[0].text
    assert "Next up: push ups" in cues[0].text

    # Test exercise segment with reps
    ex_seg = SegmentResult(
        segment_index=0,
        exercise="squat",
        start_time=0.0,
        end_time=15.0,
        duration_s=15.0,
        is_exercise=True,
        rep_count=5,
        rep_times=[2.5, 5.0, 7.5, 10.0, 12.5],
        pace_s_per_rep=2.5,
    )
    ex_cues = generate_segment_cues(ex_seg)
    assert len(ex_cues) >= 6  # 1 start + 5 reps + 1 end
    assert ex_cues[0].cue_type == "exercise_start"
    assert "squats" in ex_cues[0].text
    assert ex_cues[-1].cue_type == "exercise_end"
    assert "Completed 5 squats" in ex_cues[-1].text


def test_resolve_features_input(tmp_path: Path):
    # Test DataFrame input
    df = pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "knee_best": [170.0, 160.0, 170.0]})
    out_df, vid = resolve_features_input(df)
    assert vid == "in_memory_video"
    assert len(out_df) == 3

    # Test CSV path input
    csv_file = tmp_path / "custom_angles.csv"
    df.to_csv(csv_file, index=False)
    out_df2, vid2 = resolve_features_input(csv_file)
    assert vid2 == "custom"
    assert len(out_df2) == 3


def test_pipeline_on_synth_test_0001():
    pipeline = WorkoutPipeline()
    result = pipeline.analyze("synth_test_0001", show_gt=True)

    assert isinstance(result, WorkoutAnalysisResult)
    assert result.video_id == "synth_test_0001"
    assert result.duration_s > 100.0
    assert result.summary.total_reps > 50
    assert len(result.segments) >= 5

    # Check that segments have timestamps and rep counts
    for s in result.segments:
        assert s.end_time > s.start_time
        if s.is_exercise:
            assert s.exercise in ("squat", "push_up", "jump_jack")
            assert len(s.rep_times) == s.rep_count
        else:
            assert s.exercise == "transition"
            assert s.rep_count == 0

    # Check narration cues
    cues = result.to_narration_script()
    assert len(cues) > 20
    assert cues[0]["cue_type"] == "workout_start"
    assert cues[-1]["cue_type"] == "workout_end"

    # Verify JSON export
    j_str = result.to_json()
    parsed = json.loads(j_str)
    assert parsed["video_id"] == "synth_test_0001"
    assert "summary" in parsed
    assert "segments" in parsed
    assert "narration_timeline" in parsed


def test_pipeline_save_json(tmp_path: Path):
    out_file = tmp_path / "test_out.json"
    result = analyze_workout("synth_test_0001", show_gt=False)
    result.save_json(out_file)
    assert out_file.exists()
    content = json.loads(out_file.read_text(encoding="utf-8"))
    assert content["video_id"] == "synth_test_0001"
    assert content["summary"]["total_reps"] > 0
