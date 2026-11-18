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
    df = pd.DataFrame({"time_s": [0.,1.,2.], "knee_best": [170.,160.,170.]})
    with pytest.raises(ValueError, match="run_id=v2"):
        resolve_features_input(df)
    csv_file = tmp_path / "custom_angles.csv"
    df.to_csv(csv_file,index=False)
    with pytest.raises(ValueError, match="run_id != v2"):
        resolve_features_input(csv_file)
    df["run_id"] = "v2"
    out, vid = resolve_features_input(df)
    assert vid == "in_memory_video" and len(out)==3


def test_pipeline_with_frozen_fixture(synthetic_bundle):
    artifact=synthetic_bundle
    pipeline=WorkoutPipeline(model_path=artifact["model"],windows_path=artifact["windows"],
                             features_dir=artifact["features"],synth_dir=artifact["synth"],rep_method='state_machine')
    vid=next(vid for vid,meta in artifact["videos"].items() if meta["split"]=="train")
    result=pipeline.analyze(vid)
    assert result.run_id=="v2" and result.experiment_id=="fixture"
    assert result.duration_s==24.
    assert result.segments[0].start_time==0.
    assert result.segments[-1].end_time==24.
    assert result.ground_truth is None
    assert result.to_narration_script()[0]["cue_type"]=="workout_start"
    with pytest.raises(ValueError,match="frozen CV"):
        WorkoutPipeline(model_path=artifact["model"],windows_path=artifact["windows"],
                        features_dir=artifact["features"],tau=.99,min_len=1)


def test_pipeline_save_json(tmp_path: Path,synthetic_bundle):
    artifact=synthetic_bundle
    vid=next(vid for vid,meta in artifact["videos"].items() if meta["split"]=="test")
    result=analyze_workout(vid,show_gt=True,model_path=artifact["model"],
                           windows_path=artifact["windows"],features_dir=artifact["features"],
                           synth_dir=artifact["synth"],rep_method='state_machine')
    target=tmp_path/"result.json";result.save_json(target)
    content=json.loads(target.read_text())
    assert content["run_id"]=="v2" and content["experiment_id"]=="fixture"
    assert content["comparison_scope"]=="final_test"
    assert content["duration_s"]==24.
    assert content["ground_truth"]["segmentation_metrics"]
