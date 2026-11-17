import copy
import json

import pytest

from src.narration.integration import analysis_to_segments, narrate_analysis


def analysis():
    return {"run_id": "v2", "video_id": "prediction", "experiment_id": "fixture",
            "duration_s": 30., "segments": [
                {"exercise": "squat", "is_exercise": True, "start_time": 0.,
                 "end_time": 10., "rep_count": 3, "rep_times": [3., 6., 9.]},
                {"exercise": "transition", "is_exercise": False, "start_time": 10.,
                 "end_time": 20., "rep_count": 0, "rep_times": []},
                {"exercise": "push_up", "is_exercise": True, "start_time": 20.,
                 "end_time": 25., "rep_count": 0, "rep_times": []},
                {"exercise": "transition", "is_exercise": False, "start_time": 25.,
                 "end_time": 30., "rep_count": 0, "rep_times": []}]}


def test_prediction_schedule_preserves_boundaries_and_zero_reps(tmp_path):
    data = analysis()
    before = copy.deepcopy(data)
    rendered = []
    result = narrate_analysis(data, tmp_path, synth=lambda text: "fake.mp3",
                              dur=lambda path: .5,
                              render=lambda *args: rendered.append(args))
    assert data == before
    assert len(result["segments"]) == 2
    assert result["segments"][0]["rep_times"] == [3., 6., 9.]
    assert result["segments"][1]["rep_times"] == []
    transition = next(t for t in result["timeline"] if t["event"] == "transition")
    assert transition["scheduled"] == 20.
    assert result["timeline"][-1]["scheduled"] == 30.
    assert rendered[0][1] == 30.
    assert json.loads((tmp_path / "narration.json").read_text())["experiment_id"] == "fixture"
    with pytest.raises(FileExistsError):
        narrate_analysis(data, tmp_path)


@pytest.mark.parametrize("field,value", [("rep_count", 4), ("rep_times", [3., 12., 31.]),
                                         ("rep_times", [6., 3., 9.])])
def test_invalid_rep_outputs_rejected(field, value):
    data = analysis()
    data["segments"][0][field] = value
    with pytest.raises(ValueError):
        analysis_to_segments(data)


def test_counter_end_padding_is_preserved():
    data = analysis()
    data["segments"][0]["rep_times"][-1] = 10.2
    _, segments = analysis_to_segments(data)
    assert segments[0]["end_time"] == 10.
    assert segments[0]["rep_times"][-1] == 10.2


def test_live_pipeline_result(synthetic_bundle, tmp_path):
    from src.pipeline import WorkoutPipeline
    a = synthetic_bundle
    pipeline = WorkoutPipeline(model_path=a["model"], windows_path=a["windows"],
                               features_dir=a["features"], synth_dir=a["synth"])
    vid = next(iter(a["videos"]))
    result = pipeline.analyze(vid)
    report = narrate_analysis(result, tmp_path, synth=lambda text: "fake.mp3",
                             dur=lambda path: .5, render=lambda *args: None)
    assert report["video_id"] == vid
    assert sum(len(s["rep_times"]) for s in report["segments"]) == result.summary.total_reps
