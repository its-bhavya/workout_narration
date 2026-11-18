from dataclasses import asdict
import json

import pytest
import pandas as pd

from src.data_pipeline.v2_protocol import sha256
from src.rep_counting.evaluate_peak_test import evaluate, freeze_settings
from src.rep_counting.peak_counter import PeakSettings
from src.rep_counting.peak_improvements import ImprovementSettings
from src.pipeline import WorkoutPipeline


def reports(tmp_path):
    exercises = ("jump_jack", "push_up", "squat")
    selected = tmp_path / "training.json"
    old = tmp_path / "old.json"
    selected.write_text(json.dumps(dict(split="train", source_hashes={}, exercises={
        ex: dict(selected_settings=asdict(ImprovementSettings())) for ex in exercises})))
    old.write_text(json.dumps(dict(split="train", exercises={ex: dict(families=dict(unguided=dict(
        settings=asdict(PeakSettings(period_distance_fraction=0.))))) for ex in exercises})))
    return selected, old


def test_freeze_rejects_test_selection(tmp_path):
    selected, old = reports(tmp_path)
    data = json.loads(selected.read_text()); data["split"] = "test"
    selected.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="training-only"):
        freeze_settings(selected, old)


def test_test_evaluation_freezes_before_inference_and_preserves_features(synthetic_bundle, tmp_path, monkeypatch):
    artifact = synthetic_bundle
    # Add the extra cached channels required by the experimental signal builder
    # to this temporary test fixture only. Training provenance stays unchanged.
    for vid, meta in artifact["videos"].items():
        if meta["split"] == "test":
            path = artifact["features"] / f"{vid}_angles.csv"
            frame = pd.read_csv(path)
            for name in ("hip_L", "hip_R", "ankle_dist_norm"):
                frame[name] = 0.
            frame.to_csv(path, index=False)
    selected, old = reports(tmp_path)
    output = tmp_path / "test_eval"
    original = WorkoutPipeline.analyze
    seen = []

    def analyze(self, vid, **kwargs):
        frozen = json.loads((output / "frozen_settings.json").read_text())
        assert frozen["selection_split"] == "train"
        assert not frozen["parameters_tuned_on_test"]
        assert kwargs["show_gt"] is False
        assert self.videos[vid]["split"] == "test"
        seen.append(vid)
        return original(self, vid, **kwargs)

    monkeypatch.setattr(WorkoutPipeline, "analyze", analyze)
    result = evaluate(output, selected, old, model_path=artifact["model"], windows_path=artifact["windows"],
                       features_dir=artifact["features"], synth_dir=artifact["synth"])
    assert len(seen) == result["n_videos"] == 1
    assert result["test_inputs_verified_unchanged"]
    for method in result["methods"].values():
        assert method["end_to_end_video_exercise"]["overall"]["n"] == 3
        assert method["oracle_gt_segments"]["overall"]["n"] == 3
    for item in result["inputs"]:
        assert sha256(artifact["features"] / f"{item['video_id']}_angles.csv") == item["features_sha256"]
    with pytest.raises(FileExistsError):
        evaluate(output, selected, old)
