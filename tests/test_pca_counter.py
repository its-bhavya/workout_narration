from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.rep_counting.pca_counter import ANGLE_CHANNELS, PCASettings, project_motion, count_pca_reps
from src.rep_counting.rep_counter import EXERCISES


def frame_with_cycles():
    t = np.arange(900) / 15
    values = {"time_s": t}
    for i, name in enumerate(ANGLE_CHANNELS):
        values[name] = 100 - (20 + i) * np.cos(np.pi * t)
        values[name + "_3d"] = values[name].copy()
        values[name + "_vis"] = np.ones(len(t))
    return pd.DataFrame(values)


def test_recovers_shared_motion_and_does_not_mutate_inputs():
    frame = frame_with_cycles()
    original = frame.copy(deep=True)
    config = deepcopy(EXERCISES)
    result = count_pca_reps(frame, "squat", 0, 60)
    assert result["reps"] == 30
    assert result["pca"]["explained_variance"] > .99
    assert len(result["pca"]["channels"]) == len(ANGLE_CHANNELS)
    assert "prominence_score" in result["runs"][0]
    assert "prominence_deg" not in result["runs"][0]
    pd.testing.assert_frame_equal(frame, original)
    assert config == EXERCISES


def test_annotation_columns_do_not_affect_projection():
    frame = frame_with_cycles()
    expected = project_motion(frame)
    frame["seg_label"] = np.arange(len(frame)) * 100
    frame["rep_count_annotated"] = 9999
    frame["fold"] = 4
    actual = project_motion(frame)
    np.testing.assert_allclose(expected["signal"], actual["signal"])
    assert actual["channels"] == list(ANGLE_CHANNELS)


def test_long_tracking_gap_is_not_projected_as_motion():
    frame = frame_with_cycles()
    frame.loc[100:130, list(ANGLE_CHANNELS)] = np.nan
    result = project_motion(frame)
    assert np.isnan(result["signal"][100:131]).all()
    assert result["valid_fraction"] < 1


def test_low_visibility_and_motionless_channels_are_excluded():
    frame = frame_with_cycles()
    frame["knee_L_vis"] = .1
    frame["hip_R"] = 100.
    result = project_motion(frame)
    assert "knee_L" not in result["channels"]
    assert "hip_R" not in result["channels"]


def test_missing_3d_columns_do_not_fall_back():
    with pytest.raises(ValueError, match="fallback forbidden"):
        project_motion(frame_with_cycles().drop(columns="elbow_R_3d"), PCASettings(dimensions="3d"))


def test_insufficient_observation_returns_zero_instead_of_fabricating_reps():
    frame = frame_with_cycles()
    frame[list(ANGLE_CHANNELS)] = np.nan
    result = count_pca_reps(frame, "squat", 0, 60)
    assert result["reps"] == 0
    assert result["pca"]["flag"] == "insufficient_motion_channels"


def test_dimensions_have_separate_projections():
    frame = frame_with_cycles()
    for i, name in enumerate(ANGLE_CHANNELS):
        frame[name + "_3d"] = 100 - (20 + i) * np.cos(np.pi * frame.time_s / 2)
    result = count_pca_reps(frame, "squat", 0, 60, pca_settings=PCASettings(dimensions="3d"))
    assert result["reps"] == 15


def test_validation_reads_only_train_and_preserves_existing_inputs(synthetic_cv_data, tmp_path, monkeypatch):
    import json
    from src.data_pipeline.v2_protocol import sha256
    from src.rep_counting import validate_pca_counter as validation
    data = synthetic_cv_data
    reference = tmp_path / "reference"
    reference.mkdir()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for name in ("synthetic_index.csv", "splits_v2.csv", "folds.csv"):
        (manifests / name).write_text("fixture\n")
    inputs, rows = [], []
    for vid, meta in data["videos"].items():
        if meta["split"] != "train":
            continue
        path = data["features"] / f"{vid}_angles.csv"
        frame = pd.read_csv(path)
        for name in ANGLE_CHANNELS:
            if name not in frame:
                frame[name] = 100.
            frame[name + "_3d"] = frame[name]
        frame.to_csv(path, index=False)
        inputs.append(dict(video_id=vid, features_sha256=sha256(path),
                           annotations_sha256=sha256(data["synth"] / "train" / f"{vid}.json")))
        for i, seg in enumerate(meta["segments"]):
            if seg["type"] != "exercise":
                continue
            seg["clip_id"] = f"{vid}_{seg['label']}"
            seg["rep_pairs_out"] = []
            method, dimensions = {"squat": ("state_machine", "3d"),
                "push_up": ("state_machine", "2d"), "jump_jack": ("peaks", "2d")}[seg["label"]]
            rows.append(dict(video_id=vid, fold=meta["fold"], segment_index=i, clip_id=seg["clip_id"],
                             exercise=seg["label"], annotated=0, predicted=0, error=0,
                             absolute_error=0, hits=0, rep_times="[]", method=method, dimensions=dimensions))
    (reference / "report.json").write_text(json.dumps(dict(inputs=inputs)))
    pd.DataFrame(rows).to_csv(reference / "heldout_predictions.csv", index=False)
    monkeypatch.setattr(validation, "catalog", lambda: data["videos"])
    monkeypatch.setattr(validation, "FEATURES_DIR", data["features"])
    monkeypatch.setattr(validation, "SYNTH_DIR", data["synth"])
    monkeypatch.setattr(validation, "MANIFESTS", manifests)
    monkeypatch.setattr(validation, "REFERENCE_DIR", reference)
    original = validation.feature_frame
    loaded = []

    def train_only(path, meta):
        assert meta["split"] == "train"
        loaded.append(meta["video_id"])
        return original(path, meta)

    monkeypatch.setattr(validation, "feature_frame", train_only)
    report = validation.validate(tmp_path / "pca_output", plots=False)
    assert len(loaded) == 5
    assert report["runtime_verified_projections"] == 30
    assert report["existing_inputs_verified_unchanged"]
    for path, digest in report["protected_file_hashes"].items():
        assert sha256(path) == digest
