from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.v2_protocol import sha256
from src.rep_counting.coordinate_counter import SIGNALS, CoordinatePeakSettings, count_coordinate_peaks, choose
from src.rep_counting.extract_coordinate_training import signature, verify_case, case_name, extract_case, extract_all
from src.rep_counting.rep_counter import EXERCISES


def sidecar(times):
    motion = np.cos(2 * np.pi * (times - times[0]) / 3.)
    data = dict(frame=np.round(times * 30).astype(int), time_s=times, n_poses=np.ones(len(times), dtype=int))
    for name in SIGNALS:
        data[name] = (120 + 35 * motion) if "elbow" in name else (1 + .3 * motion)
        data[name + "_confidence"] = np.ones(len(times))
    return pd.DataFrame(data)


def test_coordinate_counter_is_read_only_and_keeps_absolute_completion_times():
    frame = sidecar(np.arange(180, 360, 2) / 30.)
    before, config = frame.copy(deep=True), deepcopy(EXERCISES)
    result = count_coordinate_peaks(frame, CoordinatePeakSettings(signal="shoulder_to_wrist_height"))
    assert result["reps"] == 2
    assert all(6. < t < 12. for t in result["rep_times"])
    pd.testing.assert_frame_equal(frame, before)
    assert EXERCISES == config


def test_quality_selection_does_not_maximize_count_and_median_uses_actual_timeline():
    prepared = {name: dict(usable=True, score=1.) for name in SIGNALS}
    prepared["elbow_R"]["score"] = 10.
    detections = {name: list(range(10)) for name in SIGNALS}
    detections["elbow_R"] = [1., 2.]
    times, selected = choose(prepared, detections, "fresh_elbow_quality")
    assert selected == "elbow_R" and times == [1., 2.]
    detections["shoulder_height"] = list(range(40))
    times, selected = choose(prepared, detections, "median")
    assert len(times) == 10 and times == detections[selected]


def test_coordinate_sidecar_has_no_silent_canonical_fallback():
    with pytest.raises(ValueError, match="no canonical fallback"):
        count_coordinate_peaks(sidecar(np.arange(90) / 15.).drop(columns="head_height"))


def test_extraction_blocks_test_frames_and_non_output_paths(tmp_path):
    with pytest.raises(ValueError, match="test frames"):
        extract_case(dict(split="test"))
    with pytest.raises(ValueError, match="outputs"):
        extract_all(tmp_path / "canonical_features")


def test_resume_verification_rejects_changed_sidecar(tmp_path):
    case = dict(video_id="fixture", segment_index=3)
    csv = tmp_path / f"{case_name(case)}.csv"
    csv.write_text("frame,time_s\n0,0\n")
    metadata = dict(plan_signature="frozen", case=case, csv_sha256=sha256(csv))
    (tmp_path / f"{case_name(case)}.json").write_text(json.dumps(metadata))
    assert verify_case(tmp_path, case, "frozen") == metadata
    csv.write_text("frame,time_s\n0,1\n")
    with pytest.raises(ValueError, match="provenance"):
        verify_case(tmp_path, case, "frozen")


def test_fold_validation_uses_complete_training_sidecars_only(synthetic_cv_data, tmp_path, monkeypatch):
    from src.rep_counting import validate_coordinate_training as validation
    data = synthetic_cv_data
    root = tmp_path / "experiment"
    sidecars = root / "outputs/sidecars"; sidecars.mkdir(parents=True)
    reference = root / "reference"; reference.mkdir()
    cases, protected, previous_rows, previous_inputs = [], {}, [], []
    for vid, meta in data["videos"].items():
        if meta["split"] != "train":
            continue
        index, segment = next((i, s) for i, s in enumerate(meta["segments"]) if s["label"] == "push_up")
        segment.update(clip_id=f"{vid}_push", rep_count_annotated=2, rep_pairs_out=[[180, 240], [240, 360]])
        annotation = data["synth"] / "train" / f"{vid}.json"
        annotation.write_text(json.dumps(meta))
        feature = data["features"] / f"{vid}_angles.csv"
        case = dict(video_id=vid, segment_index=index, split="train", fold=meta["fold"], clip_id=segment["clip_id"],
                    start_frame=180, end_frame=360, fps=30., annotation_path=str(annotation),
                    canonical_features_path=str(feature), canonical_features_sha256=sha256(feature))
        cases.append(case)
        for p in (annotation, feature): protected[str(p)] = sha256(p)
        previous_inputs.append(dict(video_id=vid, features_sha256=sha256(feature)))
        previous_rows.append(dict(video_id=vid, fold=meta["fold"], segment_index=index, clip_id=segment["clip_id"],
                                  exercise="push_up", annotated=2, predicted=0, error=-2, absolute_error=2, hits=0, rep_times="[]"))
    plan = dict(protocol=validation.PROTOCOL, split="train", cases=cases, protected_hashes=protected, source_hashes={})
    sig = signature(plan)
    (sidecars / "plan.json").write_text(json.dumps(dict(plan_signature=sig, plan=plan)))
    (sidecars / "extraction_report.json").write_text(json.dumps(dict(plan_signature=sig, n_windows=len(cases))))
    for case in cases:
        csv = sidecars / f"{case_name(case)}.csv"
        sidecar(np.arange(180, 360, 2) / 30.).to_csv(csv, index=False)
        (sidecars / f"{case_name(case)}.json").write_text(json.dumps(dict(plan_signature=sig, case=case, csv_sha256=sha256(csv))))
    pd.DataFrame(previous_rows).to_csv(reference / "segments.csv", index=False)
    (reference / "report.json").write_text(json.dumps(dict(inputs=previous_inputs)))
    monkeypatch.setattr(validation, "ROOT", root)
    monkeypatch.setattr(validation, "REFERENCE", reference)
    original = validation.feature_frame
    loaded = []
    def training_only(path, meta):
        assert meta["split"] == "train"
        loaded.append(meta["video_id"])
        return original(path, meta)
    monkeypatch.setattr(validation, "feature_frame", training_only)
    result = validation.validate(sidecars, root / "outputs/validation", plots=False)
    assert len(loaded) == 5 and result["n_windows"] == 5
    assert result["candidate_count"] == 324
    assert result["inputs_verified_unchanged"] and result["runtime_verified_windows"] == 5
