import json

from src.rep_counting import calibrate_pushups as calibration


def test_calibration_never_loads_test_features(synthetic_cv_data, monkeypatch):
    data = synthetic_cv_data
    for meta in data["videos"].values():
        for segment in meta["segments"]:
            if segment["type"] == "exercise":
                segment["clip_id"] = f"{meta['video_id']}_{segment['label']}"
    monkeypatch.setattr(calibration, "catalog", lambda: data["videos"])
    monkeypatch.setattr(calibration, "FEATURES_DIR", data["features"])
    original = calibration.feature_frame
    loaded = []

    def train_only(path, meta):
        assert meta["split"] == "train"
        loaded.append(meta["video_id"])
        return original(path, meta)

    monkeypatch.setattr(calibration, "feature_frame", train_only)
    target = data["out"] / "pushups"
    result = calibration.calibrate(target)
    assert len(loaded) == 5
    assert result["n_pushup_segments"] == 5
    assert len(result["heldout_fold_checks"]) == 5
    assert json.loads((target / "selection.json").read_text())["split"] == "train"
