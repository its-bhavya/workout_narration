from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.rep_counting import compare_signal_dimensions as comparison
from src.rep_counting.rep_counter import EXERCISES


def angle_frame():
    t = np.arange(600) / 15
    return pd.DataFrame(dict(time_s=t, knee_L=120 + 40 * np.cos(np.pi * t),
                             knee_R=120 + 40 * np.cos(np.pi * t),
                             knee_L_3d=120 + 40 * np.cos(np.pi * t / 2),
                             knee_R_3d=120 + 40 * np.cos(np.pi * t / 2)))


@pytest.mark.parametrize("method", ["state_machine", "peaks"])
def test_signal_source_changes_count_without_mutating_defaults(method):
    frame = angle_frame()
    before = deepcopy(EXERCISES)
    a = comparison.count_variant(frame, "squat", 0, 40, dimensions="2d", method=method)
    b = comparison.count_variant(frame, "squat", 0, 40, dimensions="3d", method=method)
    assert a["reps"] == 20
    assert b["reps"] == 10
    assert b["signal"].endswith("_3d")
    assert EXERCISES == before


def test_missing_3d_column_never_falls_back():
    with pytest.raises(ValueError, match="fallback forbidden"):
        comparison.count_variant(angle_frame().drop(columns="knee_L_3d"), "squat", 0, 40)


def test_nan_3d_signal_remains_a_failed_count_not_a_2d_count():
    frame = angle_frame()
    frame[["knee_L_3d", "knee_R_3d"]] = np.nan
    result = comparison.count_variant(frame, "squat", 0, 40)
    assert result["reps"] == 0 and not result["usable"]


def test_comparison_loads_training_only(synthetic_cv_data, monkeypatch):
    data = synthetic_cv_data
    for meta in data["videos"].values():
        for seg in meta["segments"]:
            if seg["type"] == "exercise":
                seg["clip_id"] = f"{meta['video_id']}_{seg['label']}"
                seg["rep_pairs_out"] = []
        if meta["split"] == "train":
            path = data["features"] / f"{meta['video_id']}_angles.csv"
            frame = pd.read_csv(path)
            for cfg in EXERCISES.values():
                for column in cfg["columns"]:
                    frame[f"{column}_3d"] = frame[column]
            frame.to_csv(path, index=False)
    monkeypatch.setattr(comparison, "catalog", lambda: data["videos"])
    monkeypatch.setattr(comparison, "FEATURES_DIR", data["features"])
    monkeypatch.setattr(comparison, "SYNTH_DIR", data["synth"])
    original = comparison.feature_frame
    loaded = []

    def only_train(path, meta):
        assert meta["split"] == "train"
        loaded.append(meta["video_id"])
        return original(path, meta)

    monkeypatch.setattr(comparison, "feature_frame", only_train)
    report = comparison.compare(data["out"] / "dimensions", plots=False)
    assert len(loaded) == 5 and report["n_segments"] == 15
    for methods in report["exercises"].values():
        for dimensions in methods.values():
            assert dimensions["2d"]["tuned_other_folds"] == dimensions["3d"]["tuned_other_folds"]
