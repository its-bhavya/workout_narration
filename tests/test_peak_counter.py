import numpy as np
import pandas as pd
import pytest

from src.rep_counting.peak_counter import PeakSettings, detect_peaks, estimate_period, count_peaks_for_exercise
from src.rep_counting.validate_peak_counter import match_intervals


@pytest.mark.parametrize("pattern,sign", [("down_up", 1), ("low_high_low", -1)])
def test_counts_full_cycles_at_recovery_instead_of_extremum(pattern, sign):
    t = np.arange(900) / 15
    signal = 100 + sign * 40 * np.cos(np.pi * t)
    result = detect_peaks(signal, t, pattern, .2)
    assert result["reps"] == 30
    assert all(a < b for a, b in zip(result["peak_times"], result["rep_times"]))
    assert all(.2 <= r["distance_s"] <= 1. for r in result["runs"])
    assert result["runs"][0]["period_s"] == pytest.approx(2., abs=1 / 15)


def test_no_repetition_across_missing_data():
    t = np.arange(100) / 15
    signal = np.r_[np.linspace(160, 60, 40), np.full(20, np.nan), np.linspace(60, 160, 40)]
    result = detect_peaks(signal, t, "down_up", .2)
    assert result["reps"] == 0
    assert len(result["runs"]) == 2


def test_width_rejects_single_frame_spikes():
    t = np.arange(150) / 15
    signal = np.full(150, 150.)
    signal[20::20] = 60
    assert detect_peaks(signal, t, "down_up", .2, PeakSettings(min_width_s=.2))["reps"] == 0


def test_low_motion_and_invalid_period_have_safe_fallback():
    t = np.arange(150) / 15
    signal = 100 + .1 * np.cos(t)
    assert detect_peaks(signal, t, "down_up", .2)["reps"] == 0
    assert estimate_period(np.ones(100), 15) == (None, 0.)
    assert estimate_period(np.full(100, np.nan), 15) == (None, 0.)


def test_public_api_preserves_absolute_timestamps():
    t = np.arange(150) / 15 + 20
    angle = 120 + 40 * np.cos(np.pi * (t - 20))
    frame = pd.DataFrame(dict(time_s=t, elbow_L=angle, elbow_R=angle))
    result = count_peaks_for_exercise(frame, "push_up", 20, 30)
    assert result["reps"] == 5
    assert all(20 <= x < 30 for x in result["rep_times"])


def test_matching_does_not_count_one_rep_twice_or_steal_match():
    assert match_intervals([1., 5.], [(5., 6.), (0., 7.)], tolerance=0) == 2
    assert match_intervals([1.], [(0., 1.), (1., 2.)], tolerance=0) == 1


def test_invalid_times_rejected():
    with pytest.raises(ValueError, match="timestamps"):
        detect_peaks([1, 2, 3], [0, 0, 1], "down_up", .2)


def test_training_validation_does_not_read_test_features(synthetic_cv_data, monkeypatch):
    from src.rep_counting import validate_peak_counter as validation
    data = synthetic_cv_data
    for meta in data["videos"].values():
        for segment in meta["segments"]:
            if segment["type"] == "exercise":
                segment["clip_id"] = f"{meta['video_id']}_{segment['label']}"
                segment["rep_pairs_out"] = []
    monkeypatch.setattr(validation, "catalog", lambda: data["videos"])
    monkeypatch.setattr(validation, "FEATURES_DIR", data["features"])
    monkeypatch.setattr(validation, "SYNTH_DIR", data["synth"])
    original = validation.feature_frame
    loaded = []

    def only_train(path, meta):
        assert meta["split"] == "train"
        loaded.append(meta["video_id"])
        return original(path, meta)

    monkeypatch.setattr(validation, "feature_frame", only_train)
    report = validation.validate(data["out"] / "peak_test", plots=0)
    assert len(loaded) == 5
    assert report["runtime_verified_segments"] == 15
    assert set(report["exercises"]) == {"push_up", "squat", "jump_jack"}
    assert report["candidate_count_per_exercise"] == 48
