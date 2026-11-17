import sys
from pathlib import Path

import pytest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_pipeline"))

from src.data_pipeline.pose_utils import PosePersonTracker


def test_tracker_assigns_persistent_ids():
    tracker = PosePersonTracker()

    tracker.reset_for_segment("squat", exercise_signal=None)

    # Two people detected.
    tracker._update_tracks([
        (0.1, 0.1, 0.3, 0.5),
        (0.6, 0.1, 0.8, 0.5),
    ])

    first_ids = tracker.detection_track_ids.copy()

    # Same people, slightly shifted.
    tracker._update_tracks([
        (0.11, 0.1, 0.31, 0.5),
        (0.61, 0.1, 0.81, 0.5),
    ])

    second_ids = tracker.detection_track_ids.copy()

    assert first_ids == second_ids
    assert len(set(second_ids)) == 2


def test_movement_score_prefers_more_active_candidate():
    tracker = PosePersonTracker(min_observations=5)

    tracker.reset_for_segment("squat", exercise_signal=None)

    tracker.tracks = {
        0: {
            "angles": [170, 169, 171, 170, 170],
            "visibility": [0.95] * 5,
        },
        1: {
            "angles": [170, 140, 110, 140, 170],
            "visibility": [0.95] * 5,
        },
    }

    selected = tracker._score_candidates()

    assert selected == 1


def test_visibility_fallback_when_movement_history_is_insufficient():
    tracker = PosePersonTracker(min_observations=5)

    tracker.reset_for_segment("squat", exercise_signal=None)

    tracker.tracks = {
        0: {
            "angles": [170, 170],
            "visibility": [0.4, 0.4],
        },
        1: {
            "angles": [170, 170],
            "visibility": [0.95, 0.95],
        },
    }

    selected = tracker._score_candidates()

    assert selected is None


def test_tracker_selects_person_actually_exercising():
    tracker = PosePersonTracker(
        observation_frames=5,
        min_observations=3,
    )

    def exercise_signal(pose):
        return pose[0].z, pose[0].visibility

    tracker.reset_for_segment(
        "squat",
        exercise_signal=exercise_signal,
    )

    def make_pose(x_min, x_max, angle):
        return [
            SimpleNamespace(
                x=x_min if i % 2 == 0 else x_max,
                y=0.1 if i < 16 else 0.8,
                z=angle,
                visibility=0.95,
            )
            for i in range(33)
        ]

    stationary_angles = [170, 169, 171, 170, 170]
    exercising_angles = [170, 140, 110, 140, 170]

    for stationary, exercising in zip(
        stationary_angles,
        exercising_angles,
    ):
        poses = [
            make_pose(0.05, 0.35, stationary),
            make_pose(0.65, 0.85, exercising),
        ]

        tracker.select_person(poses)

    assert tracker.locked is True
    assert tracker.selected_track_id == 1

from src.data_pipeline.extract_pose_features import get_clip_start_frame


def test_clip_start_before_first_join():
    assert get_clip_start_frame(299, [300, 1500]) == 0


def test_clip_start_exactly_at_join():
    assert get_clip_start_frame(300, [300, 1500]) == 300


def test_clip_start_after_multiple_joins():
    assert get_clip_start_frame(1600, [300, 1500]) == 1500


def test_tracker_recovers_when_selected_track_is_deleted():
    tracker = PosePersonTracker(max_missed_frames=1)

    tracker.reset_for_segment("squat", exercise_signal=None)

    selected_id = tracker._create_track(
        (0.1, 0.1, 0.3, 0.5)
    )
    other_id = tracker._create_track(
        (0.6, 0.1, 0.8, 0.5)
    )

    tracker.selected_track_id = selected_id
    tracker.locked = True

    tracker.tracks[other_id]["angles"] = [170, 140, 110]
    tracker.tracks[other_id]["visibility"] = [0.9, 0.9, 0.9]

    # The selected person disappears, but the other remains.
    tracker._update_tracks([(0.6, 0.1, 0.8, 0.5)])
    tracker._update_tracks([(0.6, 0.1, 0.8, 0.5)])

    assert selected_id not in tracker.tracks
    assert tracker.selected_track_id is None
    assert tracker.locked is False
    assert tracker.tracks[other_id]["angles"] == []
    assert tracker.tracks[other_id]["visibility"] == []


def test_tracker_recovers_after_consecutive_empty_frames():
    tracker = PosePersonTracker(max_missed_frames=1)

    tracker.reset_for_segment("squat", exercise_signal=None)

    selected_id = tracker._create_track(
        (0.1, 0.1, 0.3, 0.5)
    )

    tracker.selected_track_id = selected_id
    tracker.locked = True

    # No people detected for consecutive frames.
    tracker.select_person([])
    tracker.select_person([])

    assert selected_id not in tracker.tracks
    assert tracker.selected_track_id is None
    assert tracker.locked is False