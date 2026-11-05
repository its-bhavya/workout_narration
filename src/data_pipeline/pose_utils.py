"""
pose_utils.py – Pure, testable pose/angle helper functions.

All functions are stateless and dependency-free (numpy only where needed).
They are called by extract_pose_features.py but can be unit-tested in
isolation without touching MediaPipe or video I/O.

IMAGE vs VIDEO mode tradeoff
-----------------------------
IMAGE mode (default, used here): each frame is processed independently.
  + No tracking state leaks across hard cuts in stitched clips.
  + Consistent with the screening script (quality_screen.py).
  - Pose identity is not tracked across frames; instead, the primary subject
    is chosen STATELESSLY as the pose with the largest landmark bounding box.

VIDEO mode: MediaPipe propagates tracking state between frames.
  + More stable landmark positions within a single continuous shot.
  - Tracking state from one clip leaks into the next after a hard cut,
    potentially producing a plausible-but-wrong pose for the first few frames
    of the new clip.  In stitched synthetic videos this would introduce a
    systematic artifact at every clip boundary – exactly the kind of boundary
    cue we want to avoid.  For that reason IMAGE mode is the correct default.

MediaPipe landmark indices (33 total, 0-based)
----------------------------------------------
  11 = left shoulder  12 = right shoulder
  13 = left elbow     14 = right elbow
  15 = left wrist     16 = right wrist
  23 = left hip       24 = right hip
  25 = left knee      26 = right knee
  27 = left ankle     28 = right ankle
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Angle computation
# ---------------------------------------------------------------------------

def angle_deg(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> float:
    """Angle at vertex *b* in the triangle a-b-c, in degrees.

    Returns NaN when any coordinate is non-finite or when either arm of the
    angle is degenerate (zero length).

    Parameters
    ----------
    a, b, c : sequences of 2 or 3 floats (x, y[, z])
        Points in pixel space (2-D) or world space (3-D).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return float("nan")

    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    denom = norm_ba * norm_bc
    if denom < 1e-8:
        return float("nan")

    cosine = np.dot(ba, bc) / denom
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def angle_deg_batch(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Vectorised angle at *b* for N points; shape (N,) in degrees.

    Parameters
    ----------
    a, b, c : (N, D) arrays where D is 2 or 3.
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    denom = norm_ba * norm_bc

    n = len(a)
    out = np.full(n, np.nan, dtype=float)
    valid = (
        np.isfinite(a).all(axis=1)
        & np.isfinite(b).all(axis=1)
        & np.isfinite(c).all(axis=1)
        & (denom > 1e-8)
    )
    cosine = np.full(n, np.nan, dtype=float)
    cosine[valid] = (
        np.einsum("ij,ij->i", ba[valid], bc[valid]) / denom[valid]
    )
    out[valid] = np.degrees(np.arccos(np.clip(cosine[valid], -1.0, 1.0)))
    return out


# ---------------------------------------------------------------------------
# 2-D pixel coordinate conversion
# ---------------------------------------------------------------------------

def norm_to_pixels(
    pts_norm: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert normalised MediaPipe coords (x∈[0,1], y∈[0,1]) to pixels.

    Parameters
    ----------
    pts_norm : (N, 2+) array  – first two columns are x_norm, y_norm.
    width, height : frame dimensions.

    Returns
    -------
    (N, 2+) array  – x and y columns scaled; remaining columns unchanged.
    """
    out = pts_norm.copy().astype(float)
    out[:, 0] *= width
    out[:, 1] *= height
    return out


# ---------------------------------------------------------------------------
# Primary-subject selection (stateless, per-frame)
# ---------------------------------------------------------------------------

def largest_bbox_pose_index(pose_landmarks_list) -> int:
    """Return the index of the pose with the largest 2-D bounding box area.

    Uses normalised image coordinates (no need to scale to pixels for a
    relative comparison).  Returns 0 if the list is empty or has one entry.

    Parameters
    ----------
    pose_landmarks_list : list of MediaPipe NormalizedLandmarkList
        As returned by PoseLandmarkerResult.pose_landmarks.
    """
    if not pose_landmarks_list:
        return 0
    if len(pose_landmarks_list) == 1:
        return 0

    best_idx = 0
    best_area = -1.0
    for i, lm_list in enumerate(pose_landmarks_list):
        xs = [lm.x for lm in lm_list]
        ys = [lm.y for lm in lm_list]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best_idx = i
    return best_idx


def get_pose_bbox(lm_list):
    xs = [lm.x for lm in lm_list]
    ys = [lm.y for lm in lm_list]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    )


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


# class PosePersonTracker:
#     def __init__(self, max_distance=0.25, max_missed_frames=10):
#         self.previous_bbox = None
#         self.max_distance = max_distance
#         self.max_missed_frames = max_missed_frames
#         self.missed_frames = 0

#     def select_person(self, pose_landmarks_list):

#         if not pose_landmarks_list:
#             self.missed_frames += 1

#             if self.missed_frames > self.max_missed_frames:
#                 self.previous_bbox = None

#             return None

#         boxes = [
#             get_pose_bbox(pose)
#             for pose in pose_landmarks_list
#         ]

#         # First detection: choose the largest person
#         if self.previous_bbox is None:
#             idx = largest_bbox_pose_index(pose_landmarks_list)

#         else:
#             prev = self.previous_bbox

#             prev_cx = (prev[0] + prev[2]) / 2
#             prev_cy = (prev[1] + prev[3]) / 2

#             best_idx = None
#             best_score = float("inf")

#             for i, box in enumerate(boxes):
#                 cx = (box[0] + box[2]) / 2
#                 cy = (box[1] + box[3]) / 2

#                 distance = (
#                     (cx - prev_cx) ** 2 +
#                     (cy - prev_cy) ** 2
#                 ) ** 0.5

#                 iou = calculate_iou(prev, box)

#                 score = distance + 0.2 * (1 - iou)

#                 if score < best_score:
#                     best_score = score
#                     best_idx = i

#             # Reject a candidate that is too far away
#             if best_idx is None or best_score > self.max_distance + 0.2:
#                 self.missed_frames += 1

#                 if self.missed_frames <= self.max_missed_frames:
#                     return None

#                 # Tracking lost: reinitialize
#                 idx = largest_bbox_pose_index(pose_landmarks_list)

#             else:
#                 idx = best_idx

#         self.previous_bbox = boxes[idx]
#         self.missed_frames = 0

#         return idx


class PosePersonTracker:
    def __init__(
        self,
        max_distance=0.25,
        max_missed_frames=10,
        observation_frames=45,
        min_observations=15,
    ):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.observation_frames = observation_frames
        self.min_observations = min_observations

        self.segment_label = None
        self.exercise_signal = None
        # Persistent tracks: track_id -> tracking information
        self.tracks = {}
        self.next_track_id = 0

        # Current frame: detection index -> persistent track ID
        self.detection_track_ids = []

        self.frame_count = 0
        self.locked = False
        self.selected_track_id = None
        self.missed_frames = 0

    def reset_for_segment(self, segment_label, exercise_signal):
        self.segment_label = segment_label
        self.exercise_signal = exercise_signal

        self.tracks = {}
        self.next_track_id = 0
        self.detection_track_ids = []
        self.observation_history = []
        self.frame_count = 0
        self.locked = False
        self.selected_track_id = None
        self.missed_frames = 0

    @staticmethod
    def _center(box):
        return (
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
        )

    def _match_score(self, previous_box, current_box):
        """Lower score means a better match."""
        px, py = self._center(previous_box)
        cx, cy = self._center(current_box)

        distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        iou = calculate_iou(previous_box, current_box)

        return distance + 0.2 * (1 - iou)

    def _create_track(self, box):
        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = {
            "bbox": box,
            "missed": 0,
            "angles": [],
            "visibility": [],
        }

        return track_id

    def _update_tracks(self, boxes):
        """Match current detections to persistent tracks one-to-one."""
        self.detection_track_ids = [None] * len(boxes)

        possible_matches = []

        for track_id, track in self.tracks.items():
            for detection_idx, box in enumerate(boxes):
                score = self._match_score(track["bbox"], box)

                if score <= self.max_distance + 0.2:
                    possible_matches.append(
                        (score, track_id, detection_idx)
                    )

        # Greedy one-to-one matching, best matches first.
        possible_matches.sort(key=lambda item: item[0])

        matched_tracks = set()
        matched_detections = set()

        for score, track_id, detection_idx in possible_matches:
            if track_id in matched_tracks:
                continue

            if detection_idx in matched_detections:
                continue

            track = self.tracks[track_id]
            track["bbox"] = boxes[detection_idx]
            track["missed"] = 0

            self.detection_track_ids[detection_idx] = track_id

            matched_tracks.add(track_id)
            matched_detections.add(detection_idx)

        # Existing tracks not detected in this frame.
        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_tracks:
                track["missed"] += 1

                if track["missed"] > self.max_missed_frames:
                    del self.tracks[track_id]


                     # Recover if the selected person has been lost.
                    if track_id == self.selected_track_id:
                        self.selected_track_id = None
                        self.locked = False
                        self.frame_count = 0
                        self.missed_frames = 0

                        # Start fresh movement histories for remaining candidates.
                        for remaining_track in self.tracks.values():
                            remaining_track["angles"] = []
                            remaining_track["visibility"] = []

        # Every unmatched detection becomes a new persistent track.
        for detection_idx, box in enumerate(boxes):
            if detection_idx not in matched_detections:
                track_id = self._create_track(box)
                self.detection_track_ids[detection_idx] = track_id

    def _collect_movement(self, poses):
        """Store exercise movement against persistent track IDs."""
        if self.exercise_signal is None:
            return

        for detection_idx, pose in enumerate(poses):
            track_id = self.detection_track_ids[detection_idx]

            if track_id is None or track_id not in self.tracks:
                continue

            try:
                angle, visibility = self.exercise_signal(pose)
                angle = float(angle)
                visibility = float(visibility)

                if not math.isfinite(angle):
                    angle = float("nan")

                if not math.isfinite(visibility):
                    visibility = 0.0

                visibility = max(0.0, min(1.0, visibility))

            except (ValueError, TypeError, IndexError):
                angle = float("nan")
                visibility = 0.0

            history = self.tracks[track_id]
            history["angles"].append(angle)
            history["visibility"].append(visibility)

    def _movement_score(self, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) < self.min_observations:
            return 0.0

        return float(
            np.percentile(values, 90) -
            np.percentile(values, 10)
        )

    def _score_candidates(self):
        scores = {}

        for track_id, history in self.tracks.items():
            angles = np.asarray(history["angles"], dtype=float)
            visibility = np.asarray(history["visibility"], dtype=float)

            valid_angles = angles[np.isfinite(angles)]

            if len(valid_angles) >= self.min_observations:
                movement = self._movement_score(valid_angles)
                mean_visibility = (
                    float(np.mean(visibility))
                    if len(visibility) else 0.0
                )

                scores[track_id] = movement * mean_visibility

    # Primary selection: movement multiplied by visibility.
        if scores:
            return max(scores, key=scores.get)

        # Fallback: choose the most consistently visible candidate.
        visibility_scores = {}

        for track_id, history in self.tracks.items():
            visibility = np.asarray(history["visibility"], dtype=float)
            visibility = visibility[np.isfinite(visibility)]

            if len(visibility) >= self.min_observations:
                visibility_scores[track_id] = float(np.mean(visibility))

        if visibility_scores:
            return max(visibility_scores, key=visibility_scores.get)

        return None

    def _get_detection_index(self, track_id):
        """Find the current detection index belonging to a track."""
        for detection_idx, current_id in enumerate(
            self.detection_track_ids
        ):
            if current_id == track_id:
                return detection_idx

        return None

    def select_person(self, pose_landmarks_list):
        """Return the current detection index of the selected person."""
        poses = pose_landmarks_list or []

        # No pose detected in this frame.
                # No pose detected in this frame.
        if not poses:
            self.detection_track_ids = []

            # Update missed-frame counts for existing tracks.
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["missed"] += 1

                if self.tracks[track_id]["missed"] > self.max_missed_frames:
                    del self.tracks[track_id]

                    # Recover only when the selected track is actually lost.
                    if track_id == self.selected_track_id:
                        self.selected_track_id = None
                        self.locked = False
                        self.frame_count = 0
                        self.missed_frames = 0

                        # Reset candidate histories for fresh selection.
                        for remaining_track in self.tracks.values():
                            remaining_track["angles"] = []
                            remaining_track["visibility"] = []

            # Keep the selected person locked during temporary losses.
            if self.locked:
                if self.selected_track_id in self.tracks:
                    self.missed_frames += 1
                return None

            # Continue the observation phase if not locked.
            self.frame_count += 1

            if (
                self.exercise_signal is not None
                and self.frame_count >= self.observation_frames
            ):
                selected_id = self._score_candidates()

                if selected_id is not None:
                    self.selected_track_id = selected_id
                    self.locked = True

            return None
        
        boxes = [get_pose_bbox(pose) for pose in poses]

        # Assign persistent IDs to current detections.
        self._update_tracks(boxes)

        # Observation phase: collect movement for every candidate.
        if not self.locked:
            self._collect_movement(poses)
            self.frame_count += 1

            if (
                self.exercise_signal is not None
                and self.frame_count >= self.observation_frames
            ):
                selected_id = self._score_candidates()

                if selected_id is not None:
                    self.selected_track_id = selected_id
                    self.locked = True

            # During observation, temporarily use the largest person.
            if not self.locked:
                return largest_bbox_pose_index(poses)

            if self.selected_track_id is None:
                # No candidate had enough valid observations.
                return None

        # Tracking phase: follow only the selected persistent ID.
        if self.selected_track_id not in self.tracks:
            return None

        detection_idx = self._get_detection_index(
            self.selected_track_id
        )

        if detection_idx is None:
            self.missed_frames += 1
            return None

        self.missed_frames = 0
        return detection_idx



def hip_midpoint_norm(landmarks) -> tuple[float, float] | None:
    """Return (x, y) of the hip midpoint in normalised coordinates.

    Returns None if landmarks is None or visibility is 0 for both hips.
    Indices: left hip = 23, right hip = 24.
    """
    if landmarks is None:
        return None
    lh = landmarks[23]
    rh = landmarks[24]
    return (0.5 * (lh.x + rh.x), 0.5 * (lh.y + rh.y))


# ---------------------------------------------------------------------------
# Angle-vector jump (for boundary_jump_check)
# ---------------------------------------------------------------------------

def angle_vector_jump(
    v_prev: np.ndarray,
    v_curr: np.ndarray,
    min_components: int = 3,
) -> float:
    """L2 distance between two angle vectors, skipping NaN components.

    Parameters
    ----------
    v_prev, v_curr : 1-D arrays of angle values (degrees).  NaN means
        the component was not visible at that frame.
    min_components : minimum number of valid (non-NaN) shared components
        required; returns NaN if fewer are available.

    Returns
    -------
    L2 norm over valid components, or NaN.
    """
    v_prev = np.asarray(v_prev, dtype=float)
    v_curr = np.asarray(v_curr, dtype=float)
    valid = np.isfinite(v_prev) & np.isfinite(v_curr)
    if valid.sum() < min_components:
        return float("nan")
    diff = v_curr[valid] - v_prev[valid]
    return float(np.sqrt(np.dot(diff, diff)))


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------

def fill_short_gaps(series: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Linear interpolation of NaN runs up to *max_gap_frames* long.

    NaN runs longer than *max_gap_frames* are left as NaN.  Leading and
    trailing NaNs are never filled.

    Parameters
    ----------
    series : 1-D float array (may contain NaN).
    max_gap_frames : maximum gap length to interpolate (inclusive).

    Returns
    -------
    New array of the same shape with short gaps filled.
    """
    series = np.array(series, dtype=float)
    n = len(series)
    out = series.copy()

    i = 0
    while i < n:
        if not np.isnan(series[i]):
            i += 1
            continue
        # Find the start of a NaN run.
        gap_start = i
        while i < n and np.isnan(series[i]):
            i += 1
        gap_end = i  # exclusive; series[gap_end] is the first non-NaN after gap

        gap_len = gap_end - gap_start

        # Only fill if bounded on both sides and short enough.
        left_ok = gap_start > 0 and np.isfinite(series[gap_start - 1])
        right_ok = gap_end < n and np.isfinite(series[gap_end])

        if left_ok and right_ok and gap_len <= max_gap_frames:
            v_left = series[gap_start - 1]
            v_right = series[gap_end]
            for k in range(gap_len):
                alpha = (k + 1) / (gap_len + 1)
                out[gap_start + k] = v_left + alpha * (v_right - v_left)

    return out


# ---------------------------------------------------------------------------
# Best-side selection
# ---------------------------------------------------------------------------

def best_side_angle(
    left_angle: np.ndarray,
    right_angle: np.ndarray,
    left_vis: np.ndarray,
    right_vis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame: pick the side with higher visibility; tie goes to left.

    Parameters
    ----------
    left_angle, right_angle : (N,) float arrays (may contain NaN).
    left_vis, right_vis : (N,) float visibility arrays.

    Returns
    -------
    best_angle : (N,) array
    best_vis   : (N,) array
    side_mask  : (N,) bool array  – True = right side chosen
    """
    left_angle = np.asarray(left_angle, dtype=float)
    right_angle = np.asarray(right_angle, dtype=float)
    left_vis = np.asarray(left_vis, dtype=float)
    right_vis = np.asarray(right_vis, dtype=float)

    # right wins only when strictly better
    use_right = right_vis > left_vis

    best_angle = np.where(use_right, right_angle, left_angle)
    best_vis = np.where(use_right, right_vis, left_vis)
    return best_angle, best_vis, use_right
