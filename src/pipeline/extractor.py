"""extractor.py -- Pose extraction and feature input resolution."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..data_pipeline.pose_utils import (
    angle_deg_batch,
    best_side_angle,
    largest_bbox_pose_index,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = REPO_ROOT / "data" / "features"
SYNTH_DIR = REPO_ROOT / "data" / "synthetic"
MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker_full.task"

# MediaPipe landmark indices
_SH_L, _SH_R = 11, 12
_EL_L, _EL_R = 13, 14
_WR_L, _WR_R = 15, 16
_HI_L, _HI_R = 23, 24
_KN_L, _KN_R = 25, 26
_AN_L, _AN_R = 27, 28


def _min_vis(lm_list, *indices: int) -> float:
    if lm_list is None:
        return 0.0
    return min((lm_list[i].visibility for i in indices), default=0.0)


def _compute_frame_angles(img_lm, wld_lm, width: int, height: int, vis_thresh: float = 0.5) -> dict:
    """Compute 2D and 3D joint angles for a single pose detection."""
    row: dict = {}

    def vis3(a, b, c):
        return _min_vis(img_lm, a, b, c)

    def pt2d(idx):
        if img_lm is None:
            return np.array([float("nan"), float("nan")])
        lm = img_lm[idx]
        return np.array([lm.x * width, lm.y * height])

    def angle2d(a, b, c):
        pa, pb, pc = pt2d(a), pt2d(b), pt2d(c)
        return angle_deg_batch(pa[np.newaxis], pb[np.newaxis], pc[np.newaxis])[0]

    def pt3d(lm_list, idx):
        if lm_list is None:
            return np.array([float("nan")] * 3)
        lm = lm_list[idx]
        return np.array([lm.x, lm.y, lm.z])

    def angle3d(a, b, c):
        pa = pt3d(wld_lm, a)
        pb = pt3d(wld_lm, b)
        pc = pt3d(wld_lm, c)
        return angle_deg_batch(pa[np.newaxis], pb[np.newaxis], pc[np.newaxis])[0]

    # Knee (hip-knee-ankle)
    for name, sh, kn, an in [("knee_L", _HI_L, _KN_L, _AN_L), ("knee_R", _HI_R, _KN_R, _AN_R)]:
        vis = vis3(sh, kn, an)
        row[name] = angle2d(sh, kn, an) if vis >= vis_thresh else float("nan")
        row[f"{name}_3d"] = angle3d(sh, kn, an) if vis >= vis_thresh else float("nan")
        row[f"{name}_vis"] = vis

    # Elbow (shoulder-elbow-wrist)
    for name, sh, el, wr in [("elbow_L", _SH_L, _EL_L, _WR_L), ("elbow_R", _SH_R, _EL_R, _WR_R)]:
        vis = vis3(sh, el, wr)
        row[name] = angle2d(sh, el, wr) if vis >= vis_thresh else float("nan")
        row[f"{name}_3d"] = angle3d(sh, el, wr) if vis >= vis_thresh else float("nan")
        row[f"{name}_vis"] = vis

    # Hip (shoulder-hip-knee)
    for name, sh, hi, kn in [("hip_L", _SH_L, _HI_L, _KN_L), ("hip_R", _SH_R, _HI_R, _KN_R)]:
        vis = vis3(sh, hi, kn)
        row[name] = angle2d(sh, hi, kn) if vis >= vis_thresh else float("nan")
        row[f"{name}_3d"] = angle3d(sh, hi, kn) if vis >= vis_thresh else float("nan")
        row[f"{name}_vis"] = vis

    # Arm raise (hip-shoulder-wrist)
    for name, hi, sh, wr in [("arm_raise_L", _HI_L, _SH_L, _WR_L), ("arm_raise_R", _HI_R, _SH_R, _WR_R)]:
        vis = vis3(hi, sh, wr)
        row[name] = angle2d(hi, sh, wr) if vis >= vis_thresh else float("nan")
        row[f"{name}_3d"] = angle3d(hi, sh, wr) if vis >= vis_thresh else float("nan")
        row[f"{name}_vis"] = vis

    # Leg spread
    vis_ls = _min_vis(img_lm, _HI_L, _HI_R, _AN_L, _AN_R)
    if vis_ls >= vis_thresh and img_lm is not None:
        hm2d = np.array([
            0.5 * (img_lm[_HI_L].x + img_lm[_HI_R].x) * width,
            0.5 * (img_lm[_HI_L].y + img_lm[_HI_R].y) * height,
        ])
        row["leg_spread"] = angle_deg_batch(pt2d(_AN_L)[np.newaxis], hm2d[np.newaxis], pt2d(_AN_R)[np.newaxis])[0]
        if wld_lm is not None:
            hm3d = np.array([
                0.5 * (wld_lm[_HI_L].x + wld_lm[_HI_R].x),
                0.5 * (wld_lm[_HI_L].y + wld_lm[_HI_R].y),
                0.5 * (wld_lm[_HI_L].z + wld_lm[_HI_R].z),
            ])
            row["leg_spread_3d"] = angle_deg_batch(pt3d(wld_lm, _AN_L)[np.newaxis], hm3d[np.newaxis], pt3d(wld_lm, _AN_R)[np.newaxis])[0]
        else:
            row["leg_spread_3d"] = float("nan")
    else:
        row["leg_spread"] = float("nan")
        row["leg_spread_3d"] = float("nan")
    row["leg_spread_vis"] = vis_ls

    # Ankle distance normalized
    vis_ad = _min_vis(img_lm, _SH_L, _SH_R, _AN_L, _AN_R)
    if vis_ad >= vis_thresh and img_lm is not None:
        sh_width = abs(img_lm[_SH_L].x - img_lm[_SH_R].x)
        an_dist = abs(img_lm[_AN_L].x - img_lm[_AN_R].x)
        row["ankle_dist_norm"] = an_dist / sh_width if sh_width > 1e-6 else float("nan")
    else:
        row["ankle_dist_norm"] = float("nan")
    row["ankle_dist_norm_vis"] = vis_ad

    # Torso tilt
    vis_tt = _min_vis(img_lm, _SH_L, _SH_R, _HI_L, _HI_R)
    if vis_tt >= vis_thresh and img_lm is not None:
        sm2d = np.array([
            0.5 * (img_lm[_SH_L].x + img_lm[_SH_R].x) * width,
            0.5 * (img_lm[_SH_L].y + img_lm[_SH_R].y) * height,
        ])
        hm2d = np.array([
            0.5 * (img_lm[_HI_L].x + img_lm[_HI_R].x) * width,
            0.5 * (img_lm[_HI_L].y + img_lm[_HI_R].y) * height,
        ])
        torso_vec = sm2d - hm2d
        vert_vec = np.array([0.0, -1.0])
        norm_t = np.linalg.norm(torso_vec)
        if norm_t > 1e-8:
            cos_t = np.dot(torso_vec, vert_vec) / norm_t
            row["torso_tilt"] = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
        else:
            row["torso_tilt"] = float("nan")

        if wld_lm is not None:
            sm3d = np.array([
                0.5 * (wld_lm[_SH_L].x + wld_lm[_SH_R].x),
                0.5 * (wld_lm[_SH_L].y + wld_lm[_SH_R].y),
                0.5 * (wld_lm[_SH_L].z + wld_lm[_SH_R].z),
            ])
            hm3d = np.array([
                0.5 * (wld_lm[_HI_L].x + wld_lm[_HI_R].x),
                0.5 * (wld_lm[_HI_L].y + wld_lm[_HI_R].y),
                0.5 * (wld_lm[_HI_L].z + wld_lm[_HI_R].z),
            ])
            tv3 = sm3d - hm3d
            vert3 = np.array([0.0, -1.0, 0.0])
            norm_tv3 = np.linalg.norm(tv3)
            if norm_tv3 > 1e-8:
                cos3 = np.dot(tv3, vert3) / norm_tv3
                row["torso_tilt_3d"] = float(np.degrees(np.arccos(np.clip(cos3, -1.0, 1.0))))
            else:
                row["torso_tilt_3d"] = float("nan")
        else:
            row["torso_tilt_3d"] = float("nan")
    else:
        row["torso_tilt"] = float("nan")
        row["torso_tilt_3d"] = float("nan")
    row["torso_tilt_vis"] = vis_tt

    return row


def extract_features_from_video(
    video_path: Path,
    model_path: Path = MODEL_PATH,
    stride: int = 1,
    vis_thresh: float = 0.5,
) -> pd.DataFrame:
    """Extract pose angle time series directly from an MP4 video."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe pose landmarker model not found at {model_path}.\n"
            "Download pose_landmarker_full.task or provide a pre-extracted angles CSV."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=2,
        output_segmentation_masks=False,
    )

    rows = []
    frame_idx = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as lm:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lm.detect(mp_img)

            poses = res.pose_landmarks or []
            world_poses = res.pose_world_landmarks or []
            pose_detected = len(poses) > 0

            if pose_detected:
                subj_idx = largest_bbox_pose_index(poses)
                img_lm = poses[subj_idx]
                wld_lm = world_poses[subj_idx] if subj_idx < len(world_poses) else None
            else:
                img_lm = None
                wld_lm = None

            angle_row = _compute_frame_angles(img_lm, wld_lm, width, height, vis_thresh)
            row = {
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "pose_detected": pose_detected,
                "n_poses": len(poses),
                "subject_jump": False,
                **angle_row,
            }
            rows.append(row)
            frame_idx += 1

    cap.release()

    if not rows:
        raise RuntimeError(f"No frames could be decoded from {video_path}")

    df = pd.DataFrame(rows)

    # Add best-side columns (knee_best, elbow_best)
    for joint in ("knee", "elbow"):
        best, best_vis, use_right = best_side_angle(
            df[f"{joint}_L"].to_numpy(),
            df[f"{joint}_R"].to_numpy(),
            df[f"{joint}_L_vis"].to_numpy(),
            df[f"{joint}_R_vis"].to_numpy(),
        )
        df[f"{joint}_best"] = best
        df[f"{joint}_best_vis"] = best_vis
        df[f"{joint}_side"] = np.where(use_right, "R", "L")

    return df


def resolve_features_input(
    video_input: str | Path | pd.DataFrame,
    features_dir: Path = FEATURES_DIR,
    synth_dir: Path = SYNTH_DIR,
    model_path: Path = MODEL_PATH,
) -> Tuple[pd.DataFrame, str]:
    """Resolve an input (ID, video file, CSV, or DataFrame) into (features_df, video_id)."""
    if isinstance(video_input, pd.DataFrame):
        df = video_input.copy()
        video_id = "in_memory_video"
        return df, video_id

    input_str = str(video_input).strip()
    path_obj = Path(input_str)

    # 1. Direct CSV file
    if path_obj.is_file() and path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(path_obj)
        video_id = path_obj.stem.replace("_angles", "")
        return df, video_id

    # 2. Check if input_str is a video_id whose angles CSV exists in features_dir
    possible_csv = features_dir / f"{input_str}_angles.csv"
    if possible_csv.exists():
        df = pd.read_csv(possible_csv)
        return df, input_str

    # Also check without _angles
    possible_csv2 = features_dir / f"{input_str}.csv"
    if possible_csv2.exists():
        df = pd.read_csv(possible_csv2)
        return df, input_str

    # 3. Direct video file
    if path_obj.is_file() and path_obj.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        video_id = path_obj.stem
        # Check if pre-extracted angles CSV exists in features_dir
        cached_csv = features_dir / f"{video_id}_angles.csv"
        if cached_csv.exists():
            logger.info("Found cached angles CSV at %s", cached_csv)
            df = pd.read_csv(cached_csv)
            return df, video_id

        # Otherwise extract on the fly
        logger.info("Extracting pose features from %s...", path_obj)
        df = extract_features_from_video(path_obj, model_path=model_path)
        return df, video_id

    # 4. Check if it's a video_id inside synthetic directory
    for split in ("test", "val", "train"):
        candidate_mp4 = synth_dir / split / f"{input_str}.mp4"
        if candidate_mp4.exists():
            cached_csv = features_dir / f"{input_str}_angles.csv"
            if cached_csv.exists():
                df = pd.read_csv(cached_csv)
                return df, input_str
            df = extract_features_from_video(candidate_mp4, model_path=model_path)
            return df, input_str

    raise FileNotFoundError(
        f"Could not find or resolve input: '{video_input}'.\n"
        f"Looked for:\n"
        f"  - CSV file at '{path_obj}'\n"
        f"  - Precomputed features at '{possible_csv}'\n"
        f"  - Video file at '{path_obj}' or in '{synth_dir}/*/{input_str}.mp4'"
    )


def load_ground_truth_if_available(
    video_id: str,
    synth_dir: Path = SYNTH_DIR,
) -> tuple[list[dict], float, dict[str, int]] | tuple[None, None, None]:
    """Load ground truth segments, video duration, and rep counts if companion JSON exists."""
    for split in ("test", "val", "train"):
        j_path = synth_dir / split / f"{video_id}.json"
        if j_path.exists():
            with open(j_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            fps = float(meta.get("fps", 30))
            duration_s = float(meta.get("duration_s", 0.0))
            raw_segments = meta.get("segments", [])
            gt_segments = []
            annotated_reps: dict[str, int] = {}

            for s in raw_segments:
                start_s = s.get("start_s", s.get("start_frame", 0) / fps)
                end_s = s.get("end_s", s.get("end_frame", 0) / fps)
                label = s.get("label", "")
                reps = s.get("rep_count_annotated", s.get("rep_count", len(s.get("rep_pairs_out", []))))
                gt_segments.append({
                    "exercise": label,
                    "start_time": start_s,
                    "end_time": end_s,
                    "rep_count": reps,
                })
                if s.get("type") == "exercise":
                    annotated_reps[label] = annotated_reps.get(label, 0) + reps

            if not duration_s and gt_segments:
                duration_s = gt_segments[-1]["end_time"]

            return gt_segments, duration_s, annotated_reps

    return None, None, None
