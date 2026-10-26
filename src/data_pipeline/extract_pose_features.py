"""
extract_pose_features.py – Phase 2 pose extraction + joint angles.

Runs MediaPipe PoseLandmarker on synthetic videos and writes:
  data/features/<video_id>_angles.csv      (per-frame angles)
  data/manifests/pose_extraction_summary.csv

Usage
-----
    # single video
    python src/data_pipeline/extract_pose_features.py --video data/synthetic/train/synth_train_0001.mp4

    # all videos
    python src/data_pipeline/extract_pose_features.py --all

    # all videos, 4 workers, force re-extraction
    python src/data_pipeline/extract_pose_features.py --all --workers 4 --force

    # use VIDEO mode (see pose_utils.py for tradeoff)
    python src/data_pipeline/extract_pose_features.py --all --mode video

Run from repo root.

IMAGE vs VIDEO mode
-------------------
Default is IMAGE mode (--mode image).  Each frame is processed independently,
so there is no tracking state that leaks across the hard cuts between stitched
clips.  This matches the quality screening script.

VIDEO mode (--mode video) can produce smoother landmark trajectories within a
continuous shot but will propagate stale tracking state for a few frames after
each clip join, potentially creating an artificial pose at boundaries.  Since
the whole pipeline is designed to be boundary-leakage-free (pose features only,
no pixel info), VIDEO mode is not recommended for the synthetic stitched data.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mediapipe_module
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from pose_utils import (
    angle_deg_batch,
    best_side_angle,
    hip_midpoint_norm,
    largest_bbox_pose_index,
    norm_to_pixels,
)

# ---------------------------------------------------------------------------
# Paths (all relative to repo root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
SUMMARY_PATH = PROJECT_ROOT / "data" / "manifests" / "pose_extraction_summary.csv"

SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"

# MediaPipe landmark indices
_SH_L, _SH_R = 11, 12
_EL_L, _EL_R = 13, 14
_WR_L, _WR_R = 15, 16
_HI_L, _HI_R = 23, 24
_KN_L, _KN_R = 25, 26
_AN_L, _AN_R = 27, 28

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lm_xyz_norm(lm_list, idx: int) -> tuple[float, float, float]:
    """Normalised (x, y, z) from a landmark list; NaN if None."""
    if lm_list is None:
        return (float("nan"),) * 3
    lm = lm_list[idx]
    return float(lm.x), float(lm.y), float(lm.z)


def _lm_xyz_world(lm_list, idx: int) -> tuple[float, float, float]:
    if lm_list is None:
        return (float("nan"),) * 3
    lm = lm_list[idx]
    return float(lm.x), float(lm.y), float(lm.z)


def _vis(lm_list, idx: int) -> float:
    if lm_list is None:
        return 0.0
    v = getattr(lm_list[idx], "visibility", None)
    return float(v) if v is not None else 0.0


def _min_vis(lm_list, *indices) -> float:
    return min(_vis(lm_list, i) for i in indices)


# ---------------------------------------------------------------------------
# Per-frame angle extraction
# ---------------------------------------------------------------------------

def _compute_angles_for_frame(
    img_lm,          # normalised landmark list (or None)
    wld_lm,          # world landmark list (or None)
    width: int,
    height: int,
    vis_thresh: float,
) -> dict:
    """Return a dict of angle values and their _vis companions for one frame."""

    row: dict = {}

    # -----------------------------------------------------------------------
    # Helper: min visibility for the three landmarks of an angle
    # -----------------------------------------------------------------------
    def vis3(lm_list, a, b, c):
        return _min_vis(lm_list, a, b, c)

    # -----------------------------------------------------------------------
    # 2-D angles (normalised -> pixels first)
    # -----------------------------------------------------------------------
    def pt2d(idx):
        if img_lm is None:
            return np.array([float("nan"), float("nan")])
        lm = img_lm[idx]
        return np.array([lm.x * width, lm.y * height])

    def angle2d(a, b, c):
        pa, pb, pc = pt2d(a), pt2d(b), pt2d(c)
        return angle_deg_batch(
            pa[np.newaxis], pb[np.newaxis], pc[np.newaxis]
        )[0]

    # -----------------------------------------------------------------------
    # 3-D angles (world landmarks, metres)
    # -----------------------------------------------------------------------
    def pt3d(lm_list, idx):
        if lm_list is None:
            return np.array([float("nan")] * 3)
        lm = lm_list[idx]
        return np.array([lm.x, lm.y, lm.z])

    def angle3d(a, b, c):
        pa = pt3d(wld_lm, a)
        pb = pt3d(wld_lm, b)
        pc = pt3d(wld_lm, c)
        return angle_deg_batch(
            pa[np.newaxis], pb[np.newaxis], pc[np.newaxis]
        )[0]

    # -----------------------------------------------------------------------
    # knee_L, knee_R  (hip-knee-ankle)
    # -----------------------------------------------------------------------
    for name, sh, kn, an in [
        ("knee_L", _HI_L, _KN_L, _AN_L),
        ("knee_R", _HI_R, _KN_R, _AN_R),
    ]:
        vis = vis3(img_lm, sh, kn, an)
        a2 = angle2d(sh, kn, an) if vis >= vis_thresh else float("nan")
        a3 = angle3d(sh, kn, an) if vis >= vis_thresh else float("nan")
        row[name] = a2
        row[f"{name}_3d"] = a3
        row[f"{name}_vis"] = vis

    # -----------------------------------------------------------------------
    # elbow_L, elbow_R  (shoulder-elbow-wrist)
    # -----------------------------------------------------------------------
    for name, sh, el, wr in [
        ("elbow_L", _SH_L, _EL_L, _WR_L),
        ("elbow_R", _SH_R, _EL_R, _WR_R),
    ]:
        vis = vis3(img_lm, sh, el, wr)
        a2 = angle2d(sh, el, wr) if vis >= vis_thresh else float("nan")
        a3 = angle3d(sh, el, wr) if vis >= vis_thresh else float("nan")
        row[name] = a2
        row[f"{name}_3d"] = a3
        row[f"{name}_vis"] = vis

    # -----------------------------------------------------------------------
    # hip_L, hip_R  (shoulder-hip-knee)
    # -----------------------------------------------------------------------
    for name, sh, hi, kn in [
        ("hip_L", _SH_L, _HI_L, _KN_L),
        ("hip_R", _SH_R, _HI_R, _KN_R),
    ]:
        vis = vis3(img_lm, sh, hi, kn)
        a2 = angle2d(sh, hi, kn) if vis >= vis_thresh else float("nan")
        a3 = angle3d(sh, hi, kn) if vis >= vis_thresh else float("nan")
        row[name] = a2
        row[f"{name}_3d"] = a3
        row[f"{name}_vis"] = vis

    # -----------------------------------------------------------------------
    # arm_raise_L, arm_raise_R  (hip-shoulder-wrist)
    # -----------------------------------------------------------------------
    for name, hi, sh, wr in [
        ("arm_raise_L", _HI_L, _SH_L, _WR_L),
        ("arm_raise_R", _HI_R, _SH_R, _WR_R),
    ]:
        vis = vis3(img_lm, hi, sh, wr)
        a2 = angle2d(hi, sh, wr) if vis >= vis_thresh else float("nan")
        a3 = angle3d(hi, sh, wr) if vis >= vis_thresh else float("nan")
        row[name] = a2
        row[f"{name}_3d"] = a3
        row[f"{name}_vis"] = vis

    # -----------------------------------------------------------------------
    # leg_spread  (angle at hip_mid between the two ankles)
    # Approximation: treat hip_mid as the vertex.
    # 2-D: compute pixel coords of hip_mid, ankle_L, ankle_R
    # -----------------------------------------------------------------------
    vis_ls = _min_vis(img_lm, _HI_L, _HI_R, _AN_L, _AN_R)
    if vis_ls >= vis_thresh and img_lm is not None:
        hm2d = np.array([
            0.5 * (img_lm[_HI_L].x + img_lm[_HI_R].x) * width,
            0.5 * (img_lm[_HI_L].y + img_lm[_HI_R].y) * height,
        ])
        al2d = pt2d(_AN_L)
        ar2d = pt2d(_AN_R)
        row["leg_spread"] = angle_deg_batch(
            al2d[np.newaxis], hm2d[np.newaxis], ar2d[np.newaxis]
        )[0]
        # 3-D
        if wld_lm is not None:
            hm3d = np.array([
                0.5 * (wld_lm[_HI_L].x + wld_lm[_HI_R].x),
                0.5 * (wld_lm[_HI_L].y + wld_lm[_HI_R].y),
                0.5 * (wld_lm[_HI_L].z + wld_lm[_HI_R].z),
            ])
            al3d = pt3d(wld_lm, _AN_L)
            ar3d = pt3d(wld_lm, _AN_R)
            row["leg_spread_3d"] = angle_deg_batch(
                al3d[np.newaxis], hm3d[np.newaxis], ar3d[np.newaxis]
            )[0]
        else:
            row["leg_spread_3d"] = float("nan")
    else:
        row["leg_spread"] = float("nan")
        row["leg_spread_3d"] = float("nan")
    row["leg_spread_vis"] = vis_ls

    # -----------------------------------------------------------------------
    # ankle_dist_norm  (|ankle_L.x - ankle_R.x| / shoulder_width, 2-D)
    # -----------------------------------------------------------------------
    vis_ad = _min_vis(img_lm, _SH_L, _SH_R, _AN_L, _AN_R)
    if vis_ad >= vis_thresh and img_lm is not None:
        sh_width = abs(img_lm[_SH_L].x - img_lm[_SH_R].x)
        an_dist = abs(img_lm[_AN_L].x - img_lm[_AN_R].x)
        row["ankle_dist_norm"] = an_dist / sh_width if sh_width > 1e-6 else float("nan")
    else:
        row["ankle_dist_norm"] = float("nan")
    row["ankle_dist_norm_vis"] = vis_ad

    # -----------------------------------------------------------------------
    # torso_tilt  (angle between hip_mid->shoulder_mid and vertical)
    # 0° = upright, ~90° = horizontal (push-up)
    # -----------------------------------------------------------------------
    vis_tt = _min_vis(img_lm, _SH_L, _SH_R, _HI_L, _HI_R)
    if vis_tt >= vis_thresh and img_lm is not None:
        # Use 2-D pixel coords
        sm2d = np.array([
            0.5 * (img_lm[_SH_L].x + img_lm[_SH_R].x) * width,
            0.5 * (img_lm[_SH_L].y + img_lm[_SH_R].y) * height,
        ])
        hm2d = np.array([
            0.5 * (img_lm[_HI_L].x + img_lm[_HI_R].x) * width,
            0.5 * (img_lm[_HI_L].y + img_lm[_HI_R].y) * height,
        ])
        # Vector from hip_mid to shoulder_mid
        torso_vec = sm2d - hm2d
        # Vertical reference vector (upward in image coords = negative y)
        vert_vec = np.array([0.0, -1.0])
        norm_t = np.linalg.norm(torso_vec)
        if norm_t > 1e-8:
            cos_t = np.dot(torso_vec, vert_vec) / norm_t
            row["torso_tilt"] = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
        else:
            row["torso_tilt"] = float("nan")
        # 3-D version
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
            vert3 = np.array([0.0, -1.0, 0.0])  # MediaPipe world: y up = negative
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


def _apply_vis_mask(row: dict, vis_thresh: float) -> dict:
    """Set any angle to NaN if its _vis companion is below threshold."""
    angle_names = [
        "knee_L", "knee_R", "elbow_L", "elbow_R",
        "hip_L", "hip_R", "arm_raise_L", "arm_raise_R",
        "leg_spread", "ankle_dist_norm", "torso_tilt",
    ]
    for name in angle_names:
        vis_val = row.get(f"{name}_vis", 0.0)
        if vis_val < vis_thresh:
            row[name] = float("nan")
            row[f"{name}_3d"] = float("nan")
    return row


# ---------------------------------------------------------------------------
# Per-video extraction
# ---------------------------------------------------------------------------

def extract_video_angles(
    video_path: Path,
    json_path: Path,
    output_csv: Path,
    model_path: Path,
    vis_thresh: float = 0.5,
    mode: str = "image",
) -> dict:
    """Extract per-frame angles for a single synthetic video.

    Parameters
    ----------
    video_path  : path to the .mp4 file
    json_path   : path to the companion boundary JSON
    output_csv  : where to write <video_id>_angles.csv
    model_path  : pose_landmarker_full.task
    vis_thresh  : landmarks below this visibility get NaN angles
    mode        : 'image' (default, stateless) or 'video'

    Returns
    -------
    summary dict with detection stats (merged into pose_extraction_summary.csv)
    """
    t0 = time.perf_counter()
    video_id = video_path.stem

    with open(json_path) as f:
        meta = json.load(f)
    fps = float(meta["fps"])
    width = int(meta["width"])
    height = int(meta["height"])

    # Build a frame->segment lookup for quick per-frame labelling.
    seg_at_frame: dict[int, dict] = {}
    for seg in meta.get("segments", []):
        for fr in range(seg["start_frame"], seg["end_frame"]):
            seg_at_frame[fr] = seg

    clip_join_set = set(meta.get("clip_join_frames", []))

    # Build landmarker
    running_mode = (
        mp_vision.RunningMode.IMAGE
        if mode == "image"
        else mp_vision.RunningMode.VIDEO
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_poses=2,
        output_segmentation_masks=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    rows = []
    frame_idx = 0
    n_detected = 0
    prev_hip_mid: tuple[float, float] | None = None
    prev_ts_ms = -1

    with mp_vision.PoseLandmarker.create_from_options(options) as lm:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            ts_ms = round(frame_idx * 1000.0 / fps)
            if mode == "video":
                ts_ms = max(ts_ms, prev_ts_ms + 1)
                prev_ts_ms = ts_ms

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mediapipe_module.Image(
                image_format=mediapipe_module.ImageFormat.SRGB, data=rgb
            )

            if mode == "image":
                result = lm.detect(mp_img)
            else:
                result = lm.detect_for_video(mp_img, ts_ms)

            poses = result.pose_landmarks or []
            world_poses = result.pose_world_landmarks or []
            n_poses = len(poses)
            pose_detected = n_poses > 0

            if pose_detected:
                n_detected += 1
                subj_idx = largest_bbox_pose_index(poses)
                img_lm = poses[subj_idx]
                wld_lm = world_poses[subj_idx] if subj_idx < len(world_poses) else None
            else:
                img_lm = None
                wld_lm = None

            # subject_jump diagnostic
            hip_mid = hip_midpoint_norm(img_lm)
            subject_jump = False
            if hip_mid is not None and prev_hip_mid is not None:
                dx = hip_mid[0] - prev_hip_mid[0]
                dy = hip_mid[1] - prev_hip_mid[1]
                subject_jump = math.sqrt(dx * dx + dy * dy) > 0.2
            if hip_mid is not None:
                prev_hip_mid = hip_mid
            else:
                prev_hip_mid = None

            seg = seg_at_frame.get(frame_idx, {})
            angle_row = _compute_angles_for_frame(
                img_lm, wld_lm, width, height, vis_thresh
            )
            _apply_vis_mask(angle_row, vis_thresh)

            # best-side columns are computed after collecting all rows
            row = {
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "pose_detected": pose_detected,
                "n_poses": n_poses,
                "subject_jump": subject_jump,
                "seg_type": seg.get("type", ""),
                "seg_label": seg.get("label", ""),
                **angle_row,
            }
            rows.append(row)
            frame_idx += 1

    cap.release()

    if not rows:
        raise RuntimeError(f"No frames decoded from {video_path}")

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Add best-side columns
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    elapsed = time.perf_counter() - t0
    detection_rate = n_detected / frame_idx if frame_idx > 0 else 0.0

    # -----------------------------------------------------------------------
    # Per-exercise NaN stats inside exercise segments
    # -----------------------------------------------------------------------
    exercise_angle_map = {
        "squat": "knee_best",
        "push_up": "elbow_best",
        "jump_jack_arm": "arm_raise_L",
        "jump_jack_leg": "leg_spread",
    }

    summary_row: dict = {
        "video_id": video_id,
        "split": meta.get("split", ""),
        "n_frames": frame_idx,
        "detection_rate": detection_rate,
        "elapsed_s": round(elapsed, 2),
        "n_groups": len([s for s in meta.get("segments", []) if s["type"] == "exercise"]),
        "subject_jump_count": int(df["subject_jump"].sum()),
    }

    # Per-exercise gap stats
    for ex_key, angle_col in exercise_angle_map.items():
        ex_label = ex_key.split("_")[0] + ("_jack" if "jack" in ex_key else "")
        if ex_key == "jump_jack_arm":
            ex_label = "jump_jack"
        elif ex_key == "jump_jack_leg":
            ex_label = "jump_jack"

        segs = [s for s in meta.get("segments", [])
                if s["type"] == "exercise" and s["label"] == ex_label]

        # Gather the relevant angle series inside those segments
        vals = []
        for seg in segs:
            mask = (df["frame"] >= seg["start_frame"]) & (df["frame"] < seg["end_frame"])
            vals.append(df.loc[mask, angle_col].to_numpy(float))

        if not vals:
            suffix = ex_key
            summary_row[f"frac_nan_{suffix}"] = float("nan")
            summary_row[f"longest_nan_s_{suffix}"] = float("nan")
            summary_row[f"has_long_gap_{suffix}"] = False
            continue

        combined = np.concatenate(vals)
        n_total = len(combined)
        n_nan = int(np.isnan(combined).sum())
        frac_nan = n_nan / n_total if n_total > 0 else float("nan")

        # Longest consecutive NaN run
        longest = _longest_nan_run(combined)
        longest_s = longest / fps

        suffix = ex_key
        summary_row[f"frac_nan_{suffix}"] = round(frac_nan, 4)
        summary_row[f"longest_nan_s_{suffix}"] = round(longest_s, 3)
        summary_row[f"has_long_gap_{suffix}"] = longest_s > 1.0

    return summary_row


def _longest_nan_run(arr: np.ndarray) -> int:
    """Return the length of the longest consecutive NaN run."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return 0
    is_nan = np.isnan(arr).astype(int)
    # Trick: diff of cumsum gives run lengths
    max_run = 0
    cur = 0
    for v in is_nan:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


# ---------------------------------------------------------------------------
# Worker (one per process in multiprocessing pool)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> dict:
    """Top-level worker function (must be picklable -> module-level)."""
    video_path, json_path, output_csv, model_path, vis_thresh, mode, force = args

    if not force and output_csv.exists():
        try:
            existing = pd.read_csv(output_csv, nrows=2)
            if len(existing) > 0:
                log.info("SKIP %s (already exists)", output_csv.name)
                return {"video_id": video_path.stem, "status": "skipped"}
        except Exception:
            pass

    try:
        summary = extract_video_angles(
            video_path, json_path, output_csv,
            model_path, vis_thresh, mode,
        )
        summary["status"] = "success"
        log.info(
            "DONE %s | frames=%d | detection=%.3f | elapsed=%.1fs",
            video_path.name,
            summary.get("n_frames", 0),
            summary.get("detection_rate", 0),
            summary.get("elapsed_s", 0),
        )
        return summary
    except Exception as exc:
        log.exception("FAILED %s: %s", video_path.name, exc)
        return {"video_id": video_path.stem, "status": "failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _collect_videos(synth_dir: Path) -> list[tuple[Path, Path]]:
    """Return [(mp4_path, json_path)] for all synthetic videos."""
    pairs = []
    for mp4 in sorted(synth_dir.rglob("*.mp4")):
        j = mp4.with_suffix(".json")
        if j.exists():
            pairs.append((mp4, j))
        else:
            log.warning("No companion JSON for %s – skipping", mp4)
    return pairs


def main() -> None:
    import math  # noqa: F401 (used inside extract_video_angles via closure)

    parser = argparse.ArgumentParser(
        description="Phase 2: extract pose angles from synthetic videos."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to a single .mp4 file")
    group.add_argument("--all", action="store_true", help="Process all synthetic videos")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default 1)")
    parser.add_argument("--vis-thresh", type=float, default=0.5,
                        help="Visibility threshold for NaN masking (default 0.5)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if output CSV already exists")
    parser.add_argument("--mode", choices=["image", "video"], default="image",
                        help="PoseLandmarker running mode (default: image)")
    parser.add_argument("--synth-dir", type=Path, default=SYNTH_DIR,
                        help="Root of synthetic video tree (default: data/synthetic)")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR,
                        help="Root for angle CSV outputs (default: data/features)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Pose model not found: {MODEL_PATH}")

    if args.video:
        mp4 = args.video.resolve()
        j = mp4.with_suffix(".json")
        if not j.exists():
            raise SystemExit(f"No companion JSON found: {j}")
        pairs = [(mp4, j)]
    else:
        pairs = _collect_videos(args.synth_dir)

    if not pairs:
        raise SystemExit("No video/JSON pairs found.")

    log.info("Processing %d video(s) with %d worker(s), mode=%s, vis_thresh=%.2f",
             len(pairs), args.workers, args.mode, args.vis_thresh)

    work_items = []
    for mp4, js in pairs:
        video_id = mp4.stem
        output_csv = args.features_dir / f"{video_id}_angles.csv"
        work_items.append((mp4, js, output_csv, MODEL_PATH,
                           args.vis_thresh, args.mode, args.force))

    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(_worker, work_items)
    else:
        results = [_worker(item) for item in work_items]

    # -----------------------------------------------------------------------
    # Write / update summary CSV
    # -----------------------------------------------------------------------
    new_rows = [r for r in results if r.get("status") != "skipped"]
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if SUMMARY_PATH.exists():
            try:
                old_df = pd.read_csv(SUMMARY_PATH)
                # Drop rows we just re-computed
                updated_ids = set(new_df["video_id"])
                old_df = old_df[~old_df["video_id"].isin(updated_ids)]
                combined = pd.concat([old_df, new_df], ignore_index=True)
            except Exception:
                combined = new_df
        else:
            combined = new_df

        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(SUMMARY_PATH, index=False)
        log.info("Summary written to %s", SUMMARY_PATH)

    # Print table
    print("\n=== Extraction summary ===")
    for r in results:
        status = r.get("status", "?")
        vid = r.get("video_id", "?")
        dr = r.get("detection_rate", float("nan"))
        elapsed = r.get("elapsed_s", float("nan"))
        print(f"  {vid:35s}  {status:10s}  detection={dr:.3f}  elapsed={elapsed:.1f}s")


# Import math at module level for use in extract_video_angles
import math  # noqa: E402

if __name__ == "__main__":
    main()
