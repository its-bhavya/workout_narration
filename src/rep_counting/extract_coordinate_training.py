"""Resumable, training-only coordinate sidecars; canonical data is read-only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data_pipeline.v2_protocol import ROOT, SYNTH_DIR, FEATURES_DIR, MANIFESTS, catalog, sha256
from .extract_pushup_diagnostic import extract_pass, coordinate_signals


MODEL = ROOT / "models/pose_landmarker_full.task"
PROTOCOL = "pushup_coordinates_train_v1"
LANDMARKER = None


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_plan():
    protected = {str(MODEL): sha256(MODEL)}
    for name in ("synthetic_index.csv", "splits_v2.csv", "folds.csv"):
        path = MANIFESTS / name
        protected[str(path)] = sha256(path)
    cases = []
    for vid, meta in sorted(catalog().items()):
        if meta["split"] != "train":
            continue
        exercise_windows = [(i, s) for i, s in enumerate(meta["segments"]) if s.get("label") == "push_up"]
        if not exercise_windows:
            continue
        video = SYNTH_DIR / "train" / f"{vid}.mp4"
        annotation = video.with_suffix(".json")
        feature = FEATURES_DIR / f"{vid}_angles.csv"
        for path in (video, annotation, feature):
            protected[str(path)] = sha256(path)
        for index, segment in exercise_windows:
            cases.append(dict(video_id=vid, fold=int(meta["fold"]), split="train", segment_index=index,
                clip_id=segment["clip_id"], start_frame=int(segment["start_frame"]), end_frame=int(segment["end_frame"]),
                fps=float(meta["fps"]), video_path=str(video), annotation_path=str(annotation),
                canonical_features_path=str(feature), video_sha256=protected[str(video)],
                annotation_sha256=protected[str(annotation)], canonical_features_sha256=protected[str(feature)]))
    if not cases:
        raise ValueError("No registered training push-up windows")
    # A reused source clip must never cross the fold boundary.
    folds = {}
    for case in cases:
        if case["clip_id"] in folds and folds[case["clip_id"]] != case["fold"]:
            raise ValueError("Source clip occurs in multiple folds")
        folds[case["clip_id"]] = case["fold"]
    return dict(protocol=PROTOCOL, run_id="v2", split="train", cases=cases, protected_hashes=protected,
        model_path=str(MODEL), model_sha256=sha256(MODEL),
        source_hashes={name: sha256(Path(__file__).with_name(name)) for name in (
            "extract_coordinate_training.py", "extract_pushup_diagnostic.py")},
        settings=dict(mode="IMAGE", stride=2, num_poses=2, subject_selection="largest_visible_bbox_per_frame",
                      crop=False, visibility_threshold=.5, coordinate_system="original_image_pixels",
                      torso_normalization="fixed per-window median torso length", labels_used_for_pose_selection=False),
        limitations=["All registered train push-up GT windows, not arbitrary untrimmed video inference.",
                     "Person selection is per-frame largest pose; identity changes are possible.",
                     "No crops, test video frames, test feature frames, or canonical cache writes.",
                     "Coordinates are model estimates, including potentially inferred off-screen points."])


def init_worker(model_path):
    global LANDMARKER
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    LANDMARKER = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.IMAGE,
        num_poses=2, output_segmentation_masks=False))


def add_signals(frame):
    """Derived channels and confidence metadata, without labels or target counts."""
    signals, scale = coordinate_signals(frame)
    signals["head_height"][~np.isfinite(frame.nose_vis.to_numpy())] = np.nan
    for name, values in signals.items():
        frame[name] = values
    for side in ("L", "R"):
        confidence = frame[[f"{joint}_{side}_vis" for joint in ("shoulder", "elbow", "wrist")]].to_numpy(dtype=float).min(axis=1)
        frame[f"elbow_{side}_confidence"] = np.where(np.isfinite(confidence), confidence, 0.)
    left, right = frame.elbow_L.to_numpy(), frame.elbow_R.to_numpy()
    wl = np.where(np.isfinite(left), frame.elbow_L_confidence.to_numpy(), 0.)
    wr = np.where(np.isfinite(right), frame.elbow_R_confidence.to_numpy(), 0.)
    weighted = np.full(len(frame), np.nan)
    np.divide(np.nan_to_num(left) * wl + np.nan_to_num(right) * wr, wl + wr, out=weighted, where=(wl + wr) > 0)
    frame["weighted_elbows"] = weighted
    frame["weighted_elbows_confidence"] = np.maximum(wl, wr)
    def pair_confidence(name):
        vis = frame[[f"{name}_L_vis", f"{name}_R_vis"]].to_numpy(dtype=float)
        return np.where(np.isfinite(vis), vis, 0.).max(axis=1)
    frame["shoulder_height_confidence"] = pair_confidence("shoulder")
    frame["shoulder_to_wrist_height_confidence"] = np.minimum(pair_confidence("shoulder"), pair_confidence("wrist"))
    frame["head_height_confidence"] = np.where(np.isfinite(frame.nose_vis), frame.nose_vis, 0.)
    return frame, scale


def extract_case(case):
    started = time.monotonic()
    if case["split"] != "train":
        raise ValueError("Training sidecars cannot extract test frames")
    frame, _, _ = extract_pass(LANDMARKER, Path(case["video_path"]), case["start_frame"], case["end_frame"], case["fps"])
    expected = list(range(case["start_frame"] + case["start_frame"] % 2, case["end_frame"], 2))
    if frame.frame.tolist() != expected:
        raise ValueError("Incomplete or misaligned coordinate frames")
    frame, scale = add_signals(frame)
    return case, frame, dict(n_rows=len(frame), torso_scale_pixels=scale if np.isfinite(scale) else None,
                             detection_fraction=float(frame.detected.mean()), elapsed_s=time.monotonic() - started)


def case_name(case):
    return f"{case['video_id']}__segment_{case['segment_index']:03d}"


def verify_case(output, case, plan_signature):
    stem = case_name(case)
    csv_path, meta_path = output / f"{stem}.csv", output / f"{stem}.json"
    if not csv_path.exists() and not meta_path.exists():
        return None
    if not csv_path.exists() or not meta_path.exists():
        raise ValueError(f"Partial sidecar requires inspection; refusing overwrite: {stem}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["plan_signature"] != plan_signature or meta["case"] != case or meta["csv_sha256"] != sha256(csv_path):
        raise ValueError(f"Sidecar provenance mismatch: {stem}")
    return meta


def extract_all(output, workers=2, resume=False):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Sidecars must be written under outputs, never canonical data")
    if output.exists() and not resume:
        raise FileExistsError("Use a new directory or explicit --resume for verified sidecars")
    if workers < 1:
        raise ValueError("workers must be positive")
    plan = build_plan()
    plan_sig = signature(plan)
    if output.exists():
        saved = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        if saved["plan_signature"] != plan_sig or saved["plan"] != plan:
            raise ValueError("Extraction inputs, code, or policy differ from the saved plan")
    else:
        output.mkdir(parents=True)
        (output / "plan.json").write_text(json.dumps(dict(plan_signature=plan_sig, plan=plan), indent=2), encoding="utf-8")
    completed, pending = [], []
    for case in plan["cases"]:
        verified = verify_case(output, case, plan_sig)
        if verified:
            completed.append(verified)
        else:
            pending.append(case)
    print(f"Training-only plan: {len(plan['cases'])} windows, {len(completed)} verified, {len(pending)} pending; workers={workers}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(str(MODEL),)) as pool:
            futures = [pool.submit(extract_case, case) for case in pending]
            for future in as_completed(futures):
                case, frame, statistics = future.result()
                stem = case_name(case)
                csv_path = output / f"{stem}.csv"
                frame.to_csv(csv_path, index=False)
                metadata = dict(protocol=PROTOCOL, plan_signature=plan_sig, case=case, csv_sha256=sha256(csv_path), **statistics)
                (output / f"{stem}.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
                completed.append(metadata)
                print(f"[{len(completed)}/{len(plan['cases'])}] {stem}: {statistics['n_rows']} frames, detected {statistics['detection_fraction']:.1%}", flush=True)
                (output / "progress.json").write_text(json.dumps(dict(completed=len(completed), total=len(plan["cases"])), indent=2), encoding="utf-8")
    for path, digest in plan["protected_hashes"].items():
        if sha256(path) != digest:
            raise ValueError(f"Existing input changed during extraction: {path}")
    if len(completed) != len(plan["cases"]):
        raise ValueError("Incomplete training sidecar set")
    summary = dict(protocol=PROTOCOL, plan_signature=plan_sig, split="train", n_windows=len(completed),
                   n_source_clips=len({c['case']['clip_id'] for c in completed}),
                   completed_at_utc=datetime.now(timezone.utc).isoformat(), inputs_verified_unchanged=True,
                   cases=[dict(csv=case_name(c["case"])+'.csv', metadata=case_name(c["case"])+'.json') for c in completed])
    (output / "extraction_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    extract_all(args.out_dir, args.workers, args.resume)
