# src/data_pipeline
Owned by: Bhavya
Data source: RepCount (squat, push_up, jump_jack)

Scripts, run in order (for segmentation, use only steps 1–5):
1. `load_repcount.py`            — parse annotation CSVs, filter to target exercises, normalize category names → `repcount_filtered.csv`, `repcount_excluded.csv`
2. `quality_screen.py`           — MediaPipe Pose Landmarker visibility check per video → `repcount_quality.csv`
3. `apply_quality_thresholds.py` — inspect quality distributions, decide pass/fail thresholds → `quality_screening.csv`
4. `build_splits.py`             — leakage-safe train/val/test split assignment (groups by subject, RepCount only, seed 42) → `splits.csv`
5. `concatenate_videos.py`       — stitch whole clips into synthetic multi-exercise videos with ground-truth segment labels → `data/synthetic/<split>/synth_<split>_<NNNN>.{mp4,json}` + `data/manifests/synthetic_index.csv`
6. `extract_pose_features.py`    — extract per-frame pose keypoints via MediaPipe and compute joint angles → `data/features/<video_id>_angles.csv` (used by segmentation) + `data/manifests/pose_extraction_summary.csv`

(Legacy: `extract_pose.py` and `extract_features.py` produced per-clip parquet files under `data/features/repcount/{pose,derived}/` — no longer used; extraction now goes directly to per-video angle CSVs.)

Utility scripts:
- `quality_screening_checks.py`  — ad-hoc inspection of borderline quality cases
- `validate_pose.py`             — sanity-check pose extraction outputs
- `inspect_rep_boundaries.py`    — visualize frame-level repetition annotations for a single video
- `plot_features.py`             — plot extracted features for exploration
- `verify_l_columns.py`          — contact-sheet PNGs of frames at each L-value for 6 random clips (2 per exercise); outputs to `outputs/l_check/`; run to confirm L-column semantics by eye
- `preview_boundaries.py`        — render an annotated preview MP4 for a synthetic video (segment label + frame index + timeline bar); `--video-id synth_train_0001`

## extract_pose_features.py

**Input:** synthetic videos from `data/synthetic/<split>/` + boundary JSONs.
**Output:** `data/features/<video_id>_angles.csv` (per-frame joint angles, stride-2 by default) + `data/manifests/pose_extraction_summary.csv`.

Usage (run from repo root):
```
# single video
python src/data_pipeline/extract_pose_features.py --video data/synthetic/train/synth_train_0001.mp4

# all videos
python src/data_pipeline/extract_pose_features.py --all

# all videos, 4 workers, force re-extraction
python src/data_pipeline/extract_pose_features.py --all --workers 4 --force

# train+val only, stride 2, 4 workers (for faster iteration)
python src/data_pipeline/extract_pose_features.py --all --only-splits train,val --stride 2 --workers 4
```

Reads each video's companion boundary JSON to extract the split (never tuned on
test). Fills short tracking gaps: per-frame visibility filtering → NaN masking;
a few frames with low visibility per joint is normal. Per-frame angles:
`knee_{L,R,best}`, `elbow_{L,R,best}`, `hip_{L,R}`, `arm_raise_{L,R}`,
`leg_spread`, `ankle_dist_norm`, `torso_tilt` (each computed 2-D and optionally
3-D). Plus `knee_best`, `elbow_best`, `*_side` (left/right choice per frame) and
visibility stats. Also writes `data/manifests/pose_extraction_summary.csv` with
detection rate, NaN gaps per exercise, subject jumps, etc.

**For segmentation:** Use the angle CSVs as input to `src/segmentation/`. Do NOT
use the legacy parquet files under `data/features/repcount/{pose,derived}/` —
they are per-clip and carry label leakage.

## concatenate_videos.py

```
python src/data_pipeline/concatenate_videos.py \
    --splits-file data/manifests/splits.csv \
    --n-train 24 --n-val 6 --n-test 15 \
    --seed 42 --out data/synthetic \
    [--pilot]  [--margin-label {transition,exercise}]
```

Key decisions baked in:
- Pool: status in {pass, review}, fps ≥ 20. Reads `repcount_rep_boundaries.json` for pairs.
- K = 3..5 clips per video, all from the same split, adjacent exercises differ, ≥2 distinct exercises.
- val/test: each clip used at most once across the whole split. train: reuse across videos allowed, never twice within one video.
- TARGET_FPS = 30, TARGET_SIZE = 640×360, letterboxed. Nearest-frame resampling. cv2.VideoWriter, no audio.
- Rep pairs re-offset and rescaled into stitched timeline and stored in the JSON per segment.
- `--margin-label exercise` suppresses all transition segments (whole clip = exercise label).

JSON schema per video: `video_id, split, seed, fps, width, height, n_frames, splits_file, margin_label, clip_ids_in_order, clip_join_frames, segments[]`
Each segment: `type, label, start_frame, end_frame` + for exercise: `clip_id, rep_count_annotated, rep_pairs_out, src_window_frames` + for transition: `source_clips`.

## Tests

`tests/test_concatenate.py` — 20 tests covering: fps resampling index mapping, pair re-offset, contiguity (no gaps/overlaps, covers [0, n_frames)), no clip reuse in val/test, single-split videos, adjacent transition merging, no zero-length segments, clip_join_frame placement, join count == clips - 1.

Run: `python -m pytest tests/test_concatenate.py -v`
