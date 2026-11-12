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

**Multi-person tracking (November 2026):** Uses `PosePersonTracker` from
`pose_utils.py` instead of the previous `largest_bbox_pose_index` heuristic.
The tracker assigns persistent IDs across frames (greedy centre-distance + IoU
matching), observes all candidates for an observation window (45 frames by
default), then locks onto the person with the most exercise-relevant movement
× visibility. During the observation phase it temporarily uses
`largest_bbox_pose_index`. If the locked track is lost (> 10 missed frames),
it unlocks and re-observes. The tracker resets at each segment boundary.

During the observation phase, angle features for all candidates are buffered;
once the tracker locks, only the selected person's buffered rows are emitted.
This prevents a secondary subject's oscillating signal from contaminating the
angle CSV (the cause of 2× overcounting on multi-person clips like `stu10_49`).

**For segmentation:** Use the angle CSVs as input to `src/segmentation/`. Do NOT
use the legacy parquet files under `data/features/repcount/{pose,derived}/` —
they are per-clip and carry label leakage.

## pose_utils.py

Pose geometry helpers used by `extract_pose_features.py`:

| Function / class | Purpose |
|------------------|---------|
| `angle_deg` / `angle_deg_batch` | compute joint angle from 3 landmarks (2-D or 3-D) |
| `norm_to_pixels` | convert normalised landmark coords to pixel coords |
| `get_pose_bbox` | bounding box of a pose's landmarks (normalised coords) |
| `calculate_iou` | IoU between two bounding boxes |
| `largest_bbox_pose_index` | return index of the pose with the largest bounding box (legacy heuristic) |
| `hip_midpoint_norm` | midpoint of left+right hip in normalised coords (subject-jump diagnostic) |
| `fill_short_gaps` | linear interpolation of NaN gaps up to a max length |
| `best_side_angle` | pick the more visible side's angle per frame |
| `PosePersonTracker` | multi-person tracking: persistent IDs, movement-based selection, locking (see above) |

`PosePersonTracker` constructor parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `max_distance` | 0.25 | max centre-distance for track matching |
| `max_missed_frames` | 10 | frames a track survives without detection before deletion |
| `observation_frames` | 45 | frames to observe candidates before locking |
| `min_observations` | 15 | minimum valid angle samples needed to score a candidate |

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

`tests/test_person_tracker.py` — 7 tests for `PosePersonTracker` and `get_clip_start_frame`: persistent ID assignment across frames, movement-score prefers the more active candidate, visibility fallback when movement history is insufficient, selects the person actually exercising, recovers when the selected track is deleted, recovers after consecutive empty frames, clip start frame computation.

`tests/test_pose_utils.py` — tests for pose geometry helpers (`angle_deg`, `norm_to_pixels`, etc.).

Run: `python -m pytest tests/test_concatenate.py tests/test_person_tracker.py tests/test_pose_utils.py -v`
