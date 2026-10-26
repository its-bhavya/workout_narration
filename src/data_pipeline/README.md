# src/data_pipeline
Owned by: Bhavya
Data source: RepCount (squat, push_up, jump_jack)

Scripts, run in order:
1. `load_repcount.py`            — parse annotation CSVs, filter to target exercises, normalize category names → `repcount_filtered.csv`, `repcount_excluded.csv`
2. `quality_screen.py`           — MediaPipe Pose Landmarker visibility check per video → `repcount_quality.csv`
3. `apply_quality_thresholds.py` — inspect quality distributions, decide pass/fail thresholds → `quality_screening.csv`
4. `build_splits.py`             — leakage-safe train/val/test split assignment (groups by subject) → `splits.csv`
5. `extract_pose.py`             — extract per-frame pose keypoints via MediaPipe → `data/features/repcount/pose/*.parquet`
6. `extract_features.py`         — compute joint angles and exercise-specific motion features → `data/features/repcount/derived/*.parquet`

Utility scripts:
- `quality_screening_checks.py`  — ad-hoc inspection of borderline quality cases
- `validate_pose.py`             — sanity-check pose extraction outputs
- `inspect_rep_boundaries.py`    — visualize frame-level repetition annotations for a single video
- `plot_features.py`             — plot extracted features for exploration
- `verify_l_columns.py`          — contact-sheet PNGs of frames at each L-value for 6 random clips (2 per exercise); outputs to `outputs/l_check/`; run to confirm L-column semantics (start/bottom/pairs) by eye
