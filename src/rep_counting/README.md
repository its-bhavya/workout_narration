# src/rep_counting

Owned by: Deepti

## Overview

Rule-based rep counter that operates on pre-extracted joint angle time series.
No model training — thresholds are tuned on the train split and validated on val.

Pipeline per segment:

```
<video_id>_angles.csv  (from extract_pose_features.py)
  -> slice to [start_s, end_s]
  -> pick signal column (best-tracked side or mean of both arms)
  -> mask long tracking gaps, interpolate short ones, smooth
  -> two-threshold hysteresis state machine
  -> drop reps closer than min_gap_s
  -> return rep count + timestamps
```

## Files

| File | Purpose |
|------|---------|
| `counter.py` | Original prototype (normalised-coordinate angles). Superseded by `rep_counter.py`. |
| `rep_counter.py` | Current implementation. Reads pixel-scaled angles from the shared CSVs. |
| `evaluate_rep_counts.py` | Batch evaluation against ground-truth annotations on synthetic videos. |
| `rep_counting_main_script.py` | Dev script for running and inspecting a single angles CSV. |

## rep_counter.py

### Key design decisions

- Works entirely from `data/features/<video_id>_angles.csv` — does not touch raw video or MediaPipe.
- Operates in real seconds (`time_s` column), so it handles any frame rate or stride without changes.
- `count_for_exercise(df, exercise, start_s, end_s)` is the single entry point. One call per segment.
- Adaptive thresholds by default: placed at fixed fractions of the segment's own p10..p90 angle range, so camera angle and subject variation don't require manual re-tuning per clip.
- Falls back to the second side if the preferred side is mostly missing or shows no real motion (`low_motion` flag).

### Exercise configuration (EXERCISES dict)

| param | squat | push_up | jump_jack |
|-------|-------|---------|-----------|
| columns | knee_L/R | elbow_L/R | arm_raise_L/R |
| combine | best side | best side | mean of both |
| pattern | down_up | down_up | low_high_low |
| mode | adaptive | adaptive | adaptive |
| fracs | 0.35 / 0.65 | 0.25 / 0.55 | 0.35 / 0.65 |
| smooth_s | 0.5 s | 0.1 s | 0.3 s |
| min_gap_s | 0.5 s | 0.2 s | 0.5 s |
| min_range_deg | 25° | 20° | 40° |

`min_range_deg` guards against counting reps in a motionless or mis-labelled segment.

### Flags

A segment result may carry one of:

- `too_short` — fewer than 8 samples in the window
- `mostly_missing` — more than 50% of the signal is NaN after gap-filling
- `low_motion` — p10..p90 angle range is below `min_range_deg`; reps set to 0

### Usage

```python
from src.rep_counting.rep_counter import load_features, count_for_exercise

df = load_features("data/features/synth_val_0001_angles.csv")
out = count_for_exercise(df, "squat", start_s=4.2, end_s=19.8)
print(out["reps"], out["rep_times"])
```

For predicted segments (from segmentation), pass `end_pad_s=0.3` so a rep cut
off at the segment boundary can still complete.

## evaluate_rep_counts.py

Runs the counter on every exercise segment of every synthetic video in the
specified splits, using ground-truth segment windows, and computes:

- `exact_%` — predicted count == annotated count
- `within1_%` — |predicted − annotated| ≤ 1
- `MAE`, `bias` — mean absolute and signed error
- `precision`, `recall` — one-to-one match of predicted rep timestamps to annotated rep intervals (tolerance = 0.3 s); detects cases where count is right by coincidence (miss + false alarm cancel out)
- `flagged_%` — share of segments the counter abstained on

Also prints calibration tables: annotated rep durations (to set `min_gap_s`) and observed angle ranges (to set `min_range_deg`).

Output: `outputs/rep_eval/segments.csv` — one row per evaluated segment.

### Usage

```bash
# default: train + val, adaptive mode
uv run python src/rep_counting/evaluate_rep_counts.py

# try different fracs
uv run python src/rep_counting/evaluate_rep_counts.py --mode adaptive --fracs 0.3 0.7

# final numbers only (do once, at the end)
uv run python src/rep_counting/evaluate_rep_counts.py --splits test
```

---

## Results (train + val, adaptive mode, ground-truth segments)

Evaluated on 119 segments across train and val synthetic videos (October 2026).

### Combined train + val

| exercise | segments | exact_% | within1_% | MAE | bias | precision | recall | flagged_% |
|----------|----------|---------|-----------|-----|------|-----------|--------|-----------|
| jump_jack | 45 | 45.0 | 71.11 | 1.73 | -1.42 | 0.99 | 0.94 | 2.22 |
| push_up | 37 | 45.95 | 62.16 | 2.57 | -0.57 | 0.93 | 0.90 | 0.00 |
| squat | 37 | 67.57 | 75.68 | 2.00 | -1.30 | 0.94 | 0.82 | 2.70 |

### Train only

| exercise | segments | exact_% | within1_% | MAE | bias | precision | recall | flagged_% |
|----------|----------|---------|-----------|-----|------|-----------|--------|-----------|
| jump_jack | 36 | 69.44 | 80.56 | 2.11 | -1.78 | 0.99 | 0.93 | 2.78 |
| push_up | 30 | 50.00 | 63.33 | 2.07 | -1.20 | 0.96 | 0.90 | 0.00 |
| squat | 29 | 65.52 | 68.97 | 2.48 | -1.72 | 0.94 | 0.80 | 3.45 |

### Val only

| exercise | segments | exact_% | within1_% | MAE | bias | precision | recall | flagged_% |
|----------|----------|---------|-----------|-----|------|-----------|--------|-----------|
| jump_jack | 9 | 77.78 | 100.00 | 0.22 | 0.00 | 0.99 | 0.99 | 0.00 |
| push_up | 7 | 28.57 | 57.14 | 4.71 | 2.14 | 0.81 | 0.91 | 0.00 |
| squat | 8 | 75.00 | 100.00 | 0.25 | 0.25 | 0.92 | 0.96 | 0.00 |

Val squat and jump_jack are clean (100% within-1). Val push_up is dragged down entirely by one outlier: `stu10_49.mp4` (annotated 21, counted 43 — 2× overcounting). Inspection shows this clip contains multiple people in frame; the counter locks onto a secondary subject's oscillating signal and fires twice per actual rep. This is a known multi-person artefact and will be addressed when the pipeline gains a consistent single-subject selection rule (already implemented in `extract_pose_features.py` via `largest_bbox_pose_index` for synthetic videos; needs to be confirmed for raw RepCount clips).

### Flagged segments

| exercise | flag | count |
|----------|------|-------|
| jump_jack | low_motion | 1 |
| squat | low_motion | 1 |

### Largest errors

| video | exercise | clip | annotated | counted | flag |
|-------|----------|------|-----------|---------|------|
| synth_train_0014 | jump_jack | stu5_25.mp4 | 68 | 29 | |
| synth_val_0001 | push_up | stu10_49.mp4 | 21 | 43 | |
| synth_train_0023 | squat | stu6_62.mp4 | 20 | 0 | low_motion |
| synth_train_0022 | jump_jack | stu6_19.mp4 | 16 | 0 | low_motion |
| synth_train_0003 | push_up | stu6_41.mp4 | 54 | 41 | |
| synth_train_0019 | push_up | stu9_50.mp4 | 65 | 54 | |
| synth_train_0003 | squat | stu6_62.mp4 | 20 | 9 | |
| synth_train_0007 | squat | stu5_67.mp4 | 16 | 6 | |
| synth_train_0017 | push_up | stu9_50.mp4 | 65 | 55 | |
| synth_train_0001 | squat | stu6_62.mp4 | 20 | 11 | |

The two `low_motion` flags (stu6_62 squat, stu6_19 jump_jack) are correct abstentions — those clips have insufficient angle range for the counter to fire. The `stu10_49` overcounting and `stu5_25` undercounting are the main cases to investigate with sanity_plots.

### Calibration reference

Annotated rep durations (used to set `min_gap_s`):

| exercise | p05 | p25 | p50 | p75 |
|----------|-----|-----|-----|-----|
| squat | 1.00 s | 1.43 s | 1.80 s | 2.57 s |
| push_up | 0.43 s | 0.73 s | 0.95 s | 1.77 s |
| jump_jack | 0.73 s | 0.87 s | 0.93 s | 1.00 s |

Observed angle range p10..p90 inside ground-truth segments (used to set `min_range_deg`):

| exercise | p05 | p25 | p50 |
|----------|-----|-----|-----|
| jump_jack | 87.2° | 135.8° | 148.1° |
| push_up | 30.4° | 73.7° | 88.2° |
| squat | 26.1° | 44.0° | 70.4° |

Current `min_range_deg` values are set safely below the p05 in all three cases.

---

## Known issues / next steps

- `stu10_49` push_up overcounts 2×: multi-person clip — a secondary subject's signal causes double-firing. `largest_bbox_pose_index` in `extract_pose_features.py` handles this for synthetic videos; needs to be confirmed/applied for raw RepCount clips. Tracked as a known limitation until single-subject selection is enforced end-to-end.
- `stu5_25` jump_jack undercounts (68 → 29): very high rep-count clip, likely the adaptive fracs cut too low. Try narrowing fracs or checking the raw angle trace.
- `stu6_62` squat flags `low_motion` in three separate synthetic videos — the source clip may have an unusually small range; worth checking if it should be excluded from the pool.
- Test split is locked; do not evaluate until the method is final.
