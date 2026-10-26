# data/

- `raw/`       — untouched source clips from RepCount. GITIGNORED — lives on shared Google Drive.
- `manifests/` — small CSVs, source of truth, tracked in git.
- `synthetic/` — generated multi-exercise videos + ground-truth boundary JSONs. GITIGNORED — shared Drive.
- `features/`  — extracted pose keypoints and joint-angle time series. Tracked in git.

Shared Drive folder link: TODO — add link here once created

## Manifests

| File | Description |
|------|-------------|
| `repcount_filtered.csv`       | Usable RepCount clips after annotation validation; includes `rep_boundaries` as `[[start, end], ...]` pairs parsed from raw L-columns |
| `repcount_excluded.csv`       | Excluded clips with rejection reasons |
| `repcount_quality.csv`        | Raw MediaPipe quality metrics per video |
| `quality_screening.csv`       | Pass/fail/review status after threshold application |
| `splits.csv`                  | Leakage-safe train/val/test split assignments (RepCount only, seed 42) |
| `repcount_rep_boundaries.json`| `filename → [L1, L2, ...]` flat int list for every clip in splits.csv; parsed from repcount_filtered.csv |

## RepCount category normalization (6 October 2026)

Raw annotation labels were normalized to three canonical exercise names:

- `squat`: squat, squant
- `push_up`: push_up, pushups
- `jump_jack`: jump_jack, jumpjacks

The `squant` label was manually verified on a sample of 10 videos and confirmed to be the squat exercise.

After normalization and annotation validation, 375 videos matched the target exercise classes. Two were excluded (one atypical push-up variation, one incomplete squat), leaving 373 usable videos.

## Splits (26 October 2026, seed 42, RepCount only)

Countix dropped (decision O1). `build_splits.py` filters to `source == "repcount"` before splitting. 336 clips pass quality screening (pass + review).

| split | jump_jack | push_up | squat | total | independent groups |
|-------|-----------|---------|-------|-------|--------------------|
| train | 68        | 72      | 76    | 216   | 31                 |
| val   | 11        | 14      | 12    | 37    | 8                  |
| test  | 26        | 28      | 29    | 83    | 27                 |

Target fractions 65/10/25 achieved within ~1%. Leakage check passes (no group spans two splits).

## Video statistics (splits.csv clips)

- FPS: 70% at 30 fps, 15% at 24 fps, 11% at 25 fps; 12 outliers (13.9–29.9 fps).
- Resolution: 68% are 640×360; next largest are 1280×720 (9%) and 202×360 (7%).
- Portrait clips (height > width): 46 / 336 (14%).
- Frames before L1 (pre-exercise lead-in):

  | exercise  | median | min | max  |
  |-----------|--------|-----|------|
  | jump_jack | 19     | 0   | 375  |
  | push_up   | 28     | 0   | 1056 |
  | squat     | 33     | 0   | 493  |

  Large max values mean some clips contain usable idle footage before the first rep (relevant to O3).
