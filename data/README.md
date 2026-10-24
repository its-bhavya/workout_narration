# data/

- `raw/`       — untouched source clips from RepCount. GITIGNORED — lives on shared Google Drive.
- `manifests/` — small CSVs, source of truth, tracked in git.
- `synthetic/` — generated multi-exercise videos + ground-truth boundary JSONs. GITIGNORED — shared Drive.
- `features/`  — extracted pose keypoints and joint-angle time series. Tracked in git.

Shared Drive folder link: TODO — add link here once created

## Manifests

| File | Description |
|------|-------------|
| `repcount_filtered.csv`       | Usable RepCount clips after annotation validation |
| `repcount_excluded.csv`       | Excluded clips with rejection reasons |
| `repcount_quality.csv`        | Raw MediaPipe quality metrics per video |
| `quality_screening.csv`       | Pass/fail/review status after threshold application |
| `splits.csv`                  | Leakage-safe train/val/test split assignments |

## RepCount category normalization (6 October 2026)

Raw annotation labels were normalized to three canonical exercise names:

- `squat`: squat, squant
- `push_up`: push_up, pushups
- `jump_jack`: jump_jack, jumpjacks

The `squant` label was manually verified on a sample of 10 videos and confirmed to be the squat exercise.

After normalization and annotation validation, 375 videos matched the target exercise classes. Two were excluded (one atypical push-up variation, one incomplete squat), leaving 373 usable videos.
