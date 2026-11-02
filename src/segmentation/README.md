# src/segmentation
Owned by: Bhavya (Role 1)

Sliding-window classifier + smoothing + raw-vs-smoothed evaluation. Classes are
exactly `squat`, `push_up`, `jump_jack`, `transition`.

## Input

`data/features/<video_id>_angles.csv` produced by
`src/data_pipeline/extract_pose_features.py` (synthetic videos, `--stride 2`).
**Do not use** `data/features/repcount/derived/*.parquet` — those are per-clip,
exercise-specific columns and leak the label.

Windows are built from `time_s`, never from row counts (stride 2 makes row index
unreliable as a timeline). Each video's split comes from its boundary JSON
`"split"` field — never tuned on test.

## Scripts

All run from the repo root. Import convention: bare imports within the package
(no `src.segmentation` package path), matching `src/data_pipeline`. Tests add
the package dir to `sys.path` the same way `tests/test_pose_utils.py` does.

### `config.py`
All parameters live here (`SEED = 42`): `WINDOW_S`, `STRIDE_S`,
`MIN_PURITY_TRAIN`, `MIN_ROWS_FRAC`, `CLASSES`, `FEATURE_CHANNELS`, and path
constants. `feature_columns(header)` resolves the base channels plus each
`<channel>_3d` variant when that column exists in the CSV header.

`FEATURE_CHANNELS` is the **allow-list**: only these columns may be model
features. Everything else is metadata and MUST be excluded — `seg_label`,
`seg_type`, `subject_jump`, `n_poses`, `pose_detected`, `frame`, `time_s`, and
any `*_vis` column. The classifier builds its matrix from `feature_columns()`
and nothing else.

### `make_windows.py`
```
python src/segmentation/make_windows.py
```
Slides `[t, t+WINDOW_S)` windows over `time_s` (step `STRIDE_S`), drops the
final partial window and windows below `MIN_ROWS_FRAC` of the expected row
count, labels each window by the majority `seg_label` (purity = its share of
rows; ties broken by `CLASSES` order). Reads the split from each video's JSON.
Writes `outputs/segmentation/windows.csv`:

`video_id, split, win_idx, t_start, t_end, t_center, label, purity, row_start, row_end`
(`row_end` is slice-style exclusive: `df.iloc[row_start:row_end]` recovers the
window). Prints window counts per split × class and asserts one split per
video.

### `features.py`
Two stages:
1. `fill_video_nans(df, channels, max_gap_s=0.5)` — per video, before windowing:
   linear-interpolate NaN gaps up to 0.5 s via `fill_short_gaps` from
   `pose_utils` (0.5 s → rows using the median `time_s` spacing).
2. `window_features(rows_df) -> dict` — per channel, 7 nan-aware stats: mean,
   std, min, max, range, mean-absolute-change-per-second, NaN fraction. Nothing
   else.

### `train.py`
```
python src/segmentation/train.py [--tag "experiment label"]
```
Builds the feature matrix for all windows, fits per-feature NaN-fill medians on
**train only** (eligible windows, purity ≥ `MIN_PURITY_TRAIN`), trains
`RandomForestClassifier(n_estimators=200, class_weight="balanced",
random_state=SEED)`, evaluates on **val** (accuracy, macro-F1, per-class F1,
confusion matrix, majority-class baseline, top-10 feature importances to spot
leakage), and writes per-window predictions (all class probabilities + max prob)
for **val and test**. Test metrics are never computed.

Outputs (all under `outputs/segmentation/`):

| file | what |
|------|------|
| `windows.csv` | window index (from `make_windows.py`) |
| `val_window_report.txt` | latest val report (text) |
| `rf_model.joblib` | latest model bundle: `model`, `feature_cols`, `fill_values`, `classes` |
| `window_predictions_{val,test}.csv` | latest per-window predictions |

### Run versioning
Every `train.py` run archives a full snapshot to
`outputs/segmentation/runs/<run_id>/` (`run_id` = `YYYYMMDD_HHMMSS`): the
report, model bundle, val+test predictions, and `run_meta.json` (config + data
counts + metrics + confusion matrix + top-10 importances + resolved feature
channels). One row is appended to `outputs/segmentation/runs_log.csv` so runs
are comparable when you change models, data, or hyperparameters — the log records
`n_estimators`, `window_s`, `stride_s`, `min_purity_train`, window counts,
`n_features`, and all val metrics per run. The top-level files are the "latest"
copy (overwritten); `runs/` is the history.

## Tests

`tests/test_make_windows.py` — builds a fake 10 s CSV with 2 segments and checks
labels, purity, that no window crosses the end, plus a short-video and a
row-slice round-trip case.

```
python -m pytest tests/test_make_windows.py -v
```

## Current result (val)

RandomForest, window 2.0 s / stride 1.0 s, train windows with purity ≥ 0.8
(3107 of 3292 train windows eligible), 112 features.

| metric | value |
|--------|-------|
| accuracy | 0.828 |
| majority baseline | 0.216 (predict jump_jack for all) |
| macro-F1 | 0.766 |

| class | F1 |
|-------|------|
| jump_jack | 0.955 |
| push_up | 0.885 |
| squat | 0.864 |
| transition | 0.360 |

Top features are `torso_tilt` and `arm_raise` (physically meaningful), with no
`*_nanfrac` feature in the top 10 — i.e. no NaN-fraction leakage. `transition`
is the weak class; the errors are boundary windows where exercise edges get read
as transition. That is the case the smoothing / minimum-segment-length step is
meant to clean up, so a low raw-transition score is expected.
