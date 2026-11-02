# src/segmentation
Owned by: Bhavya (Role 1)

Sliding-window classifier + smoothing + raw-vs-smoothed evaluation. Classes are
exactly `squat`, `push_up`, `jump_jack`, `transition`.

---

## How to run

All commands from the **repo root**.

### Full pipeline (recommended)

```
python src/segmentation/run_all.py
```

This calls `make_windows` → `train` → `evaluate` in order and prints the final
raw-vs-smoothed results table.  Optional flags:

| Flag | Effect |
|------|--------|
| `--tag "label"` | forwarded to `train.py`; recorded in `runs_log.csv` |
| `--skip-windows` | reuse an existing `outputs/segmentation/windows.csv` |

### Individual steps

```
python src/segmentation/make_windows.py          # build windows.csv
python src/segmentation/train.py [--tag "label"] # train classifier, write predictions
python src/segmentation/evaluate.py              # grid-search val, eval test, print table
```

### Classifier + smoothing hyperparameter sweep

`hparam_sweep.py` loads the pre-built `windows.csv` and angle CSVs, builds the
feature matrix **once**, then sweeps classifiers, `min_purity_train`, and the
smoothing grid.  Windowing and NaN-fill settings are fixed (same as `train.py`).

```
# Full sweep: all 8 classifiers × 2 purity thresholds
python src/segmentation/hparam_sweep.py

# Quick sanity check: RF + LR only
python src/segmentation/hparam_sweep.py --quick

# Specific classifiers
python src/segmentation/hparam_sweep.py --classifiers rf extra_trees hist_gbt lr

# Wider smoothing search
python src/segmentation/hparam_sweep.py --tau-grid 0.3 0.4 0.5 0.6 0.7 --min-len-grid 1 2 3 4 5
```

Results are ranked by test Edit Score and written to
`outputs/segmentation/sweep_results.csv`.

**Classifiers available:** `rf`, `extra_trees`, `gbt`, `hist_gbt`, `lr`, `svm`, `knn`, `mlp`.

---

## Config parameters (`config.py`)

All parameters live in `config.py`. Do not hardcode them elsewhere.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `WINDOW_S` | `2.0` | window length in seconds |
| `STRIDE_S` | `1.0` | step between successive window starts (50 % overlap) |
| `MIN_ROWS_FRAC` | `0.5` | drop windows with fewer than this fraction of expected rows |
| `MIN_PURITY_TRAIN` | `0.8` | minimum label purity for a window to be eligible for training |
| `SEED` | `42` | random seed for all reproducibility |
| `CLASSES` | `["squat","push_up","jump_jack","transition"]` | ordered class list |
| `N_ESTIMATORS` (train.py) | `200` | trees in the Random Forest |
| `CLASS_WEIGHT` (train.py) | `"balanced"` | upweights minority classes |
| `MAX_FILL_GAP_S` (features.py) | `0.5` | NaN gaps ≤ this many seconds are linearly interpolated before feature extraction |

Smoothing parameters (`TAU_GRID`, `MIN_LEN_GRID` in `evaluate.py`) are searched
on val; the best pair is written to `outputs/segmentation/chosen_params.json`.

---

## Input columns

### Used as model features (`FEATURE_CHANNELS` allow-list)

The classifier may only use columns in this list (plus their `_3d` variants when
present). `config.feature_columns(header)` is the single source of truth.

| Column | What it captures |
|--------|-----------------|
| `knee_best` | best-side knee angle (hip–knee–ankle); primary squat signal |
| `elbow_best` | best-side elbow angle (shoulder–elbow–wrist); primary push-up signal |
| `hip_L`, `hip_R` | left/right hip flexion; secondary squat depth signal |
| `arm_raise_L`, `arm_raise_R` | left/right arm elevation; jumping-jack lateral arm signal |
| `leg_spread` | lateral leg separation; jumping-jack leg signal |
| `ankle_dist_norm` | normalised ankle-to-ankle distance; jumping-jack lower-body signal |
| `torso_tilt` | torso inclination; distinguishes push-up (forward lean) from squat/transition |
| `*_3d` variants | world-landmark equivalents of any of the above when present in the CSV |

7 statistics are computed per channel per window: `mean`, `std`, `min`, `max`,
`range`, `macps` (mean absolute change per second), `nanfrac`.  With 16 base+3d
channels resolved at runtime the typical feature vector is 112-dimensional.

### Excluded columns (and why)

| Column(s) | Reason excluded |
|-----------|----------------|
| `seg_label` | **the label itself** — using it as a feature would be direct label leakage |
| `seg_type` | metadata field that encodes exercise identity; leaks the label |
| `subject_jump` | flags a large inter-frame hip shift; administrative metadata, not an exercise signal |
| `n_poses`, `pose_detected` | detection quality metadata; would let the model exploit data-collection artefacts rather than exercise geometry |
| `frame` | raw frame index with no temporal meaning after stride-2 extraction |
| `time_s` | timeline index used for windowing, not an exercise-discriminating feature |
| `*_vis` columns | MediaPipe landmark visibility scores; highly correlated with which exercise is present (e.g. ankle visibility is low during push-ups) and would constitute indirect leakage |

---

## Output files

All outputs land under `outputs/segmentation/`.

| File | Produced by | Contents |
|------|-------------|----------|
| `windows.csv` | `make_windows.py` | one row per sliding window; columns below |
| `val_window_report.txt` | `train.py` | val accuracy, macro-F1, per-class F1, confusion matrix, top-10 importances |
| `rf_model.joblib` | `train.py` | model bundle: `model`, `feature_cols`, `fill_values`, `classes` |
| `window_predictions_val.csv` | `train.py` | per-window val predictions + class probabilities |
| `window_predictions_test.csv` | `train.py` | per-window test predictions + class probabilities |
| `runs_log.csv` | `train.py` | one row per run; config + window counts + all val metrics |
| `runs/<run_id>/` | `train.py` | full snapshot: report, model, predictions, `run_meta.json` |
| `chosen_params.json` | `evaluate.py` | `{"tau": …, "min_len": …}` from val grid search |
| `results.csv` | `evaluate.py` | mean ± std of all metrics across test videos (raw and smoothed) |
| `test_segments.json` | `evaluate.py` | smoothed segments per test video |
| `timeline_example.png` | `evaluate.py` | GT / raw / smoothed timeline for the first test video |
| `sweep_results.csv` | `hparam_sweep.py` | ranked results across all classifier + smoothing experiments |

---

## Segment format

Every segment (in `test_segments.json` and internally in `evaluate.py`) is a
dict:

```json
{"exercise": "squat", "start_time": 4.2, "end_time": 19.8}
```

`exercise` is always one of `squat`, `push_up`, `jump_jack`, `transition`.
`start_time` and `end_time` are in seconds relative to the start of the
synthetic video.  Segments are non-overlapping and cover the full video duration
(the first segment starts at 0 and the last ends at `video_end_s`).

### `windows.csv` columns

| Column | Description |
|--------|-------------|
| `video_id` | synthetic video id, e.g. `synth_train_0001` |
| `split` | `train`, `val`, or `test` (from the video's boundary JSON) |
| `win_idx` | 0-based index within the video |
| `t_start`, `t_end` | window boundaries in seconds |
| `t_center` | midpoint in seconds |
| `label` | majority `seg_label` across the window's rows |
| `purity` | fraction of rows matching the majority label |
| `row_start`, `row_end` | slice into the time-sorted angle CSV (`iloc[row_start:row_end]`); `row_end` is exclusive |

---

## Known limitations

- **Val is ~one subject (stu10).** The val split for segmentation is built from
  synthetic videos whose source clips come almost entirely from `stu10_*`
  RepCount recordings.  Val metrics reflect performance on one person's movement
  pattern; they may not generalise.

- **Test is mostly two subjects (stu2/stu4).** Similarly, the test split is
  dominated by clips from `stu2` and `stu4`.  The test Edit score and frame
  accuracy are therefore an in-distribution estimate for those subjects, not a
  population estimate.  Grouped n and subject counts should be reported alongside
  any test metric in the paper.

- **Transitions are real pre/post-rep footage so boundaries are fuzzy.** The
  `transition` class is sourced from idle frames around RepCount rep boundaries;
  these frames blend naturally into the start and end of each exercise, so the
  true ground-truth boundary is ambiguous by a few frames.  The majority-vote
  windowing absorbs some of this, but boundary windows are legitimately
  multi-class.  This explains why `transition` is the weakest class in val F1
  and why boundary error metrics should be interpreted with that ambiguity in
  mind.

- **`jump_jack` has high NaN rates.** Both arm and leg spread involve
  fast lateral movement; MediaPipe visibility drops when limbs leave the frame
  or cross the body midline.  This causes elevated `*_nanfrac` values for
  `arm_raise_L/R` and `leg_spread` during jump-jack windows.  The current
  workaround is the pre-window NaN interpolation up to `MAX_FILL_GAP_S = 0.5 s`
  and the `nanfrac` statistic in the feature vector.  The effect is visible in
  the model: jump_jack achieves a higher F1 than might be expected (the NaN
  pattern itself is diagnostic) but also the one that degrades most when a
  subject has unusual arm trajectories.

- **No augmentation.** The Random Forest is trained on unaugmented windows.
  The architecture section of the project context mentions left/right mirroring
  and time-warp as planned augmentations; these have not been applied.

- **No deep model.** The classifier is a `RandomForestClassifier` on
  hand-crafted angle statistics.  A 1D-CNN or LSTM over the raw angle sequences
  would likely improve `transition` recall and boundary precision, but is left as
  future work.

---

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

---

## Classifier sweep results (test, smoothed)

`hparam_sweep.py` was run over 8 classifiers × 2 purity thresholds (16
experiments total).  The feature matrix (112-dim, 5891 windows, fixed
windowing / NaN-fill) was built once; only classifier, `min_purity_train`, and
smoothing params (tau × min_len, grid-searched on val) were varied.

| Rank | Classifier | Purity | Best τ | min_len | Test Edit | Test Acc | Test F1@25 | Test F1@50 | Val Edit |
|------|-----------|--------|--------|---------|-----------|----------|-----------|-----------|----------|
| 1 | **lr** | 0.7 | 0.7 | 4 | **63.4** | 0.781 | 0.579 | 0.453 | 55.2 |
| 2 | lr | 0.8 | 0.7 | 4 | 60.5 | 0.773 | 0.581 | 0.459 | 54.6 |
| 3 | gbt | 0.7 | 0.7 | 1 | 60.0 | 0.785 | 0.536 | 0.441 | 59.0 |
| 4 | hist_gbt | 0.7 | 0.7 | 1 | 59.5 | **0.815** | 0.585 | 0.466 | 48.3 |
| 5 | svm | 0.7 | 0.6 | 4 | 58.8 | 0.765 | 0.575 | 0.454 | 58.0 |
| 6 | svm | 0.8 | 0.7 | 4 | 58.3 | 0.771 | 0.601 | 0.490 | 55.2 |
| 7 | extra_trees | 0.8 | 0.4 | 1 | 58.1 | 0.807 | 0.556 | 0.474 | 54.6 |
| 9 | **rf** *(baseline)* | 0.7 | 0.6 | 1 | 57.2 | 0.799 | **0.629** | **0.487** | 56.0 |
| 14 | mlp | 0.8 | 0.6 | 3 | 49.6 | 0.715 | 0.414 | 0.291 | 47.0 |
| 16 | knn | 0.7 | 0.4 | 4 | 48.0 | 0.757 | 0.490 | 0.370 | 56.4 |

Full results in `outputs/segmentation/sweep_results.csv`.

**Key findings:**
- Logistic Regression with purity ≥ 0.7 and aggressive smoothing (τ=0.7, min_len=4) achieves
  the best test Edit Score (+6.2 points over RF baseline), suggesting that smoother decision
  boundaries combined with strong temporal post-processing outperforms RF's local partitioning.
- HistGBT achieves the best frame accuracy (0.815) but lower Edit Score — it places
  individual frames correctly but produces more fragmented sequences.
- RF retains the best segment-level F1@25 (0.629), indicating the most accurate boundary placement.
- MLP and KNN are clear underperformers on this dataset size.
- Lower purity threshold (0.7 vs 0.8) generally wins, as including more boundary windows
  exposes the model to transition-adjacent patterns.
- Best smoothing params cluster at τ=0.6–0.7 and min_len=1–4 depending on classifier.

---

## Tests

`tests/test_make_windows.py` — builds a fake 10 s CSV with 2 segments and checks
labels, purity, that no window crosses the end, plus a short-video and a
row-slice round-trip case.

```
python -m pytest tests/test_make_windows.py -v
```
