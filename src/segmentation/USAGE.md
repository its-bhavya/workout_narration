# Segmentation Pipeline — Quick Start

From the repo root, build the sliding-window classifier end-to-end:

## 1. Build window index

```bash
python src/segmentation/make_windows.py
```

Outputs: `outputs/segmentation/windows.csv` (5891 windows: 3292 train, 541 val, 2058 test).
Prints window counts per split × class.

## 2. Train and evaluate

```bash
python src/segmentation/train.py --tag "experiment_label"
```

Outputs:
- `outputs/segmentation/val_window_report.txt` — latest val metrics
- `outputs/segmentation/rf_model.joblib` — latest model bundle (feature_cols, fill_values, classes)
- `outputs/segmentation/window_predictions_{val,test}.csv` — per-window predictions (all class probs + max)
- `outputs/segmentation/runs/<run_id>/` — archived full snapshot (report, model, predictions, metadata)
- `outputs/segmentation/runs_log.csv` — master log, one row per run

`--tag` is optional; recorded in the log so you can compare experiments.

## Current results (val)

RandomForest, window 2.0 s, stride 1.0 s, train purity ≥ 0.8, 112 features.

| metric | value |
|--------|-------|
| accuracy | 0.828 |
| macro-F1 | 0.766 |

| class | F1 |
|-------|------|
| jump_jack | 0.955 |
| push_up | 0.885 |
| squat | 0.864 |
| transition | 0.360 |

## Inspect past runs

```bash
# Compare experiments: open outputs/segmentation/runs_log.csv
# See all runs' model params, window counts, and metrics side-by-side

# Drill into a run:
cat outputs/segmentation/runs/<run_id>/run_meta.json
cat outputs/segmentation/runs/<run_id>/val_window_report.txt
```

## Modify hyperparameters

All parameters are in `src/segmentation/config.py`: `WINDOW_S`, `STRIDE_S`,
`MIN_PURITY_TRAIN`, `MIN_ROWS_FRAC`, `SEED`. Change any and re-run `train.py` —
the new values are logged and every run is archived, so you have full history.

## Next steps

After the classifier, implement:
- OnlineTAS-style smoothing (confidence threshold, minimum segment length)
- Raw-vs-smoothed evaluation metrics (frame accuracy, Edit score, F1@{10, 25, 50}, boundary error)
- Per-segment rep counting via the rep_counter from `src/rep_counting/`
