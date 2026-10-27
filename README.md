# workout-narration

Silent workout video to auto-generated coaching narration (text + audio).
Exercises in scope: squat, pushup, jumping jack.

## Team / Ownership
- Bhavya — src/data_pipeline, src/segmentation
- Deepti — src/rep_counting
- Ishita — src/narration, src/evaluation, paper/

## Setup
1. Install uv, then run: `uv sync`
2. Raw video data lives on shared Google Drive (not in git) — see data/README.md
3. Manifests in data/manifests/ are the source of truth for what data is usable

## Data
Single source: RepCount dataset (squat, push_up, jump_jack). See data/README.md for the full manifest breakdown and pipeline notes.

## Structure
See data/README.md and each src/ subfolder's README for details.

## Rep counting status (October 2026)
Rule-based counter (adaptive thresholds + hysteresis state machine) on joint angle CSVs.
Evaluated on train+val synthetic videos using ground-truth segment windows.
See `src/rep_counting/README.md` for full results and known issues.

| exercise | exact_% | within1_% | MAE |
|----------|---------|-----------|-----|
| squat | 67.6 | 75.7 | 2.0 |
| jump_jack | 45.0 | 71.1 | 1.7 |
| push_up | 46.0 | 62.2 | 2.6 |

Test split is locked until the method is final.
