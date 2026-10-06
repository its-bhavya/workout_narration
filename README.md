# workout-narration

Silent workout video to auto-generated coaching narration (text + audio).
Minor project, 4-week timeline. Exercises in scope: squat, pushup, jumping jack.

## Team / Ownership
- Bhavya — src/data_pipeline, src/segmentation
- Deepti — src/rep_counting
- Ishita — src/narration, src/evaluation, paper/

## Setup
1. Install uv, then run: uv sync
2. Raw/synthetic video data lives on shared Google Drive (not in git) — see data/README.md
3. Manifests in data/manifests/ are the source of truth for what data is usable

## Structure
See data/README.md and each src/ subfolder's README for details.
