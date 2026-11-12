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

## Rep counting status (November 2026)
Rule-based counter (adaptive thresholds + hysteresis state machine with temporal
confirmation) on joint angle CSVs. Evaluated on train+val synthetic videos using
ground-truth segment windows. See `src/rep_counting/README.md` for full results
and known issues.

Temporal confirmation (`confirm_s`) was added to the state machine: a threshold
crossing must hold for a set duration (0.10–0.15 s depending on exercise) before a
rep is counted, which filters brief false crossings from pose jitter. Jump_jack
exact-match improved from 45% to 58% and within-1 from 71% to 84% with this change.

| exercise | exact_% | within1_% | MAE | precision | recall |
|----------|---------|-----------|-----|-----------|--------|
| squat | 64.9 | 75.7 | 2.2 | 0.98 | 0.80 |
| jump_jack | 57.8 | 84.4 | 2.3 | 0.99 | 0.91 |
| push_up | 48.6 | 59.5 | 2.9 | 0.93 | 0.89 |

Multi-person tracking (`PosePersonTracker` in `pose_utils.py`) is implemented in
the pose extraction pipeline to lock onto the exercising subject across frames.
Angle CSVs need re-extraction to benefit from this; the `stu10_49` multi-person
overcounting case persists until then.

Test split is locked until the method is final.

## Segmentation status (November 2026)
Sliding-window classifier + temporal smoothing. The training pipeline now supports
8 classifier types via `--model` (rf, extra_trees, gbt, hist_gbt, lr, svm, knn,
mlp). Smoothing was improved: transition labels are preserved during low-confidence
filtering, short exercise blips absorb into neighbouring transitions, and
sandwiched transitions between identical exercises are collapsed (e.g.
`push_up | transition | push_up` merges into one `push_up` run). On the test split
with HistGradientBoosting, these changes improved Edit score from 56.2 to 64.1 and
transition frame-F1 from 0.36 to 0.50. See `src/segmentation/README.md` and
`outputs/segmentation/smoothing_improvement_report.md` for details.

## Integrated Pipeline & Narration Engine Interface

The segmentation and rep counting modules are integrated into an end-to-end pipeline in `src/pipeline` and runnable via `main.py`.

Given any multi-exercise video (video file, angles CSV, or video ID), the pipeline:
1. Resolves input features (or extracts MediaPipe joint angles).
2. Runs sliding-window classification and OnlineTAS temporal smoothing to predict exercise and transition segments.
3. Runs per-segment hysteresis state-machine rep counting with boundary padding.
4. Generates structured output, workout summaries, and a timestamp-aligned narration cue sheet ready for TTS engines.

### Running via CLI

```bash
# Analyze a synthetic video (default: synth_test_0001)
uv run python main.py synth_test_0001

# Analyze an MP4 video or CSV and export results for the narration engine
uv run python main.py path/to/video.mp4 --export workout_narration.json

# Output raw JSON to pipe directly into downstream narration / TTS tools
uv run python main.py synth_test_0001 --json
```

### Python API

```python
from src.pipeline import WorkoutPipeline, analyze_workout

# Single-call execution
result = analyze_workout("synth_test_0001")

# High-level summary
print(result.summary.exercise_counts)  # e.g. {'jump_jack': 37, 'push_up': 58}
print(result.summary.total_reps)       # e.g. 95

# Segments breakdown
for seg in result.segments:
    print(f"{seg.exercise}: {seg.start_time:.1f}s - {seg.end_time:.1f}s | Reps: {seg.rep_count}")

# Formatted narration cue sheet ready for TTS / narration engine
cues = result.to_narration_script()
# Each cue has: {"time_s": ..., "cue_type": ..., "text": ..., "exercise": ...}
```

