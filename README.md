# V2 source-clip CV, evaluation and integration

For current methods and the GT-window and post-segmentation CV/test results, see the [v2 pipeline results](docs/results/V2_PIPELINE_RESULTS.md).

For a single video, follow [the end-to-end PowerShell runbook](docs/END_TO_END.md):
registered v2 video/features -> frozen segmentation model -> per-segment rep
counting -> text cues -> scheduled gTTS narration WAV. It includes the bridge
from analysis JSON to audio, output locations and optional cached LLM phrases.
The integrated flow uses registered v2 videos. Audio is optional via
`--narration-dir`, and `--burn-in` adds speech and subtitles to a video copy.

Current training/evaluation and integration guide: [V2_TRAINING.md](docs/V2_TRAINING.md).

## Current pipeline

The integrated v2 pipeline uses the frozen random forest with 2-second windows,
1-second stride, and OnlineTAS smoothing (tau=0.5, min_len=3). The default rep
counter uses unguided 2D arm-raise peaks for jumping jacks, unguided
confidence-weighted 2D hip-angle peaks for squats, and frozen coordinate
median voting for push-ups. Use `--rep-method state_machine` for the legacy
hysteresis counter.

Squat and jumping-jack counting reuse the cached angle features. Push-up
voting derives three motion signals from raw pose coordinates in each
predicted push-up window. Narration optionally turns the completion timestamps
into duration-aware speech and subtitles, with audio bounded by the video.
The earlier angle-only peak and state-machine results are retained in the
[v2 pipeline results](docs/results/V2_PIPELINE_RESULTS.md).

## Data preparation and v2 protocol

Active protocol and provenance: [PROJECT_CONTEXT](docs/PROJECT_CONTEXT.md).

RepCount only, pass/review, fps >=20: train is original train (239 clips);
test is original val+test (94 clips). Countix is excluded. Contributor prefixes
are not subjects. There is no separate validation split.

Five train source-clip folds use stratified seed 3969, selected on total margin
balance only. Construction uses seed 42: 24 videos/fold, K=3..5, train cap 5,
test cap 1, 21 test videos. Exercise slots stay within +/-10%; clips never
repeat within a video. Train reuse spread is <=1 per exercise/fold.

Phase 4 uses per-video features in data/features/v2/ and existing window code
in outputs/segmentation_v2/. Run-ID guards reject missing/non-v2 provenance and
archived data. V1 data is frozen in data/_archive_v1/. Native clip caches are
deferred; separate-val workflows are retired; source-clip CV training/evaluation is now available.

From the repository root, resume the entire preparation workflow:

```powershell
uv run --no-sync python -B src/data_pipeline/phase4.py
```

This extracts train and test at stride 2 with one worker, without --force;
then rebuilds summary/long-gap diagnostics, reports boundary ratios, builds
windows and reports counts before/after training purity 0.8. Test counts stay
unfiltered. `src/segmentation/cv.py:get_cv_split` supplies training folds !=k
and validation fold k, asserting zero source-clip overlap. No models are trained
or evaluated. Test is extraction and counting only.

To rerun only reports/windows after extraction is complete:

```powershell
uv run --no-sync python -B src/data_pipeline/phase4.py --postprocess-only
```

Historical documentation is retained under docs/history/ and is not an active
workflow. See docs/V2_GENERATION.md for rendering/resume details.

## Model workflow handoff

Train-only CV selects the classifier, purity and smoothing. A separate explicit
refit creates the all-train bundle; final test is a separate explicit command.
Prediction and narration use that same bundle and exact n_frames/fps duration.
Ground-truth comparison is opt-in. Dataset run_id=v2 and experiment_id remain
distinct, with input hashes and feature-schema checks. Model compatibility was
verified on synthetic unit-test fixtures; this is not a new benchmark result.

<!-- v2-model-results:start -->
## Completed v2 model experiment — 17 November 2026

8-model train-only CV selected **rf** (Edit 71.54 ± 2.90 fold SD; tau=0.5, min_len=3). After all-train refit, only that frozen winner was evaluated on 21 test videos: Edit 72.51, frame accuracy 86.66%, segment F1@25 0.750. Test was not used for model selection or tuning. See [full results, caveats and reproducibility tables](docs/results/V2_MODEL_RESULTS.md).
<!-- v2-model-results:end -->
