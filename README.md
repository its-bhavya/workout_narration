# V2 source-clip CV, evaluation and integration

For a single video, follow [the end-to-end PowerShell runbook](docs/END_TO_END.md):
registered v2 video/features -> frozen segmentation model -> per-segment rep
counting -> text cues -> scheduled gTTS narration WAV. It includes the bridge
from analysis JSON to audio, output locations and optional cached LLM phrases.
Arbitrary new MP4 ingestion is still deferred; `main.py` alone exports text,
not audio or a narrated MP4.

Current training/evaluation and integration guide: [V2_TRAINING.md](docs/V2_TRAINING.md).

## Rep-counting alternatives: current findings (18 November 2026)

The integrated pipeline adopts the frozen experimental combo as the default
rep counter (`rep_method='experimental'`: unguided arm-raise peaks for jumping
jacks, unguided hip-angle peaks for squats, and frozen coordinate median
voting for push-ups), while preserving the legacy 2D state machine via
`rep_method='state_machine'`. Upstream segmentation is kept as is. Experimental
peak counting detects prominent, sufficiently wide motion extrema, optionally uses
capped autocorrelation-guided spacing, and timestamps recovery after the
extremum. Cached world-3D angles are compared with 2D using the same signal
cleaning and arm-selection rules; missing 3D columns cannot fall back to 2D.

Five source-disjoint training folds evaluate 460 GT-window exercise segments.
MAE averages source clips within each fold, then averages folds; it is not
end-to-end test MAE. Peak-family settings are selected on the other four folds:

| Exercise | Current 2D state machine | Guided peaks | Unguided peaks |
|---|---:|---:|---:|
| Jumping jacks | 5.288 | 4.413 | 4.204 |
| Push-ups | 2.865 | 2.981 | 2.827 |
| Squats | 1.594 | 1.153 | 1.150 |

A separate paired 2D/3D study tunes parameters independently on other folds.
Its peak search includes both guided and unguided settings:

| Exercise | State machine 2D | State machine 3D | Peaks 2D | Peaks 3D |
|---|---:|---:|---:|---:|
| Jumping jacks | 4.562 | 4.913 | 4.435 | 5.414 |
| Push-ups | 2.865 | 5.205 | 2.993 | 4.968 |
| Squats | 1.370 | 0.928 | 1.197 | 0.992 |

3D state-machine squats, 2D state-machine push-ups, and 2D peaks for jumping
jacks are the strongest development combination here. It has not replaced
the pipeline default. Guidance itself did not consistently beat unguided peaks.
These are exploratory GT-window training results, not new test scores or an
unbiased nested-CV estimate. Pose person selection remains annotation-assisted.

Methods, fixed-settings results, tuning grids, and commands are documented in
[the rep-counting README](src/rep_counting/README.md). Existing
[peak](docs/results/PEAK_COUNTER_EXPERIMENT.md) and
[2D/3D](docs/results/SIGNAL_DIMENSIONS_COMPARISON.md) reports retain plots,
per-segment predictions, timing metrics, source hashes, and selected settings.

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
