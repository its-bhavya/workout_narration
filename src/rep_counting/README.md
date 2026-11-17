# Rep counting on v2 pose features

Current training/evaluation and integration guide: [V2_TRAINING.md](../../docs/V2_TRAINING.md).

## Data preparation and v2 protocol

Active protocol and provenance: [PROJECT_CONTEXT](../../docs/PROJECT_CONTEXT.md).

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

## Fixed v2 final-test evaluation

```powershell
uv run --no-sync python -m src.rep_counting.evaluate_final --out-dir outputs/rep_eval_v2/new_run
```

Evaluates every indexed test video using the frozen segmentation model and current
counter defaults. Records end-to-end video/exercise counts, video totals, and a
separate ground-truth-window counter diagnostic, with model/input hashes and
settings. No calibration is performed; do not tune from test results. Existing
output directories are preserved. See [recorded results](../../docs/results/V2_REP_COUNT_RESULTS.md).

## Push-up local adaptation

Push-ups now use local adaptive thresholds selected on training folds. See the
[change and follow-up results](../../docs/results/PUSHUP_COUNTER_UPDATE.md).
Reproduce training-only selection with:

```powershell
.venv/Scripts/python.exe -m src.rep_counting.calibrate_pushups --out-dir outputs/rep_pushup_dev/new_selection
```

This reports settings; it never rewrites defaults automatically. The original
final-test baseline is preserved separately from the follow-up evaluation.

## Motion-aware push-up arm selection and diagnostic plots

Push-ups select a usable elbow signal using visibility, continuity, and periodic
motion quality. Scores and both arm candidates are exported with segment results.
Use `count_for_exercise(..., arm_selection="visibility")` to compare the previous
arm preference. The scorer does not use predicted or annotated rep counts.

```powershell
.venv/Scripts/python.exe -m src.rep_counting.validate_arm_selection --out-dir outputs/arm_selection/new_validation --plots 8
```

This selects and validates quality weights using training folds only, checks
runtime agreement, and writes per-arm/segment CSVs, fold metrics, and diagnostic
plots. It does not change defaults automatically or evaluate test features.
See [results and diagnostic plots](../../docs/results/MOTION_ARM_SELECTION.md).
