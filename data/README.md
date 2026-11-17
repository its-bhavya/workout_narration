# Workout narration: active v2 protocol

Active protocol and provenance: [PROJECT_CONTEXT](../docs/PROJECT_CONTEXT.md).

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
deferred; no separate-val training/evaluation command is active.

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
