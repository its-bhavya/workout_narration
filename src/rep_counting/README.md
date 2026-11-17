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

## Rep-counting alternatives: methods and results (17 November 2026)

### Implemented logic and default status

The default remains a rule-based hysteresis state machine on cleaned 2D
angles: knee angles for squats, elbow angles for push-ups, and mean available
arm-raise angles for jumping jacks. Push-ups use local six-second normalization
and motion-aware arm selection. Peak counting and 3D counting are opt-in
experiments; the pipeline has not adopted the best experimental combination.

The peak alternative reuses those cleaned signals and arm-selection rules.
`scipy.signal.find_peaks` detects bend minima for squats/push-ups (invert the
angle signal) and raised-arm maxima for jumping jacks. Prominence must exceed
both 8 degrees and a selected fraction of the run's p10/p90 range. Minimum
width and spacing are converted from seconds to samples. Each finite tracking
run is processed independently, so no cycle spans missing pose data.

Autocorrelation uses a detrended run to find the first strong period between
0.3 and 5 seconds, requiring at least three observed cycles and correlation
>= 0.35. It can increase minimum spacing using a selected fraction of that
period, capped at one second without reducing the exercise's existing minimum.
Unreliable estimates fall back to existing spacing. A zero spacing fraction
is the unguided ablation. A motion extremum is not a completed rep: the angle
must recover by 70% of peak prominence before the next detected peak or run
end. Recovery times, rather than extrema, are exported for narration.

### Evaluation protocol

Both studies use the same 460 annotated training exercise segments: 155
jumping-jack segments from 66 unique clips, 155 push-up segments from 81 clips,
and 150 squat segments from 92 clips. Five source-disjoint training folds are
used. Repeated synthetic uses of a source clip receive one weight within their
fold; folds are weighted equally. Settings are selected on four folds and
scored on the fifth, with deterministic candidate-order ties. These are
exploratory development checks, not unbiased nested-CV estimates: the algorithm
families and existing baseline were also developed using training data.

Exercise windows are ground truth, isolating counting from segmentation.
No test feature frames were loaded for either study. The MAE unit is reps per
source clip averaged across folds, not the video/exercise count unit of the
21-video test results. The previous final-test baseline and follow-up results
remain separate. Upstream pose person selection is annotation-assisted.

### Peak detection and autocorrelation ablation

The peak search has 48 settings per exercise: prominence fractions
0.15/0.25/0.35/0.45; widths 1/15, 2/15, or 0.2 seconds; autocorrelation spacing
fractions 0/0.35/0.5/0.65. Guided and unguided families are selected separately
for the table below. The all-candidate search is also retained in report.json
and has different held-out results; it must not be substituted for a family
score. The guided settings selected on all training folds use prominence
fraction 0.25 for squats/jumping jacks and 0.45 for push-ups, width 1/15 second,
and spacing fraction 0.35. Per-held-out-fold choices are recorded separately.

| Exercise | Current 2D state machine | Guided peaks | Unguided peaks |
|---|---:|---:|---:|
| Jumping jacks | 5.288 | 4.413 | 4.204 |
| Push-ups | 2.865 | 2.981 | 2.827 |
| Squats | 1.594 | 1.153 | 1.150 |

### 2D versus world-3D signals

The dimensional comparison uses cached MediaPipe world-landmark 3D angles,
not a newly trained estimator or motion-capture truth. It first holds all
counter settings fixed while changing 2D to 3D; then it tunes parameters
separately for each exercise, method, and dimension on other training folds.
State-machine tuning searches fractions 0.2/0.55, 0.35/0.65, 0.45/0.75 and
confirmation durations 0/0.1/0.15 seconds (nine settings). Peak tuning uses the
same 48-setting grid, including guided and unguided candidates together.

Smoothing, gap handling, motion-amplitude guards, minimum spacing, local-window
length, and arm-quality weights remain fixed. The same arm-scoring rule is used,
but it may choose a different arm from each dimensional signal. Strict column
checks forbid silent 3D-to-2D fallback. Unusable signals remain zero counts;
segments are not dropped. Raw finite coverage is identical in both dimensions
(jumping jacks 88.26%, push-ups 62.98%, squats 80.23%, averaged over segment
occurrences and both raw joint channels), so 3D does not recover missing poses.
Usable 2D/3D segment counts are 148/143 for jumping jacks, 153/149 for push-ups,
and 146/146 for squats under the unchanged guards.

Fixed settings: change signal representation only (descriptive training comparison).

| Exercise | State machine 2D | State machine 3D | Peaks 2D | Peaks 3D |
|---|---:|---:|---:|---:|
| Jumping jacks | 5.288 | 5.531 | 4.413 | 5.328 |
| Push-ups | 2.865 | 4.782 | 2.845 | 4.765 |
| Squats | 1.594 | 1.050 | 1.101 | 0.954 |

Settings selected on the other four folds (peak candidates include both guidance modes).

| Exercise | State machine 2D | State machine 3D | Peaks 2D | Peaks 3D |
|---|---:|---:|---:|---:|
| Jumping jacks | 4.562 | 4.913 | 4.435 | 5.414 |
| Push-ups | 2.865 | 5.205 | 2.993 | 4.968 |
| Squats | 1.370 | 0.928 | 1.197 | 0.992 |

### Interpretation and reporting limits

Guided peaks reduce jumping-jack MAE by 16.5% and squat MAE by 27.6% relative
to the current untuned 2D state-machine baseline, but worsen push-ups by 4.0%.
Unguided peaks are as good or better in those aggregate comparisons; added
value from autocorrelation guidance is not established.

3D helps squats under both fixed and independently tuned settings, but worsens
push-ups and jumping jacks. Tuned squat state-machine MAE improves from 1.370
in 2D to 0.928 in 3D (32.3% lower); peak MAE improves from 1.197 to 0.992
(17.1% lower). The best development combination in the dimensional study is
3D state-machine squats, 2D state-machine push-ups, and 2D peaks for jumping
jacks. This is a candidate for future integration, not a deployed default or
an independently evaluated winner. A fresh holdout is needed for a new final
generalization claim. Do not advertise these training MAEs as test accuracy.

Timing precision/recall/F1 use one-to-one matching of predicted completion
times to annotated rep intervals expanded by 0.3 seconds at either end. Timing
scores pool segment occurrences, including reused source clips. Exact match,
within-one, and WAPE in the dimensional report also pool segment occurrences;
they do not use source-weighted MAE aggregation. WAPE is absolute count error
divided by annotated reps, not one minus a rep-detection accuracy.

| Exercise | Counter / signal | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Jumping jacks | state machine / 2D | 96.85% | 82.66% | 89.20% |
| Jumping jacks | state machine / 3D | 97.29% | 80.77% | 88.26% |
| Jumping jacks | peaks / 2D | 96.47% | 84.45% | 90.06% |
| Jumping jacks | peaks / 3D | 95.66% | 80.09% | 87.18% |
| Push-ups | state machine / 2D | 91.29% | 87.85% | 89.54% |
| Push-ups | state machine / 3D | 91.09% | 75.90% | 82.80% |
| Push-ups | peaks / 2D | 91.38% | 87.18% | 89.23% |
| Push-ups | peaks / 3D | 89.96% | 78.28% | 83.71% |
| Squats | state machine / 2D | 97.85% | 90.62% | 94.10% |
| Squats | state machine / 3D | 97.76% | 94.23% | 95.97% |
| Squats | peaks / 2D | 95.91% | 92.35% | 94.10% |
| Squats | peaks / 3D | 95.86% | 93.48% | 94.65% |

Existing detail/artifact records: [peak experiment](../../docs/results/PEAK_COUNTER_EXPERIMENT.md),
[peak settings, fold checks, and hashes](../../docs/results/peak_counter_20261117/report.json),
[2D/3D experiment](../../docs/results/SIGNAL_DIMENSIONS_COMPARISON.md),
[2D/3D settings, metrics, and hashes](../../docs/results/signal_dimensions_20261117/report.json).
Plots and per-segment prediction CSVs are linked from those existing reports.

### Reproduce the experiments

Run from the repository root with a new directory for each experiment:

```powershell
.venv/Scripts/python.exe -m src.rep_counting.validate_peak_counter --out-dir outputs/peak_counter/new_experiment
.venv/Scripts/python.exe -m src.rep_counting.compare_signal_dimensions --out-dir outputs/signal_dimensions/new_comparison
```

Both commands preserve existing directories and verify selected predictions
against the public experimental APIs. The dimensional comparison also checks
fixed state-machine predictions against the existing counter and asserts that
global defaults remain unchanged. No test features are evaluated.

The opt-in APIs are `peak_counter.count_peaks_for_exercise(..., settings=PeakSettings(...))`
and `compare_signal_dimensions.count_variant(..., dimensions="3d", method="state_machine",
parameters=...)`. The latter expects canonical paired features; select parameters
from the saved report rather than assuming all-training settings produced every
held-out score. The integrated `WorkoutPipeline` still calls the default counter.
At the dimensional experiment checkpoint, all 219 repository tests passed.
