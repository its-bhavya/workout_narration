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

## Segment-local PCA plus peaks (17 November 2026)

This opt-in experiment leaves input data, existing results, and production
defaults unchanged. `pca_counter.py` projects ten explicitly allowed angle
channels (both knees/elbows/hips/arm raises, leg spread, torso tilt) onto PC1.
Only 2D angles or only actual world-3D angles are used in a projection; missing
columns raise an error rather than falling back to another dimensional signal.
Annotation labels, boundaries, IDs, and fold columns are excluded from PCA.

Per segment, confidence below 0.5 masks observations. Channels need >=50%
coverage and >=10 degrees of p10/p90 motion range; at least three channels are
required. The existing short-gap cleaner and 0.3-second requested smoothing
are applied. Retained channels are standardized, remaining missing cells use
the channel mean in standardized space, and frames with <60% channel coverage
are masked. PCA fits only adequate-coverage rows. PC1 scores are rescaled to
unit standard deviation; explained variance, channel loadings, centers/scales,
coverage, imputation fraction, and abstention reasons are exported.

This is a bundled PCA-plus-preprocessing alternative, not an isolated PCA-only
ablation of the earlier joint counter. A mean-imputed coordinate is not recovered
physical motion, and missingness/posture changes can dominate a component.
The largest-magnitude loading fixes the otherwise arbitrary PC sign. Peak
polarity is selected on other training folds; a generic PCA recovery event is
not guaranteed to correspond to anatomical rep completion. It must not drive
narration without further phase validation.

Each 2D/3D PCA variant searches 24 peak settings per exercise: absolute
score-space prominence 0.5/1/1.5 (also at least 15% of local PC range), width
1/15 or 2/15 seconds, autocorrelation spacing fraction 0 or 0.35, and polarity
+1 or -1. Degree-valued prominence defaults are not reused on PCA scores.
The same missing-run separation and post-peak recovery logic is retained.
Settings are selected on four source-disjoint training folds and evaluated on
the fifth over the same 460 GT-window segments, with source-weighted fold MAE.
PCA/scaling fit inside each individual segment without rep annotations;
whole-segment lookahead makes this an offline experiment.

| Exercise | Previous selected comparator MAE | PCA 2D + peaks | PCA 3D + peaks |
|---|---:|---:|---:|
| Jumping jacks | 4.435 | 4.580 | 4.781 |
| Push-ups | 2.865 | 4.889 | 5.152 |
| Squats | 0.928 | 1.109 | 0.899 |

The previous selected comparators are the dimensional study's 2D peaks for
jumping jacks (full peak search), 2D state machine for push-ups, and 3D state
machine for squats. They were chosen from inspected development results, so
this is exploratory comparison, not an independent test or unbiased nested CV.

PCA worsens push-up counts and does not beat the previous selected jumping-jack
counter on MAE. Squat PCA 3D slightly lowers count MAE (0.928 to 0.899, about
3.1%), but rep-timing F1 falls from 95.97% to 95.22%; this is insufficient
evidence to replace the simpler squat counter. Timing matches use the existing
one-to-one interval matching with 0.3-second tolerance and pool reused segment
occurrences, unlike source-weighted MAE. The default remains unchanged.

No test feature frames were evaluated. Upstream pose selection remains
annotation-assisted. Before/after SHA-256 verification confirmed all 245
protected training input, manifest, and reference-result files were unchanged.
Selected settings also matched the public API for all 920 segment/dimension
projections. Synthetic tests additionally check input nonmutation, annotation
exclusion, long gaps, visibility, strict 3D columns, and training-only access.

```powershell
.venv/Scripts/python.exe -m src.rep_counting.validate_pca_counter --out-dir outputs/pca_counter/new_experiment
```

Existing output directories are preserved. The local run is under
`outputs/pca_counter/verified_train_experiment_20261117/` (ignored by Git), containing
[report/settings/hash verification](../../outputs/pca_counter/verified_train_experiment_20261117/report.json),
[projection diagnostics](../../outputs/pca_counter/verified_train_experiment_20261117/projection_diagnostics.csv),
[held-out predictions](../../outputs/pca_counter/verified_train_experiment_20261117/heldout_predictions.csv),
and [comparison plot](../../outputs/pca_counter/verified_train_experiment_20261117/comparison.png).
Three fixed first-segment examples show projected signals, channel motion, and
annotated versus predicted recovery events. Existing reference results were
read only; no data/features/model files were rewritten.

## Unguided-peak error analysis and targeted improvements (18 November 2026)

Work branch: `experiments/peak-detection-improvements`. The default integrated
counter and all existing feature/annotation files remain unchanged. These are
opt-in, training-only experiments, not new end-to-end test results.

### Error analysis before counter changes

`analyze_peak_errors.py` reconstructs the exact unguided peak settings selected
on other source-disjoint training folds and checks count MAE and temporal errors
against the saved reference. It writes signed errors per segment/source clip,
video/exercise counts, full-video totals, and every unmatched annotated/detected
rep. Position diagnostics include normalized first/middle/last segment bins,
first/last annotated reps, edge proximity, and missing-signal overlap. Missingness
and position are associations, not proof of a particular failure mechanism.

The 120 GT-window training videos contain 8,654 annotated reps versus 8,052
baseline detections. Full-video mean signed error is -5.02 reps (undercounting),
MAE 8.10, and 32.50% of videos are within one rep. Video totals can hide errors
cancelling across exercises and do not include segmentation error.

| Exercise | Missed reps | Extra detections | Active videos | Signed error / active video | Active videos within +/-1 |
|---|---:|---:|---:|---:|---:|
| Jumping jacks | 543 | 114 | 100 | -4.29 | 66.00% |
| Push-ups | 386 | 281 | 65 | -1.62 | 29.23% |
| Squats | 139 | 71 | 120 | -0.57 | 75.00% |

All-video per-exercise within-one rates (including absent-class zero rows) are
71.67% jumping jacks, 61.67% push-ups, and 75.00% squats. Active-only rates above
avoid those exact-zero matches inflating the apparent performance. No missing
or failed segment is excluded from count evaluation.

For jumping jacks, 432 of 543 misses occur in the middle 80% of segment time;
the middle miss rate is 14.82%, versus 12.75% in the first 10% and 15.43% in
the last 10%. Edge effects alone do not explain the main deficit. First/last
annotated-rep miss rates are 12.90%/19.35% for jumping jacks, 20.65%/12.26% for
push-ups, and 13.33%/12.00% for squats. Misses overlapping missing cleaned
tracking are 200/543, 247/386, and 101/139 respectively. These are pooled
segment-occurrence counts, including synthetic reuse, not independent clips.

Opposite-phase peak/recovery counts are also computed without changing the
reported count. Disagreement >1 flags 18 jumping-jack, 40 push-up, and 28 squat
segments. Of those flagged segments, 83.33%, 85.00%, and 50.00% respectively
have count error >1. This is a diagnostic association, not a validated decision
threshold or an instruction to pick whichever phase produces more reps.

### What the cached data supports

The cache has arm-raise, leg-spread, knee, hip, and elbow angles with visibility,
and ankle distance normalized by horizontal shoulder width. It does not store
raw wrist/hip positions needed to derive wrist height, wrist-to-wrist distance,
or normalized hip height. Those would require a separate extraction stage;
no pose extraction or existing feature modification was performed here.

`peak_improvements.py` evaluates three related changes:

- Smoothing/spacing on the existing selected signal, with genuine smoothing-off
  support rather than the old cleaner's minimum five-sample filter.
- Conservative arm/ankle fusion for jumping jacks, confidence-weighted elbow
  averaging for push-ups, and knee/hip-angle fusion for squats. Signals are
  independently standardized. Fusion requires all components at a frame to
  avoid changes in contributors manufacturing a cycle. Elbow visibility weights
  ignore unavailable/low-confidence joints rather than inserting zero angles.
- Median-count voting among three usable component signals. The returned event
  timeline belongs to an actual candidate having the median count; event times
  are never fabricated or truncated. With fewer than three voters it falls back
  to the baseline signal.

Motion guards remain per raw feature (including 0.15 shoulder-width units for
ankle spacing). Prominence is adaptive already: a fraction of each finite
run's robust p10/p90 range. After standardization, this experiment uses a
0.1 score-unit minimum rather than the earlier 8-degree floor. This is a bundled
signal/preprocessing comparison, not an isolated fusion-only ablation.
Autocorrelation spacing is disabled throughout; no signal is reflected or padded
at segment edges and the existing 70%-prominence recovery convention is unchanged.

The grid uses smoothing 0/0.3/0.6 seconds, prominence fractions
0.15/0.25/0.35/0.45, minimum width 1/15 second, and two minimum spacings per
exercise: jumping jacks 0.25/0.5, push-ups 0.2/0.35, squats 0.35/0.5 seconds.
Four signal families for jumping jacks/squats give 96 candidates each; push-ups
use three distinct families (72 candidates). Parameters and family are selected
on four training folds and evaluated on the fifth, equally weighting source
clips within folds. This remains exploratory development, not unbiased nested CV.

### Results and trade-offs

| Exercise | Previous source-weighted fold MAE | New other-fold selected MAE | Previous active videos within +/-1 | New active videos within +/-1 |
|---|---:|---:|---:|---:|
| Jumping jacks | 4.204 | 2.877 | 66.00% | 64.00% |
| Push-ups | 2.827 | 2.989 | 29.23% | 29.23% |
| Squats | 1.150 | 0.975 | 75.00% | 85.83% |

Jumping-jack MAE improves by 31.6%, but the within-one video fraction falls by
two percentage points. Rep-matching F1 improves from 90.54% to 93.73%. A lower
MAE is therefore not a universal per-video improvement. Whole-video count MAE
across all exercises drops from 8.10 to 5.90 and within-one rises from 32.50%
to 36.67%, with possible cancellation between class errors.

The all-training selected jumping-jack setting keeps the existing arm signal,
disables smoothing, uses prominence fraction 0.25 and minimum spacing 0.25 s.
Within the same normalized baseline-signal family, fixing the old 0.5 s gap
yields other-fold MAE 4.210 versus 2.877 when spacing is selectable. Fixing
smoothing at 0.3 s still gives 2.992. The spacing constraint is the strongest
supported lever; the gain is not evidence that arm/leg fusion helped.

Family-specific other-fold MAEs:

| Exercise | Existing signal, retuned | Alternate signal | Fused signal | Median voting |
|---|---:|---:|---:|---:|
| Jumping jacks | 2.877 | 5.197 | 4.579 | 3.729 |
| Push-ups | 2.821 | same as fused | 3.489 | 2.978 |
| Squats | 1.216 | 0.963 | 0.943 | 0.994 |

For push-ups, weighted averaging and voting did not outperform the previous
selected-arm approach. For squats, hip angles/fusion are promising, but the best
family selected after inspecting all fold scores is itself a development choice,
not an independent estimate. The all-family other-fold selector is the main
comparison table above. Existing 3D squat results are separate and unchanged.

No new default is promoted: retain the prior push-up counter, consider a frozen
spacing/smoothing update for jumping jacks, and investigate squat hip-angle
fusion further. Automatic edge completion or taking the larger peak/valley
count would risk counting partial/noisy motion and was not introduced.

### Reproduce and inspect

```powershell
.venv/Scripts/python.exe -m src.rep_counting.analyze_peak_errors --out-dir outputs/peak_improvements/new_error_analysis
.venv/Scripts/python.exe -m src.rep_counting.validate_peak_improvements --baseline-dir outputs/peak_improvements/new_error_analysis --out-dir outputs/peak_improvements/new_comparison
```

The programs refuse existing output directories, validate against reference
input hashes, verify selected public-API outputs on all 460 segments, and assert
that training data and global counter defaults remain unchanged. No test feature
frames are read. The opt-in API is `count_improved_peaks(..., settings=ImprovementSettings(...))`.

Local outputs (ignored by Git):

- [Signed, positional, tracking, video, and phase-error report](../../outputs/peak_improvements/error_analysis_with_phase_20261118/report.json)
- [Per-video/exercise baseline counts](../../outputs/peak_improvements/error_analysis_with_phase_20261118/video_exercise.csv)
- [Every temporal error](../../outputs/peak_improvements/error_analysis_with_phase_20261118/temporal_errors.csv)
- [Verified alternative settings and metrics](../../outputs/peak_improvements/verified_cached_signal_comparison_20261118/report.json)
- [All candidate/fold scores](../../outputs/peak_improvements/verified_cached_signal_comparison_20261118/candidates.csv)
- [Other-fold selected rep timestamps](../../outputs/peak_improvements/verified_cached_signal_comparison_20261118/heldout_segments.csv)
- [Comparison plot](../../outputs/peak_improvements/verified_cached_signal_comparison_20261118/comparison.png)

No annotations, cached features, existing reference results, or integrated
counter defaults were rewritten. No commits or pushes were performed.

## Frozen unguided-peak test follow-up (18 November 2026)

`evaluate_peak_test.py` freezes both configurations from training records and
writes `frozen_settings.json` before the first test inference. The candidate
uses each exercise's all-training selected settings from the cached-signal
experiment; the baseline uses the earlier unguided-peak settings selected on
all training folds. No test-driven parameter edits, signal changes, or automatic
adoption occur. This is a follow-up on the same already-inspected test set,
not a new untouched holdout.

All 21 indexed synthetic test videos use the same frozen RF segmentation
bundle. Both counters receive identical predicted exercise windows, with the
existing 0.3-second end padding; GT-window diagnostics use zero padding.
No test videos or failed segments are silently skipped. Separate per-class
temporal matching penalizes detections in the wrong exercise/time interval.

### Segmentation followed by counting

MAE is reps per exercise per video: 21 units per exercise, 63 overall, including
absent-class zero rows. This is not the source-weighted training-fold MAE.

| Exercise | Previous predicted / annotated | Proposed predicted / annotated | Previous MAE | Proposed MAE | Previous within +/-1 | Proposed within +/-1 |
|---|---:|---:|---:|---:|---:|---:|
| Jumping jacks | 782 / 761 | 786 / 761 | 2.714 | 2.905 | 52.38% | 52.38% |
| Push-ups | 430 / 468 | 430 / 468 | 4.381 | 4.381 | 61.90% | 61.90% |
| Squats | 357 / 361 | 349 / 361 | 2.381 | 2.095 | 52.38% | 61.90% |
| Overall | 1569 / 1590 | 1565 / 1590 | 3.159 | 3.127 | 55.56% | 58.73% |

The overall video/exercise MAE improvement is about 1.0%, driven by squats.
Jumping jacks regress slightly and push-ups are unchanged. Overall WAPE is
12.52% -> 12.39%, exact match 36.51% -> 38.10%, and class-aware timing F1
90.91% -> 91.16%. The candidate counts 1,565 of 1,590 annotated reps, but that
net total is not a detection-accuracy measure.

For complete video totals (sum across classes), MAE worsens from 7.19 to 7.76
reps/video, and within-one falls from 4/21 (19.05%) to 2/21 (9.52%). Aggregate
class errors can cancel differently when summed per video; these metrics do
not contradict a small improvement in video/exercise MAE. They prevent claiming
that the new candidate is an across-the-board improvement.

Active-only per-exercise within-one fractions are 41.18% jumping jacks (17
videos), 27.27% push-ups (11), and 61.90% squats (21) for the candidate. Baseline
rates are 41.18%, 27.27%, and 52.38%. Active-only reports and all-video reports
are both retained to expose inflation from exercise-absent exact-zero rows.

### Counter-only GT-window diagnostic (verified replay, 18 November 2026)

Both frozen configurations were replayed on all 79 annotated exercise windows
across all 21 test videos, without parameter changes or end padding. All 158
counts and completion-timestamp lists exactly reproduce the original saved
GT-window diagnostic. This is supporting error analysis, not the headline result.
The headline remains counting after predicted segmentation.

MAE below is per annotated exercise segment; exact/within-one percentages use
those same segments. This unit differs from per-video/exercise headline MAE.

| Exercise | GT windows | Previous MAE | Proposed MAE | Proposed exact | Proposed within +/-1 |
|---|---:|---:|---:|---:|---:|
| Jumping jacks | 27 | 1.370 | 1.444 | 51.85% | 77.78% |
| Push-ups | 27 | 3.778 | 3.778 | 48.15% | 59.26% |
| Squats | 25 | 1.360 | 1.200 | 68.00% | 84.00% |
| Overall | 79 | 2.190 | 2.165 | 55.70% | 73.42% |

The proposed counter detects 1,515 of 1,590 annotated reps inside GT windows.
Push-up count errors remain substantial with perfect exercise windows; removing
segmentation errors does not solve the principal counting failure. The proposed
jumping-jack training gain is absent in the GT-window test diagnostic as well.

For a comparable window-policy diagnostic, GT-window counts were summed into
21 video/exercise units per class (63 overall). Predicted windows were also
recounted with zero end padding, holding counter parameters, input video,
labels-at-inference, and aggregation fixed. These supplementary zero-padding
runs do not replace the existing headline with 0.3-second predicted-window padding.

| Exercise | Predicted windows, no padding | GT windows, no padding | Original headline, predicted windows +0.3 s |
|---|---:|---:|---:|
| Jumping jacks | 2.810 | 1.762 | 2.905 |
| Push-ups | 4.286 | 4.762 | 4.381 |
| Squats | 2.048 | 1.429 | 2.095 |
| Overall | 3.048 | 2.651 | 3.127 |

All numbers in this second table are MAE per video/exercise. Even after aligning
units and padding, differences are not an additive causal decomposition into
segmentation and counting percentages: boundaries, labels, segment fragmentation,
adaptive signal normalization, and cancellation interact. GT windows can even
worsen aggregate count error (as for push-ups) when predicted-window extra
counts had offset misses. Do not subtract the first table's per-segment MAE
from a headline per-video/exercise MAE.

The supplementary diagnostic verified all original test features, annotations,
and prior result files unchanged; source hashes and the original frozen-settings
hash are preserved in a separate local report. Production defaults and previous
headline scores are unchanged. This remains an already-inspected synthetic test
set with annotation-assisted pose extraction, not an untouched holdout or a
new claim of unassisted video accuracy.

Local supporting artifacts (ignored by Git):

- [GT-window replay and same-unit comparison report](../../outputs/peak_improvements/gt_window_diagnostic_20261118/report.json)
- [Verified per-GT-window counts and timestamps](../../outputs/peak_improvements/gt_window_diagnostic_20261118/gt_segments.csv)
- [GT counts aggregated per video/exercise](../../outputs/peak_improvements/gt_window_diagnostic_20261118/gt_video_exercise.csv)
- [Predicted-window counts with zero padding](../../outputs/peak_improvements/gt_window_diagnostic_20261118/predicted_video_exercise_zero_padding.csv)

The normal `evaluate_peak_test` command below reproduces the annotated-window
diagnostic alongside the headline. To reproduce the supplementary zero-padding
comparison, call `evaluate(..., end_pad_s=0.0)` from Python with a new output path.

### Reproduce and artifacts

```powershell
.venv/Scripts/python.exe -m src.rep_counting.evaluate_peak_test --out-dir outputs/peak_improvements/new_test_followup
```

Optional `--training-report`, `--baseline-report`, and `--model` select explicit
frozen artifacts. Existing output directories are refused. The saved report
contains model/source/input hashes and verifies test feature/annotation bytes
are unchanged after counting. No data files or previous reports are rewritten.

Local outputs (ignored by Git):

- [Report and both configurations' metrics](../../outputs/peak_improvements/test_followup_20261118/report.json)
- [Settings frozen before test inference](../../outputs/peak_improvements/test_followup_20261118/frozen_settings.json)
- [Per-video/exercise counts and matched events](../../outputs/peak_improvements/test_followup_20261118/video_exercise.csv)
- [Whole-video totals](../../outputs/peak_improvements/test_followup_20261118/video_totals.csv)
- [Predicted segments and rep timestamps](../../outputs/peak_improvements/test_followup_20261118/predicted_segments.csv)
- [GT-window diagnostic](../../outputs/peak_improvements/test_followup_20261118/oracle_segments.csv)

No commits or pushes were performed for this evaluation.

## Push-up view, timing, coordinate and crop diagnosis (18 November 2026)

This follow-up investigates the frozen GT-window push-up failures. It does not
change the counter, pose cache, annotations, benchmark scores, or defaults.
Test clips are used descriptively; all spacing/smoothing/prominence/recovery
parameter selection below uses training folds only.

### Signed errors and coarse camera views

Across 27 annotated test push-up source windows, 13 counts are exact, seven
overcount and seven undercount. Total absolute error is 102 reps; two undercount
outliers contribute 58 (56.9%). Thus the average bias (-1.78/window) hides a
mixed error distribution concentrated in a few failures, rather than a uniform
prominence or rate problem.

Three sampled frames per source window were manually inspected for coarse
view tags. Camera tags are observational, not calibrated geometry or causal
proof, and each synthetic video can contain several source views.

| Synthetic test video | Annotated | Predicted in GT windows | Signed error | Sampled source views |
|---|---:|---:|---:|---|
| synth_v2_test_0001 | 34 | 31 | -3 | front_oblique, side, side_oblique |
| synth_v2_test_0003 | 5 | 6 | +1 | side_oblique |
| synth_v2_test_0004 | 32 | 32 | +0 | oblique, side |
| synth_v2_test_0006 | 40 | 48 | +8 | oblique_edited |
| synth_v2_test_0008 | 67 | 63 | -4 | side_oblique |
| synth_v2_test_0009 | 42 | 46 | +4 | oblique, side_multiple_panels, side_oblique |
| synth_v2_test_0011 | 76 | 59 | -17 | side |
| synth_v2_test_0012 | 32 | 45 | +13 | front_oblique, oblique_edited |
| synth_v2_test_0013 | 31 | 27 | -4 | mixed_views_edited, side, side_oblique |
| synth_v2_test_0015 | 74 | 28 | -46 | front, oblique, side_tight_crop |
| synth_v2_test_0019 | 35 | 35 | +0 | side, side_oblique |

Both largest misses are broadly side-on, not front-view examples:

| Source clip | Annotated | Predicted | Selected elbow | Raw selected-elbow coverage |
|---|---:|---:|---|---:|
| stu9_52.mp4 | 57 | 16 | Right | 26.79% |
| stu4_46.mp4 | 45 | 28 | Left | 66.56% |

The opposite elbows have only 1.28% and 11.82% raw coverage respectively, so
simply choosing the other arm cannot recover these cases. Low-observation
review flags are exported rather than silently excluding failed clips. Camera
view alone does not explain the error: edited/compound movements, subject
selection, missing landmarks, and temporal conventions remain confounds.

### Test the cadence prior on training data

A minimum gap is a rejection threshold, not an assertion that the person performs
five reps/second. Training annotation timings must be checked before replacing
it with a universal physiological prior. With equal source-clip weighting,
25.24% of completion gaps are below 0.7 s and 53.29% below 1.0 s. Applying those
gaps to the annotated completion timestamps alone suppresses about 13.17% and
28.50% of annotated reps. This benchmark's video timings cannot be replaced by
an assumed 1.5-3 second repetition period.

`validate_pushup_timing.py` tests five gaps (0.2/0.35/0.5/0.7/1.0 s), four requested
smoothing windows (0.3/0.4/0.5/0.6 s), three prominence fractions
(0.25/0.35/0.45), and two recovery fractions (0.5/0.7): 120 candidates on the same
155 training push-up windows from 81 source clips. For each gap, the remaining
settings are selected on four source-disjoint folds and scored on the fifth.
The selected elbow and score normalization are unchanged from the current
normalized peak family; no new coordinate signals enter this timing sweep.

| Minimum spacing | Other-fold-selected MAE |
|---|---:|
| 0.20 s | 2.857 |
| 0.35 s | 3.460 |
| 0.50 s | 4.580 |
| 0.70 s | 5.841 |
| 1.00 s | 7.918 |

The frozen normalized-family baseline (0.2 s spacing, 0.3 s smoothing, 0.45
prominence, 0.7 recovery) has source-weighted fold MAE 2.821. Selecting from the
whole new grid gives 2.857, not a reliable improvement. All five held-out choices
retain 0.2 s spacing and 0.3 s smoothing; two choose 0.5 recovery instead of 0.7,
without an overall MAE gain. No longer-spacing or smoothing change is promoted.
This is exploratory training-data development, not an unbiased nested-CV result.

### Isolated coordinate/crop pilot

`extract_pushup_diagnostic.py` was run on two large training failures, two exact
training controls, and the two largest test misses. It reads existing MP4s and
the existing MediaPipe model but writes only new sidecars under `outputs/`.
It refuses output paths outside that directory and never calls the canonical
cache writer or overwrites a feature CSV, manifest, video, model, or annotation.

Both passes use IMAGE-mode, stride-two inference and the largest visible pose
box. A fixed per-window ROI is estimated from full-frame detections, expanded
15%, clipped to the image, then resized 2x for a second pass. This differs from
the canonical annotation-assisted subject tracker. Crop results therefore do
not isolate tracking as the sole cause, and full-frame re-extraction itself is
a confound. Some ROIs cover almost the full image and are effectively an
upsampling rather than a tighter person crop.

New signals are confidence-weighted shoulder height, shoulder-to-wrist vertical
distance, and head height, normalized by a fixed median observed torso length
per pass/window. Coordinates are mapped back to original image pixels before
normalization. Fixed scaling avoids per-frame torso-denominator oscillations.
Camera motion, cuts, inferred off-screen landmarks, and head movements remain
possible artifacts. Each new signal is counted separately with the fixed
0.2 s / 0.3 s / 0.45 / 0.7 peak configuration; no signal or settings are selected
using the two test outcomes.

Selected diagnostic examples (not a new accuracy benchmark):

| Source | Existing cached count | Fresh full-frame shoulder-height count | GT |
|---|---:|---:|---:|
| stu9_52.mp4 | 16 | 50 | 57 |
| stu4_46.mp4 | 28 | 47 | 45 |

Fresh full-frame elbow extraction also gives 36 and 46 reps on the usable arms,
respectively. On stu4_46, raw elbow coverage rises from 66.56% in the cache to
99.84% in the pilot. Improvements are therefore not attributable to replacing
elbows alone. Cropping changes those elbow counts to 39 and 57: it helps one
case and worsens the other. In the first case the ROI spans the full image.
The two coordinate examples also have different errors from GT; they are not
ready to be substituted into the final pipeline.

Training pilot shoulder-to-wrist counts are 64/62, 57/65, 22/23, and 45/45
(predicted/annotated). These selected four cases only establish feasibility;
they are not source-fold validation. The next useful experiment is to extract
these sidecars consistently over training folds and evaluate a fixed quality
selection rule. Do not choose the best signal per test clip by count error.

### Label screening, not relabeling

Ten test clips were screened using unblinded temporal contact sheets at 3 or
6 frames/second. Five sampled cycle counts agreed; five remain ambiguous due
to compound movements, shallow/fast motion, camera changes, or partial endpoints.
One alternating/compound clip shows roughly nine visible bend/recovery motions
against five annotations; this may reflect a compound-cycle convention rather
than an erroneous label. No exact corrected count or label-noise floor is claimed.

This is not a completed independent/native-frame ten-video recount. Full-speed
review and an explicit repetition convention need independent adjudication before
any correction. All original labels and all headline metrics are unchanged.
The review queue and sampled evidence are preserved for that follow-up.

### Reproduce and artifacts

```powershell
.venv/Scripts/python.exe -m src.rep_counting.validate_pushup_timing --out-dir outputs/pushup_diagnosis/new_training_timing
.venv/Scripts/python.exe -m src.rep_counting.extract_pushup_diagnostic --cases outputs/pushup_diagnosis/pilot_cases_20261118.json --out-dir outputs/pushup_diagnosis/new_coordinate_pilot
```

The extraction command is a diagnostic pilot, not the guarded canonical pose
extraction workflow. Reusing its outputs for a benchmark requires a separately
versioned, consistently extracted and validated feature set.

Local outputs (ignored by Git):

- [Diagnosis scope and data-integrity verification](../../outputs/pushup_diagnosis/initial_20261118/diagnosis_summary.json)
- [All 27 GT-window signed errors and camera views](../../outputs/pushup_diagnosis/initial_20261118/test_signed_errors_and_camera_view.csv)
- [Per-video error/view aggregation](../../outputs/pushup_diagnosis/initial_20261118/test_video_signed_errors_and_views.csv)
- [Both arms' coverage, motion and counts](../../outputs/pushup_diagnosis/initial_20261118/arm_tracking_diagnostics.csv)
- [Low-observation review flags](../../outputs/pushup_diagnosis/initial_20261118/tracking_review_flags.csv)
- [Training timing sweep, fold choices, cadence and hashes](../../outputs/pushup_diagnosis/timing_train_20261118/report.json)
- [Six-case full-frame/crop pilot and limitations](../../outputs/pushup_diagnosis/coordinate_crop_pilot_20261118/report.json)
- [All pilot signals, counts, and coverage](../../outputs/pushup_diagnosis/coordinate_crop_pilot_20261118/pilot_results.csv)
- [Ten-clip sampled label screen](../../outputs/pushup_diagnosis/initial_20261118/ten_clip_label_screen.csv)

Original training/test feature and annotation hashes, and pilot video/model
hashes, were verified unchanged. No production settings or published test scores
were updated. No commits or pushes were performed.

## Consistent training coordinate sidecars (18 November 2026)

`extract_coordinate_training.py` applies one frozen full-frame protocol to all
155 annotated training push-up windows from 81 source clips across five folds.
MediaPipe IMAGE mode evaluates every second original frame, allows two poses,
and selects the largest visible-landmark bounding box independently per frame.
No crops or test frames are used. This differs from the canonical annotation-
assisted tracker and can switch people; it is an experimental feature set.

Sidecars preserve original frame/time coordinates, raw pixel landmarks and
visibility, plus shoulder height, shoulder-to-wrist height, nose height, both
elbow angles, and confidence-weighted elbows. Heights use a fixed per-window
median torso scale. Missing observations remain missing. The signed heights
are oriented so extension is high and a bend/recovery is one completed rep.

Plans freeze input/model/code hashes and every expected window. Outputs must
stay under `outputs/`; existing sidecars require explicit `--resume` and exact
provenance/checksum matches. Completion verifies canonical inputs unchanged.
The validator rejects incomplete sets, changed inputs, frame misalignment,
test cases, and source overlap across folds. Unusable signals count zero;
no difficult windows are excluded.

`coordinate_counter.py` is an opt-in offline counter. It masks confidence below
0.5, interpolates only short gaps, applies Savitzky-Golay smoothing and robust
standardization, then detects peaks without autocorrelation-guided spacing.
Candidates include six individual signals, a count-free quality selector,
fresh-elbow-only quality selection, and median voting across shoulder height,
shoulder-to-wrist height and weighted elbows. Median voting returns a real
candidate timeline and falls back to quality selection if any voter is unusable.
Quality weights are fixed: visibility 2, continuity 1, periodicity 4.

The training search crosses smoothing 0.3/0.5 s, spacing 0.2/0.35/0.5 s,
prominence 0.25/0.35/0.45 of robust amplitude, and recovery 0.5/0.7, with width
1/15 s: 36 settings per family, 324 candidates overall. Each fold is scored
using settings selected on the other four folds. MAE weights source clips
equally within each fold, then folds equally. The previous peak result,
canonical fixed elbows, fresh elbows through the identical old counter path,
and fresh elbows with the new quality-ranking path are separate baselines.
These are exploratory development checks on GT exercise windows,
not final-test results or unbiased nested CV. Whole-window normalization and
pose extraction are offline; production counter defaults remain unchanged.

```powershell
.venv/Scripts/python.exe -m src.rep_counting.extract_coordinate_training --out-dir outputs/pushup_coordinates/train_v1_20261118 --workers 2
# Only for resuming/verifying exactly the same frozen extraction:
.venv/Scripts/python.exe -m src.rep_counting.extract_coordinate_training --out-dir outputs/pushup_coordinates/train_v1_20261118 --workers 2 --resume
.venv/Scripts/python.exe -m src.rep_counting.validate_coordinate_training --sidecars outputs/pushup_coordinates/train_v1_20261118 --out-dir outputs/pushup_coordinates/train_v1_validation_20261118
```

### Completed training results

All 155 windows completed and passed alignment/provenance checks. Verified
resume found 155 valid sidecars and zero pending windows. Canonical videos,
annotations, feature caches, model and manifests passed unchanged-hash checks;
the validator also verified sidecars and reference results unchanged.

| Comparison | Source-weighted fold MAE | Timing F1 | Active videos within +/-1 |
|---|---:|---:|---:|
| Previous other-fold-selected peaks | 2.827 | 0.8910 | 29.23% |
| Canonical elbows, fixed normalized counter | 2.821 | 0.8912 | 29.23% |
| Fresh elbows, identical old counter path | 2.471 | 0.9045 | 35.38% |
| Fresh elbows, new fixed quality-ranking path | 2.477 | 0.9047 | 36.92% |
| Family + parameters selected on other folds | 2.850 | 0.9188 | 40.00% |
| Median family, parameters selected on other folds | 1.940 | 0.9382 | 44.62% |

The broad family search does **not** improve count MAE: it selects quality on
folds 0-2 and shoulder height on folds 3-4, with held-out MAEs 2.625, 2.500,
2.219, 2.938 and 3.969. Its active-video MAE also worsens from 4.662 to 6.400,
despite improved timing F1 and within-one rate. Do not replace that score with
the best family after inspecting held-out results.

Exact and within-one count comparisons (unweighted synthetic occurrences;
within-one includes exact matches):

| Method | Exact windows /155 | Windows within +/-1 /155 | Exact active videos /65 | Active videos within +/-1 /65 |
|---|---:|---:|---:|---:|
| Previous other-fold-selected peaks | 52 (33.55%) | 72 (46.45%) | 9 (13.85%) | 19 (29.23%) |
| Canonical fixed elbows | 52 (33.55%) | 72 (46.45%) | 9 (13.85%) | 19 (29.23%) |
| Fresh elbows, identical old counter | 56 (36.13%) | 81 (52.26%) | 11 (16.92%) | 23 (35.38%) |
| Fresh elbows, fixed new quality path | 55 (35.48%) | 81 (52.26%) | 11 (16.92%) | 24 (36.92%) |
| Broad other-fold selection | 76 (49.03%) | 100 (64.52%) | 14 (21.54%) | 26 (40.00%) |
| Median family, other-fold parameters | 83 (53.55%) | 100 (64.52%) | 17 (26.15%) | 29 (44.62%) |

Window scores compare each annotated push-up window independently. Video
scores first sum all push-up window counts per video; errors can cancel.
The 65-video denominator includes only videos containing annotated push-ups,
avoiding credit for the 55 exercise-absent videos. These percentages do not
use the source weighting of the fold-MAE column.

Per-family development results (parameters selected on other folds):

| Family | MAE |
|---|---:|
| Median: shoulder, wrist-relative, weighted elbow | 1.940 |
| Head height | 2.058 |
| Shoulder-to-wrist height | 2.095 |
| Count-free quality selection across six signals | 2.194 |
| Shoulder height | 2.424 |
| Fresh elbow-only quality selection | 2.633 |
| Confidence-weighted elbows | 3.363 |
| Left elbow alone | 7.796 |
| Right elbow alone | 8.615 |

Median is the most promising family in this development comparison (31.4%
lower source-weighted MAE than previous peaks). Its held-out MAEs are 1.688,
2.324, 1.531, 2.406 and 1.750; all five choose 0.5 s smoothing, 0.2 s spacing,
0.5 recovery, and prominence 0.25 or 0.35. Segment exact/within-one rates are
53.55%/64.52%; active-video MAE is 4.292 over 65 videos. Selecting this family
after seeing these scores introduces development-selection optimism; no new
test claim or production promotion follows.

The all-training search chooses quality, 0.3 s smoothing, 0.2 s spacing,
0.45 prominence, 0.5 recovery and 1/15 s width. Its **in-sample** MAE is 1.690,
not a held-out estimate. This gap is further evidence against reporting the
best all-training grid score as generalization performance.

Mean raw/cleaned coverage across window occurrences is 94.18%/98.56% for
shoulder height, 92.09%/97.56% for wrist-relative height, and 94.05%/98.48%
for head height. All pass usability guards in 155/155 windows. Individual
elbows pass in only 106/155 (left) and 102/155 (right); weighted elbows pass in
151/155. Confidence and amplitude guards still apply, and model visibility
does not prove correct person identity or tracking accuracy.

MAE is source-weighted within folds; timing pools synthetic occurrences with
the existing 0.3 s matching tolerance. Active-video percentages aggregate all
push-up GT windows per video and exclude the 55 videos without push-ups. They
are different units from segment/source MAE. The report also retains the full
120-video denominator, including exercise-absent zeros.

Artifacts (local, ignored by Git):

- [Frozen extraction plan](../../outputs/pushup_coordinates/train_v1_20261118/plan.json)
- [Completed extraction and integrity report](../../outputs/pushup_coordinates/train_v1_20261118/extraction_report.json)
- [Fold comparison, settings, hashes and environment](../../outputs/pushup_coordinates/train_v1_validation_20261118/report.json)
- [Every candidate and per-fold score](../../outputs/pushup_coordinates/train_v1_validation_20261118/candidates.csv)
- [Per-window signal coverage and quality](../../outputs/pushup_coordinates/train_v1_validation_20261118/signal_coverage.csv)
- [Other-fold-selected predictions](../../outputs/pushup_coordinates/train_v1_validation_20261118/heldout_predictions.csv)
- [Per-family held-out predictions](../../outputs/pushup_coordinates/train_v1_validation_20261118/family_heldout_predictions.csv)
- [Comparison plot](../../outputs/pushup_coordinates/train_v1_validation_20261118/comparison.png)

All 246 tests pass; the six coordinate tests also pass after final validator
updates. No test video/feature inference, canonical writes, commits or pushes
were performed by this experiment.

### Tagged freeze and second exploratory test evaluation

The subsequent user-authorized freeze is commit
`454144e0a59d9f1879f1bc493b11378e1a245ac1`, annotated tag
`coordinate-voting-v1-20261118`, committed 18 November 2026 at 10:39:48 +05:30,
before the new evaluation plan at 10:41:42 +05:30. No push was performed.
The tracked `coordinate_voting_frozen.json` fixes the median family at 0.5 s
smoothing, 0.2 s spacing, 0.35 relative prominence, 0.5 recovery and 1/15 s
minimum width (candidate 187, best all-training setting within that family).
Voters, quality fallback, extraction policy, code/model hashes, comparison
policy and bootstrap seed are recorded in that file. The evaluator checks the
tagged JSON and frozen implementation before inference; no retuning follows.

Frozen-config training sensitivity uses every occurrence of the two largest
previous-peak source-mean absolute errors, `stu9_51.mp4` and `stu9_50.mp4`.
The same two source identities are excluded from both methods:

| Training diagnostic | Previous peak source-weighted fold MAE | Frozen voting MAE |
|---|---:|---:|
| All 155 windows | 2.827 | 1.799 |
| Excluding those two source clips | 2.432 | 1.718 |

The frozen training MAEs are **in-sample**, distinct from the median-family
other-fold-parameter estimate of 1.940. Mean signed window error changes from
-0.677 to -0.452 reps. Under/exact/over window counts change from 59/52/44 to
40/80/35. Nine previous undercounts and three previous exact windows become
overcounts: reduced total overcounting does not imply no newly introduced
overcounts. The two largest baseline misses change from -19/-18 reps to
mean -3.5/-6.5 reps across their reused windows.

The second evaluation reuses the saved frozen segmentation predictions from
the first peak follow-up, with the same 0.3 s end padding. GT diagnostics have
zero padding. Only the push-up counter changes; jumping jacks and squats
retain the first follow-up's proposed frozen counters. Coordinates are freshly
extracted with the tagged full-frame protocol independently in each window.
The test set has already been inspected, including two coordinate-pilot cases;
this is explicitly **a second exploratory evaluation**, not an untouched
holdout. No settings were tuned on the new evaluation results.

Paired 95% percentile bootstrap intervals use 10,000 video-cluster draws,
seed 20261118, keeping both methods and all windows/classes within each
sampled video together. Intervals describe new-minus-old MAE; negative favors
voting. Only 21 videos are available, and source reuse may further reduce
independence. Top-two test sensitivity excludes the previously diagnosed
`stu9_52.mp4` and `stu4_46.mp4` from GT-window results only; headline scores
retain all windows. No artificial source assignment is made to predicted
segments crossing GT boundaries.

```powershell
.venv/Scripts/python.exe -m src.rep_counting.evaluate_coordinate_frozen --out-dir outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118 --workers 2
```

This command refuses existing output directories. The one completed run's
artifacts must be preserved rather than overwritten for a preferred result.

#### Completed exploratory test results

All 55 new pose windows completed once (28 predicted push-up windows and 27
GT push-up windows). All 21 registered test videos remain in evaluation.
Other exercises replay the first follow-up without modification. Canonical
test videos/features/annotations, the frozen config/model and every previous
test artifact passed unchanged-hash verification. No production defaults
changed. The local freeze commit/tag are unchanged; the subsequent evaluator,
analysis and documentation are recorded in a separate post-freeze commit.

Both test runs, with the older comparator from the first run retained:

| Counter result | JJ video MAE | Push-up video MAE | Squat video MAE | Overall video/exercise MAE |
|---|---:|---:|---:|---:|
| First run: previous unguided peaks | 2.714 | 4.381 | 2.381 | 3.159 |
| First run: proposed cached peaks | 2.905 | 4.381 | 2.095 | 3.127 |
| Second exploratory run: frozen coordinate voting for push-ups | 2.905 | 3.143 | 2.095 | 2.714 |

Each exercise video MAE uses all 21 videos, including exercise-absent zeros;
overall uses 63 video/exercise units. Paired changes below compare the second
run with the first run's **proposed** counters:

| Evaluation unit | First run MAE | Frozen run MAE | Change | Paired 95% CI for change |
|---|---:|---:|---:|---:|
| All 63 video/exercise totals, predicted segmentation | 3.127 | 2.714 | -0.413 | [-1.556, +0.429] |
| Push-up totals, 21 videos, predicted segmentation | 4.381 | 3.143 | -1.238 | [-4.667, +1.286] |
| Push-up totals, 11 active videos only | 8.364 | 6.000 | -2.364 | [-8.727, +2.455] |
| Whole-workout totals, 21 videos | 7.762 | 6.905 | -0.857 | [-4.381, +2.000] |
| Push-up GT windows, 27 windows | 3.778 | 2.481 | -1.296 | [-2.846, +0.520] |
| All exercise GT windows, 79 windows | 2.165 | 1.722 | -0.443 | [-1.047, +0.171] |

Point estimates improve, but every nonzero-change confidence interval includes
zero. This does not establish a reliable population improvement. The push-up
GT diagnostic has only 11 contributing video clusters; exercise-absent videos
do not enter its bootstrap. GT and predicted-window MAEs are different units
and must not be subtracted as an additive segmentation-error estimate.

| Push-up count comparison | First run exact | Frozen exact | First run within +/-1 | Frozen within +/-1 |
|---|---:|---:|---:|---:|
| GT windows /27 | 13 (48.15%) | 15 (55.56%) | 16 (59.26%) | 19 (70.37%) |
| All test videos /21 | 11 (52.38%) | 12 (57.14%) | 13 (61.90%) | 15 (71.43%) |
| Active push-up videos /11 | 1 (9.09%) | 2 (18.18%) | 3 (27.27%) | 5 (45.45%) |

Within-one includes exact matches. The all-video row includes ten push-up-
absent videos that both methods count correctly as zero. Across all 63
video/exercise units, exact changes 24/63 to 25/63 and within-one 37/63 to
39/63. Whole-workout exact remains 0/21; within-one changes 2/21 to 3/21.

| Push-up GT-window sensitivity | Windows | First run MAE | Frozen MAE |
|---|---:|---:|---:|
| Full diagnostic (headline unchanged) | 27 | 3.778 | 2.481 |
| Without stu9_52 and stu4_46 | 25 | 1.760 | 1.760 |

Outside these two previously diagnosed cases there is **no net MAE gain**;
the paired difference CI is [-0.846, +1.429]. Their counts change 16 to 36
(GT 57) and 28 to 47 (GT 45). These two cases explain the full 35-rep reduction
in summed absolute GT error. The known pilot exposure makes this especially
important when interpreting the exploratory improvement.

Test GT under/exact/over windows change 7/13/7 to 7/15/5; mean signed error
changes -1.778 to -1.222 reps. Two new overcounts occur. Notable regressions
are stu10_44 (48 to 20, GT 40; error +8 to -20) and val901 (7 to 14, GT 7).
In training, stu10_51 also worsens from 40 to 22 against GT 54. These failures
remain included, and no settings are changed to repair them after evaluation.
The evidence supports a promising outlier-specific improvement, not a robust
general solution or automatic production adoption.

Local artifacts:

- [Tagged frozen config](coordinate_voting_frozen.json)
- [First test run, preserved](../../outputs/peak_improvements/test_followup_20261118/report.json)
- [Second run report, intervals and both sensitivities](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/report.json)
- [Pre-inference plan and tag/commit provenance](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/evaluation_plan.json)
- [Every training window's signed errors](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/training_window_signed_errors.csv)
- [Every test push-up GT window's signed errors](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/gt_pushup_window_signed_errors.csv)
- [Video/exercise count comparisons](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/video_exercise_comparison_counts_only.csv)
- [All GT-window count comparisons](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/all_gt_window_comparison_counts_only.csv)
- [Active-video and signed-transition supplement](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/supplement.json)
- [Signed-error and confidence plots](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/signed_errors_and_confidence.png)

The count-only exports avoid stale earlier-run auxiliary timing fields retained
in raw replay comparison tables. Frozen push-up timelines are in the dedicated
window files and sidecars; no frozen timing-F1 claim is made here. The supplement
is derived without repeating pose inference or counting:

```powershell
.venv/Scripts/python.exe -m src.rep_counting.summarize_coordinate_frozen --run-dir outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118
```

All 250 tests pass, including paired-bootstrap sign/clustering, identical
source-exclusion and raw missing-observation checks. The post-freeze commit
is separate from the tagged freeze; nothing was pushed.

#### Claim and per-clip failure analysis

**Reporting claim:** In development, median voting over coordinate signals
reduced source-weighted push-up MAE from 2.83 to 1.94. Side-on clips with
missing elbow keypoints motivated this approach; the development comparison
does not establish camera view as the cause of its gain. In a second
exploratory evaluation on a previously seen test set, point estimates favored
voting (GT-window MAE 3.78 to 2.48), but confidence intervals included zero
and the net MAE gain disappeared when the two previously diagnosed clips were
excluded. Both of those test clips were coarsely tagged side-on and had
substantial missing canonical elbow observations. Their earlier pilot exposure
limits the independence of this evidence.

The **first frozen test evaluation** changed overall video/exercise MAE
3.16 to 3.13 (previous versus proposed cached peaks). The **second exploratory
test evaluation** changed 3.13 to 2.71 by replacing only push-ups with the
tagged coordinate-voting config. These are separate evaluations, not two
stages of an untouched test. Both remain in the table above; the second uses
an already-seen set and its confidence intervals do not establish a gain.

Eight largest absolute signed errors in the frozen push-up GT-window run
(ties sorted by clip ID):

| Clip | Coarse camera view | GT | Previous -> frozen count | Signed error, previous -> frozen | Missing old elbow | Missing frozen selected signal | Selected signal |
|---|---|---:|---:|---:|---:|---:|---|
| stu9_52 | Side, tight crop | 57 | 16 -> 36 | -41 -> -21 | 73.21% | 47.71% | Weighted elbows |
| stu10_44 | Oblique, edited | 40 | 48 -> 20 | +8 -> -20 | 9.00% | 5.12% | Shoulder-to-wrist height |
| val901 | Front, elevated support | 7 | 7 -> 14 | 0 -> +7 | 62.94% | 16.08% | Shoulder height |
| train2748 | Front/oblique | 5 | 9 -> 9 | +4 -> +4 | 12.88% | 9.85% | Shoulder height |
| stu4_45 | Oblique, edited | 16 | 21 -> 19 | +5 -> +3 | 0.12% | 0.12% | Weighted elbows |
| stu5_47 | Side/oblique | 30 | 25 -> 27 | -5 -> -3 | 23.78% | 13.89% | Shoulder height |
| stu8_45 | Front/oblique | 11 | 15 -> 8 | +4 -> -3 | 12.91% | 11.71% | Shoulder height |
| stu4_46 | Side | 45 | 28 -> 47 | -17 -> +2 | 33.44% | 0.00% | Shoulder height |

Camera tags come from the prior manual review of three sampled frames per
window, not calibrated geometry or a full temporal audit. stu10_44 contains
sampled cuts/pauses; stu4_45 includes talking/reset content; stu8_45 shows
compound movement. These are observations, not established error causes or
label corrections. Low missingness in stu10_44 and stu4_45 shows that failure
is not explained by absent keypoints alone.

Missing fraction is one minus the fraction of raw sampled signal values that
are finite with finite model visibility/confidence >=0.5, before interpolation
or smoothing. The old-elbow figure uses the saved canonical selected-arm raw
coverage. The frozen figure describes the actual candidate timeline selected
by median voting, not all voters; it mixes coordinate and angle channels and
is not a controlled same-signal tracking comparison. Model confidence does
not verify landmark accuracy, off-screen inference or subject identity.

The [failure table CSV](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/largest_signed_error_failures.csv)
also retains fresh left/right elbow missingness and review notes. Its
[provenance](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/largest_signed_error_failures_provenance.json)
records source checksums and definitions. It is derived without repeating
inference/counting or changing the frozen configuration:

```powershell
.venv/Scripts/python.exe -m src.rep_counting.coordinate_failure_table --run-dir outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118
```

The post-freeze evaluator, derived analyses, tests and reporting are committed
separately after the tagged freeze, preserving freeze-first/evaluation-second
history. No push is performed.
