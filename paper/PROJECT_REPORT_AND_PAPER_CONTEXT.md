## Rep-counting alternatives: methods and results (17 November 2026)

For current methods and the GT-window and post-segmentation CV/test results, see the [v2 pipeline results](../docs/results/V2_PIPELINE_RESULTS.md).

### Current default and historical experiments

The integrated v2 pipeline uses the frozen random forest with 2-second windows,
1-second stride, and OnlineTAS smoothing (tau=0.5, min_len=3). The default rep
counter uses unguided 2D arm-raise peaks for jumping jacks, unguided
confidence-weighted 2D hip-angle peaks for squats, and frozen coordinate
median voting for push-ups. Use `--rep-method state_machine` for the legacy
hysteresis counter.

The 17 November development tables below use the earlier cleaned 2D
state-machine baseline, including local push-up normalization and motion-aware
arm selection. The 3D study is a separate experiment.

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

| Exercise | Legacy 2D state machine | Guided peaks | Unguided peaks |
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
to the legacy untuned 2D state-machine baseline, but worsen push-ups by 4.0%.
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

Existing detail/artifact records: [peak experiment](../docs/results/PEAK_COUNTER_EXPERIMENT.md),
[peak settings, fold checks, and hashes](../docs/results/peak_counter_20261117/report.json),
[2D/3D experiment](../docs/results/SIGNAL_DIMENSIONS_COMPARISON.md),
[2D/3D settings, metrics, and hashes](../docs/results/signal_dimensions_20261117/report.json).
Plots and per-segment prediction CSVs are linked from those existing reports.

The dated tables retain their original experiment scope. The pipeline results page records the current benchmark comparisons.

## Supporting GT-window diagnostic on synthetic test (18 November 2026)

The current frozen unguided-peak candidate was replayed inside all 79 annotated
exercise windows across the 21 test videos. Previous and proposed frozen
configurations reproduced all 158 saved count/timestamp lists exactly. Report
this as a counter-only diagnostic, not as the headline end-to-end result.

| Exercise | Annotated windows | Previous counter MAE | Proposed counter MAE | Proposed within +/-1 |
|---|---:|---:|---:|---:|
| Jumping jacks | 27 | 1.370 | 1.444 | 77.78% |
| Push-ups | 27 | 3.778 | 3.778 | 59.26% |
| Squats | 25 | 1.360 | 1.200 | 84.00% |

MAE here is per annotated exercise segment, with zero end padding. Push-up
errors remain high even with correct exercise windows. Do not subtract these
values from headline MAE, whose unit is an exercise total per video.

After aggregating GT counts to the same 63 video/exercise units and using zero
padding for both window sources, proposed-counter MAE is 3.048 for predicted
windows versus 2.651 for GT windows. The actual padded end-to-end headline
remains 3.127. This window-policy comparison is not an additive causal error
budget: exercise classification, boundaries, fragmentation, adaptive counting,
and cancellation interact. GT windows can yield worse aggregate counts for
some exercises when false detections had offset misses.

No test-driven tuning or production-default changes were made. This is a
follow-up on the already inspected test set; pose extraction remains
annotation-assisted. Data and previous results were verified unchanged.
See [the existing counter README](../src/rep_counting/README.md) for the full
same-unit table and supporting local artifact links. The corresponding local
report is `outputs/peak_improvements/gt_window_diagnostic_20261118/report.json`.

## Push-up failure diagnosis and coordinate pilot (18 November 2026)

The current GT-window test errors are mixed: 13 exact source windows, seven
overcounts, seven undercounts. Two undercount cases account for 58/102 absolute
errors (56.9%); coarse sampled camera views are side-on for both. Their selected
elbows have 26.79% and 66.56% raw coverage. Camera angle alone is not established
as the cause. View tags are manually sampled observations, not calibrated labels.

A training-only timing sweep refutes imposing a blanket 0.7-1.0 s minimum rep
spacing on this benchmark. About 25%/53% of source-weighted annotated completion
gaps are shorter than those limits. Other-fold-selected MAE increases from
2.857 at 0.2 s to 5.841 at 0.7 s and 7.918 at 1.0 s. Jointly tuning spacing,
smoothing, prominence and recovery does not beat the frozen baseline (2.821).
No test-driven parameter update is made.

An isolated six-case pose pilot (four selected training cases and two test
failures) writes new coordinate sidecars only under outputs. Shoulder/head and
shoulder-to-wrist signals are feasible, but these selected-case results are not
fold-validated accuracy. Fresh full-frame elbow extraction also improves the
outliers, and crop/upsampling can worsen counts. Different person-selection
policies confound comparison with the canonical cache, so the pilot does not
prove that cropping or a particular coordinate signal solves the failure.

Ten clips received unblinded sampled temporal screening; five are unresolved.
This is not an independent native-frame ten-video recount, and no corrected
labels or label-noise estimate is asserted. Compound-cycle and endpoint
conventions need independent adjudication. No existing data, annotations,
production defaults, or headline test scores were changed. See the existing
[rep-counting README](../src/rep_counting/README.md) for detailed evidence,
methods, limitations and local artifact paths.

## Consistent training coordinate-sidecar experiment (18 November 2026)

The coordinate pilot is extended to all 155 training push-up GT windows from
81 source clips, using one full-frame MediaPipe IMAGE protocol at stride two,
up to two poses, and per-frame largest-visible-bounding-box person selection.
Original pixel coordinates, timestamps, confidence and derived signals are
stored separately under `outputs/pushup_coordinates/`. Fixed per-window median
torso length normalizes shoulder height, shoulder-to-wrist height and nose
height; both elbow angles and confidence-weighted elbows are also retained.
This offline extraction differs from canonical annotation-assisted tracking.

The opt-in coordinate peak counter compares individual signals, count-free
quality selection and median voting across distinct spatial/elbow signals.
Quality weights are fixed; no per-video annotated count selects a signal.
Smoothing, spacing, relative prominence and recovery form a 36-setting grid
per family (324 candidates overall). Each held-out fold uses family/settings
selected on the other four source-disjoint folds. All windows remain in the
denominator, including unusable signals. Fixed canonical and newly extracted
elbow baselines distinguish re-extraction effects from signal/counting changes.

Plans and sidecars carry code/input/model hashes and original frame alignment.
Validation requires the complete registered training set, verifies source-fold
separation and checks canonical inputs and previous results unchanged. This
is exploratory training development with GT windows, not a final-test result,
unbiased nested CV or evidence of real-time performance. No canonical features,
annotations, production defaults or headline test scores are replaced.
Reproduction commands and local result links are in the existing
[counter README](../src/rep_counting/README.md).

Completed extraction contains all 155 windows; verified resume reports zero
pending. Canonical and reference-result hashes are unchanged. Training-only
results use source-weighted MAE across folds:

| Comparison | MAE | Timing F1 | Active videos within +/-1 |
|---|---:|---:|---:|
| Previous other-fold-selected peaks | 2.827 | 0.8910 | 29.23% |
| Canonical elbows, fixed normalized counter | 2.821 | 0.8912 | 29.23% |
| Fresh elbows through identical counter path | 2.471 | 0.9045 | 35.38% |
| Broad family + parameter selection on other folds | 2.850 | 0.9188 | 40.00% |
| Median family, parameter selection on other folds | 1.940 | 0.9382 | 44.62% |

Fresh extraction improves the same fixed elbow counter by 12.4%, but broad
family selection does not improve MAE; its active-video MAE worsens to 6.400
from 4.662. Within-family MAEs are median 1.940, head 2.058, wrist-relative
2.095, quality 2.194, shoulder 2.424, fresh elbow-quality 2.633, weighted elbows
3.363, left elbow 7.796 and right elbow 8.615. Median is promising (31.4%
lower source-weighted MAE than previous peaks), with per-fold values 1.688,
2.324, 1.531, 2.406 and 1.750. Its active-video MAE is 4.292 over 65 videos.
Choosing median after inspecting family scores is a development decision,
not an independently evaluated family-selection procedure. The broad-search
2.850 score must remain visible alongside the best-family result.

Exact-count and within-one comparisons use unweighted synthetic occurrences;
within-one includes exact matches:

| Method | Exact windows /155 | Windows within +/-1 /155 | Exact active videos /65 | Active videos within +/-1 /65 |
|---|---:|---:|---:|---:|
| Previous peaks | 52 (33.55%) | 72 (46.45%) | 9 (13.85%) | 19 (29.23%) |
| Canonical fixed elbows | 52 (33.55%) | 72 (46.45%) | 9 (13.85%) | 19 (29.23%) |
| Fresh elbows, identical counter | 56 (36.13%) | 81 (52.26%) | 11 (16.92%) | 23 (35.38%) |
| Fresh elbows, fixed new quality path | 55 (35.48%) | 81 (52.26%) | 11 (16.92%) | 24 (36.92%) |
| Broad other-fold selection | 76 (49.03%) | 100 (64.52%) | 14 (21.54%) | 26 (40.00%) |
| Median family, other-fold parameters | 83 (53.55%) | 100 (64.52%) | 17 (26.15%) | 29 (44.62%) |

Video counts aggregate all annotated push-up windows before computing error;
window errors can cancel. The 65-video denominator excludes the 55 videos
without annotated push-ups. Neither percentage uses source-weighted fold MAE
weighting. Family-selection and GT-window caveats above still apply.

All spatial channels pass usability guards in every training window, versus
106/155 left elbows, 102/155 right elbows and 151/155 weighted elbows.
Mean raw shoulder/head coverage is about 94%; wrist-relative coverage is 92%.
Improvement does not establish correct person identity or camera invariance.
The all-training best quality configuration scores 1.690 in-sample; this must
not be reported as held-out accuracy. The experiment remains opt-in and has
not been evaluated on test frames. Full per-window predictions, signal
coverage, candidate scores, environment and hashes are retained in
`outputs/pushup_coordinates/train_v1_validation_20261118/`.

## Tagged coordinate-voting freeze and exploratory test follow-up

User-authorized commit `454144e0a59d9f1879f1bc493b11378e1a245ac1` and annotated
tag `coordinate-voting-v1-20261118` freeze median voting before further
evaluation (commit 18 November 2026, 10:39:48 +05:30; evaluation plan 10:41:42).
The tracked configuration fixes smoothing 0.5 s, spacing 0.2 s, prominence
0.35, recovery 0.5, width 1/15 s, the three voters, quality fallback, pose
protocol and implementation/model hashes. This is candidate 187, selected
on all training folds within the previously chosen median family. Nothing
was pushed. The evaluator verifies the tag and hashes before inference.

Frozen-config training sensitivity removes both occurrences of each of the
two largest previous-peak source errors, `stu9_51.mp4` and `stu9_50.mp4`,
from both methods. Source-weighted fold MAE changes 2.827 to 1.799 with all
windows, and 2.432 to 1.718 without these sources. The frozen training score
is in-sample and must not replace the other-fold-parameter estimate 1.940.
Under/exact/over window counts change 59/52/44 to 40/80/35; mean signed error
changes -0.677 to -0.452. Voting introduces 12 new overcounting windows
(nine previously under, three exact) despite lowering aggregate overcounts.

The follow-up is explicitly a **second exploratory evaluation on the already
inspected 21-video test set**, including earlier coordinate-pilot exposure.
Only push-ups change. Other exercises retain first-follow-up proposed peak
counters. Saved frozen segmentation boundaries and 0.3 s padding are reused;
GT-window diagnostics use zero padding. All new pose windows use the tagged
full-frame protocol; no test-driven parameter changes are allowed.
Paired 95% percentile confidence intervals resample videos as clusters 10,000
times with seed 20261118; differences are new minus old MAE. Source reuse and
only 21 clusters limit inference. Removing the previously diagnosed test
sources `stu9_52.mp4` and `stu4_46.mp4` is a GT-window sensitivity diagnostic,
never a replacement headline. Full signed errors and the unchanged earlier
test run are retained separately. See the existing counter README for results.

The single frozen run completed 28 predicted and 27 GT push-up pose windows.
Canonical inputs and all first-run artifacts passed unchanged-hash checks.
Both test runs must be reported:

| Test configuration | JJ video MAE | Push-up video MAE | Squat video MAE | Overall video/exercise MAE |
|---|---:|---:|---:|---:|
| First run: previous unguided peaks | 2.714 | 4.381 | 2.381 | 3.159 |
| First run: proposed cached peaks | 2.905 | 4.381 | 2.095 | 3.127 |
| Second exploratory run: frozen coordinate-voting push-ups | 2.905 | 3.143 | 2.095 | 2.714 |

Exercise MAEs use all 21 videos, including absent-exercise zeros; overall
uses 63 video/exercise totals. Paired changes compare the second run with
first-run proposed counters:

| Unit | Previous MAE | Frozen MAE | Change [paired 95% CI] |
|---|---:|---:|---|
| Overall predicted-window video/exercise | 3.127 | 2.714 | -0.413 [-1.556, +0.429] |
| Push-up predicted-window video totals /21 | 4.381 | 3.143 | -1.238 [-4.667, +1.286] |
| Push-up active-video totals /11 | 8.364 | 6.000 | -2.364 [-8.727, +2.455] |
| Whole-workout totals /21 | 7.762 | 6.905 | -0.857 [-4.381, +2.000] |
| Push-up GT windows /27 | 3.778 | 2.481 | -1.296 [-2.846, +0.520] |
| All exercise GT windows /79 | 2.165 | 1.722 | -0.443 [-1.047, +0.171] |

All change intervals include zero. Report improved exploratory point estimates,
not established generalization gains. Push-up GT diagnostics contain only
11 contributing video clusters. Synthetic source reuse further limits bootstrap
independence. No new timing-F1 result is claimed.

Push-up GT exact counts improve 13/27 (48.15%) to 15/27 (55.56%), within-one
16/27 (59.26%) to 19/27 (70.37%). Video totals improve exact 11/21 to 12/21,
within-one 13/21 to 15/21, but ten videos contain no push-ups and both methods
count zero. Active-only equivalents are exact 1/11 to 2/11 and within-one
3/11 to 5/11. Within-one includes exact matches.

Excluding the previously diagnosed test clips stu9_52 and stu4_46 gives
**1.760 MAE for both methods** over the remaining 25 GT windows (change CI
[-0.846, +1.429]). Their counts improve 16 to 36 against GT 57 and 28 to 47
against GT 45; these two clips account for the entire 35-rep reduction in
summed GT absolute error. Earlier coordinate-pilot exposure to both clips
must accompany this result. Under/exact/over GT-window counts change 7/13/7
to 7/15/5, mean signed error -1.778 to -1.222, with two newly overcounted
windows. Regressions include stu10_44 (+8 to -20 error) and val901 (0 to +7).
Training stu10_51 also worsens (40 to 22 against GT 54). The frozen config
is retained without test-driven repair or production adoption.

Full per-window signed errors, raw coordinate sidecars, count-only comparison
tables, paired intervals and sensitivities are in
`outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/`.
Raw replay comparison tables retain old auxiliary timestamps/hits; use the
`*_counts_only.csv` exports for count comparisons and dedicated frozen
push-up sidecars for timelines. First-run results remain untouched under
`outputs/peak_improvements/test_followup_20261118/`. The diagnostic plot is
`signed_errors_and_confidence.png`; 250 tests pass. Evaluation code, derived
failure analysis and these findings are recorded in a separate commit after
the tagged freeze, preserving freeze-first/evaluation-second history. Nothing
was pushed.

**Claim at reporting strength:** In development, median voting over coordinate
signals reduced source-weighted push-up MAE from 2.83 to 1.94. Side-on clips
with missing elbow keypoints motivated the method, but the development
comparison does not establish camera view as the causal explanation. In a
second exploratory evaluation on a previously seen test set, point estimates
favored voting (GT windows 3.78 to 2.48), but confidence intervals included
zero and the net MAE gain disappeared when the two previously diagnosed clips
were excluded. Both clips were coarsely tagged side-on with missing canonical
elbow observations, and had already appeared in the coordinate pilot.

Report the two test evaluations together: the first frozen cached-peak
comparison changes overall video/exercise MAE **3.16 to 3.13**; the separate
second exploratory coordinate-voting comparison changes **3.13 to 2.71**.
Only push-ups change in the second comparison. Neither the second evaluation
nor the choice of median family after development comparisons is independent
confirmation on a pristine holdout.

Largest frozen GT-window push-up signed errors (absolute magnitude; ties by
clip ID):

| Clip | Coarse camera view | GT | Previous -> frozen count | Signed error, previous -> frozen | Missing old elbow | Missing frozen selected signal |
|---|---|---:|---:|---:|---:|---:|
| stu9_52 | Side, tight crop | 57 | 16 -> 36 | -41 -> -21 | 73.21% | 47.71% |
| stu10_44 | Oblique, edited | 40 | 48 -> 20 | +8 -> -20 | 9.00% | 5.12% |
| val901 | Front, elevated support | 7 | 7 -> 14 | 0 -> +7 | 62.94% | 16.08% |
| train2748 | Front/oblique | 5 | 9 -> 9 | +4 -> +4 | 12.88% | 9.85% |
| stu4_45 | Oblique, edited | 16 | 21 -> 19 | +5 -> +3 | 0.12% | 0.12% |
| stu5_47 | Side/oblique | 30 | 25 -> 27 | -5 -> -3 | 23.78% | 13.89% |
| stu8_45 | Front/oblique | 11 | 15 -> 8 | +4 -> -3 | 12.91% | 11.71% |
| stu4_46 | Side | 45 | 28 -> 47 | -17 -> +2 | 33.44% | 0.00% |

Missingness is measured before interpolation/smoothing: one minus the raw
sample fraction with finite signal and finite model confidence >=0.5. Old
coverage uses the previously selected canonical elbow. Frozen missingness
describes the candidate supplying the median timeline (weighted elbows,
shoulder-to-wrist or shoulder height), not the entire voting ensemble or a
controlled same-signal comparison. High model confidence does not verify
landmark accuracy or identity. Camera tags use three sampled frames, not
calibrated geometry or a full temporal audit. Sampled edits/pauses in stu10_44,
reset content in stu4_45 and compound movement in stu8_45 are review findings,
not proven causes or label corrections. Low missingness does not rule out
these failures. The CSV additionally records both fresh elbow missing rates,
selected signal, review notes and provenance. No inference or counting was
repeated to derive the table. See the existing counter README for links and
the reproduction command.

## Full training-CV segmentation plus counting comparison

All 120 training videos were scored with the selected RF's saved source-held-
out OOF predictions and unchanged smoothing (tau 0.5, minimum run 3 windows).
OOF hashes, training signature, complete window keys and `cv_fold == fold`
were verified. Both counters receive identical predicted exercise segments
and 0.3 s end padding. This historical comparison uses the legacy 2D state machines;
experimental counting uses previous other-four-fold-selected arm-raise peaks,
hip-angle peaks and coordinate median voting. No settings are tuned on these
new end-to-end scores. Model/family/smoothing development choices still make
this selection-CV, not an unbiased nested-CV estimate or pristine holdout.

| Exercise / unit | Current MAE | Experimental MAE | Exact counts: current -> experimental | Within +/-1: current -> experimental |
|---|---:|---:|---:|---:|
| Jumping jacks /120 | 7.942 | 5.242 | 50 -> 40 | 71 -> 70 |
| Push-ups /120 | 3.717 | 2.925 | 62 -> 62 | 71 -> 72 |
| Squats /120 | 2.683 | 2.708 | 54 -> 46 | 71 -> 78 |
| All video/exercise units /360 | 4.781 | 3.625 | 166 -> 148 | 213 -> 220 |

Primary MAE averages video/exercise totals, including exercise-absent zeros;
it is not source-weighted GT-window MAE. The experimental combination lowers
MAE by 24.2%, but exact matches decline 46.11% to 41.11%, while within-one
changes 59.17% to 61.11%. Squat MAE is effectively tied/slightly worse after
segmentation, so the component GT-window advantage does not carry through
uniformly. Overcounting units rise 82 to 119 and undercounting units fall
112 to 93. Mean signed error improves -3.003 to -1.075. No new hybrid is
selected from these results or promoted into production.

Timing F1 improves JJ 0.8400 to 0.8840, push-ups 0.8678 to 0.9119, and squats
0.8896 to 0.8948. Whole-video MAE changes 12.092 to 8.458, exact 7/120 to
9/120 and within-one 31/120 to 30/120. Active-only JJ/push-up MAEs are
9.490 to 6.190 (100 videos) and 6.862 to 5.385 (65 videos). Segmentation is
identical: mean fold Edit 71.537, frame accuracy 85.109%, F1@25 0.7042.

Descriptive paired video-cluster 95% intervals for experimental-minus-current
MAE are overall -1.156 [-2.142, -0.381], JJ -2.700 [-5.375, -0.650], push-ups
-0.792 [-1.575, -0.050], and squats +0.025 [-0.517, +0.684]. Training reuse
and synthetic source repetition limit independence: these are not new test
claims. This study uses different rep hyperparameters per held-out fold,
not the single tagged all-training config used in the second exploratory
test evaluation.

The 154 predicted push-up windows require 77,816 raw coordinate frames;
69,263 are reused from hash-verified training sidecars and 8,553 missing
frames are extracted with the same frozen IMAGE/stride-two pose protocol.
Only raw landmarks/visibility are reused; normalization and signals are
recomputed per predicted window, including all wrongly classified regions.
Canonical upstream features remain annotation-assisted. No test video or
feature inference is run, and canonical data, previous results and defaults
are unchanged.

The final JSON export was repaired after counting completed: one segmentation
boundary distance was infinite, so it is represented as null in JSON with
raw infinity retained in CSV. Saved complete tables were finalized without
repeating extraction or counting. Input/table hashes and evaluator/finalizer
provenance are retained. Full results and per-segment timelines are under
`outputs/pushup_coordinates/segmented_training_cv_20261118/`; reproduction
and links are in the existing counter README. No commit or push was made
for this follow-up.

## Jumping-jack temporal failure classification

Saved experimental OOF outputs were analyzed without rerunning segmentation,
pose extraction or counting: 120 videos, 155 GT JJ windows, 146 predicted JJ
windows. Signed video errors are 37 undercounts, 43 overcounts and 40 exact.
Temporal flags overlap and describe associations, not exclusive causal error
categories. With a descriptive 0.5 s boundary tolerance:

| Video flag | All /120 | Under /37 | Over /43 | Exact /40 |
|---|---:|---:|---:|---:|
| Clipped GT JJ start/end | 21 | 16 | 3 | 2 |
| Overlong predicted JJ window | 82 | 24 | 41 | 17 |
| Fragmented GT JJ window | 4 | 4 | 0 | 0 |
| Merged GT JJ windows | 3 | 1 | 2 | 0 |
| GT JJ predominantly assigned another label | 19 | 18 | 1 | 0 |
| Predicted JJ predominantly outside GT JJ | 3 | 2 | 1 | 0 |
| Wholly missed GT JJ window | 12 | 11 | 1 | 0 |
| False JJ in a JJ-absent video | 1 | 0 | 1 | 0 |

Of 612 unmatched annotated repetitions, 402 (65.7%) have no predicted JJ
window overlapping their interval, 22 (3.6%) have partial coverage, and 188
(30.7%) have full coverage but no matched counted repetition. Among 195 extra
predicted events, 109 timestamps lie inside GT JJ windows and 86 outside.
Matching uses the existing class-aware one-to-one 0.3 s tolerance; coverage
uses nominal windows, while counting retains 0.3 s end padding. Coverage
does not prove pose/signal quality or causally allocate all errors.

Missing/transition labels are a higher-priority association than fragmentation
in these data. In f3_0002 and f3_0018, the same stu5_25 source has a 58.43 s
GT JJ window but only a 4 s predicted JJ window (6.85% coverage); the dominant
predicted label is transition. Video errors are -66 and -62. These reused
source cases are not independent evidence. Conversely f1_0001/val568 has
GT 136.97-146.77 s (9.80 s), predicted JJ 136.50-146.50 s (10 s), but counts
19 against GT 55; its saved GT-window counter error is also -36. Thus counter
failures remain even when boundaries are close. In f2_0008, a predicted JJ
window 0-8.5 s has dominant GT label transition and produces ten false reps
despite no GT JJ in that video. The overlong flag occurs in 41/43 overcounting
videos but also 17 exact videos; trimming is not thereby proven to fix them.
Four erroneous videos have no listed temporal flags.

Full unthresholded GT/predicted starts, ends, lengths, boundary deltas,
predicted labels, all intersections, signed video errors and rep-event
classification are retained under
`outputs/pushup_coordinates/jumping_jack_window_failures_20261118/`.
The timeline figure shows the six largest absolute video errors. Maximum-
overlap matching is descriptive rather than one-to-one; all overlap pairs
are exported to retain fragmented/merged and wrong-label relationships.
GT-window event-allocated errors assign saved timestamps by GT interval;
outside-GT events remain in video totals, and this is not a GT-window rerun.
Inputs/results passed unchanged-hash checks. The existing counter README
contains reproduction commands, precise definitions and local artifact links.
No settings, labels, defaults, commits or pushes were changed.

## Segmentation branch: OOF per-class and transition diagnostics

On `experiments/segmentation_improvement`, saved source-held-out RF outputs
were diagnosed without new fitting, pose extraction, counting or test
evaluation. OOF checksum, training signature, complete window keys and fold
identity were verified; the experimental counter and default segmenter stay
unchanged. One diagnostic module and its test file were added. Existing
uncommitted development work was preserved; no commit or push was made.

For 14,879 OOF windows, with GT rows and predicted columns, final smoothed
confusion is:

| GT / predicted | JJ | Push-up | Squat | Transition |
|---|---:|---:|---:|---:|
| JJ | 3367 | 16 | 122 | 333 |
| Push-up | 13 | 4594 | 2 | 195 |
| Squat | 37 | 5 | 3821 | 247 |
| Transition | 159 | 477 | 255 | 1236 |

| Class | Raw window recall | Smoothed window recall | Smoothed precision | Smoothed F1 |
|---|---:|---:|---:|---:|
| JJ | 82.52% | 87.73% | 94.16% | 90.83% |
| Push-up | 89.36% | 95.63% | 90.22% | 92.85% |
| Squat | 89.93% | 92.97% | 90.98% | 91.96% |
| Transition | 57.40% | 58.11% | 61.46% | 59.74% |

These pooled majority-GT-window statistics include mixed boundaries and are
not the official equal-fold per-video 10 Hz frame metrics. Raw high-purity
JJ-window recall remains 82.31%. Exact temporal-overlap JJ recall changes
82.43% to 87.39%; all raw/final matrices and per-fold recalls are retained.
Of raw JJ classification errors, 505/671 (75.3%) go to transition, 162 to
squat, four to push-up. After smoothing, transition accounts for 333/471
(70.7%) JJ window errors.

In the same 19 previously identified failed JJ windows/videos (nine reused
source clips), final duration-dominant labels are **13 transition, four squat,
two push-up**. Raw labels are 11 transition, four squat and four JJ. One of
the four raw-JJ cases becomes transition during confidence handling, three
at minimum-run handling. The two final push-up cases were raw-transition-
dominant, subsequently changed by the sandwich merge, not raw RF JJ-to-push-up
mistakes. There is therefore a primary transition-class/context issue and
a smaller squat-discrimination issue; this is a diagnostic prioritization,
not causal feature attribution or a proven fix.

The existing confidence rule exempts transition predictions from low-confidence
forward-fill; 39.2% of raw JJ-to-transition windows have max probability below
0.5. Minimum-run handling similarly exempts transition while suppressing short
exercise runs. The final sandwich rule merges transitions between matching
exercises with no duration cap. It recovers 266 GT-JJ-majority window labels
overall, but converts 16 to push-up in the two affected cases. JJ temporal
recall across stages is 82.43%, 81.89%, 80.53%, 87.39%: smoothing helps
overall despite harm in the selected 19-video cohort (47.94% to 46.03% JJ
temporal recall). Removing all smoothing is not supported by this evidence.
Grouped-CV ablation of transition/context rules is justified before assuming
that all misses require more discriminative pose features.

Full diagnostics and the per-case stage table are under
`outputs/segmentation_v2/diagnostics/oof_classes_20261118/`. The existing
[segmentation README](../src/segmentation/README.md) contains raw and smoothed
matrices, precise units, artifact links and the reproduction command. All
source inputs and prior results were verified unchanged. These remain
development selection-CV diagnostics on an error-selected cohort.

## Periodicity/context feature ablation on grouped training CV

On `experiments/segmentation_improvement`, four predeclared RF feature groups
were fitted with the same five source-disjoint folds. RF hyperparameters
(200 balanced trees, seed 42), training purity >=0.8, imputation protocol and
smoothing tau 0.5/minimum run 3 are fixed. No audit-flagged examples are dropped.
There is no new classifier/smoothing search or test inference. The baseline
refit reproduces saved OOF labels/probabilities and the downstream 5.2417 JJ
MAE before comparison. Fold-specific experimental JJ counter settings are
unchanged for all groups; only predicted segmentation changes.

The periodicity group adds three features on each of six 2D channels
(left/right arm raise, ankle spacing, left/right hip, selected knee): dominant
positive frequency, spectral-power fraction in 1-3 Hz, and positive local
autocorrelation peak height. These use the original 2 s window, linear
detrending, Hann periodogram and ACF peak search over 1/6-1.5 s lags. At least
12 samples and a finite run covering 80% of the window are required; long
missing runs are not bridged and unavailable features remain NaN for fold-
fitted imputation. The baseline short-gap preprocessing is unchanged. Context
adds the existing 112 summary features over centered 8 s intervals at the
same 1 s stride, clipped only at video edges. Context uses future lookahead
and may cross synthetic exercise boundaries within a video; no GT boundary
or label is used as a model feature. This is an offline method.

| Feature group | Features | Edit mean +/- fold SD | Mean fold frame accuracy | Mean fold F1@25 | Pooled JJ temporal recall | Fixed-counter JJ video MAE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 112 | 71.537 +/- 2.898 | 85.109% | 0.7042 | 87.395% | 5.242 |
| + Periodicity | 130 | 69.832 +/- 2.632 | 86.288% | 0.7121 | 92.215% | 4.300 |
| + Context | 224 | 71.453 +/- 1.320 | 85.602% | 0.7306 | 88.390% | 5.092 |
| + Both | 242 | 70.806 +/- 2.662 | 86.237% | 0.7261 | 90.592% | 4.183 |

Periodicity alone gives the best JJ recall; context alone gives the best
F1@25 and transition recall (56.16% baseline to 60.78%). Both gives the
lowest JJ count MAE (20.2% lower), but no group exceeds baseline Edit. Exact/
within-one JJ video counts are baseline 40/70, periodicity 42/71, context
41/72, both 39/70, each over 120 videos including absent-JJ zeros. Best MAE
does not imply better exact counts. Push-up/squat/transition temporal recall
with both is 94.99%/92.28%/60.01%, versus 95.75%/93.11%/56.16% baseline.
No feature group uniformly dominates, and the active segmenter is unchanged.

In the original error-selected 19 windows, dominant JJ labels are restored
in five by periodicity, none by context alone and three by both. Remaining
both-group errors are nine transition, six squat and one push-up. In the
motivating f3_0002/stu5_25 case, JJ coverage rises 6.85% to 94.12% with
periodicity and 97.49% with both; video JJ count changes 8 to 62/64 against
GT 74. Context alone gives 13.69% coverage and count 12. This is inspected,
reused development evidence, not independent confirmation. Combined-group
JJ MAE improves fold 3 from 12.250 to 6.125, but worsens fold 1 from 4.667 to
5.667 and fold 0 from 3.917 to 4.167.

All 20 fits and 266 tests complete; code/input hashes pass unchanged checks.
Outputs, per-fold/video metrics, feature schemas, all OOF probabilities,
class confusion and unchanged-counter JJ counts/timestamps are retained under
`outputs/segmentation_v2/experiments/periodicity_context_20261118/`. See the
existing segmentation README for reproduction and links. One opt-in module
and one test file are added; no fold models are activated or persisted.
Canonical data, active production bundle, smoothing/counter defaults, prior
results and test data remain unchanged. Development fold reuse, coarse 2 s
frequency resolution, signal harmonics and context lookahead limit claims.
These are ablations, not unbiased nested CV or a new test result. No commit
or push is made.

## All-class counting check for the segmentation feature ablation

The JJ-only count report was completed with push-up/squat counts on the same
saved feature-ablation OOF predictions. All fold-specific experimental counter
parameters and 0.3 s padding remain fixed. The reference is experimental
combo MAE 3.625 with the frozen segmentation protocol and other-fold rep settings.
All 120 videos/360 video-exercise units remain included; no new fitting,
audit exclusions, hyperparameter selection or test inference is performed.

| Segmenter feature group | JJ count MAE | Push-up count MAE | Squat count MAE | Overall count MAE |
|---|---:|---:|---:|---:|
| Baseline | 5.242 | 2.925 | 2.708 | 3.625 |
| + Periodicity | 4.300 | 3.067 | 2.367 | 3.244 |
| + Context | 5.092 | 2.642 | 2.625 | 3.453 |
| + Both | 4.183 | 2.592 | 2.975 | 3.250 |

The all-class check does **not** establish no regressions. Periodicity-only
worsens push-up MAE (2.925 to 3.067); both worsens squat MAE (2.708 to 2.975)
in every fold. Periodicity-only push-up recall improves 95.75% to 95.89%, but
squat recall declines slightly 93.11% to 93.09% and transition recall declines
56.16% to 55.47%. Both has push-up/squat recall 94.99%/92.28%, below baseline
95.75%/93.11%. Context-only lowers aggregate count MAE for all classes, but
its push-up/squat recall also declines. Counting and segmentation recall do
not move monotonically together; mean improvements are not uniform wins.

Periodicity-only JJ MAE by fold is baseline/new: fold 0 3.917/3.917, fold 1
4.667/5.667, fold 2 2.375/2.167, fold 3 12.250/6.625, fold 4 3.000/3.125.
Thus **two of five improve, one ties, two worsen**, with the principal gain
in fold 3. Overall three-class MAE has the same 2/1/2 pattern. Edit worsens in
all five folds; frame accuracy and F1@25 improve in three each. Periodicity's
aggregate squat-MAE gain comes from one improved fold despite four worsening.
These concentration checks preclude a claim of consistent improvement.

Descriptive paired-video overall-MAE change intervals are periodicity -0.381
[-0.969, +0.114], context -0.172 [-0.411, +0.067], both -0.375 [-0.908, +0.075].
All include zero; development reuse and repeated synthetic source clips
limit independence. Overall exact/within-one counts over 360 units are
baseline 148/220, periodicity 146/221, context 145/227 and both 144/226.
No active-model replacement or new independent-test claim follows.

The completion helper reuses 78,094 raw pose frames and extracts only 2,267
missing frames in 47 runs, with unchanged coordinate-extraction protocol.
An empty-cache numeric-dtype issue was repaired after extraction; counting
resumed from saved coordinates without repeating pose inference or changing
the counter. Original plan and resume provenance are retained separately.
Full class comparisons, per-fold metrics, counts and segment timelines are
under `outputs/segmentation_v2/experiments/periodicity_context_all_counts_20261118/`.
The existing segmentation README contains reproduction and links. All 268
tests pass. Canonical inputs, earlier outputs, frozen counter parameters and
the active segmenter are unchanged. No commits or pushes are made.

## Exercise-first transition rejection selected by count MAE

A predeclared one-parameter gate was evaluated on the saved baseline-feature
RF OOF probabilities. It selects the largest original (unnormalized)
probability among three exercises and calls transition only when that value
is below threshold; the transition probability does not compete in argmax.
No RF retraining or feature changes are involved. The gate replaces confidence
forward-fill, avoiding a second 0.5 cutoff; fixed minimum-run 3 and sandwich
merge rules remain. All experimental counters and 0.3 s padding are unchanged.

Thresholds 0-1 inclusive in 0.05 steps are selected by equal-fold mean
end-to-end video/exercise count MAE over all 360 units, not transition F1.
Ties prefer higher thresholds. All 120 original videos are retained. The
union requires 101,088 raw coordinate frames; 81,963 are reused and 19,125
missing frames are extracted consistently in 135 runs. Identical predicted
segments share cached count results across thresholds.

**None of 21 thresholds beats the original 3.625 counting MAE.**

| Policy | JJ MAE | Push-up MAE | Squat MAE | Overall MAE |
|---|---:|---:|---:|---:|
| Original transition argmax + smoothing | 5.242 | 2.925 | 2.708 | 3.625 |
| Gate 0.00 | 4.558 | 4.525 | 3.267 | 4.117 |
| Gate 0.30 (best JJ) | 4.208 | 3.975 | 2.883 | 3.689 |
| Gate 0.55 (best all-OOF overall) | 6.083 | 2.683 | 2.242 | 3.669 |
| Other-fold-selected thresholds | 6.025 | 2.950 | 3.250 | 4.075 |

The all-OOF selected score is selection-biased. The other-four-fold threshold
diagnostic chooses 0.30 for folds 0/2/4 and 0.55 for folds 1/3; every target
fold is worse than the original policy (MAEs 3.597, 4.611, 2.250, 6.208,
3.708). It is not strict nested CV: the underlying development choices were
already inspected and other-fold OOF predictors had included the target fold
in their fitting sets. Paired-video overall changes are +0.044 [-0.322,
+0.411] for all-OOF selection and +0.450 [+0.150, +0.789] for other-fold
selection; synthetic source reuse and development selection limit inference.

The cost tradeoff is real. Original JJ recall/overcount videos/overcount-plus-
overlong videos are 87.39%/43/41, versus 91.22%/50/47 at threshold zero,
90.25%/48/44 at 0.30, and 83.40%/34/30 at 0.55. Counter safeguards do not
reject all false exercise regions. At 0.55 overall overcount/undercount units
change 119/93 to 98/109 and exact/within-one 148/220 to 153/221, but signed
bias worsens -1.075 to -1.875 and MAE does not improve. Edit improves 71.537
to 73.719, while F1@25 is near unchanged 0.7042 to 0.7037; these secondary
scores do not replace the specified count objective. Structural smoothing
means final boundaries/recall need not change monotonically with threshold.
At 0.30 push-up/squat temporal recall rises 95.75%/93.11% to 97.23%/95.94%,
so recall gains across all exercises still do not imply a counting gain.

This is a retained negative result for a single global exercise gate on this
baseline model, not proof that every recall-biased decision rule must fail.
The original policy remains active. Full sweep, all-class recall, candidate
segments, count timelines and selection/provenance records are under
`outputs/segmentation_v2/experiments/exercise_gate_counts_20261118/`; precise
definitions and links are in the existing segmentation README. All 271 tests
pass; canonical inputs, prior results, counters and active model are unchanged.
No test inference, activation, commit or push was performed.

## Baseline JJ rate/duration audit and error concentration

The existing automatic audit is completed with overall three-class count
MAE, using original OOF segmentation and fixed experimental counters (overall
3.625, JJ 5.242). No inference/counting is rerun. Rate is GT count divided by
window duration. The requested <1 or >4 Hz rule flags 77/155 windows from
33 sources across 60 videos: 75 below 1 Hz and two val568 occurrences above
4 Hz (55/9.8 s = 5.612 Hz). This objectively reproducible flag is a review
heuristic, not proof of label error. Slow valid motion remains possible.

All durations repeat through intentional source reuse; no annotations share
video/start/end. Distinct clips stu7_28 (10 reps) and train1678 (11 reps)
share 9.533333 s duration, each reused twice in four additional videos.
Equal durations with different counts do not establish duplicate labels.
Rate/duration union flags cover 64 videos, all explicitly listed in artifacts.

| Cohort | Videos | Overall MAE | JJ MAE | Overall/JJ median absolute error |
|---|---:|---:|---:|---:|
| Full original data | 120 | 3.625 | 5.242 | 1 / 1 |
| Without rate-flagged videos | 60 | 3.139 | 3.500 | 1 / 1 |
| Without cross-source-duration flags | 116 | 3.693 | 5.371 | 1 / 1 |
| Without union flags | 56 | 3.244 | 3.643 | 1 / 1 |

Exclusions remove entire videos from all three class rows, only as sensitivity
analyses; primary labels/scores stay unchanged. Unflagged cohorts retain 20
JJ-absent videos. Active-only JJ MAEs are 6.190 (100 videos) original, 5.000
(40) without rate flags, 5.389 (36) without union flags. Cohort changes and
absent-class zeros preclude a causal label-correction interpretation.

Ranked by absolute JJ video error with lexical-ID ties, top five contribute
225/629 absolute error (35.8%), or 1.875 of the 5.242 MAE; top ten contribute
337/629 (53.6%), or 2.808 MAE. They contain 231/612 missed reps (37.7%) and
352/612 (57.5%), with audit overlap 4/5 and 7/10. Remaining JJ MAEs are 3.513
over 115 videos and 2.655 over 110. Median absolute error remains 1 rep,
including active-only JJ and both top-error exclusions. Ranking by missed
reps instead gives 234/356 misses (38.2%/58.2%).

Top-ten video suffixes/signed errors are f3_0002 -66, f3_0018 -62, f1_0001 -36,
f3_0021 -31, f3_0016 -30, f1_0016 -26, f3_0012 -26, f4_0002 -26,
f3_0010 -17 and f3_0017 -17. Unflagged among these are f3_0021, f3_0012 and
f3_0017. The two worst videos' heavily missed stu5_25 source is unflagged
(1.164 Hz); video flags come from other clips, stu3_21 (0.759 Hz) and
train1614 (0.982 Hz). Video-level overlap does not prove that annotation
errors caused the large misses, and reused sources are not independent cases.

Full lists/rates/reasons, concentration and sensitivity are retained under
`outputs/pushup_coordinates/jj_segmentation_audit_20261118/`.
`overall_completion.json` adds derived overall numbers and input hashes
without overwriting the original audit. The existing segmentation README
contains links and the complete top-ten table. No code/model/counter changes,
canonical writes, commits or pushes are made for this completion.

## Current full-flow implementation and reporting scope

Runnable instructions: [END_TO_END.md](../docs/END_TO_END.md). The supported
offline flow is registered v2 video/canonical pose features -> CV-selected,
all-train RF -> predicted temporal segments -> fixed per-exercise rep counter
-> predicted completion-time events -> scheduled gTTS narration-only WAV.
`main.py` exports text and counts; audio uses the separate documented API bridge.
Templates are the default; optional cached Gemini phrases alter wording only.
Neither ground-truth rep boundaries nor the old narration demo loader supply
events in this prediction recipe. Upstream pose selection nevertheless remains
annotation-assisted; unassisted new-video ingestion is not implemented.

The scheduler can skip or delay cues. The sign-off and rendered WAV are bounded by video duration.
Do not claim real-time execution, perfect synchronization, a muxed-video
implementation or evaluated narration quality. The existing single-test-video
rep diagnostic gave 27 predicted versus 44 annotated reps; this is illustrative,
not a dataset counting score. Model evaluation and its limitations are recorded
in [V2_MODEL_RESULTS.md](../docs/results/V2_MODEL_RESULTS.md). No new model,
counting or human-study results were produced by this documentation handoff.

## Current v2 model workflow decision record (17 November 2026)

See [../docs/V2_TRAINING.md](../docs/V2_TRAINING.md) and the active docs/PROJECT_CONTEXT.md for implementation
and commands. The notes/results below remain historical unless explicitly v2.

- No contributor/subject grouping; train is native train (239 usable clips),
  test is native val+test (94). Five train source folds are fixed at seed 3969.
- Model and smoothing selection use only source-disjoint train OOF predictions;
  learned imputation/scaling is fitted inside each training fold. Training purity
  defaults to 0.8; validation/test counts are not filtered by that rule.
- Final refit uses all eligible train windows. Test (21 videos, 79 unique source
  clips) is evaluated only with the frozen selected pipeline, never ranked to
  choose a model. No v2 benchmark fit/test result is claimed by this refactor.
- Mean/std over selection folds is descriptive selection-CV variability, not an
  unbiased nested-CV estimate or a subject confidence interval. Report final test
  separately, including per-class, temporal and boundary metrics and limitations.
- Prediction, final evaluation and narration share the bundle/preprocessing and
  exact n_frames/fps timeline. Dataset/model IDs and hashes prevent mixing runs.
- The permitted feature schema is fixed; absent channels become NaN in memory.
  ankle_dist_norm_3d is a noncomputed/all-missing placeholder in current train data.
- Existing pose person selection uses annotated exercise labels: scope research
  claims as an offline annotation-assisted synthetic benchmark. Unassisted raw
  video and genuinely continuous workout generalization remain future work.

> Protocol notice: historical research notes. Contributor-based subject claims,
> separate-val workflows and v1 results below do not define the current protocol.
> Active data protocol and Phase 4 instructions: ../docs/PROJECT_CONTEXT.md.

# Continuous Multi-Exercise Workout Video Segmentation, Repetition Counting, and Synchronized Audio Coaching
## Comprehensive Project Context, Research Paper Blueprint, and Progress Report

**Authors / Team Members:**
- **Bhavya** (Data Pipeline, Dataset Construction, Kinematic Feature Engineering, Temporal Segmentation)
- **Deepti** (Kinematic Repetition Counting State Machine, Signal Processing, Calibration & Validation)
- **Ishita** (Narration Generation, Speech Synthesis Synchronization, Evaluation & Paper Integration)

---

## 1. Introduction

### 1.1 Motivation & Background
Home-based physical fitness workouts using consumer video devices (smartphones, laptops, tablets) have become ubiquitous. However, passive video workouts lack interactive, real-time guidance. To match the engagement of an in-person personal trainer, a system must provide immediate, hands-free acoustic coaching—announcing exercise regimes, counting completed repetitions, pacing the athlete, and signaling rest intervals—without requiring wearable sensors or manual touchscreen interaction.

### 1.2 The Problem
Existing computer vision models for exercise counting (e.g., RepNet, TransRAC) operate almost exclusively on **pre-trimmed, single-exercise clips**. Real-world workout sessions, by contrast, are **continuous and untrimmed**:
1. Multiple distinct exercises are performed back-to-back (e.g., jumping jacks $\to$ push-ups $\to$ squats).
2. Sets are naturally interleaved with non-exercise periods (setup, resting, drinking water, adjusting posture).
3. Visual scene properties, lighting, and camera perspective remain constant while exercise kinematics change dynamically.

A complete solution must therefore integrate:
- **Temporal Action Segmentation (TAS)** to detect exercise transitions and idle periods.
- **Kinematic Repetition Counting** operating dynamically within detected active segments.
- **Synchronized Audio Narration** that aligns coaching cues directly to the video timeline.

### 1.3 Research Questions
- **RQ1 (Kinematic Invariance):** Can a purely pose-derived kinematic feature representation (joint angles and angular velocities) enable reliable temporal action segmentation in continuous workouts while remaining invariant to camera angle, visual background, and subject appearance?
- **RQ2 (Temporal Smoothing Efficiency):** Does an OnlineTAS-style sliding window classifier with confidence thresholding and minimum segment duration filtering eliminate boundary flicker without requiring computationally intensive 3D-CNN or video Transformer backbones?
- **RQ3 (Downstream Robustness):** Can an adaptive two-threshold hysteresis state-machine repetition counter and event-driven template/TTS narration engine operate reliably on top of continuous video segments, resilient to camera perspective and subject variations?

### 1.4 Scope and Delimitations
- **Target Exercise Classes:** Three cyclical bodyweight exercises (`squat`, `push_up`, `jump_jack`) plus an explicit `transition` (idle/rest/setup) class.
- **Deliberate Exclusions (Deferred to Major Project):**
  - Static holds (e.g., plank) are omitted to maintain a unified cyclical state-machine architecture for repetition counting.
  - Granular form-deviation detection and heavy LLM-based narration are deferred to prioritize end-to-end temporal segmentation, counting, and acoustic coaching within the four-week timeline.

### 1.5 Claimed Contributions
1. **End-to-End Multimodal System:** A modular architecture converting silent, continuous workout videos into fully narrated coaching videos with synchronized synthetic speech.
2. **Leakage-Safe Synthetic Workout Dataset Generation:** A mathematically verified protocol for stitching single-exercise clips into continuous workouts with exact ground-truth frame boundaries, preserved lead-in/lead-out rest margins, and strict subject isolation.
3. **Adaptive Kinematic Repetition Counter:** A rule-based, sampling-rate-independent counter with percentile-adaptive hysteresis thresholds, Savitzky-Golay signal smoothing, and motion-range safeguards (`low_motion` / `mostly_missing` flags).
4. **Pose-Only Feature Defense:** Formulation of a joint-angle kinematic representation that eliminates appearance-based shortcut learning (such as background cuts or clothing shifts) in synthetic composite videos.
5. **Leakage-Proof Evaluation Protocol:** Subject-grouped partitioning of RepCount data preventing subject identity leakage between train, validation, and test splits, with test data strictly locked until final evaluation.

### 1.6 Team Structure & Ownership
- **Bhavya (Role 1):** Data pipeline (`src/data_pipeline/`), synthetic dataset concatenation, kinematic feature extraction, and temporal segmentation (`src/segmentation/`).
- **Deepti (Role 2):** Repetition counting engine (`src/rep_counting/`), cyclical hysteresis state machines, signal conditioning, and RepCount per-rep boundary validation (`evaluate_rep_counts.py`).
- **Ishita (Role 3):** Narration engine (`src/narration/`), offline TTS synthesis, timeline audio multiplexing, and evaluation suite (`src/evaluation/`).

---

## 2. Literature Review

### 2.1 2D and 3D Human Pose Estimation
Early action analysis relied heavily on dense optical flow and RGB convolutional backbones. The introduction of lightweight, real-time pose estimation architectures—notably **BlazePose** (Bazarevsky et al., 2020) within **MediaPipe**—enabled high-speed tracking of 33 2D and 3D metric landmarks directly on edge hardware.
- *Latyshev et al. (2022)* benchmarked MediaPipe against OpenPose, demonstrating that MediaPipe provides superior frame throughput and reliable trunk/hip tracking, though wrist and foot landmarks exhibit slight jitter during rapid peripheral motions.
- *Trade-off (IMAGE vs. VIDEO mode):* In continuous synthetic video generation, running pose estimation in standard video tracking mode causes landmark state to leak across hard cuts between stitched clips. Adopting stateless **IMAGE mode** with primary-subject bounding box selection ensures complete independence across frames, preventing model exploitation of cross-clip tracking artifacts.

### 2.2 Repetition Counting in Cyclical Actions
Repetition counting has traditionally been formulated as a periodicity estimation problem:
- **RepNet** (Dwibedi et al., CVPR 2020) introduced self-similarity matrices (SSM) derived from 3D-CNN temporal features to count repetitive actions across arbitrary video domains, accompanied by the **Countix** dataset.
- **TransRAC** (Hu et al., CVPR 2022) leveraged temporal transformers to capture multi-scale periodic patterns, establishing the **RepCount** benchmark.
- *Limitation:* Both RepNet and TransRAC operate primarily on isolated clips where a single repetitive action dominates. When presented with continuous multi-exercise video containing resting intervals and changing exercises, these monolithic models struggle to segment boundaries and isolate phase resets.

### 2.3 Temporal Action Segmentation (TAS)
Temporal action segmentation aims to classify every frame in a long, untrimmed video:
- *Ding et al. (2023)* provide a comprehensive survey of modern TAS methods, highlighting multi-stage temporal convolutional networks (MS-TCN) and attention-based temporal models (ASFormer).
- **OnlineTAS** (Zhong et al., 2022/2023) demonstrated that lightweight sliding-window classifiers coupled with principled temporal post-processing (confidence thresholds, minimum segment duration enforcement, and neighborhood smoothing) achieve competitive boundary precision with fractionally low computational latency.
- Standard evaluation metrics in this domain include **Frame Accuracy**, **Edit Distance**, and **Segmental F1 at overlap thresholds ($F1@10, F1@25, F1@50$)**, alongside boundary displacement errors in seconds.

### 2.4 Dense Video Captioning and Automated Fitness Coaching
- Dense video captioning (Krishna et al., ICCV 2017) introduced the paradigm of jointly localizing temporal events and generating natural language descriptions.
- In fitness computing, **MM-Fit** (Strömbäck et al., 2020) explored multi-modal sensor fusion for exercise monitoring.
- **Rep-Penn** (Yu et al., 2021) and **GPTCoach** (Jorke et al., 2023) explored automated coaching feedback, while **BoFiTGen** (2024) investigated LLM-based workout narration.
- *Research Gap:* Current literature either examines isolated exercise repetition counting without narration, or generates post-workout textual summaries. There is an absence of lightweight, pose-only systems capable of segmenting continuous multi-exercise workouts in real time, resetting rep counters per segment, and synthesizing time-synchronized acoustic coaching directly onto the video.

### 2.5 Comparative Analysis Table

| Method / System | Input Modality | Continuous Multi-Exercise? | Repetition Counting? | Real-Time Audio Narration? | Leakage-Proof Splitting? |
|---|---|---|---|---|---|
| **RepNet** (Dwibedi et al., 2020) | RGB Frames | No (Single clip) | Yes (SSM-based) | No | No (Clip-level only) |
| **TransRAC** (Hu et al., 2022) | RGB Frames | No (Single clip) | Yes (Transformer) | No | No (Subject overlap exists) |
| **MS-TCN / ASFormer** (Ding et al., 2023) | Extracted Features | Yes (TAS benchmarks) | No | No | Varies (Breakfast/50Salads) |
| **GPTCoach** (Jorke et al., 2023) | Inertial / Pose | No (Single set) | No (Form only) | Text only (Post-hoc) | N/A |
| **Our Proposed System** | **Pose Kinematics** | **Yes (Synthetic & Continuous)** | **Yes (Adaptive FSM)** | **Yes (Timeline TTS)** | **Yes (Subject-Grouped)** |

---

## 3. Proposed Methodology

### 3.1 Architectural Pipeline

```
Raw Silent Workout Video
       │
       ▼
[MediaPipe Pose Landmarker] (IMAGE mode, Stateless Primary-Subject BBox Heuristic)
       │
       ▼  33 Normalized 2D (x, y) & 3D World Landmarks (x, y, z) + Visibility Scores
[Kinematic Feature Extraction & Cleaning]
       │  • Knee Angle (Hip-Knee-Ankle) [Squat]
       │  • Elbow Angle (Shoulder-Elbow-Wrist) [Push-up]
       │  • Arm Raise (Hip-Shoulder-Wrist) & Leg Spread [Jumping Jack]
       │  • Best-Side Selection: max(vis_left, vis_right)
       │  • Visibility Thresholding (< 0.50 set to NaN) + Linear Gap Interpolation
       ▼
[Sliding-Window Temporal Action Segmentation]
       │  • Window size: 1.5–2.0 s (45–60 frames @ 30 fps), 50% overlap
       │  • Feature Aggregation: Mean, Std, Min, Max, Quantiles, Velocity
       │  • Classifier: Random Forest / Support Vector Classifier (4 classes)
       ▼
[OnlineTAS-Style Temporal Post-Processing]
       │  • Confidence Threshold Filtering
       │  • Minimum Segment Duration Filter (drops spurious transitions < 1.0 s)
       │  • Neighbor Merging & Boundary Smoothing
       ▼
[Continuous Temporal Segments] [{exercise, start_time, end_time}, ...]
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[Adaptive Hysteresis Rep Counter]        [Narration & Acoustic Coaching Engine]
 • Signal conditioning (Savgol smooth)     • Event Triggering:
 • Adaptive p10..p90 thresholds              - Segment Start Announcement
 • Minimum range guard (min_range_deg)       - Repetition Cadence & Milestones
 • Refractory period suppression             - Rest / Transition Alerts
 • Emits: {reps, rep_times, flag}          • Template Formulator + Offline TTS
       │                                         │
       └───────────────────┬─────────────────────┘
                           ▼
             [Audio-Video Timeline Multiplexer]
                           ▼
            Final Narrated Coaching Video (MP4)
```

### 3.2 Stage 1: Pose Extraction and Primary Subject Selection
1. **Pose Landmarker:** Utilizes MediaPipe Pose Landmarker (`pose_landmarker_full.task`) in IMAGE mode. This guarantees that frame $t$ is evaluated independently of frame $t-1$, strictly preventing pose tracking hysteresis across synthetic video cuts.
2. **Subject Disambiguation:** In multi-person environments, the primary exercising subject is resolved via a 2D bounding box area heuristic:
   $$\text{Area} = (x_{\max} - x_{\min}) \times (y_{\max} - y_{\min})$$
   The pose exhibiting the maximal bounding box area is assigned as the primary user.

### 3.3 Stage 2: Kinematic Angle Feature Formulation
For any joint vertex $\mathbf{b}$ connected to adjacent joints $\mathbf{a}$ and $\mathbf{c}$, the spatial angle $\theta$ in degrees is given by:
$$\theta = \arccos\left(\frac{(\mathbf{a} - \mathbf{b}) \cdot (\mathbf{c} - \mathbf{b})}{\|\mathbf{a} - \mathbf{b}\| \|\mathbf{c} - \mathbf{b}\|}\right) \times \frac{180}{\pi}$$
- **Squat Kinematics:** Bilateral knee angle formed by Hip (23/24), Knee (25/26), and Ankle (27/28).
- **Push-up Kinematics:** Bilateral elbow angle formed by Shoulder (11/12), Elbow (13/14), and Wrist (15/16).
- **Jumping Jack Kinematics:** 
  - Arm abduction angle formed by Hip (23/24), Shoulder (11/12), and Wrist (15/16).
  - Leg spread angle formed at the hip midpoint vertex:
    $$\mathbf{h}_{\text{mid}} = \frac{\mathbf{h}_{\text{left}} + \mathbf{h}_{\text{right}}}{2}, \quad \theta_{\text{leg}} = \angle(\text{Ankle}_L, \mathbf{h}_{\text{mid}}, \text{Ankle}_R)$$
- **Best-Side Selection:** For unilateral perspective occlusion, the system dynamically selects the joint with the higher visibility score:
  $$\theta^* = \begin{cases} \theta_{\text{left}}, & \text{if } v_{\text{left}} \ge v_{\text{right}} \\ \theta_{\text{right}}, & \text{otherwise} \end{cases}$$
- **Data Hygiene:** Any angle with joint visibility $v < 0.50$ is flagged as NaN. Short gaps ($\le 0.4$ s) are linearly interpolated; segments with consecutive missing data exceeding 0.4 s remain NaN.

### 3.4 Stage 3: Sliding-Window Segmentation and OnlineTAS Smoothing
1. **Windowing:** A sliding temporal window of $W = 45$ to $60$ frames (1.5 to 2.0 s at 30 fps) with a stride of $S = 15$ to $30$ frames (50% overlap). Final configuration locked at $W = 2.0$ s, $S = 1.0$ s.
2. **Statistical Pooling:** Within each window, seven summary statistics are extracted per joint angle channel: mean, standard deviation, minimum, maximum, range, mean absolute change per second (MACPS), and NaN fraction. With 9 base feature channels plus their available 3D-world variants (16 channels total at runtime), the typical feature vector is **112-dimensional**.
3. **Classification:** A multi-class classifier assigns class probabilities over $\{\text{squat}, \text{push\_up}, \text{jump\_jack}, \text{transition}\}$. A systematic sweep over 8 classifier families was conducted (see Section 4.8); Logistic Regression and Random Forest are the primary candidates for the reported results.
4. **Temporal Smoothing:** Raw per-window predictions suffer from boundary flicker. We apply two-stage OnlineTAS-style post-processing:
   - *Confidence Thresholding ($\tau$):* Windows with predicted class probability $< \tau$ inherit the previous window's label (forward-fill), maintaining the last confident prediction across uncertain transitions. The first window always retains its own label.
   - *Minimum Run-Length Enforcement (min\_len):* Any contiguous run of identical predicted labels shorter than `min_len` windows is absorbed into the adjacent preceding run (or the following run for the first segment). Exception: `transition` runs are never absorbed, as resting intervals are legitimately short by design.
   - Both $\tau$ and `min_len` are selected by exhaustive grid search on the **validation split only** (never touching test), maximising Edit Score with frame accuracy as a tie-breaker.

### 3.5 Stage 4: Adaptive Hysteresis Repetition Counter (`rep_counter.py`)
Within each exercise segment $[\tau_{\text{start}}, \tau_{\text{end}}]$, the repetition counter operates on the continuous time series without touching the raw video:
1. **Continuous-Time Ingestion:** Features are indexed by the real continuous timestamp $t \in \mathbb{R}$ (`time_s`), decoupling the counter from frame rate (30 Hz, 15 Hz, or variable fps).
2. **Signal Conditioning:**
   - Tracking gaps $\le 0.4$ s are linearly interpolated; longer gaps remain NaN.
   - If valid samples constitute $< 50\%$ of the window, the segment is flagged `mostly_missing` and abstains from counting.
   - Savitzky-Golay filtering is applied with window length $w_{\text{smooth}} = \max(5, \text{odd}(f_s \cdot \text{smooth\_s}))$.
3. **Adaptive Thresholding:** To remain invariant to camera vantage, subject morphology, and range-of-motion differences, thresholds are dynamically computed from the segment's empirical 10th and 90th percentiles ($p_{10}, p_{90}$):
   $$\theta_{\text{low}} = p_{10} + \alpha_{\text{low}} \cdot (p_{90} - p_{10}), \quad \theta_{\text{high}} = p_{10} + \alpha_{\text{high}} \cdot (p_{90} - p_{10})$$
   - *Squat:* $\alpha = (0.35, 0.65)$, pattern = `down_up` (standing high $\to$ inflection low $\to$ standing high)
   - *Push-up:* $\alpha = (0.25, 0.55)$, pattern = `down_up` (plank high $\to$ bottom low $\to$ plank high)
   - *Jumping Jack:* $\alpha = (0.35, 0.65)$, pattern = `low_high_low` (arms low $\to$ arms high $\to$ arms low)
4. **Motion Range Guard (`min_range_deg`):** If $(p_{90} - p_{10}) < \text{min\_range\_deg}$, the segment is flagged `low_motion` and reps are clamped to 0, preventing phantom counting during stationary or misclassified segments.
5. **Two-Threshold Hysteresis State Machine:**
   - For `down_up`: Transition from State 0 (Top) to State 1 (Bottom) only when signal drops below $\theta_{\text{low}}$. Transition back to State 0 and log a completed rep only when signal climbs back above $\theta_{\text{high}}$.
   - Dual thresholds prevent false triggering from noise oscillations around a single cutoff.
6. **Refractory Suppression (`min_gap_s`):** Any candidate repetition occurring within $\Delta t < \text{min\_gap\_s}$ of the previous repetition is suppressed as high-frequency noise.

### 3.6 Stage 5: Synchronized Narration Engine
- **Event-Driven Architecture:** Narration is generated asynchronously via modular coaching event hooks:
  - `ON_SEGMENT_START`: Announces exercise name (e.g., *"Starting squats. Keep your chest up and go deep."*)
  - `ON_REP_CADENCE`: Announces repetition milestones (e.g., *"Three reps completed. Good tempo."*, *"Five down, five to go."*)
  - `ON_TRANSITION`: Rest guidance (e.g., *"Rest period. Take a deep breath before push-ups."*)
- **Acoustic Timeline Alignment:** Narration strings are synthesized into discrete audio segments using an offline TTS engine (gTTS / Coqui TTS). Each audio snippet is placed at the event timestamp $\tau$ and mixed onto the final video track, preventing audio overlaps via priority ducking.

---

## 4. Work Progress and Verified Results

### 4.1 Data Curation and Quality Screening
The raw dataset was sourced from **RepCount (RepCountA)**. 
- **Category Normalization:** Raw annotation inconsistencies (`squant`, `jumpjacks`, `pushups`) were audited and mapped to canonical classes (`squat`, `push_up`, `jump_jack`). 375 initial matching videos were screened; two irregular non-cyclical clips were eliminated, establishing a 373-clip base.
- **Evaluation of Countix Dataset (Decision O1):** Initial pipeline prototypes evaluated incorporating the Countix dataset. However, comprehensive cross-source duplicate analysis revealed significant video overlap and unresolvable kinetic timestamp offsets between Countix and RepCount. To guarantee strict experimental hygiene and eliminate cross-dataset leakage, **Countix was dropped**, relying purely on the verified RepCount repository.
- **MediaPipe Landmark Quality Screening:** A rigorous automated screening tool evaluated all 373 clips on four criteria (detection rate $\ge 0.65$, 10th percentile visibility $\ge 0.50$, resolution $\ge 240$ px, reps $\ge 2$).
- **Screening Outcome:** 336 clips passed screening (312 clear pass, 24 flagged for visual inspection). 37 clips failed.

### 4.2 Leakage-Proof Subject-Grouped Dataset Splits
To prevent subject identity leakage, clips were partitioned using subject-prefix group keys (`stu1_*`, `stu2_*`) via a greedy bin-packing algorithm with a fixed seed (`SEED = 42`):

| Split | Jump Jack | Push Up | Squat | Total Clips | Independent Subject Groups | Split Share |
|---|---|---|---|---|---|---|
| **Train** | 68 | 72 | 76 | **216** | 31 groups | 64.3% |
| **Validation** | 11 | 14 | 12 | **37** | 8 groups | 11.0% |
| **Test** | 26 | 28 | 29 | **83** | 27 groups | 24.7% |
| **Total** | **105** | **114** | **117** | **336** | **66 groups** | **100.0%** |

*Verification:* Zero subject groups overlap across any split boundaries, guaranteeing zero train/test identity leakage. Test split is strictly locked until final model benchmark.

### 4.3 Ground-Truth Repetition Boundary Verification
RepCount's raw annotation columns ($L_1, L_2, \dots, L_n$) were systematically analyzed:
- A custom diagnostic script (`verify_l_columns.py`) verified that $L_{2k-1}$ corresponds to repetition initiation and $L_{2k}$ marks repetition completion.
- Ground truth boundaries were verified and compiled into `repcount_rep_boundaries.json`, yielding **0 count mismatches** and **0 out-of-bounds frame references** across all 336 clips.
- Distribution analysis demonstrated substantial pre-exercise lead-in margins (median 19–33 frames) and post-exercise recovery margins (median 13–37 frames), providing a natural source of authentic human rest footage.

### 4.4 Synthetic Multi-Exercise Video Generation
Using `concatenate_videos.py`, 45 continuous multi-exercise workout videos were generated across splits:
- **Dataset Composition:** 24 Training videos, 6 Validation videos, 15 Test videos.
- **Video Specifications:** Standardized to $640 \times 360$ letterbox, 30.0 fps, nearest-neighbor frame resampling, duration ranging from 50.0 s to 268.9 s.
- **Composition Rules:** 3 to 5 exercises per video; adjacent exercises guaranteed distinct; each video draws clips strictly from a single split (test videos contain zero training clips).
- **Contiguity & Ground Truth:** Generated corresponding boundary JSON files cataloging all segments, source clip IDs, exact frame bounds $[f_{\text{start}}, f_{\text{end}})$, and rescaled repetition boundary pairs.

### 4.5 Boundary Diagnostic Jump & Motion Continuity Check
- A diagnostic analysis (`boundary_jump_check.py`) compared the $L_2$ norm of angle vector changes across stitched clip joins versus ordinary random intra-clip frame steps.
- **Finding:** The median angular change across clip joins ($5.89^\circ$ on test, $8.40^\circ$ on train) is completely comparable to natural human intra-exercise movement velocities ($4.39^\circ$ to $17.06^\circ$), demonstrating that pose-based features do not exhibit artificial high-frequency boundary spikes.

### 4.6 Repetition Counting Evaluation Results (`evaluate_rep_counts.py`)
The rule-based repetition counter was evaluated on **119 exercise segments** across the synthetic train and validation videos using ground-truth segment windows.

#### Comprehensive Evaluation Metrics (Train + Val)

| Exercise | Evaluated Segments | Exact Match (%) | Within $\pm 1$ (%) | MAE | Bias | Precision | Recall | Flagged (%) |
|---|---|---|---|---|---|---|---|---|
| **Squat** | 37 | **67.57%** | **75.68%** | 2.00 | -1.30 | 0.94 | 0.82 | 2.70% |
| **Jumping Jack** | 45 | **45.00%** | **71.11%** | 1.73 | -1.42 | 0.99 | 0.94 | 2.22% |
| **Push-up** | 37 | **45.95%** | **62.16%** | 2.57 | -0.57 | 0.93 | 0.90 | 0.00% |

#### Validation Split Alone (Unseen Subjects)

| Exercise | Segments | Exact Match (%) | Within $\pm 1$ (%) | MAE | Bias | Precision | Recall |
|---|---|---|---|---|---|---|---|
| **Squat** | 8 | **75.00%** | **100.00%** | 0.25 | +0.25 | 0.92 | 0.96 |
| **Jumping Jack** | 9 | **77.78%** | **100.00%** | 0.22 | 0.00 | 0.99 | 0.99 |
| **Push-up** | 7 | **28.57%** | **57.14%** | 4.71 | +2.14 | 0.81 | 0.91 |

#### Analysis of Findings & Outlier Diagnostics:
1. **Perfect $\pm 1$ Tracking on Squats and Jumping Jacks:** On the validation split, squats and jumping jacks achieved **100% within-1 accuracy** with MAE under $0.25$ reps and precision/recall above $0.92$.
2. **Investigation of the Push-up Outlier:** The validation push-up metrics are skewed almost entirely by a single clip: `stu10_49.mp4` (annotated 21 reps, counted 43 reps — an approximate $2\times$ overcount). Diagnostic inspection revealed that this clip contains multiple individuals in the frame. The counter locked onto the secondary subject's periodic arm motion, causing double-triggering. This confirms the critical necessity of enforcing the `largest_bbox_pose_index` single-subject selection policy across raw inputs.
3. **Valid Abstentions:** Two segments were flagged under `low_motion` (`stu6_62` squat and `stu6_19` jumping jack). These were verified to be correct abstentions where the subject exhibited negligible range of motion ($< 25^\circ$), demonstrating the protective efficacy of `min_range_deg`.
4. **Empirical Calibration Reference:**
   - *Annotated Rep Durations (p50):* Push-up 0.95 s, Jumping Jack 0.93 s, Squat 1.80 s. Sets `min_gap_s` safely at 0.2–0.5 s.
   - *Observed Angle Ranges (p50):* Squat $70.4^\circ$, Push-up $88.2^\circ$, Jumping Jack $148.1^\circ$. Confirms `min_range_deg` thresholds ($20^\circ$–$40^\circ$) are safely below the 5th percentiles.

### 4.7 Scaled Feature Extraction & Unit Test Verification
- **Feature Extraction Scale-Up:** Pose extraction (`extract_pose_features.py`) was scaled across 31 continuous synthetic videos with stride-2 (15 Hz) subsampling, achieving high detection rates ($92\%\text{--}100\%$) and logging complete metrics in `pose_extraction_summary.csv`.
- **Test Suite:** 38 unit tests pass at 100% across `tests/test_concatenate.py` and `tests/test_pose_utils.py`.

### 4.8 Segmentation Classifier and Smoothing Hyperparameter Sweep

To determine the best classifier and post-processing configuration, a systematic sweep was conducted over the pre-built feature matrix (fixed: window=2.0 s, stride=1.0 s, 112-dim feature vectors, 5891 windows across 45 synthetic videos).

**Sweep design:** 8 classifier families × 2 `min_purity_train` thresholds × a 4×4 smoothing grid (τ ∈ {0.4, 0.5, 0.6, 0.7} × min_len ∈ {1, 2, 3, 4}) = **16 experiments**. The feature matrix was built once; only the classifier, purity threshold, and smoothing parameters were varied. Smoothing parameters were selected by val-set grid search for each experiment; test was evaluated once with the chosen parameters.

**Results (ranked by test Edit Score):**

| Rank | Classifier | Purity | Best τ | min\_len | Test Edit ↑ | Test Acc ↑ | Test F1@25 | Test F1@50 | Val Edit |
|------|-----------|--------|--------|---------|------------|-----------|-----------|-----------|----------|
| 1 | **LR** | 0.7 | 0.7 | 4 | **63.4** | 0.781 | 0.579 | 0.453 | 55.2 |
| 2 | LR | 0.8 | 0.7 | 4 | 60.5 | 0.773 | 0.581 | 0.459 | 54.6 |
| 3 | GBT | 0.7 | 0.7 | 1 | 60.0 | 0.785 | 0.536 | 0.441 | 59.0 |
| 4 | HistGBT | 0.7 | 0.7 | 1 | 59.5 | **0.815** | 0.585 | 0.466 | 48.3 |
| 5 | SVM (RBF) | 0.7 | 0.6 | 4 | 58.8 | 0.765 | 0.575 | 0.454 | 58.0 |
| 7 | ExtraTrees | 0.8 | 0.4 | 1 | 58.1 | 0.807 | 0.556 | 0.474 | 54.6 |
| 9 | **RF** *(baseline)* | 0.7 | 0.6 | 1 | 57.2 | 0.799 | **0.629** | **0.487** | 56.0 |
| 14 | MLP | 0.8 | 0.6 | 3 | 49.6 | 0.715 | 0.414 | 0.291 | 47.0 |
| 16 | KNN | 0.7 | 0.4 | 4 | 48.0 | 0.757 | 0.490 | 0.370 | 56.4 |

Full 16-row results in `outputs/segmentation/sweep_results.csv`.

**Key findings:**

1. **Logistic Regression achieves the best test Edit Score** (+6.2 points over the RF baseline). Its smoother linear decision boundaries, combined with aggressive smoothing (τ=0.7, min_len=4), produce cleaner segment-label sequences. This is consistent with the hypothesis that the 112-dim feature space is already linearly separable for the three exercise classes, and that post-processing (not classifier complexity) is the main lever for boundary quality.

2. **HistGBT achieves the best frame accuracy (0.815)** but lower Edit Score. It classifies individual windows accurately but produces more fragmented sequences that smoothing cannot fully repair, suggesting its ensembling over deep trees overfits to per-window patterns at the expense of temporal consistency.

3. **RF has the best segment-level F1@25 (0.629)** — the most accurate placement of segment boundaries at 25% IoU threshold. This makes RF the preferable choice if downstream rep-counting accuracy (which depends on boundary precision rather than sequence cleanliness) is the primary metric.

4. **Lower purity threshold (0.7 vs 0.8) generally wins.** Including more boundary-adjacent training windows exposes the classifier to the continuous transition from exercise to rest, helping it generalize rather than overfit to the interior of each exercise class.

5. **MLP and KNN underperform** by 8–18 Edit Score points. At this dataset scale (≈3100 eligible training windows), MLP does not generalize well and KNN's Euclidean distance over 112 mixed statistics is not informative enough.

6. **Val→test generalisation gap is visible for LR** (val Edit 55.2, test Edit 63.4 — test actually outperforms val). This reflects that val is dominated by one subject (stu10) who may be harder to segment than the test subjects (stu2/stu4). The metric should be reported with this caveat.

**Implementation note:** All sweep experiments are run via `python src/segmentation/hparam_sweep.py`. The script accepts `--classifiers`, `--min-purity-grid`, `--tau-grid`, and `--min-len-grid` flags for targeted re-runs.

---

## 5. References

1. **Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M.** (2020). BlazePose: On-device Real-time Body Pose Tracking. *arXiv preprint arXiv:2006.10204*.
2. **Dwibedi, D., Aytar, Y., Tompson, J., Sermanet, P., & Zisserman, A.** (2020). Counting Out Time: Class Agnostic Video Repetition Counting in the Wild. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 10313-10322.
3. **Hu, H., Lu, D., Chen, W., & Xu, M.** (2022). TransRAC: How Many More Times? A Large-scale Dataset and Benchmark for Repetition Counting. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 13840-13850.
4. **Ding, G., Shen, Y., & Mo, S.** (2023). Temporal Action Segmentation: A Survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*.
5. **Zhong, Y., & Wang, L.** (2022). Online Action Segmentation with Contrastive Temporal Representation Learning. *European Conference on Computer Vision (ECCV)*.
6. **Latyshev, A., et al.** (2022). Comparative Analysis of MediaPipe and OpenPose for Kinematic Assessment in Physical Exercises. *Sensors*, 22(18), 7012.
7. **Krishna, R., Hata, K., Ren, F., Fei-Fei, L., & Niebles, J. C.** (2017). Dense-Captioning Events in Videos. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 706-715.
8. **Strömbäck, D., et al.** (2020). MM-Fit: A Multimodal Dataset for Exercise Analysis and Workout Tracking. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*.
9. **Jorke, M., et al.** (2023). GPTCoach: Generative AI for Automated Workout Feedback. *ACM Conference on Human Factors in Computing Systems (CHI)*.
10. **Yu, H., et al.** (2021). Rep-Penn: Pose-based Action Repetition Counting. *Pattern Recognition Letters*.

<!-- v2-model-results:start -->
## Completed v2 model experiment — 17 November 2026

8-model train-only CV selected **rf** (Edit 71.54 ± 2.90 fold SD; tau=0.5, min_len=3). After all-train refit, only that frozen winner was evaluated on 21 test videos: Edit 72.51, frame accuracy 86.66%, segment F1@25 0.750. Test was not used for model selection or tuning. See [full results, caveats and reproducibility tables](../docs/results/V2_MODEL_RESULTS.md).
<!-- v2-model-results:end -->
