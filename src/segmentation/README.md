# Segmentation: v2 source-clip CV

Current training/evaluation and integration guide: [V2_TRAINING.md](../../docs/V2_TRAINING.md).

Model selection now uses five train source-clip folds. Per-fold imputation
and scaling are fitted only on eligible training windows. Held-out fold windows
are unfiltered; test angle files are not read by CV or all-train refitting.

The active entry points are train.py (CV/check-only/refit), hparam_sweep.py
(train-only ranking), evaluate.py (OOF report or explicit final test), and
predict_video.py (the frozen all-train bundle). run_all.py delegates to train.py;
it never silently evaluates test. See the guide for exact commands and artifacts.

Data preparation remains `uv run --no-sync python -B src/data_pipeline/phase4.py`.
Per-video features are data/features/v2/; windows and model experiments are
under outputs/segmentation_v2/. Models, windows, features and frozen archive data
are not overwritten by these commands. No benchmark model was fitted or tested
as part of the compatibility refactor.

<!-- v2-model-results:start -->
## Completed v2 model experiment — 17 November 2026

8-model train-only CV selected **rf** (Edit 71.54 ± 2.90 fold SD; tau=0.5, min_len=3). After all-train refit, only that frozen winner was evaluated on 21 test videos: Edit 72.51, frame accuracy 86.66%, segment F1@25 0.750. Test was not used for model selection or tuning. See [full results, caveats and reproducibility tables](../../docs/results/V2_MODEL_RESULTS.md).
<!-- v2-model-results:end -->

## OOF per-class and jumping-jack diagnostics (18 November 2026)

Branch `experiments/segmentation_improvement` retains the existing uncommitted
development work. `diagnose_oof_classes.py` reads the selected RF's saved OOF
predictions, verifies their checksum, training signature, complete window keys
and `cv_fold == fold`, and reconstructs every existing smoothing stage. No
training, pose extraction, rep counting, counter-parameter changes or test
evaluation is performed. Only a diagnostic module and its tests are added;
artifacts live under ignored `outputs/` paths. No commit or push is performed.

There are 14,879 OOF windows from 120 training videos. Matrices below use
**GT rows and predicted columns**, in window counts, including mixed-boundary
windows. These are pooled window statistics, not the official equal-fold,
per-video 10 Hz frame scores. A separate high-purity (>=0.8) diagnostic and
exact temporal-overlap matrices are retained to distinguish those units.

Raw RF confusion:

| GT / predicted | Jumping jacks | Push-ups | Squats | Transition |
|---|---:|---:|---:|---:|
| Jumping jacks | 3167 | 4 | 162 | 505 |
| Push-ups | 17 | 4293 | 13 | 481 |
| Squats | 37 | 17 | 3696 | 360 |
| Transition | 199 | 449 | 258 | 1221 |

After unchanged temporal smoothing:

| GT / predicted | Jumping jacks | Push-ups | Squats | Transition |
|---|---:|---:|---:|---:|
| Jumping jacks | 3367 | 16 | 122 | 333 |
| Push-ups | 13 | 4594 | 2 | 195 |
| Squats | 37 | 5 | 3821 | 247 |
| Transition | 159 | 477 | 255 | 1236 |

| Class | Raw recall | Smoothed recall | Smoothed precision | Smoothed F1 |
|---|---:|---:|---:|---:|
| Jumping jacks | 82.52% | 87.73% | 94.16% | 90.83% |
| Push-ups | 89.36% | 95.63% | 90.22% | 92.85% |
| Squats | 89.93% | 92.97% | 90.98% | 91.96% |
| Transition | 57.40% | 58.11% | 61.46% | 59.74% |

Of 671 raw JJ classification errors, 505 (75.3%) go to transition, 162 to
squat and four to push-up. After smoothing, 333/471 JJ errors (70.7%) go to
transition. Raw JJ recall on high-purity windows is still 82.31%, so this
is not explained solely by majority-label ambiguity at boundaries. Exact
temporal-overlap JJ recall changes 82.43% to 87.39%; corresponding final
recalls are push-ups 95.75%, squats 93.11%, transition 56.16%.

### The 19 previously identified mislabeled JJ windows/videos

Duration-dominant predictions within the same recorded 19 GT windows:

| Stage | Dominant JJ | Dominant transition | Dominant squat | Dominant push-up |
|---|---:|---:|---:|---:|
| Raw RF | 4 | 11 | 4 | 0 |
| Confidence handling | 3 | 12 | 4 | 0 |
| Minimum-run handling | 0 | 15 | 4 | 0 |
| Final sandwich-transition merge | 0 | 13 | 4 | 2 |

Thus the final 19 split **13 transition / 4 squat / 2 push-up**. Fifteen
were already dominated by the wrong class in raw RF output (11 transition,
four squat); four were raw-JJ-dominant but lost during smoothing (one at
confidence handling, three at minimum-run handling). The two final push-up
cases were raw-transition-dominant and become push-up during the sandwich
merge; they are not evidence of raw RF JJ-to-push-up confusion. The 19 cases
reuse nine source clips, so their video count is not independent evidence.

Current smoothing has asymmetric rules that matter for this diagnosis:

- Confidence <0.5 forward-fills **non-transition** predictions only. Raw
  transition predictions are exempt, even at low confidence. About 39.2% of
  raw JJ-to-transition windows have max probability below 0.5.
- Runs shorter than three windows are absorbed only for **non-transition**
  classes; transition runs are exempt. Across all GT-majority-JJ windows this
  step changes 40 JJ labels to transition, plus 14 to squat.
- A transition run between matching exercises is merged without a maximum
  duration. This recovers 266 JJ labels overall, but also changes 16 GT-JJ
  windows to push-up in the two affected failure cases.

JJ temporal recall over the full data is raw 82.43%, after confidence 81.89%,
after minimum-run 80.53%, after final merge 87.39%. Smoothing helps overall;
removing it wholesale is not supported. In the selected 19-video cohort,
JJ temporal recall changes 47.94% to 46.03%, illustrating concentrated harm.
This cohort statistic includes all GT JJ windows in those videos, not only
the selected failed windows.

The primary diagnostic points to a **transition-class/context problem**, with
a smaller genuine squat-discrimination problem. It does not yet identify
causal feature contributions or prove periodic motion in every missed region.
A grouped-CV ablation of transition handling/context is justified; improved
motion/periodicity features remain candidates for the raw RF misses. The
experimental rep counter stays fixed and the segmenter defaults are unchanged.

```powershell
.venv/Scripts/python.exe -m src.segmentation.diagnose_oof_classes --out-dir outputs/segmentation_v2/diagnostics/oof_classes_20261118
```

Local artifacts:

- [Report, recalls, probabilities, fold metrics and provenance](../../outputs/segmentation_v2/diagnostics/oof_classes_20261118/report.json)
- [Raw confusion matrix](../../outputs/segmentation_v2/diagnostics/oof_classes_20261118/raw_window_confusion.csv)
- [Smoothed confusion matrix](../../outputs/segmentation_v2/diagnostics/oof_classes_20261118/final_window_confusion.csv)
- [All 19 windows, per-stage labels and class-duration fractions](../../outputs/segmentation_v2/diagnostics/oof_classes_20261118/mislabelled_19_windows.csv)
- [JJ label changes at each smoothing stage](../../outputs/segmentation_v2/diagnostics/oof_classes_20261118/jj_smoothing_label_flow.csv)
- [Every saved OOF window with stage labels](../../outputs/segmentation_v2/diagnostics/oof_classes_20261118/oof_window_stage_labels.csv)

All source inputs and previous diagnostics were verified unchanged. These
are development diagnostics on source-held-out OOF predictions, not a new
independent test result or a segmentation improvement claim.

## Periodicity and centered-context feature ablation

`periodicity_context_ablation.py` is an opt-in, training-only experiment.
It leaves canonical feature CSVs, windows, active model, default smoothing
and rep-counter parameters unchanged. One experiment module and one test
file are added; fitted fold models are not activated or persisted. Existing
results are preserved and a new output directory is required.

Four predeclared feature groups use identical RF hyperparameters (200 balanced
trees, seed 42), training purity >=0.8, five source-disjoint folds, and existing
smoothing tau 0.5/minimum run 3. Imputation is fitted on each fold's training
partition only; every held-out window is evaluated. No audit-flagged videos
are excluded and no new classifier/smoothing/grid search is performed.

| Group | Features |
|---|---:|
| Existing 2 s summary statistics | 112 |
| Baseline + 2 s periodicity | 130 |
| Baseline + centered 8 s context statistics | 224 |
| Baseline + periodicity + context | 242 |

Periodicity channels are 2D `arm_raise_L`, `arm_raise_R`, `ankle_dist_norm`,
`hip_L`, `hip_R` and `knee_best`. Each adds dominant positive-frequency bin
(Hz), fraction of positive-frequency spectral power in 1-3 Hz, and strongest
positive local autocorrelation peak at lags 1/6 to 1.5 s. Signals are linearly
detrended; a Hann periodogram is computed over the original 2 s window.
Features require at least 12 samples and one finite run covering >=80% of
the window. Remaining long gaps are not bridged. Flat signals yield zeros;
unavailable features remain NaN for fold-fitted imputation. The original
0.5 s short-gap fill is retained. Two-second spectra have coarse frequency
resolution; dominant signal frequency need not equal repetition frequency.

Context adds the original 112 statistics over a centered 8 s interval at each
existing 1 s stride. Context is clipped only to the current video extent,
never GT exercise boundaries, and does not cross into another video. Future
lookahead makes this an **offline** feature. Neighboring synthetic exercise
regions within the same video can contribute; source-disjoint folds prevent
training/held-out source reuse. Context does not use labels, split/fold IDs,
visibility diagnostics or annotated rep counts as model features.

The freshly fitted baseline must reproduce the saved RF OOF labels and
probabilities before any ablation is interpreted. A downstream JJ-only count
diagnostic uses each video's same prior other-fold-selected experimental
peak settings, with 0.3 s padding; no counter parameters are changed. The
baseline must also reproduce the saved 5.2417 video JJ MAE. This diagnostic
is not a full three-exercise pipeline reevaluation.

```powershell
.venv/Scripts/python.exe -m src.segmentation.periodicity_context_ablation --out-dir outputs/segmentation_v2/experiments/periodicity_context_20261118
```

This is development ablation on previously used training folds, not unbiased
nested CV or new independent-test evidence. The 19-window error cohort is
tracked descriptively only. No test inference, label changes, production
activation, commits or pushes are performed.

### Completed four-group ablation

All 20 fits (four groups x five source-disjoint folds) completed. The baseline
reproduced the previous OOF labels/probabilities and fixed-counter JJ MAE.
All 120 training videos remain included; no audit exclusions or new tuning
were applied. The only new tracked-source candidates are this experiment
module and its test file; reports and prediction tables remain under outputs.

| Feature group | Edit mean +/- fold SD | Mean fold frame accuracy | Mean fold F1@25 | Pooled JJ temporal recall | Fixed-counter JJ video MAE |
|---|---:|---:|---:|---:|---:|
| Baseline | 71.537 +/- 2.898 | 85.109% | 0.7042 | 87.395% | 5.242 |
| + Periodicity only | 69.832 +/- 2.632 | 86.288% | 0.7121 | 92.215% | 4.300 |
| + Context only | 71.453 +/- 1.320 | 85.602% | 0.7306 | 88.390% | 5.092 |
| + Both | 70.806 +/- 2.662 | 86.237% | 0.7261 | 90.592% | 4.183 |

Periodicity is the stronger addition for JJ recall/counting. Context alone
has the best segment F1@25 and helps transition recall, but barely improves
JJ MAE. Both groups give the lowest JJ video MAE, 20.2% below baseline, while
periodicity alone lowers it 18.0%. No group dominates: baseline still has the
highest Edit score. Fold SD is descriptive, not a confidence interval. Pooled
temporal recall weights durations and is a different aggregation from the
equal-fold/per-video frame accuracy column.

Per-class pooled exact temporal recall:

| Feature group | JJ | Push-up | Squat | Transition |
|---|---:|---:|---:|---:|
| Baseline | 87.39% | 95.75% | 93.11% | 56.16% |
| + Periodicity | 92.22% | 95.89% | 93.09% | 55.47% |
| + Context | 88.39% | 94.95% | 92.81% | 60.78% |
| + Both | 90.59% | 94.99% | 92.28% | 60.01% |

For JJ video counts over all 120 videos, exact/within-one numbers are baseline
40/70, periodicity 42/71, context 41/72, both 39/70. Thus the best mean-error
group does not improve exact or within-one counts. All include exercise-absent
zeros and remain downstream diagnostics rather than an all-exercise rep
evaluation. The same existing fold-specific JJ counter parameters are used
for every group, with no changed peak spacing, recovery or prominence.

Of the originally failed 19 windows, duration-dominant JJ labels are restored
in five with periodicity, none with context alone, and three with both.
Remaining dominant labels are periodicity: nine transition/three squat/two
push-up; context: 14 transition/four squat/one push-up; both: nine transition/
six squat/one push-up. This error-selected cohort is descriptive and shows
that adding context is not a uniform repair for class confusion.

The default active segmenter is **not replaced**. These results justify
continuing grouped-CV development with periodicity, while retaining the Edit,
transition and exact-count tradeoffs. No test inference or new independent
generalization claim follows. All 266 tests pass, and source/input hashes
were verified unchanged at completion.

For the motivating stu5_25 window in f3_0002, JJ coverage changes from 6.85%
to 94.12% with periodicity, 13.69% with context, and 97.49% with both. The
same video's fixed-counter JJ count changes 8 to 62/12/64 respectively
against GT 74. This is a useful development example, not independent evidence:
the source is reused and was inspected before designing the experiment.
The combined group's fold-3 JJ MAE falls 12.250 to 6.125, but fold 1 worsens
4.667 to 5.667 and fold 0 worsens 3.917 to 4.167. Gains are not uniform.

Artifacts:

- [Comparison table](../../outputs/segmentation_v2/experiments/periodicity_context_20261118/comparison.csv)
- [Report, per-class metrics and feature schemas](../../outputs/segmentation_v2/experiments/periodicity_context_20261118/report.json)
- [Pre-fit plan, policies, environment and input/code hashes](../../outputs/segmentation_v2/experiments/periodicity_context_20261118/plan.json)

Each group subdirectory retains its OOF probabilities/labels, fold/video
segmentation metrics, smoothed window-count and exact-duration confusion matrices,
fixed-counter JJ counts/timestamps and the original 19-window diagnostic.
No production bundle, canonical data or previous results were overwritten.
Changes remain uncommitted; nothing was pushed.

### All-class counting completion and fold consistency

The initial ablation counted JJ only. `complete_ablation_counts.py` subsequently
completes push-up/squat counting from those same saved OOF predictions, using
the unchanged fold-specific experimental combo. The baseline here is the
**experimental combo with original segmentation**, overall MAE 3.625, not
the state-machine pipeline. All 120 videos and 360 video/exercise units remain
included, with absent-exercise zeros. There is no new model fit, tuning or
test inference.

| Segmentation features | JJ counting MAE | Push-up counting MAE | Squat counting MAE | Overall counting MAE |
|---|---:|---:|---:|---:|
| Baseline | 5.242 | 2.925 | 2.708 | 3.625 |
| + Periodicity | 4.300 | 3.067 | 2.367 | 3.244 |
| + Context | 5.092 | 2.642 | 2.625 | 3.453 |
| + Both | 4.183 | 2.592 | 2.975 | 3.250 |

Other classes are **not uniformly protected**: periodicity-only worsens
push-up MAE 2.925 to 3.067; both worsens squat MAE 2.708 to 2.975. Both's
squat MAE worsens in all five folds. Context-only improves aggregate counting
MAE for all three classes, but its push-up/squat recall declines (table above)
and exact-count performance is not uniformly better. Periodicity-only has
push-up temporal recall 95.75% to 95.89%, squat 93.11% to 93.09% (a tiny
decline), and transition 56.16% to 55.47%. Lower overall MAE therefore does
not establish a uniform net win or justify automatic production adoption.

Per-fold JJ counting MAE for **periodicity-only**, holding its counter fixed:

| Held-out fold | Baseline | + Periodicity | Change |
|---|---:|---:|---:|
| 0 | 3.917 | 3.917 | 0.000 |
| 1 | 4.667 | 5.667 | +1.000 |
| 2 | 2.375 | 2.167 | -0.208 |
| 3 | 12.250 | 6.625 | -5.625 |
| 4 | 3.000 | 3.125 | +0.125 |

Only **2/5 folds improve**, one ties and two worsen. The main gain is fold 3.
Periodicity-only overall three-exercise counting MAE also improves in two
folds, ties one and worsens two. Edit worsens in **all five folds**; frame
accuracy and F1@25 improve in three each. A lower pooled MAE is not evidence
of consistent improvement. For periodicity-only squat counting, one large
fold gain offsets worsening in four folds; its better aggregate MAE should
also be read with that concentration in mind.

Descriptive paired-video 95% intervals for overall MAE change (new minus
baseline) are periodicity -0.381 [-0.969, +0.114], context -0.172 [-0.411,
+0.067], both -0.375 [-0.908, +0.075]. All include zero; development reuse
and synthetic source repetition further limit inference. Overall exact/
within-one counts over 360 units are baseline 148/220, periodicity 146/221,
context 145/227 and both 144/226. The active default model is unchanged.

All-class completion reuses 78,094/80,361 required raw pose frames and extracts
2,267 missing frames in 47 runs with the same frozen protocol. JJ counts
replay the ablation artifacts. New push-up signals are recomputed per
predicted window; squat peaks use the same canonical feature inputs. A numeric
dtype issue from concatenating an empty cache was repaired after all pose
frames were saved. Counting resumed from those saved coordinates without
repeating pose inference or changing counter logic/parameters. The original
plan and the resume-code provenance are retained separately.

```powershell
.venv/Scripts/python.exe -m src.segmentation.complete_ablation_counts --out-dir outputs/segmentation_v2/experiments/periodicity_context_all_counts_20261118 --workers 2
# Only to resume an incomplete run with all coordinates already saved:
.venv/Scripts/python.exe -m src.segmentation.complete_ablation_counts --out-dir outputs/segmentation_v2/experiments/periodicity_context_all_counts_20261118 --resume-counts
```

- [All-class report, paired intervals and fold consistency](../../outputs/segmentation_v2/experiments/periodicity_context_all_counts_20261118/report.json)
- [Per-fold counting comparisons for every class](../../outputs/segmentation_v2/experiments/periodicity_context_all_counts_20261118/per_fold_count_comparison.csv)
- [Periodicity-only video/exercise counts](../../outputs/segmentation_v2/experiments/periodicity_context_all_counts_20261118/periodicity_video_exercise_counts.csv)
- [Context-only video/exercise counts](../../outputs/segmentation_v2/experiments/periodicity_context_all_counts_20261118/context_video_exercise_counts.csv)
- [Combined-feature video/exercise counts](../../outputs/segmentation_v2/experiments/periodicity_context_all_counts_20261118/both_video_exercise_counts.csv)
- [Counted push-up/squat segment timelines](../../outputs/segmentation_v2/experiments/periodicity_context_all_counts_20261118/counted_pushup_squat_segments.csv)

The first-run feature-ablation report and all baseline results remain intact.
One completion helper and tests in the existing feature-test file were added;
no new standalone context document or model bundle was created. No commits
or pushes are performed.

## Exercise-first transition gate, selected by counting MAE

`exercise_gate_sweep.py` uses the saved **baseline-feature RF OOF probabilities**,
not the periodicity/context models. No model is refitted. For each window,
choose the largest original probability among JJ, push-up and squat, and
call it transition only when that largest exercise probability is below a
single threshold. The original transition probability does not compete in
argmax and is not redistributed: exercise probabilities are not renormalized.
Ties among exercises use JJ/push-up/squat order, including the threshold-zero
endpoint when every exercise probability is zero.

The new gate replaces the old 0.5 confidence-forward-fill stage, rather than
adding another cutoff. The existing minimum-run rule (3 windows) and sandwich
transition merge remain fixed. Those structural rules may change raw gated
labels, so the final segments are not simply threshold masks. Features, RF,
all per-fold experimental counters and 0.3 s end padding remain unchanged.

The predeclared grid is 0 to 1 inclusive in 0.05 steps (21 candidates). The
objective is equal-fold mean **video/exercise counting MAE**, including all
360 units across 120 training videos and absent-exercise zeros. It is not
transition F1, frame accuracy, or whole-workout error where classes can cancel.
Exact objective ties prefer a higher threshold, a conservative rejection rule.
No audit examples are removed and no post-result grid refinement is planned.

The best threshold on all OOF outcomes has a selection-biased score. A second
diagnostic chooses the threshold from four folds and applies it to the fifth.
That is still not fully nested CV: the development RF/smoothing/counter
families were previously selected, and OOF models for the other folds included
the target fold in their fitting data. Both diagnostics are labeled accordingly;
there is no new untouched-test or independent-generalization claim.

The sweep records all-class count MAEs, exact/within-one fractions, signed
bias and over/under counts, plus JJ overlong/overcount flags and temporal recall.
This exposes the asymmetric cost balance without assuming counter safeguards
reject every non-exercise region. The original experimental combo at 3.625 MAE
remains the comparison; no gate is activated in production.

Coordinate extraction uses the existing frozen protocol only for missing
requested frames. The union of candidates needs 101,088 unique frames, with
81,963 reused and 19,125 additional frames in 135 runs. Counts for identical
exercise/video/start/end windows are cached across thresholds. No pose frame
is extracted separately for every threshold. Plans and input/code hashes are
recorded before extraction/counting.

```powershell
.venv/Scripts/python.exe -m src.segmentation.exercise_gate_sweep --out-dir outputs/segmentation_v2/experiments/exercise_gate_counts_20261118 --workers 2
```

Only an incomplete run with fully saved coordinate files and exactly matching
plan/source can use `--resume-counts`; existing completed reports are refused.
Canonical data, test features/videos, active model and counter defaults are
unchanged. No commits or pushes are performed.

### Completed exercise-first gate sweep: no counting gain

All 21 thresholds completed on the same 120 training videos with unchanged
experimental counters. **None improves the original 3.625 all-class counting
MAE.** The best all-OOF threshold is 0.55, scoring 3.669; this is a selected
development score, not independent validation. A lower threshold of 0.30
gives the best JJ MAE but sacrifices the other classes:

| Decision policy | JJ MAE | Push-up MAE | Squat MAE | Overall MAE |
|---|---:|---:|---:|---:|
| Original four-class argmax + existing smoothing | 5.242 | 2.925 | 2.708 | 3.625 |
| Exercise-first threshold 0.00 | 4.558 | 4.525 | 3.267 | 4.117 |
| Exercise-first threshold 0.30 | 4.208 | 3.975 | 2.883 | 3.689 |
| Exercise-first threshold 0.55 (best all-OOF) | 6.083 | 2.683 | 2.242 | 3.669 |
| Other-fold-selected thresholds | 6.025 | 2.950 | 3.250 | 4.075 |

The threshold selected without each target fold is 0.30 for folds 0/2/4 and
0.55 for folds 1/3. Held-out MAEs are respectively 3.597, 4.611, 2.250,
6.208 and 3.708 in fold order; all five are worse than the original policy.
This instability supports retaining the original policy, but is a development
threshold-selection diagnostic, not strict nested CV (see limitations above).

The selected 0.55 rule has paired-video MAE change +0.044 with descriptive
95% interval [-0.322, +0.411]. The other-fold-selection change is +0.450
[+0.150, +0.789]. These intervals inherit synthetic source reuse and model/
threshold development selection; they are not independent population claims.

The recall/overcount balance is visible rather than assumed:

| Policy | JJ temporal recall | JJ overcount videos | JJ overcount + overlong | Overcount / undercount units (all 360) |
|---|---:|---:|---:|---:|
| Original | 87.39% | 43 | 41 | 119 / 93 |
| Threshold 0.00 | 91.22% | 50 | 47 | 161 / 70 |
| Threshold 0.30 | 90.25% | 48 | 44 | 139 / 76 |
| Threshold 0.55 | 83.40% | 34 | 30 | 98 / 109 |

At 0.55, overall exact/within-one counts change 148/220 to 153/221 out of
360, but mean signed error worsens -1.075 to -1.875. Edit improves 71.537
to 73.719 while F1@25 is essentially unchanged (0.7042 to 0.7037); these are
secondary diagnostics and do not override the specified counting objective.
Low rejection thresholds increase exercise recall but the counter's motion/
prominence guards do not reject all false exercise intervals. Raw exercise
acceptance is monotone in threshold, but final segmentation/count metrics
need not be, because fixed minimum-run and sandwich rules change boundaries.
The result does not support replacing transition argmax with this single
global gate on the baseline RF probabilities.

At threshold 0.30, push-up/squat temporal recall also rises to 97.23%/95.94%
(baseline 95.75%/93.11%), alongside JJ's 90.25%. Thus all exercise recalls
can improve while all-class counting MAE worsens. At 0.55 those recalls fall
to 94.69%/89.55%. Recall alone does not capture the added false-exercise counts.

Artifacts:

- [Full 21-threshold sweep, class MAEs and signed-error balance](../../outputs/segmentation_v2/experiments/exercise_gate_counts_20261118/sweep.csv)
- [Selected and other-fold-selected reports](../../outputs/segmentation_v2/experiments/exercise_gate_counts_20261118/report.json)
- [Pre-evaluation decision rule, objective and provenance](../../outputs/segmentation_v2/experiments/exercise_gate_counts_20261118/plan.json)
- [Selected all-OOF per-video/class counts](../../outputs/segmentation_v2/experiments/exercise_gate_counts_20261118/selected_all_oof_counts.csv)
- [Other-fold-selected per-video/class counts](../../outputs/segmentation_v2/experiments/exercise_gate_counts_20261118/other_fold_selected_counts.csv)
- [All candidate segmentation windows](../../outputs/segmentation_v2/experiments/exercise_gate_counts_20261118/all_candidate_segments.csv)
- [Counted segments and completion timestamps](../../outputs/segmentation_v2/experiments/exercise_gate_counts_20261118/counted_segments.csv)
- [All-class temporal recall curve](../../outputs/segmentation_v2/experiments/exercise_gate_counts_20261118/all_class_temporal_recall.csv)
- [Threshold tradeoff plot](../../outputs/segmentation_v2/experiments/exercise_gate_counts_20261118/threshold_tradeoffs.png)

The plot shows threshold 0-0.70 detail; every threshold through 1.00 remains
in the sweep CSV. Input hashes passed unchanged checks; all 271 tests pass.
No labels, counter parameters, active model, defaults or prior results were
changed. The negative result is retained. Nothing was committed or pushed.

## JJ audit and concentration: original segmenter + experimental counters

The saved audit is reused, without rerunning inference/counting. This scope
has overall MAE 3.625 and JJ MAE 5.2417, not the newer feature/gate results.
For all 155 JJ windows (66 source clips), rate is annotated count divided by
`(end_frame-start_frame)/fps`. The requested <1 or >4 Hz review rule flags
77 windows from 33 sources across 60 videos: 75 below 1 Hz and two above 4 Hz
(both val568, 55 reps/9.8 s = 5.612 Hz). These are reproducible review flags,
not verified label errors; slow valid motion and labeling conventions remain
possible. No labels or primary scores are changed.

All 155 durations repeat through expected synthetic source reuse. Zero
annotations share the same video/start/end. The only exact duration shared
by distinct sources is 9.533333 s: stu7_28 (GT 10) and train1678 (GT 11),
each appearing twice in f2_0007/f2_0013 and f3_0005/f3_0015 respectively.
These four additional videos are review flags, not confirmed duplicate labels.

| Cohort | Videos | Overall MAE | JJ MAE | Overall median absolute error | JJ median absolute error |
|---|---:|---:|---:|---:|---:|
| Full original data | 120 | 3.625 | 5.242 | 1 | 1 |
| Without rate-flagged videos | 60 | 3.139 | 3.500 | 1 | 1 |
| Without cross-source-duration flags | 116 | 3.693 | 5.371 | 1 | 1 |
| Without either flag (union) | 56 | 3.244 | 3.643 | 1 | 1 |

Exclusions remove entire videos from all three class rows; overall has three
units per remaining video, JJ one. The unflagged cohorts retain 20 JJ-absent
videos: active-only JJ MAEs are 6.190 over 100 original active videos, 5.000
over 40 without rate flags and 5.389 over 36 without union flags. Different
cohorts/zero-count proportions prevent treating sensitivity gains as a label
correction or new headline score. Median JJ error on active videos is also 1.

Rank by absolute JJ video error, ties by lexical video ID. Total absolute
error is 629 reps (629/120 = 5.2417); 612 missed reps come from the separate
interval-matching diagnostic and are not the same quantity.

| Group | Absolute error | Share of 629 | Contribution to 5.242 MAE | Missed reps /612 | Audit overlap | Remaining JJ MAE |
|---|---:|---:|---:|---:|---:|---:|
| Top 5 | 225 | 35.8% | 1.875 | 231 (37.7%) | 4/5 | 3.513 /115 videos |
| Top 10 | 337 | 53.6% | 2.808 | 352 (57.5%) | 7/10 | 2.655 /110 videos |

Median JJ absolute error remains 1 after either top-error removal. Ranked
directly by missed-rep count instead, top 5/top 10 contribute 234/356 misses
(38.2%/58.2%); the main table uses absolute-error ranking consistently.

| Video suffix (prefix synth_v2_train_) | Signed JJ error | Missed reps | Audit flag |
|---|---:|---:|---|
| f3_0002 | -66 | 66 | Yes |
| f3_0018 | -62 | 63 | Yes |
| f1_0001 | -36 | 36 | Yes |
| f3_0021 | -31 | 36 | No |
| f3_0016 | -30 | 30 | Yes |
| f1_0016 | -26 | 33 | Yes |
| f3_0012 | -26 | 26 | No |
| f4_0002 | -26 | 28 | Yes |
| f3_0010 | -17 | 17 | Yes |
| f3_0017 | -17 | 17 | No |

Overlap is at video level, not necessarily the failing source. The large
stu5_25 miss in the two worst videos passes the rate check (1.164 Hz).
f3_0002 is flagged for stu3_21 at 0.759 Hz; f3_0018 for train1614 at 0.982 Hz.
Thus 4/5 overlap does not establish that annotation errors caused the misses.
Repeated use of stu5_25 is also not independent evidence.

Artifacts under `outputs/pushup_coordinates/jj_segmentation_audit_20261118/`:

- [Every window and its rate/duration flags](../../outputs/pushup_coordinates/jj_segmentation_audit_20261118/all_jj_window_audit.csv)
- [All flagged windows](../../outputs/pushup_coordinates/jj_segmentation_audit_20261118/flagged_windows.csv)
- [All 64 flagged videos, clips, rates and reasons](../../outputs/pushup_coordinates/jj_segmentation_audit_20261118/flagged_videos.csv)
- [Overall/JJ sensitivity and medians](../../outputs/pushup_coordinates/jj_segmentation_audit_20261118/overall_and_jj_mae_sensitivity.csv)
- [Top ten with signed errors and audit overlap](../../outputs/pushup_coordinates/jj_segmentation_audit_20261118/top_10_by_absolute_error.csv)
- [Concentration and denominator checks](../../outputs/pushup_coordinates/jj_segmentation_audit_20261118/concentration_summary.csv)
- [Derived completion and protected-input hashes](../../outputs/pushup_coordinates/jj_segmentation_audit_20261118/overall_completion.json)

The original audit/results are preserved. No label is fixed or silently
dropped; no code/model/counter change, commit or push is made for this
numerical completion.
