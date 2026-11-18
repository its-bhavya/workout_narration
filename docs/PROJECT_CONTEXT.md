# Active v2 construction protocol

## Rep-counting alternatives: methods and results (17 November 2026)

### Implemented logic and default status

As of 18 November 2026, the pipeline adopts the frozen training-selected
experimental counter combo as the default (`rep_method='experimental'`).
This combination uses unguided arm-raise peaks for jumping jacks, unguided
hip-angle peaks for squats, and frozen coordinate median voting for push-ups.
The legacy rule-based hysteresis state machine remains available via
`rep_method='state_machine'`. Upstream segmentation (windowing, classifier,
and OnlineTAS smoothing) remains unchanged.

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

Existing detail/artifact records: [peak experiment](results/PEAK_COUNTER_EXPERIMENT.md),
[peak settings, fold checks, and hashes](results/peak_counter_20261117/report.json),
[2D/3D experiment](results/SIGNAL_DIMENSIONS_COMPARISON.md),
[2D/3D settings, metrics, and hashes](results/signal_dimensions_20261117/report.json).
Plots and per-segment prediction CSVs are linked from those existing reports.

This dated section is the current rep-counter experiment record; older counter tables and workflow notes below are historical unless explicitly identified as current v2.

## Current full-flow handoff (17 November 2026)

Run instructions: [END_TO_END.md](END_TO_END.md). Current supported flow is
registered v2 media + cached canonical pose features -> frozen all-train RF
bundle -> smoothed exercise/transition segments -> fixed exercise-specific rep
counter -> text cue export -> separate narration event/speech scheduler -> WAV.
`main.py` does not synthesize audio. The runbook's API bridge uses predicted
boundaries and absolute rep completion times, retains zero-rep exercises and
omits transition segments from the exercise-only speech input. It does not use
the legacy ground-truth narration demo or its old splits.csv loader.

Deterministic templates and seed 42 are the default; an optional cached Gemini
phrase bank changes wording only. gTTS needs network access for uncached text,
and caches sentence MP3s. Scheduling may skip/delay cues and extend the track
beyond video duration; output is narration-only WAV, not a muxed MP4. No speech
quality or human-study result is claimed. This documentation update did not
invoke external synthesis or generate new benchmark outputs.

Model results are recorded in [V2_MODEL_RESULTS.md](results/V2_MODEL_RESULTS.md).
The [single-video rep diagnostic](results/V2_FULL_VIDEO_REP_RESULTS.json) gave
27 predicted versus 44 annotated reps; it is not an aggregate counting score.
Arbitrary unannotated-video ingestion remains disabled. Existing pose person
selection is annotation-assisted upstream; do not present the current flow as
validated unassisted raw-video deployment. V1 archive remains frozen.

Approved on 17 November 2026. The root PROJECT_CONTEXT.md contains historical
v1 results; its contributor/subject grouping claims do not define v2.

- RepCount only; status pass/review; fps >=20. Missing manual_exclusions.csv
  is an empty set and is logged as "no manual exclusions file found".
- Native original_split=train becomes train (239 clips: 66/81/92 for
  jump_jack/push_up/squat). Native val+test becomes test (94: 38/31/25).
- No separate validation split. Five source-clip folds inside train prevent
  source reuse across CV folds. Contributor prefixes are not subjects.

## Approved fold redraw

`data/manifests/folds.csv` uses StratifiedKFold(n_splits=5, shuffle=True,
random_state=3969), with train clips sorted by clip_id before assignment.
The former seed-42 assignments are preserved as
`data/manifests/folds_seed42_archived.csv`.

Seed 3969 was selected from seeds 43 through 10042, inclusive, by minimum
maximum absolute deviation from the mean total margin-frame count per fold;
ties use smaller standard deviation, then smaller seed. Margins are pre/post
repetition-core frames resampled to the constructor's 30 fps frame space,
counting each source clip once. Exercise stratification is mandatory. No model
results, test performance or model selection scores were used in this choice.
The five selected fold margin totals are 7484, 7379, 7293, 7364, 7237 frames.
This balances margin totals, not necessarily their fractions of clip duration.

## Approved render targets

- run_id=v2; 24 train videos per fold (120 total); cap 5 per train clip.
- 21 test videos; cap 1; unused test clips remain unused; balance is not relaxed.
- A dedicated random.Random(42) draws uniform K in {3,4,5}; the same K schedule
  is used independently in each train fold, and test starts its own stream.
- Train per fold: 13 K3, 2 K4, 9 K5; 92 slots, 31/31/30 exercise slots.
- Test: 12 K3, 2 K4, 7 K5; 79 slots, 27/27/25 exercise slots. The next draw
  would require 84 slots and at least 26 squat clips to retain +/-10% balance,
  exceeding the 25 available; no redraw of K or relaxation is allowed.
- Every train video uses only one fold; no repeated clip within a video;
  least-used selection keeps per-exercise usage spread <=1 within each fold.
- Whole clips, no inserted footage; 30 fps, 640x360 letterbox, margins labeled
  transition; adjacent exercises differ; exact exclusive-end segment JSON and
  clip_join_frames. Source decode failures abort instead of substituting footage.

V1 generated artifacts are preserved under data/_archive_v1 with original
repository-relative paths. V2 IDs contain run_id and train fold. The retained
batch_checkpoint.json stores the complete frozen plan, input SHA-256 hashes,
parameters, completed IDs and completion status. Re-running identical arguments
resumes; changed inputs are rejected. Interrupted media is archived, never deleted.

## Phase 4 preparation

Per-clip caching is explicitly deferred. The active extractor writes per-video
angle CSVs and provenance/summary sidecars under `data/features/v2/`. Run it at
stride 2, workers 1, IMAGE mode, resume only, without --force. Test videos are
extracted too; test is used for extraction and counting only, never decisions.

Run the resumable no-model workflow from the repository root:

```powershell
uv run --no-sync python -B src/data_pipeline/phase4.py
```

The first-video benchmark was 66.35 seconds; all 141 per-video features are now complete.
The runner resumes verified CSVs, logs per-video timing to
`outputs/segmentation_v2/pose_extraction.log`, rebuilds
`data/manifests/pose_extraction_summary_v2.csv`, lists has_long_gap videos,
and reports boundary-jump ratios by split without tuning any settings.

The existing sliding-window algorithm is retained (2-second window, 1-second
stride). Its index adds fold/run_id and is written to the new
`outputs/segmentation_v2/windows.csv`. Counts before/after train purity >=0.8
are reported by exercise, split and fold, explicitly including transition.
Test and held-out CV validation windows are not purity-filtered.

`src/segmentation/cv.py:get_cv_split(windows, k)` returns train videos from
folds !=k (training purity filter applied) and validation videos from fold k.
Both are drawn only from the train pool. Full source-clip sets must be disjoint
before purity filtering. Separate-val entry points were disabled during Phase 4; their source-clip CV
replacements are described below. No model was fitted/evaluated in Phase 4.

All active dataset readers fail on missing/non-v2 run_id or archived paths.
The final audit verifies complete feature/window coverage, source split/fold
membership, zero CV source overlap, absence of literal v1 video identifiers in
active artifacts, and an unchanged frozen archive. Historical prose is clearly
marked under docs/history/ with old video identifiers redacted.

Two summary bugs were verified and corrected: actual video JSONs consistently
use push_up (182 exercise segments), while the old summary searched for push;
and stride-2 gaps must use actual time spacing, not rows/30. Missing-pose runs
are now measured within each segment so separate segments cannot merge a gap.

The boundary diagnostic also used frame numbers as row positions, which fails
at stride 2. It now indexes jumps by the feature frame column. An unsampled
boundary maps to the next sampled frame whose preceding jump interval straddles
that boundary; shared intervals are counted once. Random comparison frames are
sampled frames, retaining the original +/-2-frame boundary exclusion, seed and
thresholds. The existing split aggregate remains the sample-count-weighted
average of per-video medians (an approximation, not the pooled global median).

Phase 4 extraction and final data audit completed successfully; details follow.

## Phase 4 completed run

V2 per-video features: 141 videos, stride 2, workers 1, IMAGE mode, resume only. Windows: 17607. No model training/evaluation. Summary, long-gap list, boundary ratios, window counts and CV partition counts are saved in outputs/segmentation_v2/. Consistency checks passed; archive unchanged.

## CV training/evaluation/integration compatibility ? 17 November 2026

Authoritative details and commands: [V2_TRAINING.md](V2_TRAINING.md).

The retired separate-val scripts have been replaced by train-only five-fold
source-clip CV. Candidate ranking and smoothing choice use mean fold per-video
Edit, then frame accuracy, with deterministic tie breaks. They never use test
scores. Each fold fits fresh median imputation and, for LR/SVM/KNN/MLP, scaling;
training purity defaults to 0.8 while validation remains unfiltered. MLP/HistGBT
random window-level early stopping is disabled. A separate explicit all-train
refit freezes the selected pipeline and smoothing; final test is explicit and
never reselects parameters. Selection-CV scores are not unbiased nested-CV scores.

Dataset run_id=v2 is separate from model experiment_id. Versioned experiments,
input/feature/OOF hashes, all-train bundle guards, and matching model/parameter
hashes protect the handoff. Active-model pointers retain prior versions.
Prediction, evaluation and narration now share one inference implementation and
use n_frames/fps as the exact video end. GT comparison is opt-in, with train
comparisons labeled in-sample. Rep-counter thresholds were not changed.

The real train schema audit found ankle_dist_norm_3d absent in 2/120 CSVs and
all-NaN in the other 118; the extractor has no corresponding computed 3D
measurement. Approved resolution: fixed permitted 112-feature schema, missing
columns aligned to NaN in memory, fold-trained imputation/zero fallback, no
feature rewrites or re-extraction. The frozen archive and data assignments are
unchanged. Validation uses synthetic fit/refit/integration fixtures and a real
read-only train feature/CV preflight; no benchmark model sweep or test evaluation
is part of this change.

The existing extractor still uses annotated exercise/segment labels for person
selection. Classifier input allow-lists and label-independent inference do not
remove that upstream assistance. Paper claims remain scoped to an offline,
annotation-assisted synthetic benchmark; unassisted continuous-workout testing
is future work. Historical v1 metrics are not results for v2.

Compatibility preflight passed on the real train pool: 14,879 windows x 112
features, 120 videos, all five source-fold partitions valid. Test angle files
were not read; no real model or activation was written. Classifier settings and
software versions are recorded with future selections/refits. The frozen archive
fingerprint still matches the completed Phase 4 audit.

Validation for this compatibility change: 175 tests passed, including synthetic
fit/refit/final-evaluation/integration fixtures. Real benchmark training and test
evaluation were not run.

<!-- v2-model-results:start -->
## Completed v2 model experiment — 17 November 2026

8-model train-only CV selected **rf** (Edit 71.54 ± 2.90 fold SD; tau=0.5, min_len=3). After all-train refit, only that frozen winner was evaluated on 21 test videos: Edit 72.51, frame accuracy 86.66%, segment F1@25 0.750. Test was not used for model selection or tuning. See [full results, caveats and reproducibility tables](results/V2_MODEL_RESULTS.md).
<!-- v2-model-results:end -->
