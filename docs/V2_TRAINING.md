# V2 training, final evaluation, and integration

Implemented 17 November 2026. This change validates compatibility, not benchmark
performance: models are fitted only on small synthetic unit-test fixtures.
The real v2 benchmark receives a read-only feature/CV preflight. No production
model sweep, all-train fit, test prediction, or test evaluation was run here.

## Locked data protocol

RepCount pass/review clips, fps >=20, no manual exclusions: 239 native-train
clips and 94 native-val+test clips. There is no separate val split. Source-clip
folds are fixed in folds.csv (stratified seed 3969, selected by margin balance
only). Video construction uses seed 42: 120 train videos, 24 per fold, cap 5;
21 test videos, cap 1, using 79 clips and intentionally leaving 15 unused.
Exercise slot balance is within +/-10%. Contributor prefixes are not subjects.

Inputs remain splits_v2.csv, folds.csv, synthetic_index.csv, per-video
data/features/v2/ CSVs, and outputs/segmentation_v2/windows.csv. Every dataset
reader retains run_id=v2, source split/fold checks, and archive rejection.
The frozen data/_archive_v1 directory is never modified.

## Model selection and preprocessing

For outer/source fold k, eligible training windows come from train folds !=k;
validation uses every window of train fold k. Source-clip overlap is asserted
before the training purity filter. Test angle files are never opened for CV or
refitting. Test metadata may be validated to enforce dataset provenance.

The existing 2-second windows, 1-second stride, 0.5-second short-gap fill and
seven summary statistics are unchanged. The fixed allow-list is 16 channels /
112 features; labels, split, fold, run_id, visibility and person diagnostics
cannot become model inputs. Absent permitted measurements are aligned to NaN
in memory. Source CSVs are not rewritten. In the train-only schema audit,
ankle_dist_norm_3d was absent in 2/120 files and all-NaN in the other 118: the
extractor creates this placeholder in visibility masking but computes no 3D
ankle-normalized distance. It remains a missing-value placeholder, not an
invented measurement.

Every fold gets a fresh sklearn pipeline. Median imputation is fitted only on
that fold's eligible training windows, with zero fallback for entirely missing
features and stable feature count. LR, SVM, KNN and MLP additionally fit a
StandardScaler on those same windows. RF, ExtraTrees, GBT and HistGBT do not
scale. MLP/HistGBT automatic window-level early stopping is disabled. SVC's
built-in probability calibration uses only its fitting partition; it is not a
separately evaluated source-independent calibration benchmark.

Each classifier/purity candidate produces out-of-fold predictions for all train
windows. Smoothing is selected from the train OOF predictions. Rank candidates
by mean of five fold mean per-video Edit scores, then mean frame accuracy;
remaining ties use lexical classifier, higher purity, lower tau, lower min_len.
No test metric participates. Default purity is 0.8; an explicitly supplied
purity grid is also selected only inside train CV. Window accuracy, macro-F1,
per-class confusion matrices, majority baseline, temporal metrics and fold
variability are retained in reports.

These are **selection-CV scores**, not an unbiased nested-CV estimate. Fold
standard deviations describe variation across the fixed folds; they are not
subject confidence intervals. Final test metrics are reported separately.

## Commands (run manually from repository root)

Read-only compatibility check, no fitting:

```powershell
uv run --no-sync python -B src/segmentation/train.py --check-only
```

Single classifier CV, or a train-only multi-classifier sweep:

```powershell
uv run --no-sync python -B src/segmentation/train.py --model rf --experiment-id rf_cv_v2
uv run --no-sync python -B src/segmentation/hparam_sweep.py --quick --experiment-id quick_cv_v2
```

Choose one of these experiments; do not run both just to inspect test outcomes.
Each experiment directory is new and immutable-by-default; an existing ID is
rejected. Partial/old runs must be preserved or archived before reusing a path.

Report the selected OOF scores without test evaluation:

```powershell
uv run --no-sync python -B src/segmentation/evaluate.py --selection outputs/segmentation_v2/experiments/rf_cv_v2/selection.json
```

Refit the frozen selected candidate on all eligible train windows and activate
it for integration (no CV rerun, no test predictions):

```powershell
uv run --no-sync python -B src/segmentation/train.py --selection outputs/segmentation_v2/experiments/rf_cv_v2/selection.json --activate
```

Only after the method is fixed, explicitly run final test evaluation:

```powershell
uv run --no-sync python -B src/segmentation/evaluate.py --final-test --model outputs/segmentation_v2/experiments/rf_cv_v2/final/model.joblib
```

This scores frozen raw/smoothed predictions, with no search or ranking, and
preserves outputs in final/test_evaluation/. Existing evaluation output is
rejected rather than overwritten. The 21 test videos are a modest synthetic
benchmark, not a verified subject-independent or continuous-workout benchmark.

## Artifacts and integration

run_id always means dataset protocol v2; experiment_id identifies the model
experiment. Each experiment stores cv_results.csv, per-candidate OOF/window
reports, selected oof_predictions.csv, cv_video_metrics.csv, cv_fold_metrics.csv
and selection.json. Classifier settings and Python/sklearn/numpy/pandas versions
are recorded; refit rejects changed classifier settings. Training metadata/window/feature hashes bind selection to
refit; selected OOF predictions are hashed as well.

The final/model.joblib bundle contains the fitted preprocessing pipeline,
feature order, window/gap/extraction contract, train-input signature, frozen
smoothing choice, classes and experiment ID. chosen_params.json is optional
for consumers: if supplied it must match experiment ID, model SHA-256 and the
frozen choice. No RF-first fallback or silently chosen default smoothing remains.
Activation writes active_model.json with the model hash; prior pointers move
to activation_history rather than being deleted.

Prediction, final evaluation and narration share workflow.infer_frame. They
use n_frames/fps for the exact video end, and ground-truth labels are replaced
by placeholder window metadata before prediction. Registered feature inputs
and matching all-train bundles are required; on-the-fly/native-clip extraction
remains deferred. Ground-truth comparison is opt-in, and train-demo comparisons
are labeled in-sample. Explicit tau/min_len inputs must match the frozen choice.

```powershell
uv run --no-sync python -B main.py synth_v2_train_f0_0001
uv run --no-sync python -B main.py synth_v2_train_f0_0001 --show-gt
uv run --no-sync python -B src/segmentation/predict_video.py synth_v2_train_f0_0001
```

For the complete prediction -> rep counting -> scheduled audio recipe, see
[END_TO_END.md](END_TO_END.md). The commands above produce text cues, not speech;
the audio bridge consumes the exported predicted segments, not ground truth.

Integration exports dataset/model provenance and the same 10-Hz temporal metrics
as final evaluation when comparison is requested. It retains rule-based rep
counting and narration; no rep thresholds were tuned in this compatibility work.

## Research limitation carried forward

The existing pose extractor uses annotated exercise/segment labels to help
select the exercising person. Removing labels from classifier inputs does not
remove that upstream assistance. Describe the current results as an offline,
annotation-assisted synthetic benchmark. Unassisted raw-video person selection
and genuinely continuous workout validation remain future work. Historical v1
metrics are not v2 results and must not be reused as evidence for this protocol.

Before replacement, model workflow source was preserved in
docs/history/segmentation_before_cv/. It is historical, not an active CLI.

## Compatibility validation recorded

The real read-only train preflight passed: 120 videos, 14,879 windows, 112
features; held-out fold counts 2574/3503/2782/2721/3299 and eligible fitting counts
11662/10744/11437/11510/10939. All source-clip overlaps are zero; no test angle
file is opened by that preflight. Unit/integration tests exercise CV, leakage
rejection, refit, final-test reporting and narration with temporary synthetic
fixtures, not repository benchmark fits. The frozen archive fingerprint matches
the completed Phase 4 report. No real model/activation was written in this work.

Validation for this compatibility change: 175 tests passed, including synthetic
fit/refit/final-evaluation/integration fixtures. Real benchmark training and test
evaluation were not run.

<!-- v2-model-results:start -->
## Completed v2 model experiment — 17 November 2026

8-model train-only CV selected **rf** (Edit 71.54 ± 2.90 fold SD; tau=0.5, min_len=3). After all-train refit, only that frozen winner was evaluated on 21 test videos: Edit 72.51, frame accuracy 86.66%, segment F1@25 0.750. Test was not used for model selection or tuning. See [full results, caveats and reproducibility tables](results/V2_MODEL_RESULTS.md).
<!-- v2-model-results:end -->
