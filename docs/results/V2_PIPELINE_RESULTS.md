# V2 pipeline methods and evaluation results

The current pipeline uses cached v2 pose features and the frozen random forest with 2-second windows, 1-second stride, and OnlineTAS smoothing (tau=0.5, min_len=3). The default rep counter uses unguided 2D arm-raise peaks for jumping jacks, unguided confidence-weighted 2D hip-angle peaks for squats, and frozen coordinate median voting for push-ups. The CLI exposes the earlier state machine via `--rep-method state_machine`. Narration schedules speech from predicted completion times using measured TTS clip durations and ends at the video duration. See the [runbook](../END_TO_END.md).

## Rep counting

Each method cell shows **predicted reps / MAE / exact match / within ±1 rep**. Exact and within-one are percentages of evaluation units. GT counts are shared by all methods. On GT intervals, each unit is one annotated exercise segment with zero end padding. After segmentation, each unit is one video/exercise total, including absent-exercise zeros, using 0.3 s of counting end padding. Overall pools the exercise units. Whole workout sums exercises per video before scoring, so errors can cancel.

CV uses 120 synthetic train videos across five source-disjoint folds. GT tables pool segment occurrences, including source reuse. CV results reflect development selection. Earlier training reports also give source-weighted MAE. The current-family CV uses settings selected on other folds. The current test result uses the frozen all-training settings. Test comparisons reuse the same 21 inspected videos and are exploratory follow-ups.

The state-machine column uses the motion-aware legacy counter in CV and the local-adaptive follow-up in test, which predates motion-aware arm selection. The original test state machine is shown separately. The earlier 2D peaks use the angle-only unguided detector. Refined 2D peaks are the intermediate comparator from the frozen coordinate test, with refined jumping-jack and squat signals/settings and no coordinate push-up voting.

### Training CV on GT segments

| Exercise | Units | GT reps | State machine | 2D peaks | Refined 2D peaks | Current combination |
|---|---:|---:|---:|---:|---:|---:|
| Jumping jacks | 155 | 3,686 | 3,019 / 5.400 / 60.65% / 73.55% | 3,257 / 4.135 / 60.65% / 74.19% | 3,526 / 2.645 / 59.35% / 70.97% | 3,526 / 2.645 / 59.35% / 70.97% |
| Push-ups | 155 | 3,112 | 2,995 / 2.974 / 36.13% / 49.68% | 3,007 / 2.923 / 33.55% / 46.45% | 3,015 / 3.090 / 33.55% / 46.45% | 3,113 / 1.981 / 53.55% / 64.52% |
| Squats | 150 | 1,856 | 1,672 / 1.533 / 64.67% / 76.00% | 1,788 / 1.187 / 64.00% / 79.33% | 1,785 / 0.940 / 70.67% / 87.33% | 1,785 / 0.940 / 70.67% / 87.33% |
| Overall | 460 | 8,654 | 7,686 / 3.322 / 53.70% / 66.30% | 8,052 / 2.765 / 52.61% / 66.52% | 8,326 / 2.239 / 54.35% / 68.04% | 8,424 / 1.865 / 61.09% / 74.13% |

### Training CV after segmentation

| Exercise | Units | GT reps | State machine | 2D peaks | Refined 2D peaks | Current combination |
|---|---:|---:|---:|---:|---:|---:|
| Jumping jacks | 120 | 3,686 | 2,857 / 7.942 / 41.67% / 59.17% | 3,012 / 7.050 / 35.00% / 56.67% | 3,269 / 5.242 / 33.33% / 58.33% | 3,269 / 5.242 / 33.33% / 58.33% |
| Push-ups | 120 | 3,112 | 2,984 / 3.717 / 51.67% / 59.17% | 2,994 / 3.150 / 50.00% / 60.83% | 3,039 / 3.042 / 49.17% / 60.00% | 3,119 / 2.925 / 51.67% / 60.00% |
| Squats | 120 | 1,856 | 1,732 / 2.683 / 45.00% / 59.17% | 1,842 / 2.667 / 34.17% / 57.50% | 1,879 / 2.708 / 38.33% / 65.00% | 1,879 / 2.708 / 38.33% / 65.00% |
| Overall | 360 | 8,654 | 7,573 / 4.781 / 46.11% / 59.17% | 7,848 / 4.289 / 39.72% / 58.33% | 8,187 / 3.664 / 40.28% / 61.11% | 8,267 / 3.625 / 41.11% / 61.11% |
| Whole workout | 120 | 8,654 | 7,573 / 12.092 / 5.83% / 25.83% | 7,848 / 10.650 / 7.50% / 25.00% | 8,187 / 8.825 / 6.67% / 24.17% | 8,267 / 8.458 / 7.50% / 25.00% |

### Test on GT segments

| Exercise | Units | GT reps | Original state machine | State machine | 2D peaks | Refined 2D peaks | Current combination |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jumping jacks | 27 | 761 | 732 / 1.963 / 44.44% / 70.37% | 732 / 1.963 / 44.44% / 70.37% | 756 / 1.370 / 51.85% / 77.78% | 760 / 1.444 / 51.85% / 77.78% | 760 / 1.444 / 51.85% / 77.78% |
| Push-ups | 27 | 468 | 354 / 4.593 / 44.44% / 59.26% | 410 / 4.148 / 44.44% / 62.96% | 420 / 3.778 / 48.15% / 59.26% | 420 / 3.778 / 48.15% / 59.26% | 435 / 2.481 / 55.56% / 70.37% |
| Squats | 25 | 361 | 313 / 1.920 / 68.00% / 76.00% | 313 / 1.920 / 68.00% / 76.00% | 339 / 1.360 / 48.00% / 88.00% | 335 / 1.200 / 68.00% / 84.00% | 335 / 1.200 / 68.00% / 84.00% |
| Overall | 79 | 1,590 | 1,399 / 2.848 / 51.90% / 68.35% | 1,455 / 2.696 / 51.90% / 69.62% | 1,515 / 2.190 / 49.37% / 74.68% | 1,515 / 2.165 / 55.70% / 73.42% | 1,530 / 1.722 / 58.23% / 77.22% |

### Test after segmentation

| Exercise | Units | GT reps | Original state machine | State machine | 2D peaks | Refined 2D peaks | Current combination |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jumping jacks | 21 | 761 | 747 / 2.952 / 38.10% / 57.14% | 747 / 2.952 / 38.10% / 57.14% | 782 / 2.714 / 28.57% / 52.38% | 786 / 2.905 / 23.81% / 52.38% | 786 / 2.905 / 23.81% / 52.38% |
| Push-ups | 21 | 468 | 362 / 5.333 / 61.90% / 61.90% | 419 / 4.714 / 61.90% / 66.67% | 430 / 4.381 / 52.38% / 61.90% | 430 / 4.381 / 52.38% / 61.90% | 452 / 3.143 / 57.14% / 71.43% |
| Squats | 21 | 361 | 332 / 2.143 / 38.10% / 76.19% | 332 / 2.143 / 38.10% / 76.19% | 357 / 2.381 / 28.57% / 52.38% | 349 / 2.095 / 38.10% / 61.90% | 349 / 2.095 / 38.10% / 61.90% |
| Overall | 63 | 1,590 | 1,441 / 3.476 / 46.03% / 65.08% | 1,498 / 3.270 / 46.03% / 66.67% | 1,569 / 3.159 / 36.51% / 55.56% | 1,565 / 3.127 / 38.10% / 58.73% | 1,587 / 2.714 / 39.68% / 61.90% |
| Whole workout | 21 | 1,590 | 1,441 / 9.762 / 0.00% / 14.29% | 1,498 / 9.143 / 0.00% / 14.29% | 1,569 / 7.190 / 0.00% / 19.05% | 1,565 / 7.762 / 0.00% / 9.52% | 1,587 / 6.905 / 0.00% / 14.29% |

### Rep-count sources

Links under `outputs/` refer to local experiment artifacts. The new replay and segment-count tables are stored with this report.

- **CV GT**. [State-machine counts](peak_counter_20261117/baseline_segments.csv), [refined cached-angle counts](../../outputs/peak_improvements/verified_cached_signal_comparison_20261118/heldout_segments.csv), and [coordinate median family counts](../../outputs/pushup_coordinates/train_v1_validation_20261118/family_heldout_predictions.csv). The [earlier 2D peak replay](counter_comparison_20261118/cv_2d_peaks_gt.csv) uses the published unguided candidate per held-out fold and reproduces the source-weighted MAEs in the [peak report](peak_counter_20261117/report.json).
- **CV post-segmentation**. [Saved OOF state-machine and current-family comparison](../../outputs/pushup_coordinates/segmented_training_cv_20261118/report.json). The [earlier 2D peak replay](counter_comparison_20261118/cv_2d_peaks_post_segmentation.csv) uses the same saved OOF intervals and the same other-fold settings as the GT replay. The [refined comparison](counter_comparison_20261118/cv_refined_2d_peaks_post_segmentation.csv) reuses saved jumping-jack and squat counts and replays only the earlier cached-feature push-up method with its published other-fold settings.
- **Test**. [Original state machine](V2_REP_COUNT_RESULTS.md), [local-adaptive state machine](pushup_local_20261117/report.json), [2D peaks](../../outputs/peak_improvements/test_followup_20261118/report.json), and [refined peaks and frozen coordinate combination](../../outputs/pushup_coordinates/exploratory_test_frozen_v1_20261118/report.json).

## Segmentation

All rep methods use the same segmenter. CV is averaged across the five held-out folds. Test is averaged across the 21 videos using the frozen all-train refit. Segment F1 measures labeled overlap at the listed IoU threshold.

| Metric | Training CV | Test |
|---|---:|---:|
| Frame accuracy | 85.11% | 86.66% |
| Edit | 71.537 | 72.515 |
| Segment F1@10 | 0.756 | 0.791 |
| Segment F1@25 | 0.704 | 0.750 |
| Segment F1@50 | 0.618 | 0.636 |
| Frame F1 · Jumping jacks | 0.714 | 0.764 |
| Frame F1 · Push-ups | 0.497 | 0.487 |
| Frame F1 · Squats | 0.858 | 0.865 |
| Frame F1 · Transition | 0.526 | 0.484 |

### Segment counts per video

Each cell shows **GT segments / predicted segments / MAE / exact match / within ±1 segment**. Overall counts all labels per video. Count agreement complements the label and boundary scores above.

| Label | Training CV (120 videos) | Test (21 videos) |
|---|---:|---:|
| Jumping jacks | 155 / 146 / 0.175 / 82.50% / 100.00% | 27 / 29 / 0.095 / 90.48% / 100.00% |
| Push-ups | 155 / 154 / 0.042 / 95.83% / 100.00% | 27 / 28 / 0.048 / 95.24% / 100.00% |
| Squats | 150 / 148 / 0.150 / 88.33% / 96.67% | 25 / 27 / 0.095 / 90.48% / 100.00% |
| Transitions | 554 / 331 / 1.925 / 8.33% / 38.33% | 93 / 54 / 1.857 / 14.29% / 33.33% |
| Overall | 1,014 / 779 / 2.225 / 6.67% / 31.67% | 172 / 138 / 1.905 / 14.29% / 28.57% |

See [per-video segment counts](counter_comparison_20261118/segmentation_counts.csv). Sources are the [saved OOF metrics and intervals](../../outputs/pushup_coordinates/segmented_training_cv_20261118/) and the [frozen segmentation test evaluation](../../outputs/segmentation_v2/experiments/cv8_v2_seed42_p080_20261117/final/test_evaluation/). See the [model comparison](V2_MODEL_RESULTS.md) for the full selection and test protocol.
