# src/evaluation
Owned by: Ishita (Role 3)
Status: segmentation/count metric utilities are implemented. Active v2 rep-count
runners in `src/rep_counting` now evaluate canonical predictions, synthetic
annotations, training-fold counter alternatives, and rep-completion timing.
The earlier waiting-for-prediction-format note is retired.

## Input format
Segments are `(label, start_frame, end_frame)`, end EXCLUSIVE, contiguous, covering `[0, n_frames)` (see data/README.md, synthetic videos). Labels: `squat`, `push_up`, `jump_jack`, `transition`.

## Metrics (`segmentation_metrics.py`)
| Function | Meaning |
|----------|---------|
| `frame_accuracy` | share of frames with the right label |
| `edit_score` | 1 - normalised Levenshtein distance between label sequences; penalises over-segmentation |
| `f1_at_k(k)` | segment F1 at IoU >= k/100 (k = 10, 25, 50); a hit needs the same label |
| `boundary_error` | seconds between each ground-truth boundary and the nearest predicted boundary of the SAME type (same label before and after). Unmatched boundaries are counted in `n_missed` and excluded from the mean |
| `evaluate_segmentation` | all of the above in one dict |

Every label counts, including `transition`. Filter the segment lists first to score exercises only.

`rep_metrics.py`: exact match, off-by-one accuracy, MAE, normalised MAE, bias.

## V2 rep-counter experiments and metric interpretation

The existing peak and 2D/3D studies evaluate 460 GT-window training segments
across five source-disjoint folds. Select settings on four folds and score the
fifth. MAE gives each source clip one weight within its fold, then weights
folds equally. Keep those results distinct from video/exercise totals produced
by `src.rep_counting.evaluate_final` on the 21 test videos.

Peak logic: prominence and width identify extrema; capped autocorrelation may
increase spacing when a reliable period exists; a 70%-of-prominence recovery
marks completion. Missing-data runs are independent. The separate 2D/3D study
uses actual cached world-3D columns, forbids 2D fallback, retains unusable
segments as zero counts, and compares fixed settings before separate tuning.

Peak-family MAE with other-fold settings:

| Exercise | Current 2D state machine | Guided peaks | Unguided peaks |
|---|---:|---:|---:|
| Jumping jacks | 5.288 | 4.413 | 4.204 |
| Push-ups | 2.865 | 2.981 | 2.827 |
| Squats | 1.594 | 1.153 | 1.150 |

Separately tuned dimensional comparison (full peak search, both guidance modes):

| Exercise | State machine 2D | State machine 3D | Peaks 2D | Peaks 3D |
|---|---:|---:|---:|---:|
| Jumping jacks | 4.562 | 4.913 | 4.435 | 5.414 |
| Push-ups | 2.865 | 5.205 | 2.993 | 4.968 |
| Squats | 1.370 | 0.928 | 1.197 | 0.992 |

Timing precision/recall/F1 match completions one-to-one to annotated intervals
expanded by 0.3 seconds. Timing metrics pool segment occurrences, including
source reuse; exact match, within-one, and WAPE in the dimensional report use
the same pooled occurrences. They are different aggregations from source-weighted
fold MAE. Precision is matched/predicted, recall is matched/annotated, and F1
is their harmonic mean. WAPE is count error, not a detection-accuracy score.


State-machine and peak defaults have not been changed based on these studies.
These are exploratory development comparisons, not unbiased nested-CV estimates
or new final-test results. Upstream pose person selection is annotation-assisted.
For complete logic, search spaces, coverage, timing tables, and commands see
[the rep-counting README](../rep_counting/README.md),
[peak results](../../docs/results/PEAK_COUNTER_EXPERIMENT.md), and
[2D/3D results](../../docs/results/SIGNAL_DIMENSIONS_COMPARISON.md).

## Run
```
uv run python -m pytest tests/test_evaluation.py
```

## TODO
- Evaluate a deliberately frozen alternative through predicted segmentation
  before claiming an integrated counting improvement; retain prior test records.
- Human-study rating form and fixed-interval baseline (with narration).
