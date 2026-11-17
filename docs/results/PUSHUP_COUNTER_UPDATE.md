# Push-up counter update (17 November 2026)

Push-up counting now adapts thresholds to a six-second local p10/p90 elbow-angle
range, independently inside each uninterrupted tracking interval. This follows
changes in angle baseline and range of motion that a single segment-wide pair
of thresholds can miss. Low local motion is masked, missing-data gaps reset the
counter, and the existing visibility, amplitude, and minimum-spacing guards remain.
Squat and jumping-jack logic, pose features, and segmentation weights are unchanged.

The selected normalized thresholds are 0.45 / 0.75 with single-sample completion
confirmation. This is an offline method: centered local windows use lookahead.
Explicit angle thresholds and fixed mode still bypass local adaptation. Exported
segment diagnostics identify local adaptation; `thresholds` is null because no
single pair of angle thresholds describes the whole segment.

## Training selection

The reproducible search uses training features only: 155 push-up segments,
weighted equally by unique source clip within each of five source-disjoint folds.
Candidates vary local window (2/4/6/8 seconds), lower fraction (0.2/0.35/0.45),
upper fraction (0.55/0.65/0.75), and confirmation (0/0.1 seconds), plus the original
segment-wide baseline. Other counter settings are held fixed. Minimum mean
fold MAE selects the configuration with deterministic candidate-order ties.

Source-weighted mean fold MAE decreased from **4.07 to 3.02 reps** (25.8%).
For each held-out fold, settings selected on the other four folds also give mean
MAE **3.02**, versus **4.07** baseline. Fold 2 regresses from 2.28 to 2.91.
These checks support the change, but are not an unbiased nested-CV estimate:
algorithm development also inspected training results. Earlier exploratory
training-only global-threshold and local-peak candidates were less effective.
No left/right side-selection change was made.

## Follow-up evaluation on the existing test set

The following are follow-up results on the same 21 synthetic test videos already
inspected during baseline analysis, not a new untouched holdout. The configuration
was selected from training data and frozen before this follow-up. No settings
were changed after observing the follow-up test results. Original baseline files
remain unchanged.

| Metric | Original baseline | Updated |
|---|---:|---:|
| Push-up predicted reps / 468 annotated | 362 | 419 |
| Push-up MAE per video (21 videos) | 5.33 | 4.71 |
| Push-up WAPE | 23.93% | 21.15% |
| Push-up within one rep | 61.90% | 66.67% |
| Overall MAE per video/exercise (63 units) | 3.48 | 3.27 |
| Overall WAPE | 13.77% | 12.96% |
| Total predicted / 1590 annotated | 1441 | 1498 |
| Push-up GT-window diagnostic MAE (27 segments) | 4.59 | 4.15 |

Push-up video MAE improves by 11.6%. Of the 21 video/exercise units, 6 improve,
1 regresses, and 14 are unchanged. There are 10 zero/zero push-up units; on the
11 videos containing annotated push-ups, MAE changes from **10.18 to 9.00**.
Better total counts do not imply every rep is timed correctly. Some clips still
overcount, and missing pose data remains unresolved. Upstream pose extraction
is annotation-assisted. A fresh holdout is needed for a new final generalization claim.

## Reproduce and inspect

```powershell
.venv/Scripts/python.exe -m src.rep_counting.calibrate_pushups --out-dir outputs/rep_pushup_dev/new_selection
.venv/Scripts/python.exe -m src.rep_counting.evaluate_final --out-dir outputs/rep_eval_v2/new_followup
```

- [Training selection, source hashes, fold checks](pushup_local_20261117/selection.json)
- [All calibration candidates](pushup_local_20261117/candidates.csv)
- [Selected training predictions](pushup_local_20261117/selected_segments.csv)
- [Follow-up metrics, settings, model and input hashes](pushup_local_20261117/report.json)
- [Per-video comparison](pushup_local_20261117/comparison.csv)
- [Oracle segment diagnostics](pushup_local_20261117/oracle_segments.csv)
- [Original baseline](V2_REP_COUNT_RESULTS.md)

Full follow-up prediction timestamps and diagnostics remain locally under
`outputs/rep_eval_v2/pushup_local_followup_20261117/predictions.json`.
