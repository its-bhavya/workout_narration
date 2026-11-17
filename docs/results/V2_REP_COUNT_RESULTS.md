# V2 synthetic test-set rep counting ? 17 November 2026

All 21 indexed synthetic test videos were evaluated using the frozen all-train model and unchanged counter settings. No test tuning, retraining, or narration synthesis was performed.

## End-to-end segmentation ? counting

The primary unit is one video ? exercise (63 units: 21 videos ? 3 exercise classes). Repeated segments of the same exercise are summed within each video. Absent exercise classes remain included, including zero/zero matches and false positives. This evaluates class totals, not temporal rep matching.

| Exercise | Annotated reps | Predicted reps | MAE (reps/unit) | Exact | Within ?1 | WAPE |
|---|---:|---:|---:|---:|---:|---:|
| jump_jack | 761 | 747 | 2.95 | 38.10% | 57.14% | 8.15% |
| push_up | 468 | 362 | 5.33 | 61.90% | 61.90% | 23.93% |
| squat | 361 | 332 | 2.14 | 38.10% | 76.19% | 12.47% |
| Overall | 1590 | 1441 | 3.48 | 46.03% | 65.08% | 13.77% |

Summing all exercises per video gives MAE **9.76 reps/video**, exact match **0.00%**, and within ?1 **14.29%**. Video totals can hide errors cancelling across exercise classes.

WAPE is the sum of absolute unit errors divided by total annotated reps; it is not rep detection accuracy. MAE, exact match, and within-one depend on the evaluation unit.

## Counter-only diagnostic using ground-truth segments

The same counter runs within each of the 79 annotated exercise windows, with no end padding. End-to-end uses predicted windows and 0.3 seconds of padding. These have different units and boundary conditions, so their MAEs should not be interpreted as a direct segmentation penalty.

| Exercise | GT segments | Annotated reps | Counted reps | MAE (reps/segment) | Exact | Within ?1 |
|---|---:|---:|---:|---:|---:|---:|
| jump_jack | 27 | 761 | 732 | 1.96 | 44.44% | 70.37% |
| push_up | 27 | 468 | 354 | 4.59 | 44.44% | 59.26% |
| squat | 25 | 361 | 313 | 1.92 | 68.00% | 76.00% |
| Overall | 79 | 1590 | 1399 | 2.85 | 51.90% | 68.35% |

Push-ups have the largest errors in both evaluations, indicating that counting errors remain even with annotated windows. These results do not establish unassisted raw-video performance: upstream pose extraction remains annotation-assisted. Test results are for reporting; future parameter selection must use training data.

## Reproduce and inspect

```powershell
uv run --no-sync python -m src.rep_counting.evaluate_final --out-dir outputs/rep_eval_v2/new_final_test_run
```

The command refuses an existing output directory and fails on missing test inputs instead of silently skipping videos. The active model must match its frozen training provenance.

- [Machine-readable report](rep_counting_v2_test_20261117/report.json): model hash, input hashes, fixed counter configuration, metrics, and limitations.
- [Per-video exercise counts](rep_counting_v2_test_20261117/video_exercise.csv).
- [Per-video total counts](rep_counting_v2_test_20261117/video_totals.csv).
- [Ground-truth segment diagnostic](rep_counting_v2_test_20261117/oracle_segments.csv).

Full predicted segments, counter diagnostics, and rep timestamps are saved locally in `outputs/rep_eval_v2/final_test_20261117/predictions.json`. The files linked above are kept in docs because outputs are ignored by Git.

Experiment: `cv8_v2_seed42_p080_20261117`. Model SHA-256: `989cf80af9e7642c9c6593f53e43c64be64a87ea4cd9bce9416fbb1f807c3e7b`.
