# V2 8-model comparison — 17 November 2026

For the active segmenter summary, per-label segment counts, and rep-counter comparisons, see the [v2 pipeline results](V2_PIPELINE_RESULTS.md).

Experiment: `cv8_v2_seed42_p080_20261117`. Status: complete.

All 8 classifiers were compared using five source-clip folds inside train. Purity=0.8; seed=42; tau grid {0.4,0.5,0.6,0.7}; min_len grid {1,2,3,4}. Selection maximizes mean fold per-video Edit, then frame accuracy, using the documented deterministic ties. Test features were read only after selection and all-train refit.

| Model | CV Edit mean ± fold SD | CV frame accuracy | CV segment F1@25 | tau | min_len | Fit/predict seconds (5 folds) |
|---|---:|---:|---:|---:|---:|---:|
| rf | 71.54 ± 2.90 | 85.11% | 0.704 | 0.5 | 3 | 18.1 |
| mlp | 71.51 ± 1.78 | 83.66% | 0.684 | 0.4 | 3 | 101.3 |
| lr | 71.42 ± 2.42 | 84.47% | 0.701 | 0.6 | 3 | 284.3 |
| hist_gbt | 70.28 ± 2.11 | 85.70% | 0.703 | 0.5 | 3 | 95.4 |
| svm | 68.89 ± 1.63 | 84.41% | 0.714 | 0.6 | 4 | 55.0 |
| gbt | 68.68 ± 3.15 | 85.37% | 0.700 | 0.5 | 3 | 512.1 |
| extra_trees | 68.62 ± 3.47 | 85.01% | 0.698 | 0.5 | 2 | 12.2 |
| knn | 62.78 ± 3.35 | 82.42% | 0.652 | 0.4 | 4 | 2.9 |

**CV winner: rf**, tau=0.5, min_len=3 windows. Refitted on all 14,073 eligible train windows. The selected bundle is active for integration; no other model was evaluated on test.

The winner–runner-up CV Edit gap is 0.031 points. Ranking follows the fixed rule; no statistical superiority is established.

## Frozen final-test results

21 synthetic test videos, 79 unique clips; 15 unused clips remain unused. The model and smoothing choice were not changed after these results.

| Metric | Raw mean | Smoothed mean ± video SD |
|---|---:|---:|
| Edit score | 55.336 | 72.515 ± 10.716 |
| Frame accuracy (%) | 82.883 | 86.659 ± 11.033 |
| Segment F1@10 | 0.616 | 0.791 ± 0.098 |
| Segment F1@25 | 0.583 | 0.750 ± 0.141 |
| Segment F1@50 | 0.455 | 0.636 ± 0.154 |
| Nearest-boundary mean error (s) | 6.305 | 6.746 ± 4.043 |
| Nearest-boundary median error (s) | 0.990 | 1.111 ± 0.570 |
| Frame F1: squat | 0.821 | 0.865 ± 0.216 |
| Frame F1: push_up | 0.470 | 0.487 ± 0.480 |
| Frame F1: jump_jack | 0.751 | 0.764 ± 0.383 |
| Frame F1: transition | 0.400 | 0.484 ± 0.216 |
| Predicted segment count | 12.619 | 6.571 ± 2.181 |
| GT segment count | 8.190 | 8.190 ± 1.861 |

## Interpretation and limitations

- CV scores are selection-CV estimates, not unbiased nested-CV performance. Fold SD is descriptive variability, not a subject confidence interval.
- Test SD is across the 21 constructed videos, not a confidence interval. Native split provenance does not establish subject independence.
- Per-class frame F1 is averaged over videos using the existing zero-division=0 convention, including videos where a class is absent; it is not pooled frame-level F1.
- Fit/predict wall times were observed on a shared machine with each classifier's configured parallelism; they are descriptive, not a controlled runtime benchmark.
- This is a synthetic, annotation-assisted pose benchmark: annotated exercise labels still assist upstream person selection. Continuous, unassisted workout generalization was not measured.
- No test-driven parameter changes, model reranking, source exclusions or extra clip reuse were performed.
- Convergence warnings were logged for: lr. Fixed iteration budgets were retained; these scores describe the configured solvers, not guaranteed fully converged optima.
- The installed sklearn emitted an SVC probability-API deprecation warning. The predeclared implementation was retained; a future migration requires a new validated experiment, not a change to this frozen result.
- The PowerShell logging wrapper returned a nonzero status while stderr warnings were captured. Before refit, all 40 OOF folds, 128 configuration rows and the selected prediction hash were independently verified as complete; no failed fold was replaced or omitted.
- Inference-only placeholder label/purity columns were removed from the test prediction CSV after a schema audit. Original CSV and training source were preserved; prediction/probability columns, model weights, selected settings and test metrics are unchanged. Test metrics use GT segment JSONs, not placeholder window labels.

## Artifacts

Small result tables, settings and hashes: [cv8_v2_seed42_p080_20261117/](cv8_v2_seed42_p080_20261117/).
Local full log: `outputs/segmentation_v2/run_logs/cv8_v2_seed42_p080_20261117.log`.
Local model: `outputs/segmentation_v2/experiments/cv8_v2_seed42_p080_20261117/final/model.joblib`.
The frozen v1 archive remains unchanged. Full OOF predictions, model binaries and features stay local.
