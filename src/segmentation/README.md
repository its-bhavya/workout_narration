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
