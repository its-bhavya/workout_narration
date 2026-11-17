# V2 integrated prediction, rep counting and narration

Use [the end-to-end runbook](../../docs/END_TO_END.md) for a copy/paste video
example, JSON export and scheduled speech. `WorkoutPipeline` produces text cues. Add `--narration-dir outputs/my_run` to
`main.py` to synthesize scheduled speech from its predictions, or run
`python -m src.narration.integration analysis.json --output-dir outputs/my_run`
on saved results. Add `--llm` to use cached phrases with template fallback.

See [the model workflow guide](../../docs/V2_TRAINING.md) for CV, refit and final
evaluation commands. Integration requires the frozen all-train v2 bundle,
matching training provenance and canonical per-video v2 angle features.

`WorkoutPipeline(model_path=...)` loads a specific bundle; omitting the path
uses outputs/segmentation_v2/active_model.json. There is no RF-first fallback.
The bundle owns imputation/scaling, feature order and CV-selected smoothing.
An optional chosen_params.json must match the experiment ID and model hash.

`analyze_workout(video_id)` does not compare against ground truth by default.
Pass show_gt=True explicitly; train-demo comparisons are labeled in-sample,
and test comparisons are final-test comparisons, never selection criteria.
Outputs include run_id, experiment_id and comparison_scope. All temporal
comparisons use the shared 10-Hz metrics; exact duration is n_frames/fps.

The same prediction function is used by predict_video.py and evaluate.py.
Rule-based per-segment rep counting and text cue generation share those predictions.
Audio synthesis/scheduling is a separate stage; it can skip or delay callouts.
No raw-video/native-clip extraction or person-selector redesign is implemented
here. Existing pose features remain annotation-assisted upstream; this limitation
is documented in the project and paper context.

## Counter alternatives and integration status

The integrated pipeline still uses the 2D state-machine defaults, including
local push-up normalization and motion-aware arm selection. Neither the optional
peak detector nor 3D squat counting has been promoted into this flow.

Peak detection uses prominence and width, optional capped autocorrelation-guided
spacing, and post-peak recovery completion times. On source-disjoint training
folds, guided-peak MAE is 4.413/2.981/1.153 for jumping jacks/push-ups/squats,
versus current state-machine MAE 5.288/2.865/1.594. Unguided peaks do as well
or better; guidance is not an established improvement.

In a separate 2D/3D comparison with parameters selected on other folds, the best
development choices are 2D peaks for jumping jacks (MAE 4.435), 2D state machine
for push-ups (2.865), and 3D state machine for squats (0.928). This peak number
selects from both guidance modes and is different from the guided-only score.
These are source-weighted counts inside GT windows, not predicted-segment
end-to-end accuracy. No test features were used in either experiment.

See [counter methods and full tables](../rep_counting/README.md),
[peak artifacts](../../docs/results/PEAK_COUNTER_EXPERIMENT.md), and
[2D/3D artifacts](../../docs/results/SIGNAL_DIMENSIONS_COMPARISON.md).

<!-- v2-model-results:start -->
## Completed v2 model experiment — 17 November 2026

8-model train-only CV selected **rf** (Edit 71.54 ± 2.90 fold SD; tau=0.5, min_len=3). After all-train refit, only that frozen winner was evaluated on 21 test videos: Edit 72.51, frame accuracy 86.66%, segment F1@25 0.750. Test was not used for model selection or tuning. See [full results, caveats and reproducibility tables](../../docs/results/V2_MODEL_RESULTS.md).
<!-- v2-model-results:end -->
