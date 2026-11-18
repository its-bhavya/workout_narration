# V2 integrated prediction, rep counting and narration

For current methods and the GT-window and post-segmentation CV/test results, see the [v2 pipeline results](../../docs/results/V2_PIPELINE_RESULTS.md).

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
Per-segment rep counting and text cue generation share those predictions.
Audio synthesis/scheduling is a separate stage; it can skip or delay callouts.
Segmentation still requires registered v2 features. Their upstream pose person selection is annotation-assisted.

## Current rep-counting and narration methods

The integrated v2 pipeline uses the frozen random forest with 2-second windows,
1-second stride, and OnlineTAS smoothing (tau=0.5, min_len=3). The default rep
counter uses unguided 2D arm-raise peaks for jumping jacks, unguided
confidence-weighted 2D hip-angle peaks for squats, and frozen coordinate
median voting for push-ups. Use `--rep-method state_machine` for the legacy
hysteresis counter.

The push-up path extracts raw pose coordinates on demand from the registered
video, or accepts aligned raw coordinates from the caller. This is separate
from the cached v2 inputs used by the segmenter. Audio narration uses the
predicted completion times and cached clip durations, and fits the sign-off
inside the video. The earlier angle-only peak and state-machine evaluations
remain documented in the [v2 pipeline results](../../docs/results/V2_PIPELINE_RESULTS.md).

<!-- v2-model-results:start -->
## Completed v2 model experiment — 17 November 2026

8-model train-only CV selected **rf** (Edit 71.54 ± 2.90 fold SD; tau=0.5, min_len=3). After all-train refit, only that frozen winner was evaluated on 21 test videos: Edit 72.51, frame accuracy 86.66%, segment F1@25 0.750. Test was not used for model selection or tuning. See [full results, caveats and reproducibility tables](../../docs/results/V2_MODEL_RESULTS.md).
<!-- v2-model-results:end -->
