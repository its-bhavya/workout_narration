## Current v2 model workflow decision record (17 November 2026)

See [../docs/V2_TRAINING.md](../docs/V2_TRAINING.md) and the active docs/PROJECT_CONTEXT.md for implementation
and commands. The notes/results below remain historical unless explicitly v2.

- No contributor/subject grouping; train is native train (239 usable clips),
  test is native val+test (94). Five train source folds are fixed at seed 3969.
- Model and smoothing selection use only source-disjoint train OOF predictions;
  learned imputation/scaling is fitted inside each training fold. Training purity
  defaults to 0.8; validation/test counts are not filtered by that rule.
- Final refit uses all eligible train windows. Test (21 videos, 79 unique source
  clips) is evaluated only with the frozen selected pipeline, never ranked to
  choose a model. No v2 benchmark fit/test result is claimed by this refactor.
- Mean/std over selection folds is descriptive selection-CV variability, not an
  unbiased nested-CV estimate or a subject confidence interval. Report final test
  separately, including per-class, temporal and boundary metrics and limitations.
- Prediction, final evaluation and narration share the bundle/preprocessing and
  exact n_frames/fps timeline. Dataset/model IDs and hashes prevent mixing runs.
- The permitted feature schema is fixed; absent channels become NaN in memory.
  ankle_dist_norm_3d is a noncomputed/all-missing placeholder in current train data.
- Existing pose person selection uses annotated exercise labels: scope research
  claims as an offline annotation-assisted synthetic benchmark. Unassisted raw
  video and genuinely continuous workout generalization remain future work.

> Protocol notice: historical research notes. Contributor-based subject claims,
> separate-val workflows and v1 results below do not define the current protocol.
> Active data protocol and Phase 4 instructions: ../docs/PROJECT_CONTEXT.md.

# Scope Proposal: University Minor Project (8 Credits)
**Project Title:** Continuous Multi-Exercise Workout Video Segmentation, Repetition Counting, and Synchronized Audio Coaching

## 1. Scope Definition
- **Primary Goal:** Convert continuous, silent workout videos into fully narrated coaching videos featuring automated exercise identification, segment boundary detection, repetition counting, and synchronized acoustic feedback.
- **In-Scope Exercise Classes:** 
  1. `squat` (cyclical lower-body)
  2. `push_up` (cyclical upper-body)
  3. `jump_jack` (cyclical full-body)
  4. `transition` (natural idle, setup, and recovery periods)
- **Deliberate Out-of-Scope Exclusions (Future Major Project):**
  - Static holds (e.g., plank): non-cyclical; incompatible with unified phase-based FSM.
  - Granular form-deviation correction and heavy LLM narration prompting.
  - Real-time mobile deployment or multi-person simultaneous narration.

## 2. Team Division and Responsibilities
- **Bhavya (Role 1):** Data pipeline (`src/data_pipeline`), video concatenation, kinematic feature engineering, and temporal segmentation (`src/segmentation`).
- **Deepti (Role 2):** Repetition counting engine (`src/rep_counting`), cyclical state machines, and RepCount per-rep boundary validation.
- **Ishita (Role 3):** Narration text generator (`src/narration`), TTS audio synthesis, timeline synchronization, evaluation suite (`src/evaluation`), and paper integration.

## 3. Four-Week Milestones
- **Week 1:** Data curation, quality screening, subject-grouped splits, synthetic continuous video generator, and stateless pose extraction pipeline. *(Completed)*
- **Week 2:** Windowed classifier training, OnlineTAS smoothing baseline, cyclical state-machine rep counter, and narration template engine. *(In Progress)*
  - Rep counter implemented and evaluated on train + val: squat 67.6% exact / 75.7% within-1 (MAE 2.0), jump_jack 45.0% / 71.1% (MAE 1.7), push_up 46.0% / 62.2% (MAE 2.6). Val push_up depressed by one multi-person clip (`stu10_49`); known artefact, to be resolved with consistent single-subject selection. Full results in `src/rep_counting/README.md`.
- **Week 3:** End-to-end integration, boundary-tolerant repetition counting, speech synthesis audio alignment, and demo generation.
- **Week 4:** Quantitative benchmark evaluation (Frame Acc, Edit, F1@k, Rep MAE), qualitative human evaluation study, and final research paper draft.

<!-- v2-model-results:start -->
## Completed v2 model experiment — 17 November 2026

8-model train-only CV selected **rf** (Edit 71.54 ± 2.90 fold SD; tau=0.5, min_len=3). After all-train refit, only that frozen winner was evaluated on 21 test videos: Edit 72.51, frame accuracy 86.66%, segment F1@25 0.750. Test was not used for model selection or tuning. See [full results, caveats and reproducibility tables](../docs/results/V2_MODEL_RESULTS.md).
<!-- v2-model-results:end -->
