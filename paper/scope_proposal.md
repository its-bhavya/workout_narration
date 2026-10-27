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
