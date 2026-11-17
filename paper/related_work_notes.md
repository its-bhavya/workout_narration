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

# Literature Review and Related Work Notes
**Project:** Continuous Multi-Exercise Workout Video Segmentation, Repetition Counting, and Synchronized Audio Coaching

## 1. Human Pose Estimation
- **MediaPipe BlazePose (Bazarevsky et al., 2020):** 33 landmarks in 2D normalized and 3D metric world space. Real-time inference on CPU/mobile.
- **MediaPipe vs. OpenPose in Fitness (Latyshev et al., 2022):** MediaPipe achieves superior frame rates and robust torso/hip tracking; slight peripheral jitter on rapid wrist/ankle motions.
- **Architectural choice:** Use MediaPipe in **IMAGE mode** (stateless, frame-independent) rather than VIDEO mode to strictly prevent tracking history from leaking across cuts in synthetic continuous videos.

## 2. Repetition Counting in Video
- **RepNet & Countix (Dwibedi et al., CVPR 2020):** Self-similarity matrix (SSM) based periodic action counting. Generalizes across classes but trained on pre-trimmed single-activity clips.
- **TransRAC & RepCount (Hu et al., CVPR 2022):** Multi-scale transformer for dense repetition counting with comprehensive rep-boundary frame annotations.
- **Gap identified:** Neither handles continuous multi-exercise workouts with rest periods and dynamic exercise transitions without manual video trimming.

## 3. Temporal Action Segmentation (TAS)
- **TAS Survey (Ding et al., TPAMI 2023):** Review of MS-TCN, ASFormer, and dense frame classification.
- **OnlineTAS (Zhong & Wang, ECCV 2022):** Demonstrates that sliding-window models paired with confidence thresholds, minimum-segment duration pooling, and contiguous neighborhood smoothing achieve competitive boundary precision with minimal computational overhead.
- **Standard Evaluation Metrics:** Frame Accuracy, Edit Score, Segmental F1@10, F1@25, F1@50, Boundary Displacement Error (seconds).

## 4. Dense Video Captioning & Automated Coaching
- **Dense Captioning (Krishna et al., ICCV 2017):** Joint temporal event localization and caption generation.
- **Rep-Penn (Yu et al., 2021) & MM-Fit (Strömbäck et al., 2020):** Multimodal fitness analysis and rep tracking.
- **GPTCoach (Jorke et al., CHI 2023) & BoFiTGen (2024):** Explores automated fitness feedback and LLM coaching narration.
- **Our contribution:** Pure pose kinematic input, real-time segment reset, cyclical FSM counting, and time-aligned speech synthesis.

<!-- v2-model-results:start -->
## Completed v2 model experiment — 17 November 2026

8-model train-only CV selected **rf** (Edit 71.54 ± 2.90 fold SD; tau=0.5, min_len=3). After all-train refit, only that frozen winner was evaluated on 21 test videos: Edit 72.51, frame accuracy 86.66%, segment F1@25 0.750. Test was not used for model selection or tuning. See [full results, caveats and reproducibility tables](../docs/results/V2_MODEL_RESULTS.md).
<!-- v2-model-results:end -->
