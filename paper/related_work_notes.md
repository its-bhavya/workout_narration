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
