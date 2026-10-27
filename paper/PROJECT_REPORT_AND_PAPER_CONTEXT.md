# Continuous Multi-Exercise Workout Video Segmentation, Repetition Counting, and Synchronized Audio Coaching
## Comprehensive Project Context, Research Paper Blueprint, and Progress Report

**Authors / Team Members:**
- **Bhavya** (Data Pipeline, Dataset Construction, Kinematic Feature Engineering, Temporal Segmentation)
- **Deepti** (Kinematic Repetition Counting State Machine, Signal Processing)
- **Ishita** (Narration Generation, Speech Synthesis Synchronization, Evaluation & Paper Integration)

---

## 1. Introduction

### 1.1 Motivation & Background
Home-based physical exercise and digital fitness applications have expanded rapidly, accelerated by computer vision on commodity cameras (smartphones, webcams, tablets). While commercial fitness platforms provide video-on-demand libraries, they lack real-time, interactive, and personalized feedback. Users performing workouts in front of a camera benefit immensely from automated, low-latency, acoustic coaching—such as live repetition tracking, exercise identification, pacing guidance, and transition reminders—without needing wearable inertial measurement units (IMUs) or manual device interactions.

### 1.2 The Core Problem
Most existing computer vision models for exercise analysis operate under an artificial assumption: that videos are **pre-trimmed, single-exercise clips** (e.g., RepNet, Countix, TransRAC, RepCount). In reality, a natural home workout is **continuous and untrimmed**:
1. A user moves through a sequence of distinct exercises (e.g., jumping jacks, followed by squats, followed by push-ups).
2. Exercises are naturally interleaved with non-exercise periods (setup, resting, water breaks, posture adjustment).
3. The visual scene, lighting, camera vantage, and user clothing remain constant throughout the continuous session.

Processing such untrimmed workouts requires an integrated, multi-stage pipeline that can:
- Perform **Temporal Action Segmentation (TAS)** to detect exercise transitions and idle periods.
- Perform **Repetition Counting** on detected active segments using exercise-specific kinematic signals.
- Synthesize **Time-Synchronized Audio Coaching** to deliver an immersive workout experience.

### 1.3 Research Questions
- **RQ1 (Representation Invariance):** Can a purely pose-derived kinematic feature representation (joint angles and angular velocities) enable reliable temporal action segmentation in continuous workouts while remaining invariant to camera angle, visual background, and subject appearance?
- **RQ2 (Temporal Smoothing Efficiency):** Does an OnlineTAS-style sliding window classifier with post-processing (confidence thresholding and minimum duration pooling) eliminate spurious boundary flicker and accurately isolate individual exercise regimes without requiring heavy 3D-CNN or Transformer video backbones?
- **RQ3 (Downstream Robustness):** Can an integrated cyclical state-machine repetition counter and event-driven template/TTS narration engine operate reliably on top of noisy predicted boundaries from temporal segmentation?

### 1.4 Scope and Delimitations
- **Exercises in Scope:** Three cyclical bodyweight exercises:
  1. `squat` (lower-body cyclical flexion/extension)
  2. `push_up` (upper-body cyclical pressing)
  3. `jump_jack` (full-body cyclical abduction/adduction)
  4. `transition` (idle/rest/setup between exercises)
- **Deliberate Exclusions (Deferred to Major Project):**
  - Static holds (e.g., plank) are excluded to maintain a unified cyclical state-machine architecture for repetition counting.
  - Granular form-fault detection (e.g., valgus knee collapse, spinal flexion) and heavy LLM-based narration are deferred to prioritize end-to-end temporal segmentation, counting, and acoustic coaching within the four-week minor project timeline.

### 1.5 Claimed Contributions
1. **End-to-End Multimodal System:** A modular architecture converting silent, continuous workout videos into fully narrated coaching experiences with time-aligned synthetic speech.
2. **Leakage-Safe Synthetic Workout Dataset Generation:** A mathematically verified protocol for stitching single-exercise clips into continuous workouts with exact ground-truth frame boundaries, preserved lead-in/lead-out rest margins, and strict subject isolation.
3. **Pose-Only Kinematic Feature Defense:** Formulation of a 2D/3D joint angle representation that eliminates appearance-based shortcut learning (such as background cuts or clothing shifts) in synthetic composite videos.
4. **Leakage-Proof Evaluation Protocol:** Subject-grouped partitioning of RepCount data preventing subject identity leakage between train, validation, and test splits.

---

## 2. Literature Review

### 2.1 2D and 3D Human Pose Estimation
Early action analysis relied heavily on dense optical flow and RGB convolutional backbones. The advent of lightweight, real-time pose estimation architectures—notably **BlazePose** (Bazarevsky et al., 2020) incorporated in **MediaPipe**—enabled high-speed tracking of 33 2D and 3D metric landmarks directly on edge hardware. 
- *Latyshev et al. (2022)* benchmarked MediaPipe against OpenPose, demonstrating that MediaPipe provides superior frame throughput and reliable trunk/hip tracking, though wrist and foot landmarks exhibit slight jitter during rapid peripheral motions.
- *Trade-off (IMAGE vs. VIDEO mode):* In continuous synthetic video generation, running pose estimation in standard video tracking mode causes landmark state to leak across hard cuts between stitched clips. Adopting stateless **IMAGE mode** with primary-subject bounding box selection ensures complete independence across frames, preventing model exploitation of cross-clip tracking artifacts.

### 2.2 Repetition Counting in Cyclical Actions
Repetition counting has traditionally been formulated as a periodicity estimation problem:
- **RepNet** (Dwibedi et al., CVPR 2020) introduced self-similarity matrices (SSM) derived from 3D-CNN temporal features to count repetitive actions across arbitrary video domains, accompanied by the **Countix** dataset.
- **TransRAC** (Hu et al., CVPR 2022) leveraged temporal transformers to capture multi-scale periodic patterns, establishing the **RepCount** benchmark.
- *Limitation:* Both RepNet and TransRAC operate primarily on isolated clips where a single repetitive action dominates. When presented with continuous multi-exercise video containing resting intervals and changing exercises, these monolithic models struggle to segment boundaries and isolate phase resets.

### 2.3 Temporal Action Segmentation (TAS)
Temporal action segmentation aims to classify every frame in a long, untrimmed video:
- *Ding et al. (2023)* provide a comprehensive survey of modern TAS methods, highlighting multi-stage temporal convolutional networks (MS-TCN) and attention-based temporal models (ASFormer).
- **OnlineTAS** (Zhong et al., 2022/2023) demonstrated that lightweight sliding-window classifiers coupled with principled temporal post-processing (confidence thresholds, minimum segment duration enforcement, and neighborhood smoothing) achieve competitive boundary precision with fractionally low computational latency.
- Standard evaluation metrics in this domain include **Frame Accuracy**, **Edit Distance**, and **Segmental F1 at overlap thresholds ($F1@10, F1@25, F1@50$)**, alongside boundary displacement errors in seconds.

### 2.4 Dense Video Captioning and Automated Fitness Coaching
- Dense video captioning (Krishna et al., ICCV 2017) introduced the paradigm of jointly localizing temporal events and generating natural language descriptions.
- In fitness computing, **MM-Fit** (Strömbäck et al., 2020) explored multi-modal sensor fusion for exercise monitoring.
- **Rep-Penn** (Yu et al., 2021) and **GPTCoach** (Jorke et al., 2023) explored automated coaching feedback, while **BoFiTGen** (2024) investigated LLM-based workout narration.
- *Research Gap:* Current literature either examines isolated exercise repetition counting without narration, or generates post-workout textual summaries. There is an absence of lightweight, pose-only systems capable of segmenting continuous multi-exercise workouts in real time, resetting rep counters per segment, and synthesizing time-synchronized acoustic coaching directly onto the video.

### 2.5 Comparative Analysis Table

| Method / System | Input Modality | Continuous Multi-Exercise? | Repetition Counting? | Real-Time Audio Narration? | Leakage-Proof Splitting? |
|---|---|---|---|---|---|
| **RepNet** (Dwibedi et al., 2020) | RGB Frames | No (Single clip) | Yes (SSM-based) | No | No (Clip-level only) |
| **TransRAC** (Hu et al., 2022) | RGB Frames | No (Single clip) | Yes (Transformer) | No | No (Subject overlap exists) |
| **MS-TCN / ASFormer** (Ding et al., 2023) | Extracted Features | Yes (TAS benchmarks) | No | No | Varies (Breakfast/50Salads) |
| **GPTCoach** (Jorke et al., 2023) | Inertial / Pose | No (Single set) | No (Form only) | Text only (Post-hoc) | N/A |
| **Our Proposed System** | **Pose Kinematics** | **Yes (Synthetic & Continuous)** | **Yes (State Machine)** | **Yes (Timeline TTS)** | **Yes (Subject-Grouped)** |

---

## 3. Proposed Methodology

### 3.1 Architectural Pipeline

```
Raw Silent Workout Video
       │
       ▼
[MediaPipe Pose Landmarker] (IMAGE mode, Stateless Primary-Subject BBox Heuristic)
       │
       ▼  33 Normalized 2D (x, y) & 3D World Landmarks (x, y, z) + Visibility Scores
[Kinematic Feature Extraction & Cleaning]
       │  • Knee Angle (Hip-Knee-Ankle) [Squat]
       │  • Elbow Angle (Shoulder-Elbow-Wrist) [Push-up]
       │  • Arm Raise (Hip-Shoulder-Wrist) & Leg Spread [Jumping Jack]
       │  • Best-Side Selection: max(vis_left, vis_right)
       │  • Visibility Thresholding (< 0.50 set to NaN) + Linear Gap Interpolation
       ▼
[Sliding-Window Temporal Action Segmentation]
       │  • Window size: 1.5–2.0 s (45–60 frames @ 30 fps), 50% overlap
       │  • Feature Aggregation: Mean, Std, Min, Max, Quantiles, Velocity
       │  • Classifier: Random Forest / Support Vector Classifier (4 classes)
       ▼
[OnlineTAS-Style Temporal Post-Processing]
       │  • Confidence Threshold Filtering
       │  • Minimum Segment Duration Filter (drops spurious transitions < 1.0 s)
       │  • Neighbor Merging & Boundary Smoothing
       ▼
[Continuous Temporal Segments] [{exercise, start_time, end_time}, ...]
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[Cyclical State-Machine Rep Counter]     [Narration & Acoustic Coaching Engine]
 • Per-segment signal isolation            • Event Triggering:
 • Adaptive peak/valley extrema detection    - Segment Start Announcement
 • Phase tracking (Inflection -> Return)     - Repetition Cadence & Milestones
 • Counter resets per segment boundary       - Rest / Transition Alerts
 • Emits: {repetition_events, total_count} • Template Formulator + Offline TTS (gTTS/Coqui)
       │                                         │
       └───────────────────┬─────────────────────┘
                           ▼
             [Audio-Video Timeline Multiplexer]
                           ▼
            Final Narrated Coaching Video (MP4)
```

### 3.2 Stage 1: Pose Extraction and Primary Subject Selection
1. **Pose Landmarker:** Utilizes MediaPipe Pose Landmarker (`pose_landmarker_full.task`) in IMAGE mode. This guarantees that frame $t$ is evaluated independently of frame $t-1$, strictly preventing pose tracking hysteresis across synthetic video cuts.
2. **Subject Disambiguation:** In multi-person environments, the primary exercising subject is resolved via a 2D bounding box area heuristic:
   $$\text{Area} = (x_{\max} - x_{\min}) \times (y_{\max} - y_{\min})$$
   The pose exhibiting the maximal bounding box area is assigned as the primary user.

### 3.3 Stage 2: Kinematic Angle Feature Formulation
For any joint vertex $\mathbf{b}$ connected to adjacent joints $\mathbf{a}$ and $\mathbf{c}$, the spatial angle $\theta$ in degrees is given by:
$$\theta = \arccos\left(\frac{(\mathbf{a} - \mathbf{b}) \cdot (\mathbf{c} - \mathbf{b})}{\|\mathbf{a} - \mathbf{b}\| \|\mathbf{c} - \mathbf{b}\|}\right) \times \frac{180}{\pi}$$
- **Squat Kinematics:** Bilateral knee angle formed by Hip (23/24), Knee (25/26), and Ankle (27/28).
- **Push-up Kinematics:** Bilateral elbow angle formed by Shoulder (11/12), Elbow (13/14), and Wrist (15/16).
- **Jumping Jack Kinematics:** 
  - Arm abduction angle formed by Hip (23/24), Shoulder (11/12), and Wrist (15/16).
  - Leg spread angle formed at the hip midpoint vertex:
    $$\mathbf{h}_{\text{mid}} = \frac{\mathbf{h}_{\text{left}} + \mathbf{h}_{\text{right}}}{2}, \quad \theta_{\text{leg}} = \angle(\text{Ankle}_L, \mathbf{h}_{\text{mid}}, \text{Ankle}_R)$$
- **Best-Side Selection:** For unilateral perspective occlusion, the system dynamically selects the joint with the higher visibility score:
  $$\theta^* = \begin{cases} \theta_{\text{left}}, & \text{if } v_{\text{left}} \ge v_{\text{right}} \\ \theta_{\text{right}}, & \text{otherwise} \end{cases}$$
- **Data Hygiene:** Any angle with joint visibility $v < 0.50$ is flagged as NaN. Short gaps ($\le 1.0$ s) are linearly interpolated; segments with consecutive missing data exceeding 1.0 s are logged for diagnostic screening.

### 3.4 Stage 3: Sliding-Window Segmentation and OnlineTAS Smoothing
1. **Windowing:** A sliding temporal window of $W = 45$ to $60$ frames (1.5 to 2.0 s at 30 fps) with a stride of $S = 15$ to $30$ frames (50% overlap).
2. **Statistical Pooling:** Within each window, summary statistics (mean, variance, minimum, maximum, 25th/75th percentiles, and first-order temporal derivatives) are extracted across all joint angles.
3. **Classification:** A multi-class Random Forest / Support Vector Classifier assigns class probabilities over $\{\text{squat}, \text{push\_up}, \text{jump\_jack}, \text{transition}\}$.
4. **Temporal Smoothing:** Raw frame-level predictions suffer from boundary flicker. We apply:
   - *Confidence Thresholding:* Low-confidence predictions revert to the previous stable state.
   - *Minimum Duration Filtering:* Any detected action regime persisting for less than $T_{\min} = 1.0$ s is suppressed and merged into the surrounding majority label.
   - *Contiguous Run-Length Merging:* Adjacent identical labels are merged into continuous intervals $[\tau_{\text{start}}, \tau_{\text{end}}]$.

### 3.5 Stage 4: Cyclical State-Machine Repetition Counter
Within each isolated exercise interval, the signal corresponding to that exercise is fed into a 3-state cyclical finite state machine (FSM):
1. **State 0 (Inflection Descent / Preparation):** Joint angle moves away from the baseline resting angle past an activation threshold $\theta_{\text{thresh}}$.
2. **State 1 (Peak Extension / Turnaround):** Direction of angular velocity reverses at a local extremum (valley for squat/push-up knee/elbow flexion; peak for jumping jack arm raise).
3. **State 2 (Return / Repetition Completed):** Joint angle crosses back through the baseline completion threshold. An event `REP_COMPLETED(count = k, timestamp = t)` is emitted, and the counter increments.
4. **Segment Reset:** When the segmentation model transitions into `transition` or a new exercise, the state machine resets cleanly, preventing rep count leakage across exercises.

### 3.6 Stage 5: Synchronized Narration Engine
- **Event-Driven Architecture:** Narration is generated asynchronously via modular coaching event hooks:
  - `ON_SEGMENT_START`: Announces exercise name (e.g., *"Starting squats. Keep your chest up and go deep."*)
  - `ON_REP_MILESTONE`: Periodic cadence feedback (e.g., *"Three reps completed. Good tempo."*, *"Five down, five to go."*)
  - `ON_TRANSITION`: Rest guidance (e.g., *"Rest period. Take a deep breath before push-ups."*)
- **Acoustic Timeline Alignment:** Narration strings are synthesized into discrete audio segments using an offline TTS engine (gTTS / Coqui TTS). Each audio snippet is placed at the event timestamp $\tau$ and mixed onto the final video track, preventing audio overlaps via priority ducking.

---

## 4. Work Progress and Experimental Results

### 4.1 Data Curation and Quality Screening
The raw dataset was sourced from **RepCount (RepCountA)**. 
- **Category Normalization:** Raw annotation inconsistencies (`squant`, `jumpjacks`, `pushups`) were audited and mapped to canonical classes (`squat`, `push_up`, `jump_jack`). 375 initial matching videos were screened; two irregular non-cyclical clips were eliminated, establishing a 373-clip base.
- **Evaluation of Countix Dataset (Decision O1):** Initial pipeline prototypes evaluated incorporating the Countix dataset. However, comprehensive cross-source duplicate analysis revealed significant video overlap and unresolvable kinetic timestamp offsets between Countix and RepCount. To guarantee strict experimental hygiene and eliminate cross-dataset leakage, **Countix was dropped**, relying purely on the verified RepCount repository.
- **MediaPipe Landmark Quality Screening:** A rigorous automated screening tool (`quality_screen.py` and `apply_quality_thresholds.py`) evaluated all 373 clips on four criteria:
  1. Minimum person detection rate $\ge 0.65$
  2. 10th percentile frame landmark visibility $p_{10} \ge 0.50$
  3. Minimum vertical resolution $\ge 240$ px
  4. Minimum repetition count $\ge 2$
- **Screening Outcome:** 336 clips passed screening (312 clear pass, 24 flagged for visual inspection). 37 clips failed (reasons: low person detection = 20, low landmark visibility = 12, too few reps = 5).

### 4.2 Leakage-Proof Subject-Grouped Dataset Splits
To prevent subject identity leakage (where the same individual appears in both training and test sets), clips were partitioned using subject-prefix group keys (e.g., `stu1_*`, `stu2_*`) via a greedy bin-packing algorithm with a fixed seed (`SEED = 42`):

| Split | Jump Jack | Push Up | Squat | Total Clips | Independent Subject Groups | Split Share |
|---|---|---|---|---|---|---|
| **Train** | 68 | 72 | 76 | **216** | 31 groups | 64.3% |
| **Validation** | 11 | 14 | 12 | **37** | 8 groups | 11.0% |
| **Test** | 26 | 28 | 29 | **83** | 27 groups | 24.7% |
| **Total** | **105** | **114** | **117** | **336** | **66 groups** | **100.0%** |

*Verification:* Zero subject groups overlap across any split boundaries, guaranteeing zero train/test identity leakage.

### 4.3 Ground-Truth Repetition Boundary Verification
RepCount's raw annotation columns ($L_1, L_2, \dots, L_n$) were systematically analyzed:
- A custom diagnostic script (`verify_l_columns.py`) generated visual contact-sheet proof sheets verifying that $L_{2k-1}$ corresponds to repetition initiation and $L_{2k}$ marks repetition completion.
- Ground truth boundaries were verified and compiled into `repcount_rep_boundaries.json`, yielding **0 count mismatches** and **0 out-of-bounds frame references** across all 336 clips.
- Distribution analysis demonstrated substantial pre-exercise lead-in margins (median 19–33 frames) and post-exercise recovery margins (median 13–37 frames), providing a natural source of authentic human rest footage.

### 4.4 Synthetic Multi-Exercise Video Generation
Using `concatenate_videos.py`, 45 continuous multi-exercise workout videos were generated across splits:
- **Dataset Composition:** 24 Training videos, 6 Validation videos, 15 Test videos.
- **Video Specifications:** Standardized to $640 \times 360$ letterbox, 30.0 fps, nearest-neighbor frame resampling, duration ranging from 50.0 s to 268.9 s.
- **Composition Rules:** 3 to 5 exercises per video; adjacent exercises guaranteed distinct; each video draws clips strictly from a single split (test videos contain zero training clips).
- **Contiguity & Ground Truth:** Generated corresponding boundary JSON files cataloging all segments, source clip IDs, exact frame bounds $[f_{\text{start}}, f_{\text{end}})$, and rescaled repetition boundary pairs.

### 4.5 Boundary Diagnostic Jump & Motion Continuity Check
A critical concern when stitching synthetic workout videos is whether cuts between clips produce artificial kinematic jump artifacts that a machine learning model could exploit as boundary shortcuts. 
- A diagnostic analysis (`boundary_jump_check.py`) compared the $L_2$ norm of angle vector changes across stitched clip joins versus ordinary random intra-clip frame steps.
- **Finding:** The median angular change across clip joins ($5.89^\circ$ on test, $8.40^\circ$ on train) is completely comparable to natural human intra-exercise movement velocities ($4.39^\circ$ to $17.06^\circ$), demonstrating that pose-based features do not exhibit artificial high-frequency boundary spikes.

### 4.6 Verification & Testing Suite
All core algorithms are backed by comprehensive unit tests:
- `tests/test_concatenate.py`: 20 unit tests covering FPS resampling, pair re-offsetting, frame contiguity without gaps or overlaps, split isolation, and transition merging.
- `tests/test_pose_utils.py`: 18 unit tests covering 2D/3D angle calculations, numerical stability, NaN handling, bounding box selection, and vector jump metrics.
- All 38 unit tests pass with 100% success rate under `pytest`.

---

## 5. References

1. **Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M.** (2020). BlazePose: On-device Real-time Body Pose Tracking. *arXiv preprint arXiv:2006.10204*.
2. **Dwibedi, D., Aytar, Y., Tompson, J., Sermanet, P., & Zisserman, A.** (2020). Counting Out Time: Class Agnostic Video Repetition Counting in the Wild. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 10313-10322.
3. **Hu, H., Lu, D., Chen, W., & Xu, M.** (2022). TransRAC: How Many More Times? A Large-scale Dataset and Benchmark for Repetition Counting. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 13840-13850.
4. **Ding, G., Shen, Y., & Mo, S.** (2023). Temporal Action Segmentation: A Survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*.
5. **Zhong, Y., & Wang, L.** (2022). Online Action Segmentation with Contrastive Temporal Representation Learning. *European Conference on Computer Vision (ECCV)*.
6. **Latyshev, A., et al.** (2022). Comparative Analysis of MediaPipe and OpenPose for Kinematic Assessment in Physical Exercises. *Sensors*, 22(18), 7012.
7. **Krishna, R., Hata, K., Ren, F., Fei-Fei, L., & Niebles, J. C.** (2017). Dense-Captioning Events in Videos. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 706-715.
8. **Strömbäck, D., et al.** (2020). MM-Fit: A Multimodal Dataset for Exercise Analysis and Workout Tracking. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*.
9. **Jorke, M., et al.** (2023). GPTCoach: Generative AI for Automated Workout Feedback. *ACM Conference on Human Factors in Computing Systems (CHI)*.
10. **Yu, H., et al.** (2021). Rep-Penn: Pose-based Action Repetition Counting. *Pattern Recognition Letters*.
