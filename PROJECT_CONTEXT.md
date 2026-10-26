# Project Context: Silent Workout Video to Narrated Coaching

Handoff document for a new developer or an agentic IDE. Read sections 0 and 8 first.

---

## 0. How to read this document

Every claim is tagged with one of these statuses. Do not treat anything weaker than **VERIFIED** as fact.

| Tag | Meaning |
|---|---|
| **DECIDED** | Team decision, safe to build on |
| **VERIFIED** | Ran on the real data and the output was inspected |
| **PROVIDED** | Script exists and was tested on synthetic data only; never confirmed on real data |
| **ASSUMED** | Believed true, not checked. Verify before depending on it |
| **OPEN** | Undecided. Ask the team before choosing |
| **NOT STARTED** | Planned only |

Working conventions:
- Run all scripts from the **repo root**. All paths in scripts are relative to it, except a few hardcoded ones listed in section 9.
- Dev machine is **Windows** with **uv**; heavy compute (pose extraction, video stitching) runs on **Google Colab free tier**.
- Videos are large and **gitignored**. Manifests and feature CSVs are tracked in git.
- All randomness is seeded (`SEED = 42`).
- Pattern used throughout: scripts **measure** first, thresholds are applied in a **separate** step after inspecting distributions.

---

## 1. Project overview

**Goal.** Take a silent video of one person doing several exercises back-to-back, and automatically produce a narrated version: detect which exercise is happening and when, count reps per exercise, generate coaching text, convert to speech, and sync it to the video timeline.

**Context.** University minor project (8 credits), 3 students, professor-guided, 4 weeks to a first paper draft, weekly professor check-ins. A larger major project is expected to build on this one.

**Research question (proposed, confirm with professor).** Can a pose-based sliding-window classifier with temporal smoothing reliably segment continuous multi-exercise workouts, and does that segmentation support accurate rep counting and useful narration?

**Claimed contribution.**
1. A pipeline for continuous multi-exercise segmentation, rep counting and narration (most prior work handles single clips).
2. A reproducible method for building synthetic multi-exercise videos from public single-exercise datasets, with exact ground-truth boundaries.
3. A leakage-safe, grouped train/val/test protocol, plus an end-to-end demo.

**Exercises in scope (DECIDED).** `squat`, `push_up`, `jump_jack`, plus a `transition` (idle/rest) class. Plank was dropped because it is a static hold, not cyclical. All three exercises are now cyclical, so one rep-counting design covers them.

**Out of scope, listed as future work (DECIDED).** Form-deviation detection; LLM-generated narration (see open decision O2); real continuous footage; real-time or deployed app.

**Team.**

| Person | Role | Owns |
|---|---|---|
| Bhavya | Role 1 | `src/data_pipeline`, later `src/segmentation` |
| Deepti | Role 2 | `src/rep_counting` |
| Ishita | Role 3 | `src/evaluation`, `src/narration`, `paper/` |

---

## 2. Architecture (DECIDED unless tagged)

```
Silent video
  -> MediaPipe Pose (per frame landmarks + visibility)
  -> joint angles, low-visibility values set to NaN, short gaps interpolated
  -> sliding windows (1-2 s, 50% overlap) -> window classifier
       classes: squat | push_up | jump_jack | transition
  -> post-processing (OnlineTAS-style): confidence threshold + minimum
       segment length -> keep previous label; merge identical neighbours
  -> segments: [{exercise, start_time, end_time}, ...]
  -> per-segment rep counting (peak/valley on the angle signal, one unified
       cyclical state machine, counter resets per segment)
  -> events (segment start, rep done, milestone, transition)
  -> template-based narration text (+ transition phrases)
  -> TTS (Coqui or gTTS; Bark rejected as too heavy) -> audio aligned to
       timestamps -> narrated video
```

**Why pose-only features.** The classifier sees joint angles, not pixels, so background, lighting and subject differences between stitched clips cannot be exploited. This is the argument that makes synthetic stitching defensible. It must be stated in the paper.

**Window classifier.** Start with Random Forest or SVM on angle features. Move to a small 1D-CNN or LSTM only if there is a clear gain. Augmentation (training windows only): left/right mirroring, time-warp 0.8x to 1.2x, small noise on angles.

**Joint angles (DECIDED, details OPEN).**
- squat: knee angle (hip, knee, ankle)
- push_up: elbow angle (shoulder, elbow, wrist)
- jump_jack: test BOTH arm-spread and leg-spread signals, document which tracks more reliably
- For squat and push_up one clean side is enough: use the better of left/right per joint.
- 2D vs 3D (world) landmarks: OPEN (see O5).

**Interfaces between modules (proposed, agree before coding).**

Segmentation output:
```json
[{"exercise": "squat", "start_time": 4.2, "end_time": 19.8}, ...]
```
Rep counting output, per segment: `{exercise, start_time, end_time, rep_count, phase_events: [{time, phase}]}`.
Narration consumes the rep counting output and emits `[{time, text}]`, then audio clips placed at `time`.

**Evaluation (DECIDED).**
- Segmentation: frame accuracy, Edit score, F1@{10, 25, 50}, boundary error in seconds (standard temporal action segmentation metrics).
- Rep counting: compare with RepCount per-rep annotations (exact match rate, off-by-one rate, mean absolute error).
- Human study: 5 to 10 raters on naturalness, helpfulness, transition smoothness, accuracy.
- Report clip-level n AND group-level n. Prefer confidence intervals or std over bare means.
- Baseline worth including: fixed-interval narration with no pose input.

---

## 3. Repository layout (VERIFIED, created)

```
workout-narration/
  README.md  .gitignore  pyproject.toml  uv.lock
  models/                      pose_landmarker_full.task (not in git)
  data/
    README.md                  TODO: thresholds, grouping rules, seed, category map
    raw/{repcount,countix}/{squat,push_up,jump_jack}/   gitignored
    manifests/                 CSVs, tracked in git, source of truth
    synthetic/                 gitignored; videos + boundary JSONs
    features/                  joint-angle time series per synthetic video
  src/
    data_pipeline/             Role 1
    segmentation/              Role 1, from Week 2
    rep_counting/              Role 2
    narration/                 Role 3
    evaluation/                Role 3
  notebooks/  outputs/  paper/
```
Dependencies are managed with `uv` (`uv sync`, `uv add <pkg>`). Do not use pip or requirements.txt directly. Libraries in use: mediapipe, opencv-python, numpy, pandas, yt-dlp, moviepy, openpyxl.

---

## 4. Data

### 4.1 RepCount (RepCountA), downloaded locally

```
RepCountA/
  annotation/{train,valid,test}.csv
  original_data/{DownloadLink/*.txt, filename_mapping.xlsx}
  video/{train,valid,test}/*.mp4          (1057 files incl. other exercises)
```
Annotation CSV columns: `type, name, count, L1 ... Ln` (Ln count varies per row, trailing cells empty).
- `type`: exercise category. **Names are inconsistent**: `jumpjacks` and `jump_jack`, `pushups` and `push_up`, `benchpressing`, `situp`, ... A naive equality filter silently drops rows. Normalize via an explicit mapping built from `df["type"].unique()` across all three CSVs.
- `count`: number of reps in the video.
- `L1..Ln`: frame numbers marking rep boundaries. **ASSUMED** to bracket reps in pairs; the exact semantics (start of descent? bottom? end?) have NOT been confirmed (see section 8, item 1).
- Native splits are used only as `original_split` metadata. The folder name is `valid`; Countix uses `val` (normalize).
- 373 clips for the three exercises (117 jump_jack, 122 push_up, 134 squat) before quality screening.
- Typical resolution is about 360p. Some clips are low quality or oddly angled (the reason for screening).

### 4.2 Countix, downloaded (decision to keep it is OPEN, see O1)

Distributed as YouTube IDs plus timestamps. `data/manifests/countix_manifest.csv` has 402 segments from 389 videos; 377 downloaded, 25 `download_failed`.
Columns used by the scripts: `sample_id, video_id, split, exercise, count, download_status, source_path, kinetics_start, kinetics_end, repetition_start, repetition_end`.
- `repetition_start/end` are **absolute YouTube timestamps**, not file-relative. The downloaded file may be the full video or trimmed to the 10 s Kinetics window. `quality_screen.py` infers which from the file duration (`time_mode`: 362 absolute, 15 relative). The 15 relative cases are unverified.
- `sample_id` = `<video_id>_<segment index>`. 13 videos have two segments each. All segments of a video must stay in the same split.
- `source_path` uses Windows backslashes (`data\raw\countix\sources\<id>.mp4`). Scripts normalize this.

### 4.3 Known data issues

1. **RepCount and Countix overlap**: the same source video exists in both under different names. Anything that mixes them risks train/test leakage.
2. **`stuN_*` subjects**: files like `stu10_*` appear in RepCount's train, valid AND test. **ASSUMED** to be one subject or session per prefix. Verify with `filename_mapping.xlsx`.
3. **YouTube-numbered RepCount files** (`train3344.mp4`, `val912.mp4`, ...) carry no subject information, and the filename prefix does not match the folder.
4. Multi-person frames, low resolution and blur in a minority of clips.

---

## 5. Work completed

| # | Step | Script / output | Status |
|---|---|---|---|
| 1 | Repo scaffold, uv setup | repo tree above | VERIFIED |
| 2 | RepCount filtering | `load_repcount.py` -> `data/manifests/repcount_filtered.csv` (`filename, exercise, split, rep_count`) | ASSUMED (not reviewed by this doc's author; confirm category normalization and whether rep-boundary frames were preserved) |
| 3 | Countix download | `download_countix.py` -> `countix_manifest.csv` + source videos | ASSUMED (manifest inspected; script not reviewed) |
| 4 | Quality screening | `quality_screen.py` -> `repcount_quality.csv`, `countix_quality.csv` | VERIFIED |
| 5 | Pass/review/fail thresholds | `apply_quality_thresholds.py` -> `quality_screening.csv` | VERIFIED |
| 6 | Grouped, leakage-checked splits | `build_splits.py` -> `splits.csv` | VERIFIED (on both sources; rerun needed after O1) |
| 7 | Cross-source duplicate finder | `find_cross_source_duplicates.py` | PROVIDED (run status on real data unknown) |

### 5.1 `quality_screen.py`
- MediaPipe Pose Landmarker in IMAGE mode, `num_poses=2`; the main subject is the pose with the largest landmark bounding box.
- Samples up to 100 evenly spaced frames. RepCount: whole video. **Countix: only inside the annotated repetition segment.**
- Per clip: person detection rate, mean and 10th-percentile visibility (`p10_frame_visibility`), resolution, rotation metadata, multi-person rate.
- Visibility is exercise-aware: squat and push_up use `max(left, right)` per joint so side-on clips are not penalized; jump_jack averages both sides.
- Features: resume support (an output file from an older schema is renamed `*_old_schema.csv` and everything is re-screened), periodic saves, per-row error capture, path preflight check, `RUN_REPCOUNT` / `RUN_COUNTIX` flags.
- Needs `models/pose_landmarker_full.task`.

### 5.2 `apply_quality_thresholds.py`
Merges both reports into one manifest with `status` (pass / review / fail) and `fail_reasons`.

| Threshold | Value |
|---|---|
| min person detection rate | **0.65** (lowered from 0.80 after the team judged most clips clearly show a person) |
| min p10 frame visibility | 0.50 |
| min frame height | 240 px |
| min reps | 2 |
| min Countix segment | 2.0 s |
| multi-person rate | > 0.20 -> `review` (not auto-fail) |

Results (both sources): usable pool (pass + review) = 230 jump_jack, 236 push_up, 184 squat (650 total); 24 clips flagged `review`. Fail reasons: low_detection 66, low_visibility 33, download_failed 25, low_resolution 10, too_few_reps 7, short_segment 7.
Observation: Countix is noisier than RepCount, especially squat (about 26% quality-fail for Countix squat vs 13% RepCount squat). Squat is the scarcest class.

### 5.3 `build_splits.py`
- Why not native splits: Countix's public test split is unlabeled, RepCount's native test is small (13 to 26 clips per exercise), and RepCount native splits share subjects.
- Group key: Countix `yt:<video_id>`; RepCount `stuN_*` -> `rc_subject:stuN`; other RepCount -> its own group; optional `filename,youtube_id` map to link RepCount clips to Countix videos.
- Greedy group assignment to hit 65/10/25 per exercise; leakage assertion (no group in two splits); deterministic.
- Optional `manual_exclusions.csv` (`source,clip_id`) drops clips before splitting.
- Real output (both sources, before duplicate handling): train 150/153/120, val 23/24/18, test 57/59/46 (jump_jack/push_up/squat). The largest groups are `stu6` (35), `stu4` (34), `stu8` (33), `stu3` (33), `stu10` (30), so test rests on fewer independent subjects than the clip count suggests.
- Output columns: `source, clip_id, exercise, split, group, original_split, status, rep_count`.
- `review` clips are included on purpose so dropping one later cannot move other clips between splits.

### 5.4 `find_cross_source_duplicates.py`
- dHash on 12 frames per clip, same-exercise comparison, MAX_HAMMING 8, HIGH 0.60, REVIEW 0.30. Writes `cross_source_matches.csv`, `repcount_youtube_map.csv` (`filename,youtube_id` where youtube_id is the Countix video_id), and side-by-side images in `dup_review/`.
- **Limitations:** misses crops, flips, overlays, and same-video clips from different moments with changing scenes. Currently hashes only the Countix segment (not the whole file). It imports `REPCOUNT_ROOT` and `get_countix_video_path` from `quality_screen.py`, so run it from the same folder. Always review the images by eye.

### 5.5 Bugs already fixed (do not reintroduce)
- Resuming from an older-format output crashed the summary with `KeyError: 'error'` and blocked Countix. Fixed by schema validation.
- Averaging left and right visibility penalized valid side-on clips.
- Per-video failures used to kill the whole run.

---

## 6. Decisions log

| ID | Decision | Rationale |
|---|---|---|
| D1 | Synthetic multi-exercise videos stitched from public single-exercise clips | exact free boundary labels; no copyright or manual annotation; controllable |
| D2 | Pose-only (joint angle) features | removes background/subject cues across stitched clips |
| D3 | Idle/rest gap of about 2 to 3 s between exercises, explicit `transition` class | realistic; prevents forcing pauses into an exercise class |
| D4 | Plank dropped, jumping jack added | plank is non-cyclical; unified rep logic |
| D5 | Form detection and LLM narration cut | 4-week timeline; do not support the segmentation contribution |
| D6 | Sliding window plus OnlineTAS-style smoothing | simple, citable, avoids training a TAS network |
| D7 | Standard TAS metrics (frame acc, Edit, F1@k, boundary error) | rigor |
| D8 | Grouped splits over pooled data instead of native splits | subject leakage, tiny native test |
| D9 | Detection threshold 0.65, p10 visibility 0.50 | see 5.2 |
| D10 | Screening only measures; thresholds applied separately | pick cutoffs from distributions |
| D11 | `uv` for dependencies; Colab plus shared Drive for compute and video storage | team setup |

### Open decisions
- **O1. Keep Countix?** Leaning toward dropping it. RepCount alone gives about 340 usable clips (109 jump_jack, 114 push_up, 117 squat); expected splits at 65/10/25 are roughly train 71/74/76, val 11/11/12, test 27/28/29. Test is about 10 to 15 independent subject groups, so report group counts and intervals, consider grouped 5-fold cross-validation for final numbers, and state "in-the-wild testing" as future work. Dropping it removes the duplicate problem entirely. To apply: filter `df = df[df["source"] == "repcount"]` in `build_splits.py`. Countix screening results stay available as a later extra pool.
- **O2. Narration: templates vs lightweight pose-to-LLM prompting** (BoFiTGen-style). Needs the professor's sign-off.
- **O3. How to produce idle footage** between exercises. Options: (a) real idle frames, e.g. frames before the first and after the last rep boundary in RepCount clips if they show setup/rest (UNVERIFIED); (b) a frozen frame (trivial to detect, zero motion, makes the transition class too easy, would need a stated limitation). Prefer (a).
- **O4. Camera-angle policy** (`angle_policy.py`): currently planned but likely skippable, since features are pose-only. Defer unless a result needs explaining.
- **O5. 2D vs 3D landmarks** for angle computation; best-side selection for squat and push_up.
- **O6. Classifier and window parameters** (RF/SVM vs 1D-CNN; window and stride; smoothing thresholds). Tune on the val split, never on test.

---

## 7. Remaining plan

### 7.1 Week 1 remaining (Role 1), in priority order
1. Verify the L-column semantics (open a video at a few L frames).
2. Decide O1; run the duplicate finder only if Countix is kept; rerun `build_splits.py`.
3. Manually review the 24 `review` clips (keep if one subject is clearly dominant, drop if two similar-sized people are present; do not drop scarce squat clips unless clearly bad). Record drops in `data/manifests/manual_exclusions.csv`.
4. Build the **concatenation script** (spec below), generate about 10 videos first, then 30 to 50.
5. Build **pose extraction** (spec below).
6. Write `data/README.md`.

**Concatenation spec (NOT STARTED, proposed).**
- Each synthetic video draws clips only from ONE split (train, val or test), so test videos never contain a train clip.
- Normalize fps and resolution across source clips when stitching.
- Order: random exercise sequence, a few exercises per video, with an idle gap (O3) between them. Seeded.
- Per video, write a boundary JSON, for example:
```json
{"video_id": "synth_test_0001", "split": "test", "fps": 30,
 "segments": [
   {"type": "exercise", "label": "squat", "start_s": 0.0, "end_s": 8.3,
    "source": "repcount", "clip_id": "stu3_5.mp4", "rep_count": 6,
    "rep_boundary_frames": [12, 40, 66]},
   {"type": "idle", "label": "transition", "start_s": 8.3, "end_s": 11.0}]}
```
- Rep boundary frames must be re-offset into the stitched timeline (and rescaled if fps changes). Depends on the L-column check.

**Pose extraction spec (NOT STARTED).**
- Run MediaPipe on every synthetic video; output `data/features/<video_id>_angles.csv` with `frame, timestamp` plus the joint angles above.
- Per-frame visibility filtering: if any landmark used in an angle is below the visibility threshold, set that angle to NaN. Document the threshold.
- Record the **longest consecutive NaN gap** per video (the 0.65 detection cutoff lets some clips with long gaps through). Interpolate short gaps; flag or drop videos with gaps over about 1 s.
- Keep the primary-subject choice consistent frame to frame; optionally log large hip-position jumps to catch subject switching.

### 7.2 Weeks 2 to 4 (summary)

| Week | Role 1 (Bhavya) | Role 2 (Deepti) | Role 3 (Ishita) |
|---|---|---|---|
| 2 | window dataset, classifier, OnlineTAS smoothing, raw vs smoothed results | peak/valley rep counter per exercise, validate on RepCount per-rep annotations | narration templates plus transition phrases, TTS integration |
| 3 | finalize segmentation module and output format, held-out test batch, tune | counter runs per detected segment (real, not ground-truth, boundaries) | orchestration, audio sync, end-to-end demo, integration lead |
| 4 | final segmentation eval, figures, Methodology and Results sections | final rep-count eval, help design survey, write section | human study, Introduction / Related Work / Discussion, merge paper |

Ishita should have the evaluation metric functions (frame accuracy, Edit score, F1@{10,25,50}, boundary error) ready during Week 1 to 3, tested on dummy data against the agreed segment format.
Professor check-in deliverables: Week 1 scope plus dataset plus pose demo; Week 2 raw vs smoothed segmentation plus rep accuracy plus sample narration; Week 3 full silent-in, narrated-out demo; Week 4 draft review.
**Priority advice:** get a rough end-to-end run (crude classifier, crude smoothing, crude narration) as early as possible, then improve weak parts. Data hygiene must not delay this.

---

## 8. Known issues and unverified assumptions (check these first)

1. **L-column semantics** are unconfirmed. This blocks rep-boundary carry-through and Deepti's validation. Also check whether frames before the first and after the last boundary show idle footage (possible free idle windows).
2. **RepCount/Countix overlap** is known to exist but has not been quantified or resolved.
3. **`stuN` = one subject** is an assumption; the largest groups are 30 to 35 clips each.
4. **15 relative-mode Countix segments** may have been screened at the wrong timestamps if the inference was wrong.
5. **24 `review` clips** have not been manually reviewed.
6. `load_repcount.py` and `download_countix.py` have not been reviewed by the author of this document.
7. **Test set strength:** about 10 to 15 independent subjects; synthetic test videos recombine the same clips, so they are not independent samples. Report clip-level n.
8. **Synthetic transitions are artificial**, and RepCount footage is cleaner than real gym video. State both in Limitations.
9. Duplicate finder recall is imperfect (crops, flips, overlays). A planned second check compares joint-angle sequences across splits after pose extraction.
10. Within RepCount, YouTube-numbered clips from the same source video could still land in different splits.

---

## 9. Practical notes for an agent

- **Hardcoded path to fix:** `REPCOUNT_ROOT` in `quality_screen.py` points to a local Windows path (`C:\Users\Bhavya\Downloads\RepCountA (1)\LLSP`); videos are read from `<root>/video/<split>/<filename>`. `COUNTIX_ROOT` is an optional override folder for the Countix source mp4s (empty by default; `source_path` is used as-is relative to the working directory). Prefer moving these to a config or environment variable.
- Windows/Colab differences: normalize path separators; work in batches on Colab; save to Drive immediately because local Colab disk is lost on disconnect.
- Never put a clip from one `group` in two splits. Never tune on test.
- When changing any threshold, grouping rule or seed, update `data/README.md` and the decisions log.
- Exercise names are always the normalized strings `squat`, `push_up`, `jump_jack`; split names are `train`, `val`, `test`.
- Pose model file: `models/pose_landmarker_full.task` (download separately; not in git). Screening and extraction should use the same model file so results are comparable.

---

## 10. Paper plan

Sections: Abstract, Introduction, Related Work, Methodology (pose and angles, segmentation and smoothing, rep counting, synthetic dataset construction, narration), Experimental Setup (dataset, quality screening, grouped splits), Results (segmentation metrics, rep counting, human study), Discussion and Limitations, Conclusion and Future Work (form feedback, LLM narration, real continuous footage, more exercises, real time).

Related Work groups (details in the team's literature review CSVs): temporal action segmentation survey (Ding et al.) and online TAS post-processing (Zhong et al., OnlineTAS); repetition counting (RepNet / Countix by Dwibedi et al.; TransRAC / RepCount by Hu et al.); pose estimation (BlazePose, Bazarevsky et al.; Latyshev et al. on wrist/foot error and MediaPipe vs OpenPose); dense video captioning framing (Krishna et al.); closest prior systems (Yu 2021 / Rep-Penn, MM-Fit); narration and coaching design (Hegde et al., GPTCoach by Jorke et al., BoFiTGen).

---

## 11. Script index

| File | Purpose | Key constants |
|---|---|---|
| `quality_screen.py` | measure clip quality (RepCount whole video, Countix segment) | `REPCOUNT_ROOT`, `COUNTIX_ROOT`, `MAX_SAMPLED_FRAMES=100`, `RUN_REPCOUNT`, `RUN_COUNTIX`, `RESUME` |
| `apply_quality_thresholds.py` | pass / review / fail manifest | `DEFAULT_THRESHOLDS`, `EXERCISE_OVERRIDES`, `REVIEW_RELATIVE_TIME_MODE` |
| `build_splits.py` | grouped 65/10/25 splits | `SPLIT_FRACTIONS`, `SEED`, `REPCOUNT_YOUTUBE_MAP_PATH`, `EXCLUDE_CLIPS_PATH` |
| `find_cross_source_duplicates.py` | find RepCount/Countix duplicates | `MAX_HAMMING`, `HIGH_SCORE`, `REVIEW_SCORE`, `FRAMES_PER_CLIP` |
| concatenation, pose extraction, segmentation, rep counting, narration, evaluation | NOT STARTED | see section 7 |
