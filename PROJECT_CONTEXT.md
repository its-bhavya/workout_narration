## Rep-counting alternatives: methods and results (17 November 2026)

For current methods and the GT-window and post-segmentation CV/test results, see the [v2 pipeline results](docs/results/V2_PIPELINE_RESULTS.md).

### Current default and historical experiments

The integrated v2 pipeline uses the frozen random forest with 2-second windows,
1-second stride, and OnlineTAS smoothing (tau=0.5, min_len=3). The default rep
counter uses unguided 2D arm-raise peaks for jumping jacks, unguided
confidence-weighted 2D hip-angle peaks for squats, and frozen coordinate
median voting for push-ups. Use `--rep-method state_machine` for the legacy
hysteresis counter.

The 17 November development tables below use the earlier cleaned 2D
state-machine baseline, including local push-up normalization and motion-aware
arm selection. The 3D study is a separate experiment.

The peak alternative reuses those cleaned signals and arm-selection rules.
`scipy.signal.find_peaks` detects bend minima for squats/push-ups (invert the
angle signal) and raised-arm maxima for jumping jacks. Prominence must exceed
both 8 degrees and a selected fraction of the run's p10/p90 range. Minimum
width and spacing are converted from seconds to samples. Each finite tracking
run is processed independently, so no cycle spans missing pose data.

Autocorrelation uses a detrended run to find the first strong period between
0.3 and 5 seconds, requiring at least three observed cycles and correlation
>= 0.35. It can increase minimum spacing using a selected fraction of that
period, capped at one second without reducing the exercise's existing minimum.
Unreliable estimates fall back to existing spacing. A zero spacing fraction
is the unguided ablation. A motion extremum is not a completed rep: the angle
must recover by 70% of peak prominence before the next detected peak or run
end. Recovery times, rather than extrema, are exported for narration.

### Evaluation protocol

Both studies use the same 460 annotated training exercise segments: 155
jumping-jack segments from 66 unique clips, 155 push-up segments from 81 clips,
and 150 squat segments from 92 clips. Five source-disjoint training folds are
used. Repeated synthetic uses of a source clip receive one weight within their
fold; folds are weighted equally. Settings are selected on four folds and
scored on the fifth, with deterministic candidate-order ties. These are
exploratory development checks, not unbiased nested-CV estimates: the algorithm
families and existing baseline were also developed using training data.

Exercise windows are ground truth, isolating counting from segmentation.
No test feature frames were loaded for either study. The MAE unit is reps per
source clip averaged across folds, not the video/exercise count unit of the
21-video test results. The previous final-test baseline and follow-up results
remain separate. Upstream pose person selection is annotation-assisted.

### Peak detection and autocorrelation ablation

The peak search has 48 settings per exercise: prominence fractions
0.15/0.25/0.35/0.45; widths 1/15, 2/15, or 0.2 seconds; autocorrelation spacing
fractions 0/0.35/0.5/0.65. Guided and unguided families are selected separately
for the table below. The all-candidate search is also retained in report.json
and has different held-out results; it must not be substituted for a family
score. The guided settings selected on all training folds use prominence
fraction 0.25 for squats/jumping jacks and 0.45 for push-ups, width 1/15 second,
and spacing fraction 0.35. Per-held-out-fold choices are recorded separately.

| Exercise | Legacy 2D state machine | Guided peaks | Unguided peaks |
|---|---:|---:|---:|
| Jumping jacks | 5.288 | 4.413 | 4.204 |
| Push-ups | 2.865 | 2.981 | 2.827 |
| Squats | 1.594 | 1.153 | 1.150 |

### 2D versus world-3D signals

The dimensional comparison uses cached MediaPipe world-landmark 3D angles,
not a newly trained estimator or motion-capture truth. It first holds all
counter settings fixed while changing 2D to 3D; then it tunes parameters
separately for each exercise, method, and dimension on other training folds.
State-machine tuning searches fractions 0.2/0.55, 0.35/0.65, 0.45/0.75 and
confirmation durations 0/0.1/0.15 seconds (nine settings). Peak tuning uses the
same 48-setting grid, including guided and unguided candidates together.

Smoothing, gap handling, motion-amplitude guards, minimum spacing, local-window
length, and arm-quality weights remain fixed. The same arm-scoring rule is used,
but it may choose a different arm from each dimensional signal. Strict column
checks forbid silent 3D-to-2D fallback. Unusable signals remain zero counts;
segments are not dropped. Raw finite coverage is identical in both dimensions
(jumping jacks 88.26%, push-ups 62.98%, squats 80.23%, averaged over segment
occurrences and both raw joint channels), so 3D does not recover missing poses.
Usable 2D/3D segment counts are 148/143 for jumping jacks, 153/149 for push-ups,
and 146/146 for squats under the unchanged guards.

Fixed settings: change signal representation only (descriptive training comparison).

| Exercise | State machine 2D | State machine 3D | Peaks 2D | Peaks 3D |
|---|---:|---:|---:|---:|
| Jumping jacks | 5.288 | 5.531 | 4.413 | 5.328 |
| Push-ups | 2.865 | 4.782 | 2.845 | 4.765 |
| Squats | 1.594 | 1.050 | 1.101 | 0.954 |

Settings selected on the other four folds (peak candidates include both guidance modes).

| Exercise | State machine 2D | State machine 3D | Peaks 2D | Peaks 3D |
|---|---:|---:|---:|---:|
| Jumping jacks | 4.562 | 4.913 | 4.435 | 5.414 |
| Push-ups | 2.865 | 5.205 | 2.993 | 4.968 |
| Squats | 1.370 | 0.928 | 1.197 | 0.992 |

### Interpretation and reporting limits

Guided peaks reduce jumping-jack MAE by 16.5% and squat MAE by 27.6% relative
to the legacy untuned 2D state-machine baseline, but worsen push-ups by 4.0%.
Unguided peaks are as good or better in those aggregate comparisons; added
value from autocorrelation guidance is not established.

3D helps squats under both fixed and independently tuned settings, but worsens
push-ups and jumping jacks. Tuned squat state-machine MAE improves from 1.370
in 2D to 0.928 in 3D (32.3% lower); peak MAE improves from 1.197 to 0.992
(17.1% lower). The best development combination in the dimensional study is
3D state-machine squats, 2D state-machine push-ups, and 2D peaks for jumping
jacks. This is a candidate for future integration, not a deployed default or
an independently evaluated winner. A fresh holdout is needed for a new final
generalization claim. Do not advertise these training MAEs as test accuracy.

Timing precision/recall/F1 use one-to-one matching of predicted completion
times to annotated rep intervals expanded by 0.3 seconds at either end. Timing
scores pool segment occurrences, including reused source clips. Exact match,
within-one, and WAPE in the dimensional report also pool segment occurrences;
they do not use source-weighted MAE aggregation. WAPE is absolute count error
divided by annotated reps, not one minus a rep-detection accuracy.

Existing detail/artifact records: [peak experiment](docs/results/PEAK_COUNTER_EXPERIMENT.md),
[peak settings, fold checks, and hashes](docs/results/peak_counter_20261117/report.json),
[2D/3D experiment](docs/results/SIGNAL_DIMENSIONS_COMPARISON.md),
[2D/3D settings, metrics, and hashes](docs/results/signal_dimensions_20261117/report.json).
Plots and per-segment prediction CSVs are linked from those existing reports.

The dated tables retain their original experiment scope. The pipeline results page records the current benchmark comparisons.

## Current segmentation -> counting -> narration handoff

See [the runnable end-to-end guide](docs/END_TO_END.md) and
[active context](docs/PROJECT_CONTEXT.md). Registered v2 video/features feed the
frozen CV-selected RF, then predicted intervals feed the fixed rep counter.
`main.py` exports counts, absolute rep times and text cues; a documented API
bridge sends predicted exercise segments to the separate gTTS scheduler and
WAV renderer. Templates are default; cached LLM phrases are optional. This
does not ingest arbitrary new MP4s or produce a muxed narrated video. Pose
selection remains annotation-assisted, and audio quality is not yet evaluated.
Historical ground-truth narration demos are not the v2 prediction workflow.

## Current v2 model workflow decision record (17 November 2026)

See [docs/V2_TRAINING.md](docs/V2_TRAINING.md) and the active docs/PROJECT_CONTEXT.md for implementation
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

> Historical v1 context. Active protocol and commands: docs/PROJECT_CONTEXT.md.
> Contributor-prefix subject claims and separate-val workflows below are superseded.

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
  -> post-processing (OnlineTAS-style): low-confidence filtering (preserve
       transitions), minimum segment length enforcement, merge sandwiched
       transitions between identical exercises
  -> segments: [{exercise, start_time, end_time}, ...]
  -> per-segment rep counting (peak/valley on the angle signal, one unified
       cyclical state machine with temporal confirmation, counter resets
       per segment)
  -> events (segment start, rep done, milestone, transition)
  -> template-based narration text (+ transition phrases)
  -> TTS (Coqui or gTTS; Bark rejected as too heavy) -> audio aligned to
       timestamps -> narrated video
```

**Why pose-only features.** The classifier sees joint angles, not pixels, so background, lighting and subject differences between stitched clips cannot be exploited. This is the argument that makes synthetic stitching defensible. It must be stated in the paper.

**Window classifier.** Classifier sweep completed over 8 sklearn models (RF, ExtraTrees, GBT, HistGBT, LR, SVM, KNN, MLP). The `train.py` pipeline supports any of these via `--model`. LR achieves the best test Edit Score (63.4); HistGBT with improved smoothing reaches 64.1; RF has the best boundary precision (F1@25=0.629). A small 1D-CNN or LSTM is still a future option if there is a clear gain. Augmentation (training windows only): left/right mirroring, time-warp 0.8x to 1.2x, small noise on angles — not yet applied.

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
| 8 | Segmentation pipeline + classifier sweep | `make_windows.py` → `train.py` → `evaluate.py` → `hparam_sweep.py`, orchestrated by `run_all.py`; outputs: `windows.csv`, `model.joblib`, `window_predictions_{val,test}.csv`, `results.csv`, `test_segments.json`, `chosen_params.json`, `timeline_example.png`, `runs_log.csv`, `sweep_results.csv` | **VERIFIED** on the synthetic dataset. Val ≈ stu10; test ≈ stu2/stu4. Smoothing params (tau, min_len) chosen by val grid search, never touching test. Classifier sweep (16 experiments, 8 classifiers × 2 purity thresholds) completed; LR achieves best test Edit Score (63.4); RF best F1@25 (0.629). The `train.py` pipeline now supports `--model` for all 8 classifiers. Smoothing improved (transition preservation, sandwiched-transition merging); with HistGradientBoosting + improved smoothing, test Edit score reached 64.1 and transition frame-F1 improved from 0.36 to 0.50. See `src/segmentation/README.md` and `outputs/segmentation/smoothing_improvement_report.md` for full details. |
| 9 | End-to-end integration: segmentation + rep counting + narration formatting | `src/pipeline/` (`pipeline.py`, `models.py`, `extractor.py`, `narration_formatter.py`, `display.py`), `main.py`, `tests/test_pipeline.py` | **VERIFIED**. Given any multi-exercise video/ID/CSV: sliding-window TAS segmentation + OnlineTAS smoothing -> per-segment rep counting with boundary padding -> structured workout summary, per-rep timestamps, pacing, and timestamp-aligned narration cue sheet. Exportable as JSON or formatted report. Tested and verified on synthetic videos. |

### 5.6 Rep counting (`rep_counter.py`, `evaluate_rep_counts.py`) — VERIFIED on train + val

Rule-based counter. No learned model. Pipeline per segment: slice angle CSV to `[start_s, end_s]`, pick signal (best side or mean of both arms), mask long tracking gaps, interpolate short ones, smooth with Savitzky-Golay, run a two-threshold hysteresis state machine with temporal confirmation, drop reps closer than `min_gap_s`.

**Temporal confirmation (`confirm_s`):** A threshold crossing must hold for a set duration (0.15 s squat, 0.10 s push_up, 0.15 s jump_jack) before a rep is counted. NaN (lost tracking) cancels a pending completion. This filters brief false crossings from pose jitter without changing the thresholds themselves.

Evaluated on 119 exercise segments across train and val synthetic videos (ground-truth segment windows, adaptive thresholds):

| exercise | segments | exact_% | within1_% | MAE | bias | precision | recall |
|----------|----------|---------|-----------|-----|------|-----------|--------|
| squat | 37 | 64.9 | 75.7 | 2.2 | -1.9 | 0.98 | 0.80 |
| jump_jack | 45 | 57.8 | 84.4 | 2.3 | -2.0 | 0.99 | 0.91 |
| push_up | 37 | 48.6 | 59.5 | 2.9 | -1.0 | 0.93 | 0.89 |

The `confirm_s` temporal confirmation improved jump_jack exact-match from 45% to 58% and within-1 from 71% to 84% vs the previous (no-confirmation) run. Undercounting bias increased slightly (the counter is more conservative). Val squat is clean (87.5% exact, 0.25 MAE). Val push_up is still pulled down by one outlier (`stu10_49.mp4`, annotated 21, counted 43). This is a multi-person clip: the counter locks onto a secondary subject's oscillating signal and fires twice per real rep.

**Multi-person tracking (`PosePersonTracker`):** A `PosePersonTracker` class in `pose_utils.py` was added to `extract_pose_features.py` to address this. It assigns persistent track IDs across frames (greedy centre-distance + IoU matching), observes all candidates for an observation window, then locks onto the person with the most exercise-relevant movement × visibility. During the observation phase it temporarily uses `largest_bbox_pose_index`. If the locked track is lost it unlocks and re-observes. **Status:** Implemented in the extraction code, but the synthetic angle CSVs have NOT been re-extracted with this yet, so the `stu10_49` overcounting persists in current results. Re-extraction is the next step.

Two segments correctly abstained with `low_motion` flag (stu6_62 squat, stu6_19 jump_jack — insufficient angle range). Full results and calibration tables in `src/rep_counting/README.md` and `outputs/rep_eval/segments.csv`.

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
| D12 | Segmentation window = 2.0 s, stride = 1.0 s (50 % overlap) | 2 s captures a full rep at typical pace; 50 % overlap provides dense coverage without excessive redundancy; tuned on val, never on test |
| D13 | Smoothing tau = 0.7, min_len = 2 (chosen by val grid search over {0.4, 0.5, 0.6, 0.7} × {1,2,3,4}) | val grid-search maximising Edit score, tie-broken by frame accuracy; written to `outputs/segmentation/chosen_params.json`. min_len lowered from 4 to 2 after the smoothing improvements (transition preservation + sandwiched-transition merging) made the `_enforce_min_len` logic more aggressive on its own |
| D14 | Classifier sweep run over RF, ExtraTrees, GBT, HistGBT, LR, SVM, KNN, MLP × 2 purity thresholds | LR (purity=0.7, τ=0.7, min_len=4) achieves best test Edit Score (63.4); RF retained for now pending further tuning; full results in `outputs/segmentation/sweep_results.csv` |
| D15 | `train.py` supports `--model` for all 8 classifiers from the sweep catalogue | lets any classifier from the sweep be trained as the production model without code changes; reuses `_build_classifiers` from `hparam_sweep.py`; model bundle saved as `model.joblib` (was `rf_model.joblib`) |
| D16 | Smoothing improvements: transition labels preserved during low-confidence filtering; short exercise blips absorbed into neighbouring transitions; sandwiched transitions between identical exercises collapsed | prevents erasing intentional rest periods and avoids spurious exercise segments; improved test Edit score by +7.9 and transition frame-F1 by +0.14 with HistGBT. See `outputs/segmentation/smoothing_improvement_report.md` |
| D17 | Multi-person tracking via `PosePersonTracker` added to `extract_pose_features.py` | addresses overcounting on multi-person clips (e.g. `stu10_49`); tracks candidates by bbox, locks onto the person with most exercise-relevant movement × visibility; implemented but synthetic angle CSVs not yet re-extracted |
| D18 | Adopt experimental counter combo as default pipeline rep counter | Merged frozen training-selected experimental rep counter into the main pipeline as the default (`rep_method='experimental'`: unguided arm-raise peaks for jumping jacks, unguided hip-angle peaks for squats, and frozen coordinate median voting for push-ups), while preserving the legacy state machine via `rep_method='state_machine'`. Segmentation model, windowing, and OnlineTAS smoothing remain unchanged. |

### Open decisions
- **O1. Keep Countix?** Leaning toward dropping it. RepCount alone gives about 340 usable clips (109 jump_jack, 114 push_up, 117 squat); expected splits at 65/10/25 are roughly train 71/74/76, val 11/11/12, test 27/28/29. Test is about 10 to 15 independent subject groups, so report group counts and intervals, consider grouped 5-fold cross-validation for final numbers, and state "in-the-wild testing" as future work. Dropping it removes the duplicate problem entirely. To apply: filter `df = df[df["source"] == "repcount"]` in `build_splits.py`. Countix screening results stay available as a later extra pool.
- **O2. Narration: templates vs lightweight pose-to-LLM prompting** (BoFiTGen-style). Needs the professor's sign-off.
- **O3. How to produce idle footage** between exercises. Options: (a) real idle frames, e.g. frames before the first and after the last rep boundary in RepCount clips if they show setup/rest (UNVERIFIED); (b) a frozen frame (trivial to detect, zero motion, makes the transition class too easy, would need a stated limitation). Prefer (a).
- **O4. Camera-angle policy** (`angle_policy.py`): currently planned but likely skippable, since features are pose-only. Defer unless a result needs explaining.
- **O5. 2D vs 3D landmarks** for angle computation; best-side selection for squat and push_up.
- **O6. Classifier and window parameters.** Classifier sweep completed (see D14 and section 5 step 8). Best result: LR (purity=0.7, τ=0.7, min_len=4) at test Edit=63.4; RF has best boundary precision (F1@25=0.629); HistGBT with improved smoothing reaches test Edit=64.1. Window=2.0 s / stride=1.0 s locked (D12). The `train.py` pipeline now supports `--model` for all 8 classifiers (D15). Open: which classifier to report as the production model for the paper, and whether to try LightGBM or a sequence model (RNN/LSTM over windows) for further gains.

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
{"video_id": "archived-v1-video-example", "split": "test", "fps": 30,
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
11. **Multi-person clips in raw RepCount** (e.g. `stu10_49.mp4`) cause overcounting: the counter can lock onto a secondary subject. `PosePersonTracker` was added to `extract_pose_features.py` to address this — it tracks candidates by bounding box, observes movement for an observation window, then locks onto the person with the most exercise-relevant movement × visibility. **Status: implemented but not yet applied.** The synthetic angle CSVs have not been re-extracted with `PosePersonTracker` yet, so the overcounting persists in current evaluation results. Re-extraction is the next step.

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
| `src/segmentation/make_windows.py` | build `windows.csv` from angle CSVs | `WINDOW_S=2.0`, `STRIDE_S=1.0`, `MIN_ROWS_FRAC=0.5` |
| `src/segmentation/train.py` | train classifier (8 types via `--model`), write per-window predictions + run archive | `--model` (rf, extra_trees, gbt, hist_gbt, lr, svm, knn, mlp; default rf), reuses `hparam_sweep._build_classifiers`; model saved as `model.joblib` |
| `src/segmentation/evaluate.py` | val grid-search (tau × min_len), test evaluation, timeline plot | `TAU_GRID`, `MIN_LEN_GRID` |
| `src/segmentation/hparam_sweep.py` | sweep 8 classifiers × purity thresholds × smoothing grid; ranked results table | `--quick`, `--classifiers`, `--min-purity-grid`, `--tau-grid`, `--min-len-grid` |
| `src/segmentation/smooth.py` | post-process window labels: low-confidence filtering, min-length enforcement, sandwiched-transition merging | `_apply_low_confidence` (preserves transitions), `_enforce_min_len`, `_merge_sandwiched_transitions` |
| `src/segmentation/predict_video.py` | run trained model on a single video, produce segments | loads `model.joblib` + `chosen_params.json`; dynamic classifier description |
| `src/segmentation/run_all.py` | orchestrator: make_windows → train → evaluate | `--tag`, `--skip-windows` |
| `src/data_pipeline/pose_utils.py` | pose geometry helpers + `PosePersonTracker` for multi-person tracking | `PosePersonTracker(observation_frames=45, min_observations=15, max_missed_frames=10)` |
| `tests/test_person_tracker.py` | tests for `PosePersonTracker` (ID assignment, movement scoring, recovery, clip start frame) | run: `python -m pytest tests/test_person_tracker.py -v` |
| `tests/test_rep_counter.py` | tests for `run_state_machine` temporal confirmation (complete reps, brief crossings, NaN cancellation) | run: `python -m pytest tests/test_rep_counter.py -v` |
| narration, evaluation | NOT STARTED | see section 7 |

<!-- v2-model-results:start -->
## Completed v2 model experiment — 17 November 2026

8-model train-only CV selected **rf** (Edit 71.54 ± 2.90 fold SD; tau=0.5, min_len=3). After all-train refit, only that frozen winner was evaluated on 21 test videos: Edit 72.51, frame accuracy 86.66%, segment F1@25 0.750. Test was not used for model selection or tuning. See [full results, caveats and reproducibility tables](docs/results/V2_MODEL_RESULTS.md).
<!-- v2-model-results:end -->
