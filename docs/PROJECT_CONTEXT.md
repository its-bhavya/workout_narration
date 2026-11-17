# Active v2 construction protocol

Approved on 17 November 2026. The root PROJECT_CONTEXT.md contains historical
v1 results; its contributor/subject grouping claims do not define v2.

- RepCount only; status pass/review; fps >=20. Missing manual_exclusions.csv
  is an empty set and is logged as "no manual exclusions file found".
- Native original_split=train becomes train (239 clips: 66/81/92 for
  jump_jack/push_up/squat). Native val+test becomes test (94: 38/31/25).
- No separate validation split. Five source-clip folds inside train prevent
  source reuse across CV folds. Contributor prefixes are not subjects.

## Approved fold redraw

`data/manifests/folds.csv` uses StratifiedKFold(n_splits=5, shuffle=True,
random_state=3969), with train clips sorted by clip_id before assignment.
The former seed-42 assignments are preserved as
`data/manifests/folds_seed42_archived.csv`.

Seed 3969 was selected from seeds 43 through 10042, inclusive, by minimum
maximum absolute deviation from the mean total margin-frame count per fold;
ties use smaller standard deviation, then smaller seed. Margins are pre/post
repetition-core frames resampled to the constructor's 30 fps frame space,
counting each source clip once. Exercise stratification is mandatory. No model
results, test performance or model selection scores were used in this choice.
The five selected fold margin totals are 7484, 7379, 7293, 7364, 7237 frames.
This balances margin totals, not necessarily their fractions of clip duration.

## Approved render targets

- run_id=v2; 24 train videos per fold (120 total); cap 5 per train clip.
- 21 test videos; cap 1; unused test clips remain unused; balance is not relaxed.
- A dedicated random.Random(42) draws uniform K in {3,4,5}; the same K schedule
  is used independently in each train fold, and test starts its own stream.
- Train per fold: 13 K3, 2 K4, 9 K5; 92 slots, 31/31/30 exercise slots.
- Test: 12 K3, 2 K4, 7 K5; 79 slots, 27/27/25 exercise slots. The next draw
  would require 84 slots and at least 26 squat clips to retain +/-10% balance,
  exceeding the 25 available; no redraw of K or relaxation is allowed.
- Every train video uses only one fold; no repeated clip within a video;
  least-used selection keeps per-exercise usage spread <=1 within each fold.
- Whole clips, no inserted footage; 30 fps, 640x360 letterbox, margins labeled
  transition; adjacent exercises differ; exact exclusive-end segment JSON and
  clip_join_frames. Source decode failures abort instead of substituting footage.

V1 generated artifacts are preserved under data/_archive_v1 with original
repository-relative paths. V2 IDs contain run_id and train fold. The retained
batch_checkpoint.json stores the complete frozen plan, input SHA-256 hashes,
parameters, completed IDs and completion status. Re-running identical arguments
resumes; changed inputs are rejected. Interrupted media is archived, never deleted.

## Phase 4 preparation

Per-clip caching is explicitly deferred. The active extractor writes per-video
angle CSVs and provenance/summary sidecars under `data/features/v2/`. Run it at
stride 2, workers 1, IMAGE mode, resume only, without --force. Test videos are
extracted too; test is used for extraction and counting only, never decisions.

Run the resumable no-model workflow from the repository root:

```powershell
uv run --no-sync python -B src/data_pipeline/phase4.py
```

The first-video benchmark was 66.35 seconds; all 141 per-video features are now complete.
The runner resumes verified CSVs, logs per-video timing to
`outputs/segmentation_v2/pose_extraction.log`, rebuilds
`data/manifests/pose_extraction_summary_v2.csv`, lists has_long_gap videos,
and reports boundary-jump ratios by split without tuning any settings.

The existing sliding-window algorithm is retained (2-second window, 1-second
stride). Its index adds fold/run_id and is written to the new
`outputs/segmentation_v2/windows.csv`. Counts before/after train purity >=0.8
are reported by exercise, split and fold, explicitly including transition.
Test and held-out CV validation windows are not purity-filtered.

`src/segmentation/cv.py:get_cv_split(windows, k)` returns train videos from
folds !=k (training purity filter applied) and validation videos from fold k.
Both are drawn only from the train pool. Full source-clip sets must be disjoint
before purity filtering. Separate-val train/evaluate/sweep/run-all entry points
are disabled; no model is trained, evaluated or selected in this phase.

All active dataset readers fail on missing/non-v2 run_id or archived paths.
The final audit verifies complete feature/window coverage, source split/fold
membership, zero CV source overlap, absence of literal v1 video identifiers in
active artifacts, and an unchanged frozen archive. Historical prose is clearly
marked under docs/history/ with old video identifiers redacted.

Two summary bugs were verified and corrected: actual video JSONs consistently
use push_up (182 exercise segments), while the old summary searched for push;
and stride-2 gaps must use actual time spacing, not rows/30. Missing-pose runs
are now measured within each segment so separate segments cannot merge a gap.

The boundary diagnostic also used frame numbers as row positions, which fails
at stride 2. It now indexes jumps by the feature frame column. An unsampled
boundary maps to the next sampled frame whose preceding jump interval straddles
that boundary; shared intervals are counted once. Random comparison frames are
sampled frames, retaining the original +/-2-frame boundary exclusion, seed and
thresholds. The existing split aggregate remains the sample-count-weighted
average of per-video medians (an approximation, not the pooled global median).

Phase 4 extraction and final data audit completed successfully; details follow.

## Phase 4 completed run

V2 per-video features: 141 videos, stride 2, workers 1, IMAGE mode, resume only. Windows: 17607. No model training/evaluation. Summary, long-gap list, boundary ratios, window counts and CV partition counts are saved in outputs/segmentation_v2/. Consistency checks passed; archive unchanged.
