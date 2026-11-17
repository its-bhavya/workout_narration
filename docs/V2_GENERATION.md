# V2 generation and resume

Run from the repository root. Fold assignments use seed 3969; video length and
clip sampling use seed 42. The checkpoint already contains the approved 141-video
plan. The complete 141-video run finished and passed the Phase 3 audit. The first
video benchmark below is retained as historical timing information.

```powershell
uv run --no-sync python -B src/data_pipeline/concatenate_videos.py --splits-file data/manifests/splits_v2.csv --folds-file data/manifests/folds.csv --run-id v2 --n-train-per-fold 24 --n-test 21 --test-cap 1 --seed 42
```

Re-run the same command after interruption; no separate resume flag is needed.
Do not run two generators concurrently. The checkpoint stores input fingerprints
and the full ordered source plan. Completed pairs are verified and recovered into
manifests. Interrupted artifacts are moved outside the active tree to
data/_interrupted_archive/synthetic/.
The checkpoint is retained with status complete at the end.

The generator fully decodes each newly rendered video before accepting it and
prints progress after every video. Completion prints the assertions and per-fold
statistics, checked against actual video metadata, JSONs and usage manifests.

For an independent full-decode audit after completion:

```powershell
uv run --no-sync python -B src/data_pipeline/concatenate_videos.py --splits-file data/manifests/splits_v2.csv --folds-file data/manifests/folds.csv --run-id v2 --n-train-per-fold 24 --n-test 21 --test-cap 1 --seed 42 --validate-only
```

## Frozen plan statistics (not a completed-render report)

Slots are jump_jack / push_up / squat. Reuse is uses: number of source clips,
including unused clips. Margin shares are frame-weighted from exact 30 fps
resampling and exclusive-end segment construction.

| Pool | Videos | Minutes | Transition share | Direct cuts | Joins | Slots | Distinct clips | Reuse |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Train fold 0 | 24 | 43.6789 | 18.2493% | 11 | 68 | 31/31/30 | 48 | 1:7, 2:38, 3:3 |
| Train fold 1 | 24 | 59.1511 | 13.6618% | 2 | 68 | 31/31/30 | 48 | 1:9, 2:34, 3:5 |
| Train fold 2 | 24 | 47.1494 | 16.1060% | 7 | 68 | 31/31/30 | 48 | 1:9, 2:34, 3:5 |
| Train fold 3 | 24 | 46.1211 | 15.2100% | 1 | 68 | 31/31/30 | 48 | 1:9, 2:34, 3:5 |
| Train fold 4 | 24 | 55.7622 | 13.9860% | 3 | 68 | 31/31/30 | 47 | 1:7, 2:35, 3:5 |
| Test | 21 | 46.1317 | 13.7096% | 5 | 58 | 27/27/25 | 79 | 0:15, 1:79 |

All train exercise slots are within 2.18% of the mean, and test within 5.07%.
Train maximum use is 3 (below cap 5); test maximum use is 1. All 239 train clips
are represented; 15 test clips are intentionally unused. No balance relaxation.

## Benchmark and status

synth_v2_train_f0_0001: 4259 frames, 141.967 seconds, 640x360 at 30 fps;
rendering plus full decoding and checkpoint persistence took 40.9 seconds.
It has 4 joins, 1 direct cut, and 4.0855% transition frames. This first video's
exercise counts are not the fold-wide balance target; balance is enforced across
the complete 24-video fold. Phase 3 render-wide validation has completed successfully for all 141 videos. The benchmark process exited cleanly at --max-videos 1.
