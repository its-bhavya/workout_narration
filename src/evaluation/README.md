# src/evaluation
Owned by: Ishita (Role 3)
Status: segmentation and rep-count metrics implemented and tested on dummy data (`tests/test_evaluation.py`). Waiting for real prediction format from Bhavya.

## Input format
Segments are `(label, start_frame, end_frame)`, end EXCLUSIVE, contiguous, covering `[0, n_frames)` (see data/README.md, synthetic videos). Labels: `squat`, `push_up`, `jump_jack`, `transition`.

## Metrics (`segmentation_metrics.py`)
| Function | Meaning |
|----------|---------|
| `frame_accuracy` | share of frames with the right label |
| `edit_score` | 1 - normalised Levenshtein distance between label sequences; penalises over-segmentation |
| `f1_at_k(k)` | segment F1 at IoU >= k/100 (k = 10, 25, 50); a hit needs the same label |
| `boundary_error` | seconds between each ground-truth boundary and the nearest predicted boundary of the SAME type (same label before and after). Unmatched boundaries are counted in `n_missed` and excluded from the mean |
| `evaluate_segmentation` | all of the above in one dict |

Every label counts, including `transition`. Filter the segment lists first to score exercises only.

`rep_metrics.py`: exact match, off-by-one accuracy, MAE, normalised MAE, bias.

## Run
```
uv run python -m pytest tests/test_evaluation.py
```

## TODO
- Loader from the synthetic boundary JSON once its schema is confirmed.
- Per-exercise and per-split aggregation over the test set.
- Human-study rating form and fixed-interval baseline (with narration).
