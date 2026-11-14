# src/narration
Owned by: Ishita (Role 3)
Status: Week 2 in progress. Templates, gTTS wrapper, event/schedule logic and a ground-truth demo are in. gTTS timing numbers still to be measured (run `tts_smoke_test`).

## Pipeline
`segments` -> `events.build_events` -> `schedule.build_timeline` (+ `tts.synthesize`) -> `render.render_track`

Segment format (seconds): `{exercise, start_time, end_time, rep_times, clip_id?}`.
`rep_times` = moment each rep is COMPLETE (`end_frame / fps` from `repcount_rep_boundaries.json`).

## Files
| File | Purpose |
|------|---------|
| `templates.py` | phrases per event (`segment_start`, `rep`, `milestone`, `transition`), `SPOKEN_NAME`, `Narrator` (seed 42, no immediate repeats) |
| `tts.py` | `synthesize(text)` (gTTS, cached in `outputs/tts_cache/`), `duration_s(path)` |
| `tts_smoke_test.py` | generates all templates + rep numbers 1-30, writes `outputs/tts_timing.csv` |
| `loader.py` | RepCount ground truth -> segments; `clips_to_segments` mirrors `concatenate_videos.py` |
| `rep_stats.py` | rep-period percentiles per exercise -> `outputs/rep_period_stats.csv` |
| `events.py` | segments -> timed events; rate-adaptive rep cadence |
| `schedule.py` | events -> non-overlapping audio timeline; plain rep callouts are dropped if audio is still playing, milestones/transitions are delayed instead |
| `render.py` | places the clips on one audio track |
| `demo.py` | end-to-end from ground truth, no video needed |

## Run
```
uv add gtts
uv run python -m src.narration.rep_stats
uv run python -m src.narration.tts_smoke_test
uv run python -m src.narration.demo
uv run python -m pytest tests
```

## Design decisions
- Exercises: squat, push_up, jump_jack (no plank).
- Rep callouts fire at rep END, not at the bottom of the movement.
- If reps are faster than `MIN_SPOKEN_GAP_S`, only every k-th rep is announced (k in 1, 2, 3, 5, 10, chosen from the segment's median rep period). Milestone phrase on every 5th rep.
- The first exercise is announced `START_LEAD_S` before its first rep; later exercises are announced by the transition line.
- Narration runs offline on the whole video, so using the segment's median period is allowed.

## TODO
- Record gTTS durations and tune `MIN_SPOKEN_GAP_S`, `START_LEAD_S`.
- Confirm segment format with Deepti and Bhavya (idle segments, `phase_events`).
- Fixed-interval baseline narrator for the human study.
