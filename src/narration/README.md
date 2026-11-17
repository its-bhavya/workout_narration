# src/narration
Owned by: Ishita (Role 3)

Current run instructions: [segmentation -> rep counting -> narration](../../docs/END_TO_END.md).
The prediction recipe consumes `main.py` analysis JSON. The ground-truth demo
is a separate historical reference, not the v2 inference/evaluation path.

## Pipeline
`segments` -> `events.build_events` -> `schedule.build_timeline` (+ `tts.synthesize`) -> `render.render_track`

Segment format (seconds): `{exercise, start_time, end_time, rep_times, clip_id?}`.
`rep_times` are absolute seconds at predicted rep completion in the full video.
Keep the predicted start/end times, including exercises with zero detected reps;
filter out transition segments before passing the exercise list to build_events.
Do not use `predicted.prediction_to_segment` to bridge analysis JSON: it infers
new boundaries from reps and rejects empty rep lists. The runbook preserves the
actual segmentation output instead. `main.py`'s text cue sheet is distinct from
the audio module's cadence-filtered, duration-aware speech schedule.

## Files
| File | Purpose |
|------|---------|
| `templates.py` | phrases per event (`segment_start`, `rep`, `milestone`, `transition`), `SPOKEN_NAME`, `Narrator` (seed 42, no immediate repeats) |
| `tts.py` | `synthesize(text)` (gTTS, cached in `outputs/tts_cache/`), `duration_s(path)` |
| `tts_smoke_test.py` | generates all templates + rep numbers 1-30, writes `outputs/tts_timing.csv` |
| `loader.py` | Historical ground-truth demo loader; reads old splits.csv, not the v2 prediction flow |
| `rep_stats.py` | rep-period percentiles per exercise -> `outputs/rep_period_stats.csv` |
| `events.py` | segments -> timed events; rate-adaptive rep cadence |
| `schedule.py` | Non-overlapping speech; busy rep callouts and milestones delayed >2 s are skipped; other important events delayed |
| `render.py` | places the clips on one audio track |
| `demo.py` | Historical ground-truth/noise demonstration, not model predictions; --no-render still synthesizes speech |
| `llm_generator.py` | Optional external Gemini call producing a validated cached phrase bank, with fallback phrases |

## Run
```
uv sync --locked
uv run --no-sync python -B main.py synth_v2_train_f0_0001 --export outputs/my_video/analysis.json
uv run --no-sync python -B -m pytest tests/test_narration.py tests/test_evaluation.py
```

Then use the audio bridge in [END_TO_END.md](../../docs/END_TO_END.md).
It writes the final narration schedule JSON and WAV; gTTS caches sentence MP3s.
No narrated MP4 is rendered. Templates need no LLM key, but uncached gTTS calls
need internet. `use_bank=True` reads a local phrase bank and falls back to
templates if missing/corrupt; it does not request LLM text during inference.
Do not run the smoke/demo scripts just to test imports: they generate artifacts
and may access speech services. Unit tests use fake speech/durations.

## Design decisions
- Exercises: squat, push_up, jump_jack (no plank).
- Rep callouts fire at rep END, not at the bottom of the movement.
- Cadence k in 1, 2, 3, 5, 10 is chosen using median rep period and a 2.2 s target gap. Milestones occur every 5 reps at k=1, otherwise every 10; the final rep is omitted because the ending/transition line covers it.
- The first exercise is requested 1 s before its segment start (clamped to zero). Later transition lines are requested 0.2 s after the preceding exercise ends, potentially before the next exercise starts.
- Narration runs offline on the whole video, so using the segment's median period is allowed.
- Timing is duration-aware but can lag; inspect skipped events and delay statistics. Rendering extends the track to finish the last phrase.

## TODO
- Record gTTS durations and tune `MIN_SPOKEN_GAP_S`, `START_LEAD_S`.
- Validate timing/quality on predicted workouts; do not claim an evaluated audio system from the segmentation scores.
- Fixed-interval baseline narrator for the human study.

## Narrate pipeline predictions

Analyze and synthesize in one command:

```powershell
uv run --no-sync python main.py synth_v2_train_f0_0001 --export outputs/my_run/analysis.json --narration-dir outputs/my_run
```

Or narrate an existing analysis without loading a model or annotations:

```powershell
uv run --no-sync python -m src.narration.integration outputs/my_run/analysis.json --output-dir outputs/my_run
```

Add `--llm` for cached bank phrases (template fallback), and `--seed` on the
standalone CLI or `--narration-seed` on main.py for reproducible phrase choices.
Both paths write narration.wav and narration.json, and refuse to overwrite them.
The adapter preserves absolute rep completion times, including counter end
padding, and zero-rep exercise segments. Exercise transitions are scheduled at
the next predicted exercise start; completion is scheduled at video end. Audio
contention can still delay or skip events; consult the report. gTTS requires
network access for uncached phrases. See [the runbook](../../docs/END_TO_END.md).
