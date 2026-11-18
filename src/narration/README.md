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
| `schedule.py` | Non-overlapping speech; busy rep callouts and milestones delayed >2 s are skipped; short transitions and video-bounded sign-off |
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
The optional burn-in step writes a narrated video copy. Templates need no LLM key, but uncached gTTS calls
need internet. `use_bank=True` reads a local phrase bank and falls back to
templates if missing/corrupt; it does not request LLM text during inference.
Do not run the smoke/demo scripts just to test imports: they generate artifacts
and may access speech services. Unit tests use fake speech/durations.

## Scheduling

- Callouts use absolute rep completion timestamps emitted by the selected rep counter.
- Cadence is the smallest integer for which cadence ? median rep gap is at least cached p90 clip duration + 0.3 seconds. An empty cache uses a 1.9-second phrase estimate.
- Fast callouts use plain numbers. Praise is reserved for milestones.
- Transitions use their measured duration and can move earlier when the preceding exercise is finished. If the transition would overlap the first scheduled count of the next exercise, the scheduler skips it.
- A closing cue can include the segment total when the final count was missed and the cue fits. The sign-off is placed within the video and the WAV is padded with silence to match the video duration.
- Inspect the schedule JSON for skipped cues and delays. Segmentation and count benchmarks do not measure speech quality.

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
Both paths write narration.wav, narration.json, and narration.srt. Existing WAV and JSON outputs are protected from overwrite.
The adapter preserves absolute rep completion times, including counter end
padding, and zero-rep exercise segments. Exercise transitions are scheduled at
the next predicted exercise start; completion is scheduled at video end. Audio
contention can still delay or skip events; consult the report. gTTS requires
network access for uncached phrases. See [the runbook](../../docs/END_TO_END.md).
