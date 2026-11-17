# Run segmentation, rep counting and narration

Current flow, verified against the merged v2 code on 17 November 2026.
Run commands in PowerShell from the repository root.

## Supported input and prerequisites

Use a video ID from `data/manifests/synthetic_index.csv`, for example
`synth_v2_train_f0_0001`. This is a training demo, not an independent evaluation.
The matching video, JSON and canonical `data/features/v2/*_angles.csv` must be
present. A path to the same registered video or angle CSV is also accepted.
An arbitrary new MP4 is **not supported yet**: on-demand raw-video extraction
is disabled. Copying or renaming a personal video to a v2 ID is not valid.
Upstream pose person selection currently uses annotated exercise boundaries;
this is an offline, annotation-assisted synthetic benchmark.

```powershell
cd C:\Users\Bhavya\Desktop\projects\workout_narration
uv sync --locked
Test-Path outputs/segmentation_v2/active_model.json
```

The last command should print `True`. The pointer must resolve to the existing
frozen all-train model bundle. Model binaries, videos and features must be
available locally; a code-only clone is insufficient. See [V2_TRAINING.md](V2_TRAINING.md)
for deliberate CV/refit commands if the model is absent. Do not retrain just to
run a video. `uv sync --locked` installs the declared dependencies, including
gTTS; network access is needed for installation and uncached speech synthesis.

## 1. Segment and count one video

Choose a new output directory for each run to preserve earlier results:

```powershell
$video = 'synth_v2_train_f0_0001'
$out = "outputs/manual_runs/$video"
if (Test-Path $out) { throw 'Choose a new output directory; preserve the old run.' }
uv run --no-sync python -B main.py $video --export "$out/analysis.json"
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed; stop before narration.' }
```

This loads the active model, applies its fitted preprocessing and frozen
smoothing, predicts exercise/transition segments, then runs the existing
exercise-specific counter within each predicted exercise interval. It writes
one analysis JSON containing boundaries, rep counts, rep completion times,
counter diagnostics, text cues and dataset/model identifiers. Transition
segments contribute zero reps. No model is trained and ground truth is not
compared by default.

`main.py` generates text cues by default; `--narration-dir` opts into audio. `--no-narration` hides these
cues in the console; it does not remove them from the exported analysis.
`--show-gt` explicitly enables benchmark comparison; it is not required for
prediction and must not be used to tune against test videos.

## 2. Turn the predictions into scheduled speech

The supported CLI consumes the exported prediction JSON directly, without loading
annotations or rerunning the model:

```powershell
uv run --no-sync python -B -m src.narration.integration "$out/analysis.json" --output-dir $out
if ($LASTEXITCODE -ne 0) { throw 'Narration failed; inspect the error before retrying.' }
```

Or analyze and synthesize in one command:

```powershell
uv run --no-sync python -B main.py $video --export "$out/analysis.json" --narration-dir $out
```

Both paths preserve predicted exercise boundaries, absolute rep completion times,
and exercise segments with zero detected reps. Invalid timestamps or inconsistent
rep counts are rejected. Existing narration files are protected from overwrite.
Python callers can use `src.narration.integration.narrate_analysis(result, out_dir)`
with a `WorkoutAnalysisResult` or its exported mapping.

Final outputs are `analysis.json`, `narration.json` (scheduled text, timing,
delays and skipped events), and `narration.wav`. gTTS also stores reusable
sentence MP3s under `outputs/tts_cache/`; these are required synthesis cache,
not additional experiment results. No per-window exports or temporary script
files are needed. The WAV is a narration-only track: no command above creates
a narrated MP4 or mixes the source video's sound. Open the WAV to listen; play
it from time zero alongside the matching MP4 for a manual demonstration.

Narration uses an offline view of the whole video. Cadence is chosen from
1/2/3/5/10 using median rep spacing; not every rep is spoken. Ordinary rep cues
can be skipped when busy; milestones more than two seconds late are skipped.
Other important cues can be delayed. Later exercise announcements are scheduled at the next predicted exercise
start, preserving intervening rest periods. Workout completion is scheduled at
the full video duration, including trailing transitions. The final rep is intentionally not spoken separately.
Audio can extend past video duration to finish the last phrase. Inspect the
schedule report rather than assuming perfect audiovisual synchronization.

## Optional cached LLM phrases

Templates are the reproducible default above. To use an existing
`data/phrase_bank.json`, add `--llm` to either command (also retain the bank/version used for a study).
The report records the requested phrase source; bank fallback remains possible.
This reads a cached bank; it does not call an LLM per video. Missing or corrupt
banks fall back to templates; using a bank does not change predicted reps.

Generating a new bank is a separate, optional external API operation:

```powershell
$env:GEMINI_API_KEY = 'YOUR_KEY'
$env:GEMINI_MODEL = 'YOUR_ACCOUNT_SUPPORTED_MODEL_ID'
uv run --no-sync python -B -m src.narration.llm_generator
```

Keep keys out of files and Git. Preserve an existing phrase bank before
regenerating it. Check the generator's output: failures may use fallback
phrases, so requesting an LLM is not evidence that LLM phrases were used.
gTTS remains a network service even with a locally cached phrase bank.

## Evaluation and scope

Five source-disjoint training folds selected RF; only the frozen winner was
evaluated on the 21 test videos. See [model results](results/V2_MODEL_RESULTS.md).
The recorded single-video rep diagnostic is
[V2_FULL_VIDEO_REP_RESULTS.json](results/V2_FULL_VIDEO_REP_RESULTS.json):
27 predicted versus 44 annotated reps on `synth_v2_test_0001`. This is an
illustrative counting diagnostic, not aggregate counting accuracy. Narration
quality, latency and a human study have not been established by those results.

For a personal/new video, a further implementation phase is required: unassisted
person selection and canonical feature extraction, a separately guarded input
path, then the same frozen model/counter/narration interfaces. Do not fabricate
split/fold or annotation metadata to bypass the current guards. No archived v1
data should be modified or used by this flow.
