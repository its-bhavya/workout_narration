@echo off
setlocal

REM ==== Init git ====
git init

REM ==== Init uv project (creates pyproject.toml) ====
uv init --no-readme

REM ==== .gitignore ====
(
echo # Data ^& large files
echo data/raw/
echo data/synthetic/
echo *.mp4
echo *.avi
echo *.mov
echo.
echo # Python
echo __pycache__/
echo *.pyc
echo .venv/
echo .ipynb_checkpoints/
echo.
echo # OS
echo .DS_Store
echo Thumbs.db
) > .gitignore

REM ==== Root README ====
(
echo # workout-narration
echo.
echo Silent workout video to auto-generated coaching narration ^(text + audio^).
echo Minor project, 4-week timeline. Exercises in scope: squat, pushup, jumping jack.
echo.
echo ## Team / Ownership
echo - Bhavya — src/data_pipeline, src/segmentation
echo - Deepti — src/rep_counting
echo - Ishita — src/narration, src/evaluation, paper/
echo.
echo ## Setup
echo 1. Install uv, then run: uv sync
echo 2. Raw/synthetic video data lives on shared Google Drive ^(not in git^) — see data/README.md
echo 3. Manifests in data/manifests/ are the source of truth for what data is usable
echo.
echo ## Structure
echo See data/README.md and each src/ subfolder's README for details.
) > README.md

REM ==== data/ ====
mkdir data
(
echo # data/
echo.
echo - raw/        : untouched source clips from RepCount + Countix. GITIGNORED — lives on shared Google Drive.
echo - manifests/  : small CSVs, source of truth, tracked in git.
echo - synthetic/  : generated multi-exercise videos + ground-truth boundary JSONs. GITIGNORED — shared Drive.
echo - features/   : extracted joint-angle time series per synthetic video. Small, tracked in git.
echo.
echo Shared Drive folder link: TODO — add link here once created
) > data\README.md

mkdir data\raw
mkdir data\raw\repcount
mkdir data\raw\repcount\squat
mkdir data\raw\repcount\push_up
mkdir data\raw\repcount\jump_jack
mkdir data\raw\countix
mkdir data\raw\countix\squat
mkdir data\raw\countix\push_up
mkdir data\raw\countix\jump_jack

mkdir data\manifests
mkdir data\synthetic
mkdir data\features

REM ==== src/ ====
mkdir src

mkdir src\data_pipeline
(
echo # src/data_pipeline
echo Owned by: Bhavya ^(Role 1^)
echo Status: Week 1, in progress
echo.
echo Scripts, run in order:
echo 1. load_repcount.py        - parse annotation CSVs, filter to squat/push_up/jump_jack, normalize category names
echo 2. download_countix.py     - pull clips via yt-dlp, trim to timestamps
echo 3. quality_screen.py       - MediaPipe visibility check, pass/fail threshold
echo 4. angle_policy.py         - camera angle estimate + inclusion policy
echo 5. concatenate_videos.py   - stitch clips into synthetic multi-exercise videos + ground-truth boundaries
echo 6. extract_pose_features.py - MediaPipe joint angles per synthetic video + per-frame visibility filtering
) > src\data_pipeline\README.md

mkdir src\segmentation
(
echo # src/segmentation
echo Owned by: Bhavya ^(Role 1^)
echo Status: not started, begins Week 2
echo Will contain: sliding-window classifier + confidence/smoothing post-processing
) > src\segmentation\README.md

mkdir src\rep_counting
(
echo # src/rep_counting
echo Owned by: Deepti ^(Role 2^)
echo Status: not started, begins Week 1 ^(angle study + state machine design^), code from Week 2
echo Will contain: peak/valley rep-counting logic per exercise, segment-aware counting
) > src\rep_counting\README.md

mkdir src\narration
(
echo # src/narration
echo Owned by: Ishita ^(Role 3^)
echo Status: not started, begins Week 2
echo Will contain: text templates, TTS integration, orchestration logic
) > src\narration\README.md

mkdir src\evaluation
(
echo # src/evaluation
echo Owned by: Ishita ^(Role 3^)
echo Status: not started, begins Week 1 ^(metric scaffolding^)
echo Will contain: frame accuracy, Edit score, F1@{10,25,50}, boundary error functions
) > src\evaluation\README.md

REM ==== notebooks/ ====
mkdir notebooks
type nul > notebooks\bhavya_data_exploration.ipynb
type nul > notebooks\deepti_rep_counting.ipynb
type nul > notebooks\ishita_eval_tts.ipynb

REM ==== outputs/ ====
mkdir outputs
(
echo # outputs/
echo Final demo videos, figures, result tables. Keep this folder light — large files go on Drive.
) > outputs\README.md

REM ==== paper/ ====
mkdir paper
type nul > paper\related_work_notes.md
type nul > paper\scope_proposal.md
mkdir paper\draft

echo.
echo Repo structure created successfully.
endlocal