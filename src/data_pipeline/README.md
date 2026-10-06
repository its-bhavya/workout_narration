# src/data_pipeline
Owned by: Bhavya (Role 1)
Status: Week 1, in progress

Scripts, run in order:
1. load_repcount.py        - parse annotation CSVs, filter to squat/push_up/jump_jack, normalize category names
2. download_countix.py     - pull clips via yt-dlp, trim to timestamps
3. quality_screen.py       - MediaPipe visibility check, pass/fail threshold
4. angle_policy.py         - camera angle estimate + inclusion policy
5. concatenate_videos.py   - stitch clips into synthetic multi-exercise videos + ground-truth boundaries
6. extract_pose_features.py - MediaPipe joint angles per synthetic video + per-frame visibility filtering
