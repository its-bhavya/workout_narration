# data/

- raw/        : untouched source clips from RepCount + Countix. GITIGNORED — lives on shared Google Drive.
- manifests/  : small CSVs, source of truth, tracked in git.
- synthetic/  : generated multi-exercise videos + ground-truth boundary JSONs. GITIGNORED — shared Drive.
- features/   : extracted joint-angle time series per synthetic video. Small, tracked in git.

Shared Drive folder link: TODO — add link here once created

## 6 October 2026
RepCount category normalization

squat:
    squat
    squant

push_up:
    push_up
    pushups

jump_jack:
    jump_jack
    jumpjacks

The `squant` label was manually verified on a sample of 10
videos and found to correspond to the squat exercise. It was
therefore normalized to `squat`.

After exercise normalization and annotation validation, 375 videos matched the target exercise classes. Two videos were excluded during quality validation: one contained an atypical push-up variation, while another did not contain a complete squat repetition. The resulting dataset contained 373 usable videos.

