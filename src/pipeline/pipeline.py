"""pipeline.py -- End-to-end integration of segmentation, rep counting, and narration formatting."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from workflow import load_bundle, bundle_smoothing, infer_frame, verify_bundle_training
from ..segmentation.config import WINDOWS_CSV, FEATURES_DIR, SYNTH_DIR
from ..rep_counting.rep_counter import count_for_exercise
from ..segmentation.metrics import all_metrics, gt_from_json
from .extractor import load_ground_truth_if_available, resolve_features_input
from .models import (
    GroundTruthComparison,
    SegmentResult,
    WorkoutAnalysisResult,
    WorkoutSummary,
)
from .narration_formatter import generate_workout_timeline

logger = logging.getLogger(__name__)

class WorkoutPipeline:
    """End-to-end multi-exercise workout analysis pipeline.

    Connects:
    1. Video pose angles input
    2. Temporal window classification & OnlineTAS smoothing (Segmentation)
    3. Per-segment hysteresis state-machine rep counting (Rep Counting)
    4. Narration timeline cue generation (Narration Engine Interface)
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        params_path: Path | str | None = None,
        tau: float | None = None,
        min_len: int | None = None,
        end_pad_s: float = 0.3,
        windows_path: Path = WINDOWS_CSV,
        features_dir: Path = FEATURES_DIR,
        synth_dir: Path = SYNTH_DIR,
    ) -> None:
        self.end_pad_s = end_pad_s

        self.features_dir=Path(features_dir)
        self.synth_dir=Path(synth_dir)
        self.bundle,self.model_path=load_bundle(model_path)
        self.videos=verify_bundle_training(self.bundle,windows_path,self.features_dir)
        self.model=self.bundle["model"]
        self.feature_cols=self.bundle["feature_cols"]
        self.classes=self.bundle["classes"]
        self.tau,self.min_len=bundle_smoothing(self.bundle,self.model_path,params_path)
        if tau is not None or min_len is not None:
            if tau is None or min_len is None or (float(tau),int(min_len))!=(self.tau,self.min_len):
                raise ValueError("Integration must use the frozen CV smoothing parameters")

    def analyze(
        self,
        video_input: str | Path | pd.DataFrame,
        video_id: str | None = None,
        show_gt: bool = False,
    ) -> WorkoutAnalysisResult:
        """Run the full integrated pipeline on a multi-exercise workout video.

        Parameters
        ----------
        video_input : video file path, angles CSV path, video ID, or pandas DataFrame.
        video_id    : optional explicit video identifier string.
        show_gt     : if True and companion ground-truth exists, evaluate metrics.

        Returns
        -------
        WorkoutAnalysisResult
            Full structured results, segment details, and narration cue timeline.
        """
        # One canonical prediction path shared with the final evaluator.
        df,resolved_vid=resolve_features_input(video_input,self.features_dir,self.synth_dir)
        vid=video_id or resolved_vid
        if vid!=resolved_vid or vid not in self.videos:
            raise ValueError("Integration requires one indexed v2 video ID")
        meta=self.videos[vid]
        _,_,predicted_segments,video_end_s=infer_frame(self.bundle,df,meta)

        # 3. Per-Segment Rep Counting
        segment_results: list[SegmentResult] = []
        exercise_counts: dict[str, int] = {}
        exercise_durations: dict[str, float] = {}
        total_reps = 0
        total_exercise_duration = 0.0
        total_transition_duration = 0.0

        for idx, seg in enumerate(predicted_segments):
            exercise = seg["exercise"]
            start_s = float(seg["start_time"])
            end_s = float(seg["end_time"])
            duration_s = max(0.0, end_s - start_s)
            is_exercise = exercise != "transition"

            if not is_exercise:
                total_transition_duration += duration_s
                seg_res = SegmentResult(
                    segment_index=idx,
                    exercise=exercise,
                    start_time=start_s,
                    end_time=end_s,
                    duration_s=duration_s,
                    is_exercise=False,
                    rep_count=0,
                    rep_times=[],
                    flag="transition",
                )
            else:
                total_exercise_duration += duration_s
                exercise_durations[exercise] = exercise_durations.get(exercise, 0.0) + duration_s

                # Execute rule-based rep counting with end padding
                count_res = count_for_exercise(
                    df,
                    exercise,
                    start_s,
                    end_s,
                    end_pad_s=self.end_pad_s,
                )

                reps = int(count_res["reps"])
                rep_times = count_res["rep_times"]
                total_reps += reps
                exercise_counts[exercise] = exercise_counts.get(exercise, 0) + reps

                pace = None
                if reps >= 2 and len(rep_times) >= 2:
                    pace = (rep_times[-1] - rep_times[0]) / (reps - 1)
                elif reps == 1:
                    pace = duration_s

                seg_res = SegmentResult(
                    segment_index=idx,
                    exercise=exercise,
                    start_time=start_s,
                    end_time=end_s,
                    duration_s=duration_s,
                    is_exercise=True,
                    rep_count=reps,
                    rep_times=rep_times,
                    pace_s_per_rep=pace,
                    signal_used=count_res.get("signal", ""),
                    thresholds=count_res.get("thresholds"),
                    range_deg=count_res.get("range_deg"),
                    flag=count_res.get("flag", ""),
                    valid_fraction=count_res.get("valid_fraction"),
                )

            segment_results.append(seg_res)

        # 4. Build Workout Summary
        n_ex_segs = sum(1 for s in segment_results if s.is_exercise)
        n_trans_segs = len(segment_results) - n_ex_segs

        summary = WorkoutSummary(
            video_id=vid,
            total_duration_s=video_end_s,
            exercise_duration_s=total_exercise_duration,
            transition_duration_s=total_transition_duration,
            total_reps=total_reps,
            exercise_counts=exercise_counts,
            exercise_durations=exercise_durations,
            total_segments=len(segment_results),
            exercise_segments=n_ex_segs,
            transition_segments=n_trans_segs,
        )

        # 5. Narration Cue Generation
        timeline_cues = generate_workout_timeline(segment_results, summary)

        # 6. Optional Ground Truth Evaluation
        gt_comparison: GroundTruthComparison | None = None
        if show_gt:
            gt_segs, gt_dur, gt_reps = load_ground_truth_if_available(vid,self.synth_dir)
            if gt_segs is not None:
                truth,end=gt_from_json(meta)
                segmentation_metrics=all_metrics(truth,predicted_segments,end,self.classes)
                frame_acc=segmentation_metrics["frame_accuracy"]

                gt_comparison = GroundTruthComparison(
                    gt_segments=gt_segs,
                    frame_accuracy=frame_acc,
                    annotated_reps=gt_reps or {},
                    predicted_reps=exercise_counts,
                    segmentation_metrics=segmentation_metrics,
                )

        return WorkoutAnalysisResult(
            video_id=vid,
            duration_s=video_end_s,
            summary=summary,
            segments=segment_results,
            timeline_cues=timeline_cues,
            ground_truth=gt_comparison,
            run_id="v2",
            experiment_id=self.bundle["experiment_id"],
            comparison_scope=("final_test" if meta["split"]=="test" else "in_sample_train_demo") if show_gt else None,
        )


def analyze_workout(
    video_input: str | Path | pd.DataFrame,
    video_id: str | None = None,
    show_gt: bool = False,
    **kwargs: Any,
) -> WorkoutAnalysisResult:
    """Convenience helper to run the workout pipeline in a single call."""
    pipeline = WorkoutPipeline(**kwargs)
    return pipeline.analyze(video_input, video_id=video_id, show_gt=show_gt)
