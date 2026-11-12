"""pipeline.py -- End-to-end integration of segmentation, rep counting, and narration formatting."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..rep_counting.rep_counter import count_for_exercise
from ..segmentation.config import OUTPUT_DIR, STRIDE_S, WINDOW_S
from ..segmentation.features import fill_video_nans, window_features
from ..segmentation.make_windows import make_windows_for_video
from ..segmentation.smooth import smooth, to_segments
from .extractor import load_ground_truth_if_available, resolve_features_input
from .models import (
    GroundTruthComparison,
    SegmentResult,
    WorkoutAnalysisResult,
    WorkoutSummary,
)
from .narration_formatter import generate_workout_timeline

logger = logging.getLogger(__name__)

# Default model & params paths
DEFAULT_MODEL_PATH = OUTPUT_DIR / "rf_model.joblib"
ALT_MODEL_PATH = OUTPUT_DIR / "model.joblib"
DEFAULT_PARAMS_PATH = OUTPUT_DIR / "chosen_params.json"


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
    ) -> None:
        self.end_pad_s = end_pad_s

        # Resolve model path
        if model_path is not None:
            self.model_path = Path(model_path)
        elif DEFAULT_MODEL_PATH.exists():
            self.model_path = DEFAULT_MODEL_PATH
        elif ALT_MODEL_PATH.exists():
            self.model_path = ALT_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"No segmentation model found. Looked for {DEFAULT_MODEL_PATH} and {ALT_MODEL_PATH}."
            )

        # Load model bundle
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.feature_cols: list[str] = bundle["feature_cols"]
        self.fill_values: dict = bundle["fill_values"]
        self.classes: list[str] = list(bundle["classes"])

        # Resolve smoothing parameters (tau and min_len)
        self.params_path = Path(params_path) if params_path else DEFAULT_PARAMS_PATH
        if tau is not None and min_len is not None:
            self.tau = float(tau)
            self.min_len = int(min_len)
        elif self.params_path.exists():
            with open(self.params_path, encoding="utf-8") as fh:
                p = json.load(fh)
            self.tau = float(p.get("tau", 0.7))
            self.min_len = int(p.get("min_len", 2))
        else:
            self.tau = 0.7
            self.min_len = 2

    def analyze(
        self,
        video_input: str | Path | pd.DataFrame,
        video_id: str | None = None,
        show_gt: bool = True,
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
        # 1. Resolve and load features
        df, resolved_vid = resolve_features_input(video_input)
        vid = video_id or resolved_vid

        if "time_s" not in df.columns:
            raise ValueError(f"Features table must contain a 'time_s' column. Found: {list(df.columns)}")

        video_end_s = float(df["time_s"].max())

        # 2. Windowing and Segmentation
        df_filled = fill_video_nans(df)
        windows = make_windows_for_video(df_filled, vid, split="test")

        if not windows:
            raise ValueError(
                f"Video duration ({video_end_s:.1f}s) is too short to produce sliding windows of {WINDOW_S}s."
            )

        feat_rows = []
        for w in windows:
            rows_slice = df_filled.iloc[w["row_start"]:w["row_end"]]
            feat_rows.append(window_features(rows_slice))

        X = pd.DataFrame(feat_rows).reindex(columns=self.feature_cols)
        X = X.fillna(pd.Series(self.fill_values))

        # Model inference
        proba = self.model.predict_proba(X.to_numpy())
        pred_labels = self.model.classes_[proba.argmax(axis=1)]
        max_probs = proba.max(axis=1)
        t_centers = [w["t_center"] for w in windows]

        # Post-processing smoothing
        raw_labels = list(pred_labels)
        smo_labels = smooth(raw_labels, list(max_probs), tau=self.tau, min_len=self.min_len)
        raw_segs = to_segments(smo_labels, t_centers, video_end_s)

        # 3. Per-Segment Rep Counting
        segment_results: list[SegmentResult] = []
        exercise_counts: dict[str, int] = {}
        exercise_durations: dict[str, float] = {}
        total_reps = 0
        total_exercise_duration = 0.0
        total_transition_duration = 0.0

        for idx, seg in enumerate(raw_segs):
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
            gt_segs, gt_dur, gt_reps = load_ground_truth_if_available(vid)
            if gt_segs is not None:
                # Calculate frame accuracy if seg_label column exists in df
                frame_acc = None
                if "seg_label" in df.columns:
                    gt_col = df["seg_label"].astype(str).to_numpy()
                    pred_at_row = []
                    tc = np.array(t_centers)
                    for _, row in df.iterrows():
                        t = float(row["time_s"])
                        closest_idx = int(np.argmin(np.abs(tc - t)))
                        pred_at_row.append(smo_labels[closest_idx])
                    frame_acc = float(np.mean(np.array(pred_at_row) == gt_col))

                gt_comparison = GroundTruthComparison(
                    gt_segments=gt_segs,
                    frame_accuracy=frame_acc,
                    annotated_reps=gt_reps or {},
                    predicted_reps=exercise_counts,
                )

        return WorkoutAnalysisResult(
            video_id=vid,
            duration_s=video_end_s,
            summary=summary,
            segments=segment_results,
            timeline_cues=timeline_cues,
            ground_truth=gt_comparison,
        )


def analyze_workout(
    video_input: str | Path | pd.DataFrame,
    video_id: str | None = None,
    show_gt: bool = True,
    **kwargs: Any,
) -> WorkoutAnalysisResult:
    """Convenience helper to run the workout pipeline in a single call."""
    pipeline = WorkoutPipeline(**kwargs)
    return pipeline.analyze(video_input, video_id=video_id, show_gt=show_gt)
