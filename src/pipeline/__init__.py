"""Integrated segmentation, rep counting, and narration pipeline package."""
import sys
from pathlib import Path

# Ensure src/segmentation, src/data_pipeline, and src/rep_counting are on sys.path
_SRC = Path(__file__).resolve().parent.parent
for _subdir in ("segmentation", "data_pipeline", "rep_counting", "pipeline"):
    _p = str(_SRC / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .display import print_workout_report
from .extractor import extract_features_from_video, resolve_features_input
from .models import (
    GroundTruthComparison,
    NarrationCue,
    SegmentResult,
    WorkoutAnalysisResult,
    WorkoutSummary,
)
from .pipeline import WorkoutPipeline, analyze_workout

__all__ = [
    "WorkoutPipeline",
    "analyze_workout",
    "WorkoutAnalysisResult",
    "SegmentResult",
    "NarrationCue",
    "WorkoutSummary",
    "GroundTruthComparison",
    "print_workout_report",
    "extract_features_from_video",
    "resolve_features_input",
]
