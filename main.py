"""main.py -- Main CLI entry point for the integrated workout segmentation and rep counting pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path so src is importable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import WorkoutPipeline, print_workout_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workout Narration: Integrated Segmentation and Rep Counting Pipeline"
    )
    parser.add_argument(
        "video",
        nargs="?",
        default="synth_test_0001",
        help="Video ID (e.g. synth_test_0001), path to MP4 video, or path to *_angles.csv. (Default: synth_test_0001)",
    )
    parser.add_argument(
        "--export",
        "-o",
        type=Path,
        default=None,
        help="Export complete analysis and narration cues to a JSON file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON to stdout (useful for piping directly into a narration engine).",
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Skip ground-truth comparison even if companion JSON exists.",
    )
    parser.add_argument(
        "--no-narration",
        action="store_true",
        help="Hide narration cue sheet in the console output.",
    )
    parser.add_argument(
        "--all-cues",
        action="store_true",
        help="Show all narration cues without truncation in the console report.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional path to custom trained classifier (.joblib).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Confidence smoothing threshold (overrides chosen_params.json).",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=None,
        help="Minimum run length in windows (overrides chosen_params.json).",
    )

    args = parser.parse_args()

    pipeline = WorkoutPipeline(
        model_path=args.model,
        tau=args.tau,
        min_len=args.min_len,
    )

    result = pipeline.analyze(
        video_input=args.video,
        show_gt=not args.no_gt,
    )

    if args.export:
        result.save_json(args.export)
        print(f"Results exported successfully to {args.export}")

    if args.json:
        print(result.to_json())
    else:
        max_cues = None if args.all_cues else 25
        print_workout_report(
            result,
            show_narration=not args.no_narration,
            show_gt=not args.no_gt,
            max_cues=max_cues,
        )


if __name__ == "__main__":
    main()
