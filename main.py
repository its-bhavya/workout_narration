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
        default="synth_v2_train_f0_0001",
        help="Video ID (e.g. synth_v2_train_f0_0001), path to MP4 video, or path to *_angles.csv. (Default: synth_v2_train_f0_0001)",
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
    parser.add_argument("--show-gt",action="store_true",help="Explicit ground-truth comparison; training demos are in-sample")
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
        help="Frozen CV confidence threshold; if supplied, must match the selected bundle.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=None,
        help="Frozen CV minimum run length; if supplied, must match the selected bundle.",
    )

    parser.add_argument("--narration-dir", type=Path,
                        help="Synthesize narration.wav and narration.json from predicted segments.")
    parser.add_argument("--burn-in", action="store_true",
                        help="Burn subtitles and narration audio into check.mp4 (requires --narration-dir).")
    parser.add_argument('--rep-method',choices=['experimental','state_machine'],default='experimental',
                        help='Rep counter: frozen experimental combo (default), or legacy state machine.')
    parser.add_argument("--narration-seed", type=int, default=42)
    parser.add_argument("--llm", action="store_true",
                        help="Use cached narration phrase bank with template fallback.")
    args = parser.parse_args()
    if args.llm and args.narration_dir is None:
        parser.error("--llm requires --narration-dir")
    if args.burn_in and args.narration_dir is None:
        parser.error("--burn-in requires --narration-dir")

    pipeline = WorkoutPipeline(
        model_path=args.model,
        tau=args.tau,
        min_len=args.min_len,
        rep_method=args.rep_method,
    )

    result = pipeline.analyze(
        video_input=args.video,
        show_gt=args.show_gt and not args.no_gt,
    )

    if args.export:
        result.save_json(args.export)
        print(f"Results exported successfully to {args.export}", file=sys.stderr)

    if args.narration_dir:
        from src.narration.integration import narrate_analysis
        from src.narration.burn_in import burn_in_video
        narrate_analysis(result, args.narration_dir, seed=args.narration_seed, use_bank=args.llm)
        if args.burn_in:
            check_path = burn_in_video(args.video, args.narration_dir)
            print(f"Burned-in video written to {check_path}", file=sys.stderr)

    if args.json:
        print(result.to_json())
    else:
        max_cues = None if args.all_cues else 25
        print_workout_report(
            result,
            show_narration=not args.no_narration,
            show_gt=args.show_gt and not args.no_gt,
            max_cues=max_cues,
        )


if __name__ == "__main__":
    main()
