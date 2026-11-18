"""burn_in.py -- Generate subtitles and burn narration audio + subtitles into a video copy."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH_DIR = REPO_ROOT / "data" / "synthetic"


def format_srt_time(t: float) -> str:
    """Format seconds into SRT timestamp format: HH:MM:SS,mmm"""
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")


def generate_srt(timeline: Sequence[dict], srt_path: Path) -> Path:
    """Write an SRT subtitle file from timeline cues."""
    srt_path = Path(srt_path)
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(timeline, start=1):
            start_str = format_srt_time(entry.get("time", entry.get("start_s", 0.0)))
            end_val = entry.get("end", entry.get("end_s"))
            if end_val is None:
                end_val = entry.get("time", entry.get("start_s", 0.0)) + 1.5
            end_str = format_srt_time(end_val)
            ev_type = entry.get("event", entry.get("cue_type", "rep"))
            text = entry.get("text", "")
            f.write(f"{i}\n{start_str} --> {end_str}\n[{ev_type}] {text}\n\n")
    return srt_path


def resolve_video_path(video_input: str | Path) -> Path:
    """Resolve a video ID or path to an existing video file."""
    p = Path(video_input)
    if p.is_file():
        return p.resolve()

    vid = str(video_input).strip()
    if vid.endswith(".mp4"):
        vid = vid[:-4]

    for split in ("train", "test"):
        candidate = SYNTH_DIR / split / f"{vid}.mp4"
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find video file for '{video_input}'")


def burn_in_video(
    video_path: str | Path,
    narration_dir: str | Path,
    output_mp4: str | Path | None = None,
    srt_filename: str = "narration.srt",
    wav_filename: str = "narration.wav",
) -> Path:
    """Burn audio and subtitles into a new video file, leaving the original video untouched."""
    ndir = Path(narration_dir).resolve()
    vpath = resolve_video_path(video_path)
    wav_path = ndir / wav_filename
    json_path = ndir / "narration.json"
    srt_path = ndir / srt_filename

    if not wav_path.is_file():
        raise FileNotFoundError(f"Missing audio file at '{wav_path}'")

    if not srt_path.is_file():
        if not json_path.is_file():
            raise FileNotFoundError(f"Neither '{srt_path}' nor '{json_path}' exists in '{ndir}'")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        timeline = data.get("timeline", [])
        generate_srt(timeline, srt_path)

    if output_mp4 is None:
        target_mp4 = ndir / "check.mp4"
    else:
        target_mp4 = Path(output_mp4).resolve()

    if target_mp4 == vpath:
        raise ValueError("Output video path must not overwrite the original video!")

    target_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Run ffmpeg from ndir so that subtitle filename does not require Windows drive escaping
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(vpath),
        "-i", str(wav_path.name),
        "-map", "0:v",
        "-map", "1:a",
        "-vf", f"subtitles={srt_path.name}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(target_mp4),
    ]

    logger.info("Running ffmpeg burn-in in %s...", ndir)
    res = subprocess.run(cmd, cwd=ndir, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("FFmpeg failed: %s", res.stderr)
        raise RuntimeError(f"FFmpeg failed with exit code {res.returncode}:\n{res.stderr}")

    return target_mp4


def main() -> None:
    parser = argparse.ArgumentParser(description="Burn subtitles and narration audio into a video.")
    parser.add_argument("--video", "-v", required=True, help="Video path or video ID (e.g. synth_v2_train_f0_0001).")
    parser.add_argument("--narration-dir", "-d", required=True, help="Directory containing narration.json/wav.")
    parser.add_argument("--output", "-o", default=None, help="Output MP4 path (defaults to <narration-dir>/check.mp4).")
    args = parser.parse_args()

    out = burn_in_video(args.video, args.narration_dir, args.output)
    print(f"Burn-in complete: {out}")


if __name__ == "__main__":
    main()
