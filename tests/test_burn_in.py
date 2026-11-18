from pathlib import Path
import pytest
from src.narration.burn_in import format_srt_time, generate_srt, burn_in_video, resolve_video_path


def test_format_srt_time():
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(1.5) == "00:00:01,500"
    assert format_srt_time(65.123) == "00:01:05,123"
    assert format_srt_time(3661.05) == "01:01:01,050"


def test_generate_srt(tmp_path: Path):
    timeline = [
        {"time": 0.5, "end": 2.1, "event": "segment_start", "text": "Let's begin."},
        {"time": 5.0, "end": 6.2, "event": "rep", "text": "3."},
        {"time": 10.0, "end": 12.5, "event": "milestone", "text": "10 reps, great pace."},
    ]
    srt_file = tmp_path / "test.srt"
    out = generate_srt(timeline, srt_file)
    assert out.is_file()
    content = out.read_text(encoding="utf-8")
    assert "1\n00:00:00,500 --> 00:00:02,100\n[segment_start] Let's begin." in content
    assert "2\n00:00:05,000 --> 00:00:06,200\n[rep] 3." in content
    assert "3\n00:00:10,000 --> 00:00:12,500\n[milestone] 10 reps, great pace." in content


def test_burn_in_video_invokes_ffmpeg_correctly(tmp_path: Path, monkeypatch):
    import subprocess
    recorded = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None):
        recorded["cmd"] = cmd
        recorded["cwd"] = cwd
        # Simulate check.mp4 creation
        target = Path(cmd[-1])
        target.write_bytes(b"mock_mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"original_video")
    wav = tmp_path / "narration.wav"
    wav.write_bytes(b"mock_audio")
    srt = tmp_path / "narration.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")

    out = burn_in_video(video, tmp_path)
    assert out == tmp_path / "check.mp4"
    assert out.is_file()
    # Ensure original video is untouched
    assert video.read_bytes() == b"original_video"
    # Ensure run in tmp_path (avoids Windows path issues in vf)
    assert Path(recorded["cwd"]) == tmp_path
    cmd_str = " ".join(recorded["cmd"])
    assert "subtitles=narration.srt" in cmd_str
    assert "-map 0:v -map 1:a" in cmd_str
