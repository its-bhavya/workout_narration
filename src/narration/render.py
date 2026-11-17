"""Place the audio clips on one silent track so you can listen to the whole narration."""
from moviepy import AudioFileClip, CompositeAudioClip


def render_track(timeline, total_duration, out_path, fps=22050):
    if timeline:   # never truncate the last clip
        total_duration = max(total_duration, timeline[-1]["end"] + 0.5)
    clips = [AudioFileClip(t["audio"]).with_start(t["time"]) for t in timeline]
    track = CompositeAudioClip(clips).with_duration(total_duration)
    track.write_audiofile(str(out_path), fps=fps, logger=None)
    for c in clips:
        c.close()