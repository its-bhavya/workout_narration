"""Place the audio clips on one silent track so you can listen to the whole narration."""
import numpy as np
from moviepy import AudioClip, AudioFileClip, CompositeAudioClip


def render_track(timeline, total_duration, out_path, fps=22050):
    clips = [AudioFileClip(t["audio"]).with_start(t["time"]) for t in timeline]
    if not clips:
        clips.append(AudioClip(lambda t: np.zeros_like(t)[..., None], duration=total_duration, fps=fps))
    try:
        track = CompositeAudioClip(clips).with_duration(total_duration)
        track.write_audiofile(str(out_path), fps=fps, logger=None)
    finally:
        for clip in clips:
            clip.close()
