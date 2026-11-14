"""gTTS wrapper. The rest of the code only calls synthesize() and duration_s(),
so switching engine later means editing synthesize() only."""
import hashlib
import pathlib
import time

CACHE = pathlib.Path("outputs/tts_cache")
LANG = "en"
TLD = "com"      # "com" = US, "co.in" = Indian English, "co.uk" = UK
RETRIES = 3


def synthesize(text: str) -> pathlib.Path:
    """Text -> mp3 path. Cached by (lang, accent, text): each sentence hits the network once."""
    from gtts import gTTS   # imported here so tests/imports work without the package
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{LANG}|{TLD}|{text}".encode()).hexdigest()[:12]
    out = CACHE / f"{key}.mp3"
    if out.exists() and out.stat().st_size > 0:
        return out
    last_err = None
    for attempt in range(RETRIES):
        try:
            gTTS(text, lang=LANG, tld=TLD).save(str(out))
            time.sleep(0.3)          # be gentle with the free endpoint
            return out
        except Exception as e:       # network error, rate limit, ...
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"TTS failed for {text!r}: {last_err}")


def duration_s(path) -> float:
    """Length of an audio file in seconds."""
    from moviepy import AudioFileClip
    with AudioFileClip(str(path)) as a:
        return float(a.duration)
