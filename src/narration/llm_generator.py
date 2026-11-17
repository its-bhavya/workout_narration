"""Generate a validated phrase bank once, offline, using Gemini (ONE API request). Run:
    uv run python -m src.narration.llm_generator
Needs GEMINI_API_KEY in the environment. Never commit the key.
"""
import json
import os
import re
import time
from pathlib import Path

BANK = Path("data/phrase_bank.json")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
RETRIES = 3

# event type -> (required placeholders, max words, how many to request, instruction, examples)
SPEC = {
    "segment_start": ({"exercise"}, 8, 8,
                      "announce that the exercise is about to begin",
                      ["Let's begin {exercise}.", "Time for {exercise}. Here we go."]),
    "rep":           ({"n"}, 4, 10,
                      "call out the current rep number briefly",
                      ["{n}.", "Nice, {n}."]),
    "milestone":     ({"n"}, 8, 8,
                      "celebrate reaching a rep milestone",
                      ["{n} reps, great pace.", "That's {n}. You're doing great."]),
    "transition":    ({"prev", "next"}, 9, 8,
                      "praise the finished exercise and introduce the next one",
                      ["{prev} done. Next, {next}."]),
    "workout_end":   ({"exercise"}, 10, 6,
                      "congratulate the person on finishing the whole workout",
                      ["Great job on {exercise}. Workout complete."]),
}


def build_prompt():
    lines = []
    for event, (ph, max_words, count, hint, examples) in SPEC.items():
        placeholders = ", ".join("{" + p + "}" for p in sorted(ph))
        lines.append(
            f'- "{event}": {count} different phrases that {hint}. Each must contain '
            f'{placeholders} exactly as written, at most {max_words} words. '
            f'Style examples (do not copy): {" | ".join(examples)}'
        )
    return (
        "You write short lines for a voice coach that speaks during a workout video. "
        "Lines are read aloud by text-to-speech. {exercise}, {prev} and {next} are replaced "
        "by plural exercise names such as 'squats' or 'jumping jacks'; {n} is replaced by a "
        "number. No emojis, no digits, no hashtags. "
        "Return ONLY a JSON object whose keys are the event names below and whose values "
        "are arrays of strings:\n" + "\n".join(lines)
    )


def _call_llm(prompt):
    from google import genai
    from google.genai import types

    client = genai.Client(http_options=types.HttpOptions(timeout=90000))  # 90 s, in ms
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.9,
        max_output_tokens=8192,     # room for the whole answer, so the JSON isn't cut off
    )
    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
            if not resp.text:
                cand = resp.candidates[0] if resp.candidates else None
                reason = getattr(cand, "finish_reason", "unknown")
                raise RuntimeError(f"empty response (finish_reason={reason})")
            return resp.text
        except Exception as e:
            last_err = e
            msg = str(e)
            if "503" in msg or "DEADLINE" in msg:     # temporary only; never retry 429 (daily quota)
                wait = 5 * (attempt + 1)
                print(f"  temporary error, retrying in {wait}s...", flush=True)
                time.sleep(wait)
                continue
            raise
    raise last_err


def valid(phrase, placeholders, max_words):
    if not isinstance(phrase, str):
        return False
    found = set(re.findall(r"\{(\w+)\}", phrase))
    if found != placeholders:
        return False
    if len(phrase.split()) > max_words:
        return False
    if re.search(r"\d", phrase) or re.search(r"[^\w\s{}.,'!?-]", phrase):
        return False
    return True


def generate():
    print(f"Requesting all phrases from {MODEL} (1 request)...", flush=True)
    try:
        raw = _call_llm(build_prompt())
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
    except Exception as e:
        print(f"Generation failed: {e}")
        print("Keeping the existing bank (if any); templates are used otherwise.")
        return

    bank = {}
    for event, (ph, max_words, count, hint, examples) in SPEC.items():
        phrases = data.get(event, [])
        if not isinstance(phrases, list):
            phrases = []
        bank[event] = [p for p in phrases if valid(p, ph, max_words)]
        print(f"{event}: kept {len(bank[event])}/{len(phrases)}")

    if not any(bank.values()):
        print("No valid phrases; keeping the existing bank (if any).")
        return
    BANK.parent.mkdir(exist_ok=True)
    BANK.write_text(json.dumps(bank, indent=2))
    print(f"wrote {BANK}")


if __name__ == "__main__":
    generate()