"""Coaching phrases. Rep phrases are deliberately SHORT: a RepCount rep can last well under 1 s.

Deterministic TEMPLATES are the default. If use_bank=True and data/phrase_bank.json exists
(generated once by llm_generator.py), its phrases replace the templates for each event type
that has at least 2 valid phrases. Otherwise we fall back to TEMPLATES.
"""
import json
import random
from pathlib import Path

TEMPLATES = {
    "segment_start": [
        "Let's begin {exercise}.",
        "Time for {exercise}. Here we go.",
        "Starting {exercise} now.",
        "Ready? Let's do {exercise}.",
    ],
    "rep": [
        "{n}.",
        "Nice, {n}.",
        "{n}, keep it up.",
        "Good, {n}.",
    ],
    "milestone": [
        "{n} reps, great pace.",
        "That's {n}. You're doing great.",
        "{n} down, keep going.",
    ],
    "transition": [
        "{prev} done. Next, {next}.",
        "Nice work. Up next, {next}.",
        "Great job. Now, {next}.",
    ],
    "workout_end": [
        "Great job on {exercise}. Workout complete.",
        "{exercise} done. That's the workout, well done.",
    ],
}

# Repo labels -> speakable words (TTS must never read "push_up" aloud).
SPOKEN_NAME = {
    "squat": "squats",
    "push_up": "push ups",
    "jump_jack": "jumping jacks",
}

BANK_PATH = Path("data/phrase_bank.json")


def load_templates(use_bank=False):
    """Deterministic TEMPLATES, optionally overridden by the LLM phrase bank."""
    merged = {k: list(v) for k, v in TEMPLATES.items()}
    if use_bank and BANK_PATH.exists():
        try:
            bank = json.loads(BANK_PATH.read_text())
        except json.JSONDecodeError:
            return merged                      # corrupt bank -> fall back to templates
        for event, phrases in bank.items():
            # need 2+ phrases so the "never the same one twice in a row" logic works
            if event in merged and isinstance(phrases, list) and len(phrases) >= 2:
                merged[event] = phrases
    return merged


class Narrator:
    """Picks a template per event type, never the same one twice in a row."""

    def __init__(self, seed=42, use_bank=False):
        self.rng = random.Random(seed)
        self.last = {}
        self.templates = load_templates(use_bank)

    def say(self, event, **slots):
        options = self.templates[event]
        idxs = [i for i in range(len(options)) if i != self.last.get(event)]
        i = self.rng.choice(idxs)
        self.last[event] = i
        slots = dict(slots)
        for key in ("exercise", "prev", "next"):
            if key in slots:
                slots[key] = SPOKEN_NAME.get(slots[key], slots[key].replace("_", " "))
        text = options[i].format(**slots)
        return text[0].upper() + text[1:]