"""Generate every template (+ rep numbers 1-30), time them, write outputs/tts_timing.csv.

    uv run python -m src.narration.tts_smoke_test
Then listen to the clips and fill in the last column (clear y/n).
"""
import csv
import pathlib

from .templates import TEMPLATES, SPOKEN_NAME
from .tts import synthesize, duration_s

rows = []


def add(event, text):
    p = synthesize(text)
    rows.append([event, text, round(duration_s(p), 2), str(p), ""])


sample = dict(n=5, exercise=SPOKEN_NAME["squat"], prev=SPOKEN_NAME["squat"], next=SPOKEN_NAME["push_up"])
for event, options in TEMPLATES.items():
    for t in options:
        add(event, t.format(**sample))

for spoken in SPOKEN_NAME.values():
    for t in TEMPLATES["segment_start"]:
        add("segment_start", t.format(exercise=spoken))
names = list(SPOKEN_NAME.values())
for prev in names:
    for nxt in names:
        if prev != nxt:
            for t in TEMPLATES["transition"]:
                add("transition", t.format(prev=prev, next=nxt))

for n in range(1, 31):
    for t in TEMPLATES["rep"]:
        add("rep", t.format(n=n))
for n in (5, 10, 15, 20, 25, 30):
    for t in TEMPLATES["milestone"]:
        add("milestone", t.format(n=n))

seen, unique = set(), []
for r in rows:
    if r[1] not in seen:
        seen.add(r[1])
        unique.append(r)

out = pathlib.Path("outputs/tts_timing.csv")
out.parent.mkdir(exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["event", "text", "duration_s", "audio_path", "clear (y/n)"])
    w.writerows(unique)

print(f"{len(unique)} unique clips -> {out}")
by = {}
for r in unique:
    by.setdefault(r[0], []).append(r[2])
for ev, ds in by.items():
    print(f"{ev:14s} n={len(ds):3d} mean={sum(ds)/len(ds):.2f}s min={min(ds):.2f}s max={max(ds):.2f}s")
