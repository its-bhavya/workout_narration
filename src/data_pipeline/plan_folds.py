"""Exact capacity bounds with concrete, no-render source-clip witnesses.

Balance means each exercise's slot count is within 10% of the mean count.
Maximum-video witnesses use K=3, the allowed minimum. Transition share is
frame-weighted over each actual witness, not a prediction from old videos.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd

EXERCISES = ["jump_jack", "push_up", "squat"]


def capacity_plan(clips, cap, seed, balanced=True):
    # Import only helpers; no renderer, video decoder or writer is invoked.
    from concatenate_videos import build_raw_segments

    by_ex = {e: sorted([c for c in clips if c.exercise == e], key=lambda c: c.clip_id)
             for e in EXERCISES}
    capacity = {e: len(by_ex[e]) * cap for e in EXERCISES}
    total = sum(capacity.values())
    # Every video needs >=3 slots and at least one minority-exercise slot.
    upper = min(total // 3, total - max(capacity.values()))
    if balanced:
        # If V videos exist, S>=3V; every exercise needs >=0.9*S/3
        # slots, hence V <= min_exercise_capacity / 0.9.
        upper = min(upper, min(capacity.values()) * 10 // 9)
    for videos in range(upper, 0, -1):
        low = math.ceil(9 * videos / 10) if balanced else 0
        high = math.floor(11 * videos / 10) if balanced else 2 * videos
        limits = {e: min(capacity[e], high) for e in EXERCISES}
        if any(limits[e] < low for e in EXERCISES) or sum(limits.values()) < 3 * videos:
            continue
        slots = {e: low for e in EXERCISES}
        while sum(slots.values()) < 3 * videos:
            e = min((e for e in EXERCISES if slots[e] < limits[e]), key=lambda e: (slots[e], e))
            slots[e] += 1
        # Begin with ABC in each video. Each excess replaces a deficit in a
        # different video; order the repeated exercise A as ABA afterwards.
        patterns = [list(EXERCISES) for _ in range(videos)]
        excess = [e for e in EXERCISES for _ in range(max(0, slots[e] - videos))]
        deficit = [e for e in EXERCISES for _ in range(max(0, videos - slots[e]))]
        if len(excess) > videos:
            continue
        for i, (add, remove) in enumerate(zip(excess, deficit)):
            patterns[i][patterns[i].index(remove)] = add
        rng = random.Random(seed)
        rng.shuffle(patterns)
        usage = Counter({c.clip_id: 0 for c in clips})
        sequences = []
        frames = transitions = 0
        for pattern in patterns:
            counts = Counter(pattern)
            repeated = next((e for e in counts if counts[e] == 2), None)
            if repeated:
                pattern = [repeated, next(e for e in counts if e != repeated), repeated]
            chosen = []
            for e in pattern:
                candidates = [c for c in by_ex[e] if usage[c.clip_id] < cap
                              and c.clip_id not in {x.clip_id for x in chosen}]
                assert candidates, "Capacity witness could not assign distinct clips"
                least = min(usage[c.clip_id] for c in candidates)
                clip = rng.choice([c for c in candidates if usage[c.clip_id] == least])
                usage[clip.clip_id] += 1
                chosen.append(clip)
                values = [usage[c.clip_id] for c in by_ex[e]]
                assert max(values) - min(values) <= 1, "Per-exercise reuse imbalance"
            assert len({c.clip_id for c in chosen}) == 3
            assert all(a.exercise != b.exercise for a, b in zip(chosen, chosen[1:]))
            segments, nframes = build_raw_segments(chosen, "transition")
            frames += nframes
            transitions += sum(s.end - s.start for s in segments if s.seg_type == "transition")
            sequences.append(chosen)
        assert max(usage.values()) <= cap
        assert Counter(c.exercise for seq in sequences for c in seq) == Counter(slots)
        deviation = max(abs(v / videos - 1) for v in slots.values())
        assert not balanced or deviation <= .1 + 1e-12
        return dict(videos=videos, joins=2 * videos, slots=slots,
                    transition_share=transitions / frames, deviation=deviation,
                    unused=total - 3 * videos, capacity=capacity, sequences=sequences)
    raise ValueError("No feasible K=3 plan for this pool/cap")


def report_fold_plan(args):
    from concatenate_videos import load_exclusions, load_pool

    splits = pd.read_csv(args.splits_file)
    folds = pd.read_csv(args.folds_file)
    assert splits.clip_id.is_unique and folds.clip_id.is_unique
    assert set(splits.split) == {"train", "test"}
    assert set(folds.fold) == set(range(5))
    assert set(folds.clip_id) == set(splits.loc[splits.split == "train", "clip_id"])
    assert (splits.split == splits.original_split.map({"train": "train", "val": "test", "test": "test"})).all()
    with Path(args.boundaries_file).open() as handle:
        boundaries = json.load(handle)
    pool = load_pool(Path(args.splits_file), boundaries, load_exclusions(Path(args.exclusions_file)))
    fold_map = dict(zip(folds.clip_id, folds.fold))
    buckets = [(f"train fold {f}", [c for c in pool["train"] if fold_map[c.clip_id] == f], 5)
               for f in range(5)]
    buckets += [(f"test cap {cap}", pool["test"], cap) for cap in sorted({1, 2, args.test_cap})]
    print(f"PLAN ONLY run={args.run_id}, seed={args.seed}; fresh run, no legacy usage imported")
    print("All maximum-video witnesses use K=3; joins and frame-weighted transition shares are exact for these witnesses.")
    for name, clips, cap in buckets:
        plan = capacity_plan(clips, cap, args.seed)
        relaxed = capacity_plan(clips, cap, args.seed, balanced=False)
        if name.startswith("train"):
            assert len({fold_map[c.clip_id] for seq in plan['sequences'] for c in seq}) == 1
        print(f"\n{name}: pool={dict(Counter(c.exercise for c in clips))}, cap={cap}")
        for label, result in [("balanced", plan), ("balance relaxed", relaxed)]:
            print(f"  {label}: max_videos={result['videos']}, joins={result['joins']}, "
                  f"transition_frame_share={result['transition_share']:.6f}, "
                  f"exercise_slots={result['slots']}, max_deviation={100*result['deviation']:.3f}%, "
                  f"unused_capacity_slots={result['unused']}")
    print("\nNo artifacts written; no rendering performed. Targets remain unselected.")
