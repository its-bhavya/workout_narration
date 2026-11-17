"""No-render target planning with frozen, independently drawn video lengths.

K is drawn uniformly from {3,4,5} using a dedicated random.Random(seed).
The same K schedule is used in every train fold for comparable construction.
Test uses its own identical-seed stream and its longest feasible prefix; a
failed draw is never retried to favor shorter videos. No files are written.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.model_selection import StratifiedKFold

from plan_folds import EXERCISES


def draw_lengths(seed, count):
    rng = random.Random(seed)
    return [rng.randint(3, 5) for _ in range(count)]


def slot_targets(clips, cap, total):
    capacities = Counter(c.exercise for c in clips)
    mean = total / 3
    low = math.ceil(.9 * mean - 1e-9)
    limits = {e: min(capacities[e] * cap, math.floor(1.1 * mean + 1e-9)) for e in EXERCISES}
    if 3 * low > total or min(limits.values()) < low or sum(limits.values()) < total:
        raise ValueError(f"Cannot balance {total} slots at cap {cap}: capacities={dict(capacities)}")
    slots = dict.fromkeys(EXERCISES, low)
    while sum(slots.values()) < total:
        e = min((e for e in EXERCISES if slots[e] < limits[e]), key=lambda e: (slots[e], e))
        slots[e] += 1
    return slots


def plan_lengths(clips, cap, lengths, seed):
    """Find exercise patterns, then assign concrete clips by least-used first."""
    from concatenate_videos import build_raw_segments

    slots = slot_targets(clips, cap, sum(lengths))
    # Enumerate every legal pattern, retaining one representative per slot
    # vector. Integer packing enforces both the fixed K histogram and quotas.
    representatives = {}
    for k in [3, 4, 5]:
        for pattern in itertools.product(EXERCISES, repeat=k):
            if all(a != b for a, b in zip(pattern, pattern[1:])):
                vector = tuple(pattern.count(e) for e in EXERCISES)
                representatives.setdefault((k, vector), pattern)
    keys = sorted(representatives)
    matrix = np.array([[int(k == size) for k, _ in keys] for size in [3, 4, 5]]
                      + [[v[i] for _, v in keys] for i in range(3)], dtype=float)
    target = np.array([lengths.count(k) for k in [3, 4, 5]] + [slots[e] for e in EXERCISES])
    # Stable small objective breaks ties between exercise-pattern mixtures.
    solution = milp(np.arange(1, len(keys) + 1, dtype=float), integrality=np.ones(len(keys)),
                    bounds=Bounds(0, np.inf), constraints=LinearConstraint(matrix, target, target),
                    options={"mip_rel_gap": 0.0})
    if not solution.success:
        raise ValueError(f"No legal pattern packing: {solution.message}")
    amounts = np.rint(solution.x).astype(int)
    assert np.array_equal(matrix @ amounts, target)
    patterns_by_k = {k: [] for k in [3, 4, 5]}
    rng = random.Random(seed)
    for key, amount in zip(keys, amounts):
        patterns_by_k[key[0]].extend([representatives[key]] * amount)
    for patterns in patterns_by_k.values():
        rng.shuffle(patterns)
    by_ex = {e: sorted([c for c in clips if c.exercise == e], key=lambda c: c.clip_id) for e in EXERCISES}
    usage = Counter({c.clip_id: 0 for c in clips})
    sequences = []
    for k in lengths:
        pattern = patterns_by_k[k].pop()
        if rng.randrange(2):
            pattern = pattern[::-1]
        chosen = []
        for exercise in pattern:
            available = [c for c in by_ex[exercise] if usage[c.clip_id] < cap
                         and c.clip_id not in {x.clip_id for x in chosen}]
            if not available:
                raise ValueError("Cannot assign distinct clips within this video")
            least = min(usage[c.clip_id] for c in available)
            clip = rng.choice([c for c in available if usage[c.clip_id] == least])
            usage[clip.clip_id] += 1
            chosen.append(clip)
            values = [usage[c.clip_id] for c in by_ex[exercise]]
            assert max(values) - min(values) <= 1
        assert len({c.clip_id for c in chosen}) == k
        assert all(a.exercise != b.exercise for a, b in zip(chosen, chosen[1:]))
        sequences.append(chosen)
    transition_frames = frames = 0
    contributions = Counter()
    for c in clips:
        segments, count = build_raw_segments([c], "transition")
        margin = sum(s.end - s.start for s in segments if s.seg_type == "transition")
        contributions[c.clip_id] = margin * usage[c.clip_id]
        transition_frames += contributions[c.clip_id]
        frames += count * usage[c.clip_id]
    assert Counter(c.exercise for seq in sequences for c in seq) == Counter(slots)
    assert max(usage.values()) <= cap
    return dict(videos=len(lengths), lengths=lengths, slots=slots, joins=sum(lengths)-len(lengths),
                transition_frames=transition_frames, frames=frames, share=transition_frames/frames,
                usage=usage, contributions=contributions, sequences=sequences)


def longest_test_plan(clips, cap, seed):
    # Balance feasibility becomes impossible once the smallest exercise cannot
    # supply 0.3 * total slots. Because lengths are positive, no later prefix
    # can repair that capacity failure.
    draws = draw_lengths(seed, len(clips) * cap // 3 + 1)
    best = None
    capacities = Counter(c.exercise for c in clips)
    for n in range(1, len(draws) + 1):
        total = sum(draws[:n])
        if total > len(clips) * cap or math.ceil(.3 * total - 1e-9) > min(capacities.values()) * cap:
            break
        try:
            slot_targets(clips, cap, total)
        except ValueError:
            # Small prefixes can fail only because integer rounding prevents
            # +/-10% balance (e.g. 5 slots). A later prefix can repair that.
            continue
        best = plan_lengths(clips, cap, draws[:n], seed)
    if best is None:
        raise ValueError("No feasible test prefix")
    return best, draws[len(best["lengths"])]


def fold_margins(clips, fold_map):
    from concatenate_videos import build_raw_segments
    margins = np.zeros(5, dtype=np.int64)
    frames = np.zeros(5, dtype=np.int64)
    for c in clips:
        segments, count = build_raw_segments([c], "transition")
        fold = fold_map[c.clip_id]
        margins[fold] += sum(s.end-s.start for s in segments if s.seg_type == "transition")
        frames[fold] += count
    return margins, frames


def propose_redraw(clips, current_seed, candidates):
    """Select on all train source-clip margin totals; never model performance."""
    from concatenate_videos import build_raw_segments
    ordered = sorted(clips, key=lambda c: c.clip_id)
    exercises = [c.exercise for c in ordered]
    weights = []
    for c in ordered:
        segments, _ = build_raw_segments([c], "transition")
        weights.append(sum(s.end-s.start for s in segments if s.seg_type == "transition"))
    weights = np.array(weights)
    best = None
    for seed in range(current_seed + 1, current_seed + candidates + 1):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        assignment = np.empty(len(ordered), dtype=int)
        for fold, (_, held_out) in enumerate(cv.split(np.zeros(len(ordered)), exercises)):
            assignment[held_out] = fold
        totals = np.bincount(assignment, weights=weights, minlength=5)
        # Minimax total-margin deviation, then dispersion, then smaller seed.
        score = (float(np.max(np.abs(totals-totals.mean()))), float(np.std(totals)), seed)
        if best is None or score < best[0]:
            best = (score, seed, dict(zip([c.clip_id for c in ordered], assignment.tolist())))
    return best[1], best[2]


def print_plan(label, plan):
    print(f"{label}: videos={plan['videos']}, K_hist={dict(sorted(Counter(plan['lengths']).items()))}, "
          f"slots={plan['slots']}, joins={plan['joins']}, "
          f"transition_frames={plan['transition_frames']}/{plan['frames']} ({100*plan['share']:.4f}%)")


def report_target_plan(args):
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
    lengths = draw_lengths(args.seed, args.n_train_per_fold)
    print(f"PLAN ONLY run={args.run_id}: independent uniform K=3..5 stream, seed={args.seed}")
    print(f"Identical train K schedule per fold: {lengths}")
    plans = []
    for fold in range(5):
        clips = [c for c in pool['train'] if fold_map[c.clip_id] == fold]
        plan = plan_lengths(clips, 5, lengths, args.seed)
        assert all({fold_map[c.clip_id] for c in seq} == {fold} for seq in plan['sequences'])
        plans.append(plan)
        print_plan(f"train fold {fold}", plan)
    print("Fold 1 top five transition contributors (including all planned uses):")
    clip_map = {c.clip_id: c for c in pool['train']}
    for clip_id, frames in plans[1]['contributions'].most_common(5):
        uses = plans[1]['usage'][clip_id]
        print(f"  {clip_id}: exercise={clip_map[clip_id].exercise}, uses={uses}, "
              f"margin_frames_per_use={frames//uses}, total_transition_frames={frames}, seconds={frames/30:.3f}")
    test, next_k = longest_test_plan(pool['test'], args.test_cap, args.seed)
    print_plan(f"test cap {args.test_cap} (longest frozen-draw prefix)", test)
    print(f"Test K schedule: {test['lengths']}; next K={next_k}, next total slots={sum(test['lengths'])+next_k}")
    margins, frames = fold_margins(pool['train'], fold_map)
    print(f"Current unique-clip margins by fold: {margins.tolist()}; shares={(100*margins/frames).round(4).tolist()}%")
    if args.redraw_candidates:
        seed, proposed = propose_redraw(pool['train'], args.seed, args.redraw_candidates)
        margins, frames = fold_margins(pool['train'], proposed)
        print(f"PROPOSAL ONLY: minimum margin imbalance among stratified seeds {args.seed+1}..{args.seed+args.redraw_candidates}: seed={seed}")
        print(f"Proposed unique-clip margins: {margins.tolist()}; shares={(100*margins/frames).round(4).tolist()}%")
        for fold in range(5):
            clips = [c for c in pool['train'] if proposed[c.clip_id] == fold]
            print(f"  proposed fold {fold} clips={dict(Counter(c.exercise for c in clips))}")
            print_plan(f"  proposed train fold {fold}", plan_lengths(clips, 5, lengths, args.seed))
    print("No videos rendered, no manifests or fold assignments written. Await approval.")
