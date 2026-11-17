"""Validate concrete fold plans against an independent K=3..5 optimizer."""
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data_pipeline"))
from concatenate_videos import load_pool
from plan_folds import EXERCISES, capacity_plan
from plan_targets import draw_lengths, longest_test_plan, plan_lengths, slot_targets


@pytest.mark.parametrize("bucket,cap", [(f, 5) for f in range(5)] + [("test", 1), ("test", 2)])
def test_plan_attains_independent_optimum(bucket, cap):
    root = Path(__file__).resolve().parents[1]
    manifests = root / "data" / "manifests"
    boundaries = json.loads((manifests / "repcount_rep_boundaries.json").read_text())
    pool = load_pool(manifests / "splits_v2.csv", boundaries, set())
    folds = pd.read_csv(manifests / "folds.csv").set_index("clip_id").fold.to_dict()
    clips = pool["test"] if bucket == "test" else [c for c in pool["train"] if folds[c.clip_id] == bucket]
    # Enumerate all exercise patterns, including K=4 and K=5, then solve the
    # aggregate integer packing problem independently of the planner's bound.
    patterns = set()
    for k in range(3, 6):
        for seq in itertools.product(range(3), repeat=k):
            if all(a != b for a, b in zip(seq, seq[1:])):
                patterns.add(tuple(seq.count(e) for e in range(3)))
    a = np.array(sorted(patterns), dtype=float).T
    sizes = a.sum(axis=0)
    capacities = np.array([sum(c.exercise == e for c in clips) * cap for e in EXERCISES])
    for balanced in [True, False]:
        matrix = a
        lower = np.zeros(3)
        upper = capacities
        if balanced:
            matrix = np.vstack([a, a - .3 * sizes, a - (1.1 / 3) * sizes])
            lower = np.r_[lower, np.zeros(3), np.full(3, -np.inf)]
            upper = np.r_[upper, np.full(3, np.inf), np.zeros(3)]
        solution = milp(-np.ones(a.shape[1]), integrality=np.ones(a.shape[1]),
                        bounds=Bounds(0, np.inf), constraints=LinearConstraint(matrix, lower, upper))
        assert solution.success
        plan = capacity_plan(clips, cap, 42, balanced=balanced)
        assert plan["videos"] == round(-solution.fun)
        usage = Counter(c.clip_id for seq in plan["sequences"] for c in seq)
        assert max(usage.values()) <= cap
        for seq in plan["sequences"]:
            assert 3 <= len(seq) <= 5
            assert len({c.clip_id for c in seq}) == len(seq)
            assert len({c.exercise for c in seq}) >= 2
            if bucket != "test":
                assert {folds[c.clip_id] for c in seq} == {bucket}
        for exercise in EXERCISES:
            values = [usage[c.clip_id] for c in clips if c.exercise == exercise]
            assert max(values) - min(values) <= 1
        repeated = capacity_plan(clips, cap, 42, balanced=balanced)
        assert [[c.clip_id for c in s] for s in repeated["sequences"]] == [
            [c.clip_id for c in s] for s in plan["sequences"]]


def test_selected_targets_and_frozen_test_prefix():
    manifests = Path(__file__).resolve().parents[1] / "data" / "manifests"
    boundaries = json.loads((manifests / "repcount_rep_boundaries.json").read_text())
    pool = load_pool(manifests / "splits_v2.csv", boundaries, set())
    folds = pd.read_csv(manifests / "folds.csv").set_index("clip_id").fold.to_dict()
    lengths = draw_lengths(42, 24)
    assert Counter(lengths) == {3: 13, 4: 2, 5: 9}
    for fold in range(5):
        clips = [c for c in pool["train"] if folds[c.clip_id] == fold]
        plan = plan_lengths(clips, 5, lengths, 42)
        assert len(plan['sequences']) == 24
        assert [len(seq) for seq in plan['sequences']] == lengths
        assert all({folds[c.clip_id] for c in seq} == {fold} for seq in plan['sequences'])
        assert sum(plan['slots'].values()) == 92
        assert max(abs(v/(92/3)-1) for v in plan['slots'].values()) <= .1
        assert plan['joins'] == 68
        counts = Counter(c.clip_id for seq in plan['sequences'] for c in seq)
        assert max(counts.values()) <= 5
        # Check reuse spread at every construction step, including zeros.
        usage = Counter()
        for seq in plan['sequences']:
            assert len({c.clip_id for c in seq}) == len(seq)
            for c in seq:
                usage[c.clip_id] += 1
                values = [usage[x.clip_id] for x in clips if x.exercise == c.exercise]
                assert max(values)-min(values) <= 1
        expected_frames = expected_margins = 0
        for c in clips:
            n = round(c.total_frames * 30 / c.src_fps)
            start = max(0, min(round(c.first_start * 30/c.src_fps), n))
            end = max(start, min(round((c.last_end+1)*30/c.src_fps), n))
            expected_frames += counts[c.clip_id] * n
            expected_margins += counts[c.clip_id] * (start+n-end)
        assert plan['frames'] == expected_frames
        assert plan['transition_frames'] == expected_margins
        assert sum(plan['contributions'].values()) == expected_margins
    test, next_k = longest_test_plan(pool['test'], 1, 42)
    assert test['videos'] == 21
    assert sum(test['lengths']) == 79
    assert test['slots'] == {'jump_jack': 27, 'push_up': 27, 'squat': 25}
    assert max(test['usage'].values()) == 1
    assert next_k == 5
    # The next frozen draw needs 84 slots: squat would require ceil(.3*84)
    # = 26 unique clips, but only 25 exist. Later prefixes only add slots.
    with pytest.raises(ValueError):
        slot_targets(pool['test'], 1, 84)
    # A short prefix's integer-balance failure must not terminate the search.
    with pytest.raises(ValueError):
        slot_targets(pool['test'], 1, 8)
