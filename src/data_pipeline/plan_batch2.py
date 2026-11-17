"""
plan_batch2.py — Step 1 analysis for Phase 5 batch 2 (APPEND MODE).

Computes and reports:
  1.1  Per-split state: existing videos, clips used, times_used distribution,
       clips still unused, groups used.
  1.2  Max additional videos per split under cumulative caps
       (train 5, val 2, test 1; clip-never-twice-in-one-video; adjacent
       exercises differ; ≥2 distinct exercises; K=3..5; fps<20 and
       manual_exclusions excluded).
  1.3  Cumulative exercise balance (±10%), expected joins, transition-frame
       share (batch 2 vs batch 1).

Run from repo root:
    uv run python src/data_pipeline/plan_batch2.py
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

SPLITS_FILE       = Path("data/manifests/splits_v2.csv")
BOUNDARIES_FILE    = Path("data/manifests/repcount_rep_boundaries.json")
EXCLUSIONS_FILE   = Path("data/manifests/manual_exclusions.csv")
SYNTH_INDEX_FILE  = Path("data/manifests/synthetic_index.csv")

INCLUDE_STATUSES   = {"pass", "review"}
EXCLUDE_FPS_BELOW  = 20.0

CUMULATIVE_CAPS    = {"train": 5, "val": 2, "test": 1}
K_MIN, K_MAX       = 3, 5

TARGET_BATCH2      = {"train": 72, "val": 6, "test": None}  # test decided here


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_exclusions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df["clip_id"].tolist())


def build_pool(splits_df: pd.DataFrame, boundaries: dict,
               exclusions: set[str]) -> dict[str, list[dict]]:
    """Return {split -> [clip_info_dict]} after all filters."""
    pool: dict[str, list[dict]] = defaultdict(list)
    skipped = {"missing": 0, "fps": 0, "excluded": 0, "no_pairs": 0}

    for _, row in splits_df.iterrows():
        clip_id = row["clip_id"]
        if clip_id in exclusions:
            skipped["excluded"] += 1
            continue
        binfo = boundaries.get(clip_id)
        if binfo is None:
            skipped["missing"] += 1
            continue
        if binfo.get("fps", 0) < EXCLUDE_FPS_BELOW:
            skipped["fps"] += 1
            continue
        if not binfo.get("pairs"):
            skipped["no_pairs"] += 1
            continue
        pool[row["split"]].append({
            "clip_id":       clip_id,
            "exercise":      row["exercise"],
            "split":         row["split"],
            "group":         row["group"],
            "original_split": row["original_split"],
            "fps":            binfo["fps"],
            "total_frames":   binfo["total_frames"],
            "first_start":    binfo["pairs"][0][0],
            "last_end":       binfo["pairs"][-1][1],
            "rep_count":      row["rep_count"],
        })
    return pool, skipped


def compute_current_usage(index_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return {split -> {clip_id -> times_used}} from existing index."""
    usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in index_df.iterrows():
        split = row["split"]
        for clip_id in row["source_clips"].split("|"):
            usage[split][clip_id] += 1
    return usage


def max_additional_videos(pool_clips, current_usage, cap, k_min, k_max):
    """
    Upper-bound estimate of additional videos under cumulative cap.

    Returns (max_videos_upper, detail_dict).

    Method:
      - remaining_slots per clip = cap - current_usage  (≥0)
      - total_remaining = sum of all remaining_slots
      - upper = total_remaining // k_min   (every video uses at least k_min clips)
      - exercise feasibility: need ≥2 exercises with remaining slots > 0
      - minority bound: if we sort exercises by remaining slots, the
        smallest two combined must be ≥ number_of_videos (each video needs
        at least 1 clip from a non-majority exercise to satisfy ≥2 distinct
        + adjacent-differ).  A safe bound: minority_bound = sum of all but
        the largest exercise's slots.  max_videos ≤ minority_bound.
      - Final = min(upper, minority_bound).
    """
    # remaining per clip
    remaining: dict[str, int] = {}
    for c in pool_clips:
        cid = c["clip_id"]
        used = current_usage.get(cid, 0)
        rem = cap - used
        if rem > 0:
            remaining[cid] = rem

    # remaining slots per exercise
    ex_slots: dict[str, int] = defaultdict(int)
    for c in pool_clips:
        cid = c["clip_id"]
        rem = remaining.get(cid, 0)
        if rem > 0:
            ex_slots[c["exercise"]] += rem

    total_remaining = sum(ex_slots.values())
    upper = total_remaining // k_min

    # minority bound: sum of all exercises except the largest
    sorted_ex = sorted(ex_slots.values(), reverse=True)
    if len(sorted_ex) < 2:
        minority_bound = 0
    else:
        minority_bound = sum(sorted_ex[1:])  # all but the largest

    max_v = min(upper, minority_bound)

    # clips with remaining capacity
    clips_available = sum(1 for v in remaining.values() if v > 0)

    detail = {
        "total_remaining_slots": total_remaining,
        "upper_bound_div_kmin": upper,
        "minority_bound": minority_bound,
        "max_additional_videos": max_v,
        "ex_remaining_slots": dict(ex_slots),
        "clips_with_remaining": clips_available,
    }
    return max_v, detail


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    raise RuntimeError("Contributor-grouped/v1 workflow retired; use build_splits_v2.py and the fold-aware v2 planner")
    # ---- Load data ----
    splits_df = pd.read_csv(SPLITS_FILE)
    splits_df = splits_df[splits_df["status"].isin(INCLUDE_STATUSES)].copy()

    with open(BOUNDARIES_FILE) as f:
        boundaries = json.load(f)

    exclusions = load_exclusions(EXCLUSIONS_FILE)
    pool, skipped = build_pool(splits_df, boundaries, exclusions)

    index_df = pd.read_csv(SYNTH_INDEX_FILE)
    current_usage = compute_current_usage(index_df)

    print("=" * 72)
    print("PHASE 5 — STEP 1: BATCH 2 PLAN (APPEND MODE)")
    print("=" * 72)

    # ---- Exclusion summary ----
    print(f"\nPool filters: missing={skipped['missing']}, "
          f"fps<{EXCLUDE_FPS_BELOW:.0f}={skipped['fps']}, "
          f"excluded={skipped['excluded']}, "
          f"no_pairs={skipped['no_pairs']}")
    print(f"Manual exclusions file: {'exists' if EXCLUSIONS_FILE.exists() else 'absent (empty set)'}")

    # ================================================================
    # 1.1  Per-split state
    # ================================================================
    print("\n" + "=" * 72)
    print("1.1  PER-SPLIT STATE (BATCH 1)")
    print("=" * 72)

    batch2_targets = {}
    max_videos_detail = {}

    for split in ["train", "val", "test"]:
        split_pool = pool.get(split, [])
        split_videos = index_df[index_df["split"] == split]
        n_videos = len(split_videos)

        # clips used in batch 1
        usage_split = current_usage.get(split, {})
        clips_used_ids = set(usage_split.keys())
        clips_used_count = len(clips_used_ids)

        # times_used distribution
        times_dist = Counter(usage_split.values())
        dist_str = ", ".join(f"{k}x: {v} clips" for k, v in sorted(times_dist.items()))

        # clips in pool (after filters)
        pool_clip_ids = {c["clip_id"] for c in split_pool}
        pool_size = len(pool_clip_ids)

        # clips still unused (in pool but not used in any batch 1 video)
        unused_ids = pool_clip_ids - clips_used_ids
        unused_count = len(unused_ids)

        # clips used but not in pool (shouldn't happen, but check)
        used_not_in_pool = clips_used_ids - pool_clip_ids

        # groups used
        groups_in_pool = {c["group"] for c in split_pool}
        groups_used = set()
        for c in split_pool:
            if c["clip_id"] in clips_used_ids:
                groups_used.add(c["group"])
        groups_unused = groups_in_pool - groups_used

        # exercise distribution in pool
        ex_pool = Counter(c["exercise"] for c in split_pool)

        # exercise distribution of used clips (by clip slots)
        ex_slots_used = defaultdict(int)
        for c in split_pool:
            cid = c["clip_id"]
            if cid in usage_split:
                ex_slots_used[c["exercise"]] += usage_split[cid]

        # exercise distribution of unused clips
        ex_unused = defaultdict(int)
        for c in split_pool:
            if c["clip_id"] in unused_ids:
                ex_unused[c["exercise"]] += 1

        print(f"\n--- {split.upper()} ---")
        print(f"  Pool size (after filters): {pool_size} clips")
        print(f"  Exercises in pool: {dict(ex_pool)}")
        print(f"  Groups in pool: {len(groups_in_pool)} -> {sorted(groups_in_pool)}")
        print(f"  Existing videos: {n_videos}")
        print(f"  Clips used (distinct): {clips_used_count}")
        print(f"  Clip slots used (total): {sum(usage_split.values())}")
        print(f"  times_used distribution: {dist_str}")
        print(f"  Clips still unused: {unused_count}")
        print(f"  Unused clips by exercise: {dict(ex_unused)}")
        print(f"  Groups used: {len(groups_used)} -> {sorted(groups_used)}")
        print(f"  Groups unused: {len(groups_unused)} -> {sorted(groups_unused)}")
        if used_not_in_pool:
            print(f"  WARNING: {len(used_not_in_pool)} clips used in videos but NOT in pool: {sorted(used_not_in_pool)}")
        print(f"  Clip slots used by exercise: {dict(ex_slots_used)}")

        # check duplicate clips within single videos
        dup_in_video = 0
        dup_examples = []
        for _, row in split_videos.iterrows():
            clips = row["source_clips"].split("|")
            seen = set()
            for c in clips:
                if c in seen:
                    dup_in_video += 1
                    dup_examples.append((row["video_id"], c))
                seen.add(c)
        if dup_in_video:
            print(f"  NOTE: {dup_in_video} clips appear twice within a single video "
                  f"(constraint 'clip never twice in one video' was NOT enforced in batch 1)")
            for vid, cid in dup_examples[:5]:
                print(f"    {vid}: {cid}")

    # ================================================================
    # 1.2  Max additional videos
    # ================================================================
    print("\n" + "=" * 72)
    print("1.2  MAX ADDITIONAL VIDEOS UNDER CUMULATIVE CAPS")
    print("=" * 72)

    for split in ["train", "val", "test"]:
        split_pool = pool.get(split, [])
        usage_split = current_usage.get(split, {})
        cap = CUMULATIVE_CAPS[split]

        max_v, detail = max_additional_videos(
            split_pool, usage_split, cap, K_MIN, K_MAX
        )

        max_videos_detail[split] = detail

        print(f"\n--- {split.upper()} (cumulative cap = {cap} uses/clip) ---")
        print(f"  Clips with remaining capacity: {detail['clips_with_remaining']}")
        print(f"  Total remaining clip-slots: {detail['total_remaining_slots']}")
        print(f"  Remaining slots by exercise: {detail['ex_remaining_slots']}")
        print(f"  Upper bound (total_slots // K_MIN={K_MIN}): {detail['upper_bound_div_kmin']}")
        print(f"  Minority bound (all-but-largest exercise): {detail['minority_bound']}")
        print(f"  MAX ADDITIONAL VIDEOS: {detail['max_additional_videos']}")

    # Test-specific: how many unused test clips, max test videos (cap 5)
    test_pool = pool.get("test", [])
    test_usage = current_usage.get("test", {})
    test_unused = [c for c in test_pool if test_usage.get(c["clip_id"], 0) == 0]
    test_unused_by_ex = Counter(c["exercise"] for c in test_unused)
    test_max_from_unused = len(test_unused) // K_MIN
    test_max_capped = min(test_max_from_unused, 5)
    print(f"\n--- TEST (specific) ---")
    print(f"  Unused test clips: {len(test_unused)} ({dict(test_unused_by_ex)})")
    print(f"  Max videos from unused clips (// {K_MIN}): {test_max_from_unused}")
    print(f"  Capped at 5: {test_max_capped}")
    batch2_targets["test"] = test_max_capped

    batch2_targets["train"] = 72
    batch2_targets["val"] = 6

    print(f"\n  >>> BATCH 2 TARGETS: train=+{batch2_targets['train']} "
          f"(total {24 + batch2_targets['train']}), "
          f"val=+{batch2_targets['val']} (total {6 + batch2_targets['val']}), "
          f"test=+{batch2_targets['test']} (total {15 + batch2_targets['test']})")

    # feasibility check
    print(f"\n  Feasibility:")
    for split in ["train", "val", "test"]:
        target = batch2_targets[split]
        max_v = max_videos_detail[split]["max_additional_videos"]
        feasible = target <= max_v
        print(f"    {split}: target=+{target}, max={max_v} -> {'FEASIBLE' if feasible else 'NOT FEASIBLE'}")

    # ================================================================
    # 1.3  Exercise balance, joins, transition-frame share
    # ================================================================
    print("\n" + "=" * 72)
    print("1.3  EXERCISE BALANCE, JOINS, TRANSITION-FRAME SHARE")
    print("=" * 72)

    # --- Batch 1 stats ---
    for split in ["train", "val", "test"]:
        split_videos = index_df[index_df["split"] == split]
        usage_split = current_usage.get(split, {})
        split_pool = pool.get(split, [])

        # clip slots by exercise in batch 1
        ex_slots_b1 = defaultdict(int)
        for c in split_pool:
            cid = c["clip_id"]
            if cid in usage_split:
                ex_slots_b1[c["exercise"]] += usage_split[cid]

        total_slots_b1 = sum(ex_slots_b1.values())
        n_videos_b1 = len(split_videos)
        mean_slots_b1 = total_slots_b1 / len(ex_slots_b1) if ex_slots_b1 else 0

        # joins = sum of (n_clips - 1) per video
        total_joins_b1 = 0
        total_clips_b1 = 0
        trans_shares_b1 = []
        for _, row in split_videos.iterrows():
            n_clips = len(row["source_clips"].split("|"))
            total_joins_b1 += (n_clips - 1)
            total_clips_b1 += n_clips
            trans_shares_b1.append(row["transition_frame_share"])

        avg_joins_b1 = total_joins_b1 / n_videos_b1 if n_videos_b1 else 0
        avg_clips_b1 = total_clips_b1 / n_videos_b1 if n_videos_b1 else 0
        avg_trans_b1 = sum(trans_shares_b1) / len(trans_shares_b1) if trans_shares_b1 else 0

        # --- Batch 2 estimate ---
        target_b2 = batch2_targets[split]
        # estimate clip slots: batch 2 will use K=3..5, uniform assumption → avg K
        # Use same avg K as batch 1 for estimate
        est_k_b2 = avg_clips_b1 if avg_clips_b1 > 0 else 4.0
        est_total_slots_b2 = target_b2 * est_k_b2

        # remaining slots by exercise
        cap = CUMULATIVE_CAPS[split]
        ex_remaining = defaultdict(int)
        for c in split_pool:
            rem = cap - usage_split.get(c["clip_id"], 0)
            if rem > 0:
                ex_remaining[c["exercise"]] += rem
        total_remaining = sum(ex_remaining.values())

        # proportional allocation of batch 2 slots per exercise
        ex_slots_b2_est = {}
        if total_remaining > 0:
            for ex, rem in ex_remaining.items():
                frac = rem / total_remaining
                ex_slots_b2_est[ex] = round(est_total_slots_b2 * frac)
        else:
            for ex in ex_remaining:
                ex_slots_b2_est[ex] = 0

        # cumulative
        ex_slots_cum = {}
        all_ex = set(ex_slots_b1.keys()) | set(ex_slots_b2_est.keys())
        for ex in all_ex:
            ex_slots_cum[ex] = ex_slots_b1.get(ex, 0) + ex_slots_b2_est.get(ex, 0)

        total_cum = sum(ex_slots_cum.values())
        mean_cum = total_cum / len(ex_slots_cum) if ex_slots_cum else 0
        lo = mean_cum * 0.9
        hi = mean_cum * 1.1

        balance_ok = all(lo <= v <= hi for v in ex_slots_cum.values()) if ex_slots_cum else True

        # joins estimate for batch 2
        est_joins_b2 = target_b2 * (est_k_b2 - 1)
        est_trans_b2 = avg_trans_b1  # same clip population → similar margins

        print(f"\n--- {split.upper()} ---")
        print(f"  Batch 1:")
        print(f"    Videos: {n_videos_b1}")
        print(f"    Clip slots by exercise: {dict(ex_slots_b1)}")
        print(f"    Total clip slots: {total_slots_b1}")
        print(f"    Avg clips/video: {avg_clips_b1:.1f}")
        print(f"    Total joins: {total_joins_b1} (avg {avg_joins_b1:.1f}/video)")
        print(f"    Transition-frame share: mean={avg_trans_b1:.4f}, "
              f"min={min(trans_shares_b1) if trans_shares_b1 else 0:.4f}, "
              f"max={max(trans_shares_b1) if trans_shares_b1 else 0:.4f}")

        print(f"  Batch 2 estimate (+{target_b2} videos):")
        print(f"    Est avg clips/video: {est_k_b2:.1f} (assumed same as batch 1)")
        print(f"    Est total clip slots: {est_total_slots_b2:.0f}")
        print(f"    Remaining slots by exercise: {dict(ex_remaining)}")
        print(f"    Proportional batch 2 allocation: {ex_slots_b2_est}")

        print(f"  Cumulative (batch 1 + batch 2):")
        print(f"    Clip slots by exercise: {ex_slots_cum}")
        print(f"    Mean: {mean_cum:.1f}")
        print(f"    ±10% band: [{lo:.1f}, {hi:.1f}]")
        print(f"    Within ±10%: {'YES' if balance_ok else 'NO'}")
        if not balance_ok:
            for ex, v in ex_slots_cum.items():
                status = "OK" if lo <= v <= hi else "OUT OF BAND"
                print(f"      {ex}: {v} slots -> {status}")

        print(f"  Joins:")
        print(f"    Batch 1 total: {total_joins_b1} (avg {avg_joins_b1:.1f}/video)")
        print(f"    Batch 2 est:   {est_joins_b2:.0f} (avg {est_k_b2 - 1:.1f}/video)")
        print(f"    Cumulative:    {total_joins_b1 + est_joins_b2:.0f}")
        print(f"  Transition-frame share:")
        print(f"    Batch 1 mean: {avg_trans_b1:.4f}")
        print(f"    Batch 2 est:  ~{est_trans_b2:.4f} (same clip population)")
        print(f"    Note: batch 2 clips are drawn from the same pool with the same")
        print(f"    margin structure, so transition share is expected to be similar.")

    print("\n" + "=" * 72)
    print("DONE — Step 1 analysis complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
