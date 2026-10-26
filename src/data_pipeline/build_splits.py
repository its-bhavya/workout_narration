"""
Build leakage-safe train / val / test splits over RepCount clips that
survived quality screening.

Why not keep RepCount's native splits?
  * RepCount's native test split is small (13-26 clips per exercise).
  * RepCount's native splits share subjects (e.g. stu10_* appears in train,
    valid AND test), which leaks subject-specific movement style into test.

How it works
  1. Every clip gets a GROUP key. All clips with the same key always land in
     the same split:
        RepCount "stuN_*" -> the stuN prefix (ASSUMED to be one subject or
                             session; verify with filename_mapping.xlsx)
        other RepCount    -> the file itself (no subject info available)
  2. Groups are assigned greedily to train/val/test so each exercise ends up
     as close as possible to the target fractions.

Input : data/manifests/quality_screening.csv
Output: data/manifests/splits.csv
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

QUALITY_SCREENING_PATH = Path("data/manifests/quality_screening.csv")
OUTPUT_PATH = Path("data/manifests/splits.csv")

SPLIT_FRACTIONS = {"train": 0.65, "val": 0.10, "test": 0.25}

# Quality statuses to include. "review" clips are kept so that dropping
# one later (after manual review) cannot move any other clip between splits.
INCLUDE_STATUSES = {"pass", "review"}

SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STU_RE = re.compile(r"^(stu\d+)_", re.IGNORECASE)


# ---------------------------------------------------------------------
# Group keys
# ---------------------------------------------------------------------

def group_key(row: pd.Series) -> str:
    filename = row["clip_id"]

    match = STU_RE.match(filename)
    if match:
        return f"rc_subject:{match.group(1).lower()}"

    return f"rc_clip:{filename}"


# ---------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------

def assign_groups(df: pd.DataFrame) -> dict[str, str]:
    """
    Greedy assignment of whole groups to splits, minimising the squared
    deviation of every (split, exercise) count from its target.
    """

    exercises = sorted(df["exercise"].unique())
    splits = list(SPLIT_FRACTIONS)

    totals = df.groupby("exercise").size().to_dict()

    targets = {
        s: {e: SPLIT_FRACTIONS[s] * totals[e] for e in exercises}
        for s in splits
    }
    current = {s: {e: 0 for e in exercises} for s in splits}

    group_counts = (
        df.groupby(["group", "exercise"]).size().unstack(fill_value=0)
        .reindex(columns=exercises, fill_value=0)
    )

    rng = random.Random(SEED)

    # Largest groups first (hardest to place), seeded random tie-break.
    order = sorted(
        group_counts.index,
        key=lambda g: (-int(group_counts.loc[g].sum()), rng.random()),
    )

    def cost(candidate: str, add: pd.Series) -> float:
        total = 0.0
        for s in splits:
            for e in exercises:
                value = current[s][e] + (add[e] if s == candidate else 0)
                total += (value - targets[s][e]) ** 2 / max(targets[s][e], 1)
        return total

    assignment: dict[str, str] = {}

    for g in order:
        add = group_counts.loc[g]
        best = min(splits, key=lambda s: cost(s, add))

        assignment[g] = best

        for e in exercises:
            current[best][e] += int(add[e])

    return assignment


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def report(df: pd.DataFrame) -> None:
    logger.info("\n========== SPLIT SUMMARY ==========")

    counts = pd.crosstab(df["split"], df["exercise"])
    counts = counts.reindex(list(SPLIT_FRACTIONS))
    logger.info("\nClips per split / exercise:\n%s", counts.to_string())

    achieved = (counts / counts.sum()).round(3)
    logger.info("\nAchieved fractions (targets: %s):\n%s",
                SPLIT_FRACTIONS, achieved.to_string())

    sizes = df.groupby("group").size().sort_values(ascending=False)

    logger.info(
        "\nGroups: %d total, %d with more than one clip",
        len(sizes), (sizes > 1).sum(),
    )
    logger.info("Largest groups (check none dominates a split):\n%s",
                sizes.head(5).to_string())

    # Leakage check: no group may appear in more than one split.
    spread = df.groupby("group")["split"].nunique()
    leaked = spread[spread > 1]

    if len(leaked):
        raise RuntimeError(f"Leakage: {len(leaked)} groups span >1 split")

    logger.info("\nLeakage check passed: every group is in exactly one split.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(QUALITY_SCREENING_PATH)

    # RepCount-only splits (O1: Countix dropped)
    df = df[df["source"] == "repcount"].copy()

    df = df[df["status"].isin(INCLUDE_STATUSES)].copy()

    logger.info("Clips included: %d", len(df))

    # Normalise RepCount's "valid" label to "val".
    df["original_split"] = df["split"].replace({"valid": "val"})

    df["group"] = df.apply(group_key, axis=1)

    assignment = assign_groups(df)
    df["split"] = df["group"].map(assignment)

    report(df)

    out = df[[
        "clip_id", "exercise", "split", "group",
        "original_split", "status", "rep_count",
    ]].sort_values(["split", "exercise", "clip_id"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    logger.info("\nSaved: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
