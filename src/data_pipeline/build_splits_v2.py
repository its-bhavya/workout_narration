"""Build native train/test pools and source-clip CV folds; never render videos.

Run from the repository root with:
    uv run python src/data_pipeline/build_splits_v2.py

Inputs are quality_screening.csv, repcount_rep_boundaries.json and optional
manual_exclusions.csv. The legacy splits.csv is never read or written.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "data" / "manifests"
SEED = 3969  # Approved redraw; provenance in docs/PROJECT_CONTEXT.md.
EXERCISES = ["jump_jack", "push_up", "squat"]
SPLIT_COLUMNS = [
    "clip_id", "exercise", "split", "original_split", "status", "rep_count",
]
log = logging.getLogger(__name__)


def build_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    quality = pd.read_csv(MANIFESTS / "quality_screening.csv")
    pool = quality.loc[
        quality["source"].eq("repcount")
        & quality["status"].isin(["pass", "review"])
    ].copy()
    assert pool["clip_id"].is_unique, "Duplicate source clip IDs"
    pool["original_split"] = pool["split"].replace({"valid": "val"})
    assert pool["original_split"].isin(["train", "val", "test"]).all()
    assert set(pool["exercise"]) == set(EXERCISES)

    exclusions_path = MANIFESTS / "manual_exclusions.csv"
    exclusions: set[str] = set()
    if exclusions_path.exists():
        exclusions_df = pd.read_csv(exclusions_path)
        if "source" in exclusions_df.columns:
            exclusions_df = exclusions_df.loc[exclusions_df["source"].eq("repcount")]
        exclusions = set(exclusions_df["clip_id"])
        log.info("Loaded %d manual exclusions", len(exclusions))
    else:
        log.info("no manual exclusions file found")
    pool = pool.loc[~pool["clip_id"].isin(exclusions)].copy()

    with (MANIFESTS / "repcount_rep_boundaries.json").open() as handle:
        boundaries = json.load(handle)
    assert set(pool["clip_id"]) <= boundaries.keys(), "Missing boundary metadata"
    assert all(boundaries[c]["pairs"] for c in pool["clip_id"]), "Empty rep pairs"
    fps = pool["clip_id"].map(lambda clip: boundaries[clip]["fps"])
    assert fps.notna().all(), "Missing fps"
    log.info("Excluding %d clips with fps < 20", int((fps < 20).sum()))
    pool = pool.loc[fps >= 20].copy()

    expected_train = set(pool.loc[pool["original_split"].eq("train"), "clip_id"])
    expected_test = set(pool.loc[pool["original_split"].isin(["val", "test"]), "clip_id"])
    pool["split"] = pool["original_split"].map({"train": "train", "val": "test", "test": "test"})
    splits = pool[SPLIT_COLUMNS].sort_values(["split", "exercise", "clip_id"]).reset_index(drop=True)
    train = splits.loc[splits["split"].eq("train")].sort_values("clip_id").reset_index(drop=True)
    test_ids = set(splits.loc[splits["split"].eq("test"), "clip_id"])
    assert set(train["clip_id"]) == expected_train
    assert test_ids == expected_test, "Test must equal usable native val + test"
    assert expected_train.isdisjoint(test_ids), "Train/test clip overlap"
    counts = pd.crosstab(splits["split"], splits["exercise"]).reindex(
        index=["train", "test"], columns=EXERCISES, fill_value=0,
    )
    assert counts.values.tolist() == [[66, 81, 92], [38, 31, 25]], counts

    # Stable clip order plus an explicit shuffled stratifier makes regeneration
    # independent of the input CSV row order. Each source clip has one fold.
    folds = train[["clip_id"]].copy()
    folds["fold"] = -1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, held_out) in enumerate(cv.split(train["clip_id"], train["exercise"])):
        folds.loc[held_out, "fold"] = fold
    assert folds["clip_id"].is_unique
    assert set(folds["clip_id"]) == expected_train
    assert set(folds["fold"]) == set(range(5))
    fold_counts = pd.crosstab(folds["fold"], train["exercise"]).reindex(columns=EXERCISES)
    assert (fold_counts > 0).all().all(), "Every fold must contain every exercise"
    assert ((fold_counts.max() - fold_counts.min()) <= 1).all()
    log.info("Split counts:\n%s", counts.to_string())
    log.info("Source-clip fold counts (seed %d):\n%s", SEED, fold_counts.to_string())
    return splits, folds


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    splits, folds = build_manifests()
    # Refuse to replace existing artifacts. Identical regeneration is a no-op;
    # changed artifacts must first be archived explicitly, never deleted here.
    outputs = {
        MANIFESTS / "splits_v2.csv": splits.to_csv(index=False, lineterminator="\n"),
        MANIFESTS / "folds.csv": folds.to_csv(index=False, lineterminator="\n"),
    }
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise FileExistsError(f"Archive existing artifact before replacing: {path}")
    for path, content in outputs.items():
        if not path.exists():
            path.write_bytes(content.encode("utf-8"))
        log.info("Verified %s (%d rows)", path.name, len(splits) if path.name == "splits_v2.csv" else len(folds))


if __name__ == "__main__":
    main()
