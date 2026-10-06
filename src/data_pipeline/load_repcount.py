"""
Load and normalize RepCount annotations.

Outputs:
    data/manifests/repcount_filtered.csv
    data/manifests/repcount_excluded.csv

The RepCount annotations contain:
    - name: video filename
    - type: exercise category
    - count: repetition count
    - L1, L2, ...: ordered frame annotations

For the original RepCount annotations, consecutive L-values represent
the start and end frames of each repetition:

    L1, L2 -> repetition 1
    L3, L4 -> repetition 2
    ...

The final manifest stores these as:

    [[start_1, end_1],
     [start_2, end_2],
     ...]
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPCOUNT_ROOT = Path(r"C:\Users\Bhavya\Downloads\RepCountA (1)\LLSP")

ANNOTATION_DIR = REPCOUNT_ROOT / "annotation"
VIDEO_DIR = REPCOUNT_ROOT / "video"

OUTPUT_DIR = Path("data/manifests")

OUTPUT_FILTERED = OUTPUT_DIR / "repcount_filtered.csv"
OUTPUT_EXCLUDED = OUTPUT_DIR / "repcount_excluded.csv"


TARGET_EXERCISES = {
    "squat",
    "push_up",
    "jump_jack",
}


# Explicit normalization only.
#
# We do NOT automatically map ambiguous categories.
#
# "squant" was manually inspected on approximately 10 videos
# and confirmed to correspond to squat.
NORMALIZATION_MAP = {
    "squat": "squat",
    "squant": "squat",

    "push_up": "push_up",
    "pushups": "push_up",

    "jump_jack": "jump_jack",
    "jumpjacks": "jump_jack",
}


SPLITS = [
    "train",
    "valid",
    "test",
]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_split(split: str) -> pd.DataFrame:
    """Load one RepCount annotation CSV."""

    csv_path = ANNOTATION_DIR / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Avoid DataFrame fragmentation from repeated column insertion.
    df = df.assign(split=split)

    logger.info(
        "Loaded %s: %d rows, %d columns",
        split,
        len(df),
        len(df.columns),
    )

    return df


def get_raw_category_counts(df: pd.DataFrame) -> None:
    """Print the complete raw exercise vocabulary."""

    logger.info("Complete RepCount raw category vocabulary:")

    counts = (
        df["type"]
        .value_counts(dropna=False)
        .sort_index()
    )

    for category, count in counts.items():
        logger.info(
            "  %-30s %d",
            str(category),
            count,
        )


def normalize_exercise(raw_type) -> str | None:
    """
    Normalize a raw RepCount exercise category.

    Returns None for categories that are not explicitly mapped.
    """

    if pd.isna(raw_type):
        return None

    raw_type = str(raw_type).strip().lower()

    return NORMALIZATION_MAP.get(raw_type)


def get_video_path(split: str, filename: str) -> Path:
    """Return the expected video path for a split and filename."""

    return VIDEO_DIR / split / filename


def parse_rep_boundaries(row: pd.Series) -> list[list[int]]:
    """
    Parse L1, L2, L3, ... into repetition-level [start, end] pairs.

    Example:

        L1 = 20
        L2 = 55
        L3 = 82
        L4 = 117

    becomes:

        [
            [20, 55],
            [82, 117],
        ]

    The L columns are sorted numerically rather than lexicographically,
    so L2 comes before L10.
    """

    l_columns = [
        column
        for column in row.index
        if re.fullmatch(r"L\d+", str(column))
    ]

    l_columns.sort(
        key=lambda column: int(str(column)[1:])
    )

    values: list[int] = []

    for column in l_columns:
        value = row[column]

        if pd.isna(value):
            continue

        try:
            values.append(int(value))
        except (ValueError, TypeError):
            logger.warning(
                "Invalid L-value for %s: %s=%r",
                row["name"],
                column,
                value,
            )

    if len(values) % 2 != 0:
        logger.warning(
            "Odd number of L-values for %s: %d values",
            row["name"],
            len(values),
        )

    boundaries = []

    for i in range(0, len(values) - 1, 2):
        start_frame = values[i]
        end_frame = values[i + 1]

        boundaries.append(
            [start_frame, end_frame]
        )

    return boundaries


def validate_rep_boundaries(row: pd.Series) -> tuple[bool, str]:
    """
    Validate repetition annotations.

    Returns:
        (True, "") if valid
        (False, reason) otherwise
    """

    filename = row["name"]

    # -------------------------------------------------------------
    # Rep count
    # -------------------------------------------------------------

    rep_count = row["count"]

    if pd.isna(rep_count):
        return False, "missing_rep_count"

    try:
        rep_count = int(rep_count)
    except (ValueError, TypeError):
        return False, "invalid_rep_count"

    if rep_count <= 0:
        return False, "non_positive_rep_count"

    # -------------------------------------------------------------
    # Boundary pairs
    # -------------------------------------------------------------

    boundaries = row["rep_boundaries"]

    if len(boundaries) != rep_count:
        return False, (
            f"boundary_count_mismatch_"
            f"expected_{rep_count}_got_{len(boundaries)}"
        )

    # -------------------------------------------------------------
    # Frame ordering
    # -------------------------------------------------------------

    for index, boundary in enumerate(boundaries, start=1):

        if len(boundary) != 2:
            return False, (
                f"invalid_boundary_length_rep_{index}"
            )

        start_frame, end_frame = boundary

        if start_frame < 0 or end_frame < 0:
            return False, (
                f"negative_frame_index_rep_{index}"
            )

        if start_frame > end_frame:
            return False, (
                f"start_after_end_rep_{index}"
            )

    # -------------------------------------------------------------
    # Global chronological ordering
    # -------------------------------------------------------------

    for i in range(1, len(boundaries)):
        previous_end = boundaries[i - 1][1]
        current_start = boundaries[i][0]

        if current_start < previous_end:
            return False, (
                f"non_chronological_boundary_between_"
                f"rep_{i}_and_rep_{i + 1}"
            )

    return True, ""


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # 1. Load train / valid / test
    # -------------------------------------------------------------

    split_dfs = [
        load_split(split)
        for split in SPLITS
    ]

    df = pd.concat(
        split_dfs,
        ignore_index=True,
    )

    logger.info(
        "Combined dataset: %d rows",
        len(df),
    )

    # -------------------------------------------------------------
    # 2. Print complete raw category vocabulary
    # -------------------------------------------------------------

    get_raw_category_counts(df)

    # -------------------------------------------------------------
    # 3. Normalize exercise names
    # -------------------------------------------------------------

    df["exercise"] = df["type"].apply(
        normalize_exercise
    )

    # Report categories that were not explicitly mapped.

    unmapped = (
        df.loc[
            df["exercise"].isna(),
            "type"
        ]
        .value_counts(dropna=False)
    )

    if len(unmapped) > 0:

        logger.info(
            "Raw categories not mapped to target exercises:"
        )

        for category, count in unmapped.items():
            logger.info(
                "  %-30s %d",
                str(category),
                count,
            )

    # -------------------------------------------------------------
    # 4. Filter to target exercises
    # -------------------------------------------------------------

    df = df[
        df["exercise"].isin(TARGET_EXERCISES)
    ].copy()

    logger.info(
        "Target exercise rows after normalization: %d",
        len(df),
    )

    # -------------------------------------------------------------
    # 5. Parse L-values into repetition boundaries
    # -------------------------------------------------------------

    df["rep_boundaries"] = df.apply(
        parse_rep_boundaries,
        axis=1,
    )

    # -------------------------------------------------------------
    # 6. Validate annotations
    # -------------------------------------------------------------

    validation_results = df.apply(
        validate_rep_boundaries,
        axis=1,
    )

    df["boundaries_valid"] = [
        result[0]
        for result in validation_results
    ]

    df["validation_reason"] = [
        result[1]
        for result in validation_results
    ]

    # -------------------------------------------------------------
    # 7. Check that corresponding video files exist
    # -------------------------------------------------------------

    video_paths = []

    for _, row in df.iterrows():

        video_path = get_video_path(
            row["split"],
            row["name"],
        )

        video_paths.append(video_path)

    df["video_exists"] = [
        path.exists()
        for path in video_paths
    ]

    missing_videos = df[
        ~df["video_exists"]
    ]

    if len(missing_videos) > 0:

        logger.warning(
            "Missing video files: %d",
            len(missing_videos),
        )

        for _, row in missing_videos.iterrows():
            logger.warning(
                "  %s [%s]",
                row["name"],
                row["split"],
            )

    # -------------------------------------------------------------
    # 8. Final usability decision
    # -------------------------------------------------------------

    df["usable"] = (
        df["boundaries_valid"]
        & df["video_exists"]
    )

    # -------------------------------------------------------------
    # 9. Log validation problems
    # -------------------------------------------------------------

    invalid_rows = df[
        ~df["boundaries_valid"]
    ]

    if len(invalid_rows) > 0:

        logger.warning(
            "Videos with invalid annotations: %d",
            len(invalid_rows),
        )

        for _, row in invalid_rows.iterrows():

            logger.warning(
                "  %s: %s",
                row["name"],
                row["validation_reason"],
            )

    # -------------------------------------------------------------
    # 10. Create filtered manifest
    # -------------------------------------------------------------

    filtered = df[
        df["usable"]
    ].copy()

    excluded = df[
        ~df["usable"]
    ].copy()

    # -------------------------------------------------------------
    # 11. Serialize list columns as JSON
    # -------------------------------------------------------------

    filtered["rep_boundaries"] = filtered[
        "rep_boundaries"
    ].apply(json.dumps)

    excluded["rep_boundaries"] = excluded[
        "rep_boundaries"
    ].apply(json.dumps)

    # -------------------------------------------------------------
    # 12. Select manifest columns
    # -------------------------------------------------------------

    manifest_columns = [
        "name",
        "exercise",
        "split",
        "count",
        "rep_boundaries",
    ]

    excluded_columns = [
        "name",
        "exercise",
        "split",
        "count",
        "rep_boundaries",
        "boundaries_valid",
        "validation_reason",
        "video_exists",
    ]

    filtered = filtered[
        manifest_columns
    ]

    excluded = excluded[
        excluded_columns
    ]

    # Rename count to make the meaning explicit.

    filtered = filtered.rename(
        columns={
            "name": "filename",
            "count": "rep_count",
        }
    )

    excluded = excluded.rename(
        columns={
            "name": "filename",
            "count": "rep_count",
        }
    )

    # -------------------------------------------------------------
    # 13. Save manifests
    # -------------------------------------------------------------

    filtered.to_csv(
        OUTPUT_FILTERED,
        index=False,
    )

    excluded.to_csv(
        OUTPUT_EXCLUDED,
        index=False,
    )

    # -------------------------------------------------------------
    # 14. Summary
    # -------------------------------------------------------------

    logger.info("")
    logger.info("========== FINAL SUMMARY ==========")

    logger.info(
        "Target rows: %d",
        len(df),
    )

    logger.info(
        "Usable rows: %d",
        len(filtered),
    )

    logger.info(
        "Excluded rows: %d",
        len(excluded),
    )

    logger.info("")
    logger.info("Usable rows by exercise:")

    exercise_counts = (
        filtered["exercise"]
        .value_counts()
        .sort_index()
    )

    for exercise, count in exercise_counts.items():

        logger.info(
            "  %-15s %d",
            exercise,
            count,
        )

    logger.info("")
    logger.info("Usable rows by split:")

    split_counts = (
        filtered["split"]
        .value_counts()
        .reindex(SPLITS, fill_value=0)
    )

    for split, count in split_counts.items():

        logger.info(
            "  %-15s %d",
            split,
            count,
        )

    logger.info("")
    logger.info(
        "Saved usable manifest: %s",
        OUTPUT_FILTERED,
    )

    logger.info(
        "Saved excluded manifest: %s",
        OUTPUT_EXCLUDED,
    )


if __name__ == "__main__":
    main()