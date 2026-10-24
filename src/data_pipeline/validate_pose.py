
from pathlib import Path

import numpy as np
import pandas as pd

POSE_DIR = Path("data/features/repcount/pose")
FILES = [
    "test2340.parquet",
    "stu6_18.parquet",
    "stu7_49.parquet",
]


def validate_pose_file(path: Path) -> None:
    df = pd.read_parquet(path)

    landmark_cols = [
        c for c in df.columns
        if c.startswith(("image_", "world_"))
        and c.endswith(("_x", "_y", "_z"))
    ]

    detected = df["pose_detected"].astype(bool)
    missing_all = df[landmark_cols].isna().all(axis=1)
    missing_any = df[landmark_cols].isna().any(axis=1)

    frame_indices = df["frame_index"].to_numpy()
    timestamps = df["timestamp_ms"].to_numpy()

    print(f"\n{'=' * 60}")
    print(f"File: {path.name}")
    print(f"Rows: {len(df)}")
    print(f"Unique frame indices: {df['frame_index'].nunique()}")
    print(f"Frame indices sequential: "
          f"{np.array_equal(frame_indices, np.arange(len(df)))}")
    print(f"Timestamps strictly increasing: "
          f"{np.all(np.diff(timestamps) > 0)}")
    print(f"Detected frames: {detected.sum()}")
    print(f"Undetected frames: {(~detected).sum()}")
    print(f"Frames with all landmark coordinates missing: "
          f"{missing_all.sum()}")
    print(f"Frames with any landmark coordinate missing: "
          f"{missing_any.sum()}")
    print(f"Undetected frames with all coordinates missing: "
          f"{((~detected) & missing_all).sum()}")
    print(f"Detected frames with all coordinates missing: "
          f"{(detected & missing_all).sum()}")
    print(f"Undetected frames with coordinates present: "
          f"{((~detected) & ~missing_all).sum()}")

    # Check coordinate finiteness for available values.
    values = df[landmark_cols].to_numpy(dtype=float)
    print(f"All non-missing coordinates finite: "
          f"{np.isfinite(values[~np.isnan(values)]).all()}")

    # Visibility is a quality signal, not a hard validity guarantee.
    visibility_cols = [
        c for c in df.columns
        if c.endswith("_visibility")
    ]
    if visibility_cols:
        visibility = df[visibility_cols].to_numpy(dtype=float)
        valid_visibility = visibility[~np.isnan(visibility)]
        if len(valid_visibility):
            print(f"Visibility range: "
                  f"{valid_visibility.min():.4f} to "
                  f"{valid_visibility.max():.4f}")


def main() -> None:
    for filename in FILES:
        path = POSE_DIR / filename
        if not path.exists():
            print(f"\nMISSING FILE: {path}")
            continue
        validate_pose_file(path)


if __name__ == "__main__":
    main()
