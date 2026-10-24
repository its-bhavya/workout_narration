
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "data" / "manifests" / "repcount_filtered.csv"
FEATURE_DIR = ROOT / "data" / "features" / "repcount" / "derived"
OUTPUT_DIR = ROOT / "outputs" / "feature_pilots"

PILOT_VIDEOS = {
    "test2340.mp4": {
        "title": "Squat",
        "features": [
            "left_knee_angle_deg",
            "right_knee_angle_deg",
            "mean_hip_y",
        ],
    },
    "stu7_49.mp4": {
        "title": "Push-up",
        "features": [
            "left_elbow_angle_deg",
            "right_elbow_angle_deg",
        ],
    },
    "stu6_18.mp4": {
        "title": "Jumping jack",
        "features": [
            "ankle_separation_norm",
            "wrist_separation_norm",
            "left_arm_direction_deg",
            "right_arm_direction_deg",
        ],
    },
}


def parse_boundaries(value) -> list[tuple[int, int]]:
    """Parse repetition boundaries from the manifest."""
    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = ast.literal_eval(value)

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Unexpected boundary format: {value!r}")

    boundaries = []

    for boundary in value:
        if not isinstance(boundary, (list, tuple)) or len(boundary) != 2:
            raise ValueError(f"Invalid repetition boundary: {boundary!r}")

        start, end = int(boundary[0]), int(boundary[1])

        if start < 0 or end < start:
            raise ValueError(f"Invalid boundary range: {boundary!r}")

        boundaries.append((start, end))

    return boundaries


def find_manifest_row(manifest: pd.DataFrame, filename: str) -> pd.Series:
    matches = manifest.loc[manifest["filename"] == filename]

    if matches.empty:
        raise ValueError(f"{filename} was not found in {MANIFEST_PATH}")

    if len(matches) > 1:
        raise ValueError(f"Multiple manifest rows found for {filename}")

    return matches.iloc[0]


def plot_video(
    filename: str,
    config: dict,
    manifest: pd.DataFrame,
) -> None:
    feature_path = FEATURE_DIR / f"{Path(filename).stem}.parquet"

    if not feature_path.exists():
        print(f"SKIPPED {filename}: feature file not found: {feature_path}")
        return

    row = find_manifest_row(manifest, filename)
    boundaries = parse_boundaries(row["rep_boundaries"])

    df = pd.read_parquet(feature_path)

    if "frame_index" not in df.columns:
        raise ValueError(f"{feature_path} has no frame_index column")

    df = df.sort_values("frame_index").reset_index(drop=True)
    frame = df["frame_index"]

    available_features = [
        feature for feature in config["features"]
        if feature in df.columns
    ]
    missing_features = [
        feature for feature in config["features"]
        if feature not in df.columns
    ]

    if missing_features:
        print(f"WARNING {filename}: missing features: {missing_features}")

    if not available_features:
        print(f"SKIPPED {filename}: none of the requested features exist")
        return

    fig, axes = plt.subplots(
        len(available_features),
        1,
        figsize=(15, 3.2 * len(available_features)),
        sharex=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, feature in zip(axes, available_features):
        # NaNs remain gaps; no interpolation is performed.
        ax.plot(frame, df[feature], linewidth=1.0, label=feature)

        # Boundaries are shown as dashed lines. Their exact endpoint
        # convention must be verified before interpreting alignment.
        for index, (start, end) in enumerate(boundaries):
            ax.axvline(
                start,
                linestyle="--",
                linewidth=0.8,
                alpha=0.65,
                label="Annotated start" if index == 0 else None,
            )
            ax.axvline(
                end,
                linestyle=":",
                linewidth=0.8,
                alpha=0.65,
                label="Annotated end" if index == 0 else None,
            )

        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frame index")

    fig.suptitle(
        f"{config['title']} | {filename} | "
        f"{len(boundaries)} annotated repetitions",
        fontsize=13,
    )
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{Path(filename).stem}_features.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Saved: {output_path.relative_to(ROOT)} | "
        f"frames={len(df)} | "
        f"annotated_reps={len(boundaries)} | "
        f"features={len(available_features)}"
    )


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(f"Feature directory not found: {FEATURE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {"filename", "rep_boundaries"}
    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_columns)}"
        )

    for filename, config in PILOT_VIDEOS.items():
        plot_video(filename, config, manifest)

    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
