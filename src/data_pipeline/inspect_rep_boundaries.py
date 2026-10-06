from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import pandas as pd


def get_boundary_frames(row: pd.Series) -> list[int]:
    """Extract L1, L2, ... Ln as ordered frame numbers."""

    boundary_columns = [
        col
        for col in row.index
        if str(col).startswith("L")
        and str(col)[1:].isdigit()
    ]

    boundary_columns.sort(
        key=lambda col: int(str(col)[1:])
    )

    frames = []

    for col in boundary_columns:
        value = row[col]

        if pd.isna(value):
            continue

        try:
            frame = int(float(value))
        except (TypeError, ValueError):
            continue

        frames.append(frame)

    return frames


def find_annotation_row(
    annotation_csv: Path,
    filename: str,
) -> pd.Series:

    df = pd.read_csv(annotation_csv)

    matches = df[df["name"].astype(str) == filename]

    if matches.empty:
        raise ValueError(
            f"Could not find '{filename}' in {annotation_csv}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Found multiple rows for '{filename}' "
            f"in {annotation_csv}"
        )

    return matches.iloc[0]


def extract_boundary_frames(
    video_path: Path,
    boundary_frames: list[int],
    output_dir: Path,
) -> list[Path]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    saved_paths = []

    for index, frame_number in enumerate(boundary_frames, start=1):

        if frame_number < 0 or frame_number >= total_frames:
            print(
                f"WARNING: L{index} = {frame_number} "
                f"is outside video range 0-{total_frames - 1}"
            )
            continue

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number,
        )

        success, frame = cap.read()

        if not success:
            print(
                f"WARNING: Could not read frame "
                f"{frame_number}"
            )
            continue

        timestamp = (
            frame_number / fps
            if fps > 0
            else None
        )

        text = (
            f"L{index} | frame={frame_number}"
        )

        if timestamp is not None:
            text += f" | time={timestamp:.2f}s"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        output_path = (
            output_dir
            / f"L{index:02d}_frame_{frame_number:06d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame,
        )

        saved_paths.append(output_path)

    cap.release()

    return saved_paths


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    columns: int = 4,
) -> None:

    if not image_paths:
        return

    images = []

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is not None:
            images.append(image)

    if not images:
        return

    # Resize all images to a common width.
    target_width = 320
    resized = []

    for image in images:
        height, width = image.shape[:2]

        scale = target_width / width

        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (target_width, new_height),
        )

        resized.append(image)

    cell_height = max(
        image.shape[0]
        for image in resized
    )

    rows = math.ceil(
        len(resized) / columns
    )

    sheet = cv2.copyMakeBorder(
        resized[0],
        0,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    sheet = None

    cells = []

    for image in resized:

        canvas = 255 * (
            __import__("numpy").ones(
                (
                    cell_height,
                    target_width,
                    3,
                ),
                dtype="uint8",
            )
        )

        canvas[
            : image.shape[0],
            : image.shape[1],
        ] = image

        cells.append(canvas)

    while len(cells) % columns != 0:
        cells.append(
            255 * (
                __import__("numpy").ones(
                    (
                        cell_height,
                        target_width,
                        3,
                    ),
                    dtype="uint8",
                )
            )
        )

    row_images = []

    for i in range(0, len(cells), columns):

        row = cv2.hconcat(
            cells[i:i + columns]
        )

        row_images.append(row)

    sheet = cv2.vconcat(row_images)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        sheet,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Inspect RepCount L-value frame annotations."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the RepCount video.",
    )

    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
        help="Path to train.csv / valid.csv / test.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/rep_boundary_inspection"
        ),
    )

    args = parser.parse_args()

    filename = args.video.name

    print(f"Video: {filename}")
    print(f"Annotation: {args.annotation}")

    row = find_annotation_row(
        args.annotation,
        filename,
    )

    rep_count = int(row["count"])

    boundary_frames = get_boundary_frames(row)

    print()
    print(f"Annotated rep count: {rep_count}")
    print(
        f"Number of L-values: "
        f"{len(boundary_frames)}"
    )
    print(
        f"L-values: {boundary_frames}"
    )

    # Print useful pairing hypothesis, but DO NOT assume it is correct.
    print()
    print("Potential pairs:")
    
    for i in range(0, len(boundary_frames) - 1, 2):
        print(
            f"  L{i + 1}={boundary_frames[i]} "
            f"-> "
            f"L{i + 2}={boundary_frames[i + 1]}"
        )

    print()
    print(
        "NOTE: The pairs above are only a visual "
        "inspection hypothesis."
    )

    output_dir = (
        args.output_dir
        / args.video.stem
    )

    frame_dir = output_dir / "frames"

    saved_frames = extract_boundary_frames(
        args.video,
        boundary_frames,
        frame_dir,
    )

    contact_sheet_path = (
        output_dir / "contact_sheet.jpg"
    )

    create_contact_sheet(
        saved_frames,
        contact_sheet_path,
    )

    metadata = {
        "video": str(args.video),
        "annotation": str(args.annotation),
        "filename": filename,
        "rep_count": rep_count,
        "boundary_frames": boundary_frames,
        "num_boundary_frames": len(boundary_frames),
        "possible_pairs": [
            [
                boundary_frames[i],
                boundary_frames[i + 1],
            ]
            for i in range(
                0,
                len(boundary_frames) - 1,
                2,
            )
        ],
    }

    metadata_path = (
        output_dir / "inspection_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved boundary frames to: {frame_dir}"
    )
    print(
        f"Saved contact sheet to: "
        f"{contact_sheet_path}"
    )
    print(
        f"Saved metadata to: {metadata_path}"
    )


if __name__ == "__main__":
    main()