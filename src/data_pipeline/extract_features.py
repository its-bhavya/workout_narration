
from pathlib import Path

import numpy as np
import pandas as pd

POSE_DIR = Path("data/features/repcount/pose")
OUTPUT_DIR = Path("data/features/repcount/derived")

VISIBILITY_THRESHOLD = 0.5


def point(df: pd.DataFrame, prefix: str, idx: int) -> np.ndarray:
    """Return an N x 3 coordinate array for a landmark."""
    return df[
        [f"{prefix}_{idx:02d}_{axis}" for axis in ("x", "y", "z")]
    ].to_numpy(dtype=float)


def visibility(df: pd.DataFrame, prefix: str, idx: int) -> np.ndarray:
    return df[f"{prefix}_{idx:02d}_visibility"].to_numpy(dtype=float)


def joint_angle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Angle ABC in degrees. Invalid or degenerate points return NaN."""
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    denom = norm_ba * norm_bc

    result = np.full(len(a), np.nan, dtype=float)
    valid = (
        np.isfinite(a).all(axis=1)
        & np.isfinite(b).all(axis=1)
        & np.isfinite(c).all(axis=1)
        & (denom > 1e-8)
    )

    cosine = np.full(len(a), np.nan, dtype=float)
    cosine[valid] = (
        np.einsum("ij,ij->i", ba[valid], bc[valid])
        / denom[valid]
    )
    result[valid] = np.degrees(
        np.arccos(np.clip(cosine[valid], -1.0, 1.0))
    )
    return result


def valid_triplet(
    df: pd.DataFrame,
    prefix: str,
    a_idx: int,
    b_idx: int,
    c_idx: int,
) -> np.ndarray:
    """Require detected pose and sufficient visibility at all 3 points."""
    ok = df["pose_detected"].to_numpy(dtype=bool, copy=True)

    for idx in (a_idx, b_idx, c_idx):
        vis = visibility(df, prefix, idx)
        coords = point(df, prefix, idx)
        ok &= np.isfinite(vis) & (vis >= VISIBILITY_THRESHOLD)
        ok &= np.isfinite(coords).all(axis=1)

    return ok



def add_angle(
    df: pd.DataFrame,
    out: pd.DataFrame,
    name: str,
    prefix: str,
    a_idx: int,
    b_idx: int,
    c_idx: int,
) -> None:
    coords_a = point(df, prefix, a_idx)
    coords_b = point(df, prefix, b_idx)
    coords_c = point(df, prefix, c_idx)

    valid = valid_triplet(df, prefix, a_idx, b_idx, c_idx)

    # Make a writable copy before assigning NaN to invalid angles.
    angle = joint_angle(coords_a, coords_b, coords_c).copy()
    angle[~valid] = np.nan

    out[name] = angle
    out[f"{name}_valid"] = valid & np.isfinite(angle)


def extract_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    out = df[
        [
            "filename",
            "exercise",
            "split",
            "fps",
            "frame_index",
            "timestamp_ms",
            "pose_detected",
        ]
    ].copy()

    exercise = str(df["exercise"].iloc[0])

    if exercise == "squat":
        # MediaPipe landmark indices:
        # left hip/knee/ankle = 23/25/27
        # right hip/knee/ankle = 24/26/28
        for side, hip, knee, ankle in (
            ("left", 23, 25, 27),
            ("right", 24, 26, 28),
        ):
            add_angle(
                df, out, f"{side}_knee_angle_deg",
                "world", hip, knee, ankle,
            )
            add_angle(
                df, out, f"{side}_hip_angle_deg",
                "world", 11 if side == "left" else 12,
                hip, knee,
            )

        # Image y increases downward. Use normalized image coordinates
        # to track hip height relative to the image.
        for side, idx in (("left", 23), ("right", 24)):
            out[f"{side}_hip_y"] = df[f"image_{idx:02d}_y"]

        out["mean_hip_y"] = out[["left_hip_y", "right_hip_y"]].mean(
            axis=1, skipna=False
        )

    elif exercise == "push_up":
        # Shoulders/elbows/wrists = 11/13/15 and 12/14/16.
        # Hips/shoulders/ankles help describe body alignment.
        for side, shoulder, elbow, wrist, hip, ankle in (
            ("left", 11, 13, 15, 23, 27),
            ("right", 12, 14, 16, 24, 28),
        ):
            add_angle(
                df, out, f"{side}_elbow_angle_deg",
                "world", shoulder, elbow, wrist,
            )
            add_angle(
                df, out, f"{side}_shoulder_angle_deg",
                "world", elbow, shoulder, hip,
            )
            add_angle(
                df, out, f"{side}_body_alignment_angle_deg",
                "world", shoulder, hip, ankle,
            )

    elif exercise == "jump_jack":
        # Shoulder-to-wrist direction and ankle spacing.
        # Arm elevation uses image x/y so it remains interpretable in frame.
        for side, shoulder, elbow, wrist in (
            ("left", 11, 13, 15),
            ("right", 12, 14, 16),
        ):
            sx = df[f"image_{shoulder:02d}_x"].to_numpy(float)
            sy = df[f"image_{shoulder:02d}_y"].to_numpy(float)
            wx = df[f"image_{wrist:02d}_x"].to_numpy(float)
            wy = df[f"image_{wrist:02d}_y"].to_numpy(float)

            dx = wx - sx
            dy = sy - wy  # positive means wrist is above shoulder
            arm_angle = np.degrees(np.arctan2(dy, dx))
            arm_valid = (
                df["pose_detected"].to_numpy(bool)
                & (visibility(df, "image", shoulder) >= VISIBILITY_THRESHOLD)
                & (visibility(df, "image", wrist) >= VISIBILITY_THRESHOLD)
                & np.isfinite(dx) & np.isfinite(dy)
            )
            arm_angle=np.array(arm_angle, dtype=float, copy=True)
            arm_angle[~arm_valid] = np.nan

            out[f"{side}_arm_direction_deg"] = arm_angle
            out[f"{side}_arm_direction_valid"] = (
                arm_valid & np.isfinite(arm_angle)
            )

        left_x = df["image_27_x"].to_numpy(float)
        right_x = df["image_28_x"].to_numpy(float)
        ankle_valid = (
            df["pose_detected"].to_numpy(bool)
            & (visibility(df, "image", 27) >= VISIBILITY_THRESHOLD)
            & (visibility(df, "image", 28) >= VISIBILITY_THRESHOLD)
            & np.isfinite(left_x) & np.isfinite(right_x)
        )

        ankle_separation = np.abs(left_x - right_x)
        ankle_separation=np.array(ankle_separation, dtype=float, copy=True)

        ankle_separation[~ankle_valid] = np.nan
        out["ankle_separation_norm"] = ankle_separation
        out["ankle_separation_valid"] = (
            ankle_valid & np.isfinite(ankle_separation)
        )

        left_wrist_x = df["image_15_x"].to_numpy(float)
        right_wrist_x = df["image_16_x"].to_numpy(float)
        wrist_valid = (
            df["pose_detected"].to_numpy(bool)
            & (visibility(df, "image", 15) >= VISIBILITY_THRESHOLD)
            & (visibility(df, "image", 16) >= VISIBILITY_THRESHOLD)
            & np.isfinite(left_wrist_x) & np.isfinite(right_wrist_x)
        )

        wrist_separation = np.abs(left_wrist_x - right_wrist_x)
        wrist_separation=np.array(wrist_separation, dtype=float, copy=True)

        wrist_separation[~wrist_valid] = np.nan
        out["wrist_separation_norm"] = wrist_separation
        out["wrist_separation_valid"] = (
            wrist_valid & np.isfinite(wrist_separation)
        )

    else:
        raise ValueError(f"Unknown exercise label: {exercise!r}")

    # Temporal differences are calculated independently per video.
    # Temporal differences are calculated independently per video.
    feature_cols = [
        c for c in out.columns
        if c.endswith("_deg")
        or c.endswith("_norm")
        or c in ("mean_hip_y",)
    ]

    for col in feature_cols:
        if col.endswith("_arm_direction_deg"):
            # Correct angular wraparound: changes must be in [-180, 180).
            delta = (out[col].diff() + 180) % 360 - 180
        else:
            delta = out[col].diff()

        out[f"{col}_delta"] = delta

    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in sorted(POSE_DIR.glob("*.parquet")):
        output_path = OUTPUT_DIR / filename.name

        try:
            features = extract_features(filename)
            features.to_parquet(output_path, index=False)
            print(
                f"Saved {output_path.name}: "
                f"{len(features)} frames, "
                f"{len(features.columns)} columns"
            )
        except Exception as exc:
            print(f"FAILED {filename.name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
