"""Opt-in, per-segment PCA motion projection followed by offline peak counting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import rep_counter as counter
from .peak_counter import PeakSettings, detect_peaks


# Explicit allow-list: no labels, boundaries, IDs, fold, or annotation fields.
ANGLE_CHANNELS = ("knee_L", "knee_R", "elbow_L", "elbow_R", "hip_L", "hip_R",
                  "arm_raise_L", "arm_raise_R", "leg_spread", "torso_tilt")


@dataclass(frozen=True)
class PCASettings:
    dimensions: str = "2d"
    min_visibility: float = .5
    min_channel_coverage: float = .5
    min_frame_coverage: float = .6
    min_range_deg: float = 10.
    min_channels: int = 3
    smooth_s: float = .3


def project_motion(frame, settings=PCASettings()):
    """Fit scaling and PC1 inside this segment only, without rep annotations.

    Retained channels are standardized so large-angle joints do not dominate.
    Short gaps are interpolated by the existing cleaner. Remaining missing
    cells use zero (the channel mean) in standardized space; low-coverage rows
    are masked. This does not reconstruct missing physical motion.
    """
    if settings.dimensions not in ("2d", "3d"):
        raise ValueError("dimensions must be 2d or 3d")
    suffix = "_3d" if settings.dimensions == "3d" else ""
    names = [name + suffix for name in ANGLE_CHANNELS]
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"Missing PCA channels; dimensional fallback forbidden: {missing}")
    times = frame.time_s.to_numpy(dtype=float)
    result = dict(signal=np.full(len(frame), np.nan), channels=[], loadings=[], explained_variance=0.,
                  valid_fraction=0., imputed_fraction=0., flag="too_short", raw_coverage=0.)
    if len(frame) < 8:
        return result
    fs = counter.sampling_rate(times)
    if fs <= 0 or not np.all(np.diff(times) > 0):
        raise ValueError("PCA needs strictly increasing timestamps")
    columns, kept, raw_masks = [], [], []
    for base, name in zip(ANGLE_CHANNELS, names):
        values = frame[name].to_numpy(dtype=float).copy()
        visibility = f"{base}_vis"
        if visibility in frame:
            confidence = frame[visibility].to_numpy(dtype=float)
            values[~np.isfinite(confidence) | (confidence < settings.min_visibility)] = np.nan
        values[~np.isfinite(values)] = np.nan
        observed = np.isfinite(values)
        if observed.mean() < settings.min_channel_coverage:
            continue
        cleaned = counter.clean_signal(values, fs, counter.MAX_GAP_S, settings.smooth_s)
        finite = np.isfinite(cleaned)
        if finite.mean() < settings.min_channel_coverage:
            continue
        if np.nanpercentile(cleaned, 90) - np.nanpercentile(cleaned, 10) < settings.min_range_deg:
            continue
        columns.append(cleaned)
        raw_masks.append(observed)
        kept.append(name)
    result.update(channels=kept, flag="insufficient_motion_channels")
    if len(columns) < settings.min_channels:
        return result
    matrix = np.column_stack(columns)
    observed = np.isfinite(matrix)
    usable_rows = observed.mean(axis=1) >= settings.min_frame_coverage
    result["raw_coverage"] = float(np.column_stack(raw_masks).mean())
    if usable_rows.sum() < 8:
        return result | {"flag": "insufficient_joint_coverage"}
    centers = np.nanmean(matrix, axis=0)
    scales = np.nanstd(matrix, axis=0)
    scales = np.maximum(scales, 1e-6)
    standardized = (matrix - centers) / scales
    filled = np.where(observed, standardized, 0.)
    fit = filled[usable_rows]
    mean = fit.mean(axis=0)
    _, singular, vt = np.linalg.svd(fit - mean, full_matrices=False)
    component = vt[0].copy()
    # Canonical sign independent of annotations; actual exercise phase may vary.
    if component[np.argmax(np.abs(component))] < 0:
        component *= -1
    projection = (filled - mean) @ component
    score_scale = float(np.std(projection[usable_rows]))
    if score_scale <= 1e-9:
        return result | {"flag": "no_principal_motion"}
    projection /= score_scale
    projection[~usable_rows] = np.nan
    energy = float(np.square(singular).sum())
    return result | dict(signal=projection, channels=kept, loadings=component.tolist(),
                         centers=centers.tolist(), scales=scales.tolist(),
                         explained_variance=float(singular[0] ** 2 / energy) if energy else 0.,
                         valid_fraction=float(usable_rows.mean()),
                         imputed_fraction=float((~observed[usable_rows]).mean()), flag="")


def count_projection(projection, times, exercise, *, prominence=.5, width_s=1 / 15,
                     spacing_fraction=0., polarity=1):
    if polarity not in (-1, 1):
        raise ValueError("PCA polarity must be -1 or 1")
    # PCA scores are dimensionless: degree-valued prominence defaults must not
    # be reused. Use an absolute score-space floor, not a degree-based guard.
    settings = PeakSettings(prominence_fraction=.15, min_prominence_deg=prominence,
                            min_width_s=width_s, period_distance_fraction=spacing_fraction)
    result = detect_peaks(polarity * projection["signal"], times, "low_high_low",
                          counter.EXERCISES[exercise]["min_gap_s"], settings)
    for run in result["runs"]:
        run["prominence_score"] = run.pop("prominence_deg")
    return result


def count_pca_reps(frame, exercise, start_s, end_s, *, pca_settings=PCASettings(), **peak_parameters):
    segment = frame[(frame.time_s >= start_s) & (frame.time_s < end_s)]
    projection = project_motion(segment, pca_settings)
    detected = count_projection(projection, segment.time_s.to_numpy(), exercise, **peak_parameters)
    return detected | {"pca": {k: v for k, v in projection.items() if k != "signal"},
                       "exercise": exercise, "method": "pca_peaks"}
