"""Annotation-free elbow signal quality for choosing a counting arm."""
from __future__ import annotations

import numpy as np
from scipy.signal import detrend, welch


def motion_quality(signal, raw, visibility, fs, min_range_deg=20.):
    """Return bounded motion, continuity and periodicity measurements.

    Spectra are computed independently within finite tracking runs. No rep
    counts or annotations are used to score an arm.
    """
    valid = np.isfinite(signal)
    if not valid.any():
        return dict(visibility=0., continuity=0., motion=0., periodicity=0., smoothness=0.)
    span = float(np.nanpercentile(signal, 90) - np.nanpercentile(signal, 10))
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    periodicity, smoothness, weights = [], [], []
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < max(8, round(2 * fs)):
            continue
        values = detrend(signal[start:end])
        frequencies, power = welch(values, fs, nperseg=min(len(values), round(8 * fs)))
        band = (frequencies >= .2) & (frequencies <= 3.)
        total = float(power.sum())
        if total <= 1e-9 or not band.any():
            continue
        peak = frequencies[np.flatnonzero(band)[np.argmax(power[band])]]
        concentrated = np.abs(frequencies - peak) <= .2
        periodicity.append(float(power[concentrated & band].sum() / total))
        smoothness.append(float(power[frequencies <= 3.].sum() / total))
        weights.append(end - start)
    return dict(visibility=float(np.clip(visibility, 0, 1)) if np.isfinite(visibility) else 0.,
                continuity=float(np.isfinite(raw).mean()),
                motion=float(np.clip((span - min_range_deg) / 80., 0, 1)),
                periodicity=float(np.average(periodicity, weights=weights)) if weights else 0.,
                smoothness=float(np.average(smoothness, weights=weights)) if weights else 0.)


def quality_score(quality, weights):
    return float(sum(quality[key] * weight for key, weight in weights.items()))
