"""Opt-in cached-signal experiments for peak counting without period guidance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from . import rep_counter as counter
from .peak_counter import PeakSettings, detect_peaks


@dataclass(frozen=True)
class ImprovementSettings:
    signal: str = "fused"
    smooth_s: float = .3
    prominence_fraction: float = .35
    width_s: float = 1 / 15
    min_gap_s: float = .5


def clean(values, fs, smooth_s):
    """Interpolate short gaps only; allow genuinely unsmoothed comparison."""
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    s = pd.Series(values)
    missing = s.isna().to_numpy()
    if missing.all():
        return values
    groups = np.cumsum(np.r_[True, missing[1:] != missing[:-1]])
    lengths = pd.Series(missing).groupby(groups).transform("sum").to_numpy()
    long_gap = missing & (lengths > round(counter.MAX_GAP_S * fs))
    result = s.interpolate(limit_area="inside").bfill().ffill().to_numpy().copy()
    if smooth_s > 0:
        window = max(3, round(smooth_s * fs))
        if window % 2 == 0:
            window += 1
        if len(result) >= window:
            result = savgol_filter(result, window, 2)
    result[long_gap] = np.nan
    return result


def confidence_average(frame, columns):
    """Use observed joints and finite visibility; no zero-valued missing joints."""
    values, weights = [], []
    for name in columns:
        x = frame[name].to_numpy(dtype=float).copy()
        vis_name = name + "_vis"
        vis = frame[vis_name].to_numpy(dtype=float).copy() if vis_name in frame else np.ones(len(frame))
        weight = np.where(np.isfinite(x) & np.isfinite(vis) & (vis >= .5), np.clip(vis, 0., 1.), 0.)
        values.append(np.where(weight > 0, x, 0.))
        weights.append(weight)
    values, weights = np.column_stack(values), np.column_stack(weights)
    denominator = weights.sum(axis=1)
    result = np.full(len(frame), np.nan)
    np.divide((values * weights).sum(axis=1), denominator, out=result, where=denominator > 0)
    return result


def standardized(values, fs, smooth_s, min_span):
    signal = clean(values, fs, smooth_s)
    finite = np.isfinite(signal)
    if finite.mean() < counter.MIN_VALID_FRACTION:
        return np.full(len(signal), np.nan)
    if np.nanpercentile(signal, 90) - np.nanpercentile(signal, 10) < min_span:
        return np.full(len(signal), np.nan)
    scale = float(np.nanstd(signal))
    return (signal - np.nanmean(signal)) / max(scale, 1e-9)


def build_signals(segment, exercise, smooth_s):
    """Build a few physically related candidates, rather than all-angle PCA."""
    fs = counter.sampling_rate(segment.time_s.to_numpy())
    cfg = counter.EXERCISES[exercise]
    existing = counter.prepare_candidates(segment, cfg)
    chosen = next((c for c in existing if c["usable"]), None)
    raw_candidates = dict(counter.candidate_signals(segment, cfg))
    baseline_raw = raw_candidates[chosen["signal"]] if chosen else np.full(len(segment), np.nan)
    baseline = standardized(baseline_raw, fs, smooth_s, cfg["min_range_deg"])
    if exercise == "jump_jack":
        arm = standardized(confidence_average(segment, ["arm_raise_L", "arm_raise_R"]), fs, smooth_s, 40.)
        ankle = standardized(confidence_average(segment, ["ankle_dist_norm"]), fs, smooth_s, .15)
        leg = standardized(confidence_average(segment, ["leg_spread"]), fs, smooth_s, 20.)
        components = dict(arms=arm, ankle=ankle, legs=leg)
        alternate = ankle
        fusion_inputs = [arm, ankle]
    elif exercise == "push_up":
        left = standardized(confidence_average(segment, ["elbow_L"]), fs, smooth_s, 20.)
        right = standardized(confidence_average(segment, ["elbow_R"]), fs, smooth_s, 20.)
        weighted = standardized(confidence_average(segment, ["elbow_L", "elbow_R"]), fs, smooth_s, 20.)
        components = dict(left=left, right=right, weighted=weighted)
        alternate = weighted
        fusion_inputs = [weighted]
    else:
        knee = standardized(confidence_average(segment, ["knee_L", "knee_R"]), fs, smooth_s, 25.)
        hip = standardized(confidence_average(segment, ["hip_L", "hip_R"]), fs, smooth_s, 20.)
        components = dict(knee=knee, hip=hip, baseline=baseline)
        alternate = hip
        fusion_inputs = [knee, hip]
    # Conservative fusion: require every component at a frame. This avoids
    # discontinuities caused by changing contributors and does not fill occlusion.
    stacked = np.column_stack(fusion_inputs)
    fused = stacked.mean(axis=1)
    fused[~np.isfinite(stacked).all(axis=1)] = np.nan
    return dict(baseline=baseline, alternate=alternate, fused=fused, components=components)


def detect_signal(signal, times, exercise, settings):
    result = detect_peaks(signal, times, counter.EXERCISES[exercise]["pattern"], settings.min_gap_s,
        PeakSettings(prominence_fraction=settings.prominence_fraction, min_prominence_deg=.1,
                     min_width_s=settings.width_s, period_distance_fraction=0.))
    return result["rep_times"]


def count_built(signals, times, exercise, settings):
    if settings.signal != "median":
        return detect_signal(signals[settings.signal], times, exercise, settings)
    candidates = [(name, detect_signal(values, times, exercise, settings))
                  for name, values in signals["components"].items()
                  if np.isfinite(values).mean() >= counter.MIN_VALID_FRACTION]
    if len(candidates) != 3:
        return detect_signal(signals["baseline"], times, exercise, settings)
    median_count = sorted(len(times) for _, times in candidates)[1]
    # Choose an actual candidate's timeline with the median count; never
    # fabricate event times or silently truncate another candidate's timeline.
    return next(events for _, events in candidates if len(events) == median_count)


def count_improved_peaks(frame, exercise, start_s, end_s, *, settings=ImprovementSettings()):
    segment = frame[(frame.time_s >= start_s) & (frame.time_s < end_s)]
    if len(segment) < 8:
        return dict(reps=0, rep_times=[], flag="too_short")
    signals = build_signals(segment, exercise, settings.smooth_s)
    times = count_built(signals, segment.time_s.to_numpy(), exercise, settings)
    return dict(reps=len(times), rep_times=times, flag="", method="experimental_unguided_peaks")
