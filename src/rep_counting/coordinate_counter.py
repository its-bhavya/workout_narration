"""Offline counters over isolated coordinate sidecars, not canonical pose caches."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import rep_counter as counter
from .arm_quality import motion_quality, quality_score
from .peak_improvements import standardized
from .peak_counter import PeakSettings, detect_peaks


SIGNALS = ("shoulder_height", "shoulder_to_wrist_height", "head_height", "elbow_L", "elbow_R", "weighted_elbows")
FAMILIES = SIGNALS + ("quality", "median", "fresh_elbow_quality")
QUALITY_WEIGHTS = dict(visibility=2., continuity=1., periodicity=4.)
VOTERS = ("shoulder_height", "shoulder_to_wrist_height", "weighted_elbows")


@dataclass(frozen=True)
class CoordinatePeakSettings:
    signal: str = "quality"
    smooth_s: float = .3
    min_gap_s: float = .2
    prominence_fraction: float = .45
    recovery_fraction: float = .7
    width_s: float = 1 / 15


def prepare(frame, smooth_s):
    required = ["time_s"] + list(SIGNALS) + [name + "_confidence" for name in SIGNALS]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Incomplete coordinate sidecar (no canonical fallback): {missing}")
    times = frame.time_s.to_numpy(dtype=float)
    if len(times) < 8 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Coordinate counting requires at least eight ordered finite timestamps")
    fs = counter.sampling_rate(times)
    prepared = {}
    for name in SIGNALS:
        values = frame[name].to_numpy(dtype=float).copy()
        visibility = frame[name + "_confidence"].to_numpy(dtype=float)
        values[~np.isfinite(visibility) | (visibility < .5)] = np.nan
        span_guard = 20. if "elbow" in name else .1
        cleaned = standardized(values, fs, smooth_s, span_guard)
        usable = float(np.isfinite(cleaned).mean()) >= counter.MIN_VALID_FRACTION
        quality = motion_quality(cleaned, values, float(np.where(np.isfinite(visibility), visibility, 0.).mean()), fs, .1)
        prepared[name] = dict(values=cleaned, usable=usable, quality=quality,
                              score=quality_score(quality, QUALITY_WEIGHTS),
                              raw_coverage=float(np.isfinite(values).mean()),
                              cleaned_coverage=float(np.isfinite(cleaned).mean()))
    return prepared


def quality_choice(prepared, names=SIGNALS):
    eligible = [name for name in names if prepared[name]["usable"]]
    return max(eligible, key=lambda name: prepared[name]["score"]) if eligible else None


def detect_all(prepared, times, settings):
    peak_settings = PeakSettings(prominence_fraction=settings.prominence_fraction, min_prominence_deg=.1,
                                 min_width_s=settings.width_s, period_distance_fraction=0.,
                                 recovery_fraction=settings.recovery_fraction)
    return {name: detect_peaks(signal["values"], times, "down_up", settings.min_gap_s, peak_settings)["rep_times"]
                  if signal["usable"] else [] for name, signal in prepared.items()}


def choose(prepared, detections, family):
    if family not in FAMILIES:
        raise ValueError("Unknown coordinate signal family")
    if family in SIGNALS:
        name = family if prepared[family]["usable"] else None
    elif family == "fresh_elbow_quality":
        name = quality_choice(prepared, ("elbow_L", "elbow_R"))
    elif family == "quality":
        name = quality_choice(prepared)
    elif all(prepared[name]["usable"] for name in VOTERS):
        median = sorted(len(detections[name]) for name in VOTERS)[1]
        name = next(name for name in VOTERS if len(detections[name]) == median)
    else:
        name = quality_choice(prepared)
    return (detections[name], name) if name else ([], "unusable")


def count_coordinate_peaks(frame, settings=CoordinatePeakSettings()):
    prepared = prepare(frame, settings.smooth_s)
    detections = detect_all(prepared, frame.time_s.to_numpy(dtype=float), settings)
    times, selected = choose(prepared, detections, settings.signal)
    return dict(reps=len(times), rep_times=times, selected_signal=selected,
                signal_diagnostics={name: {k: v for k, v in values.items() if k != "values"}
                                    for name, values in prepared.items()})
