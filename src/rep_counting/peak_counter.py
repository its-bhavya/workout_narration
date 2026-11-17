"""Experimental offline peak counting; the state-machine default is unchanged."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import correlate, detrend, find_peaks

from . import rep_counter as counter


@dataclass(frozen=True)
class PeakSettings:
    prominence_fraction: float = .25
    min_prominence_deg: float = 8.
    min_width_s: float = .1
    period_distance_fraction: float = .5
    max_distance_s: float = 1.
    min_period_s: float = .3
    max_period_s: float = 5.
    min_autocorrelation: float = .35
    recovery_fraction: float = .7


def estimate_period(signal, fs, settings=PeakSettings()):
    """First strong autocorrelation maximum, with at least three cycles observed."""
    if len(signal) < 8 or fs <= 0:
        return None, 0.
    values = np.asarray(signal, dtype=float)
    if not np.isfinite(values).all():
        return None, 0.
    values = detrend(values)
    if np.dot(values, values) <= 1e-9:
        return None, 0.
    ac = correlate(values, values, mode="full", method="fft")[len(values) - 1:]
    ac /= ac[0]
    lo = max(1, int(np.ceil(settings.min_period_s * fs)))
    hi = min(len(values) // 3, int(settings.max_period_s * fs))
    if hi <= lo:
        return None, 0.
    peaks, _ = find_peaks(ac)
    peaks = peaks[(peaks >= lo) & (peaks <= hi)]
    if not len(peaks):
        return None, 0.
    strongest = float(ac[peaks].max())
    strong = peaks[ac[peaks] >= max(settings.min_autocorrelation, .8 * strongest)]
    if not len(strong):
        return None, strongest
    lag = int(strong[0])
    return lag / fs, float(ac[lag])


def detect_peaks(signal, time_s, pattern, min_gap_s, settings=PeakSettings()):
    """Detect motion extrema and timestamp their subsequent recovery.

    Missing runs are processed independently. A bend/opening peak is not itself
    a completed rep: the signal must recover by a fraction of its prominence.
    """
    if not 0 < settings.recovery_fraction <= 1 or settings.min_width_s <= 0:
        raise ValueError("Recovery fraction and peak width must be positive")
    signal, time_s = np.asarray(signal, dtype=float), np.asarray(time_s, dtype=float)
    if len(signal) != len(time_s) or (len(time_s) > 1 and not np.all(np.diff(time_s) > 0)):
        raise ValueError("Signal needs matching, strictly increasing timestamps")
    if pattern not in ("down_up", "low_high_low"):
        raise ValueError("Unknown motion pattern")
    fs = counter.sampling_rate(time_s)
    completions, peak_times, diagnostics = [], [], []
    edges = np.flatnonzero(np.diff(np.r_[False, np.isfinite(signal), False]))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < 8:
            continue
        values = signal[start:end]
        activity = -values if pattern == "down_up" else values
        period, confidence = estimate_period(values, fs, settings)
        distance_s = min_gap_s
        if period is not None:
            # A wrong harmonic or slow average must not suppress fast cycles.
            distance_s = max(min_gap_s, min(settings.max_distance_s, settings.period_distance_fraction * period))
        span = float(np.percentile(values, 90) - np.percentile(values, 10))
        prominence = max(settings.min_prominence_deg, settings.prominence_fraction * span)
        peaks, properties = find_peaks(activity, prominence=prominence,
                                       distance=max(1., distance_s * fs),
                                       width=max(1., settings.min_width_s * fs))
        accepted = 0
        for i, peak in enumerate(peaks):
            stop = int(peaks[i + 1]) if i + 1 < len(peaks) else len(values)
            target = activity[peak] - settings.recovery_fraction * properties["prominences"][i]
            recovered = np.flatnonzero(activity[peak + 1:stop] <= target)
            if not len(recovered):
                continue
            completion = start + int(peak) + 1 + int(recovered[0])
            if completions and time_s[completion] - completions[-1] < min_gap_s:
                continue
            completions.append(float(time_s[completion]))
            peak_times.append(float(time_s[start + peak]))
            accepted += 1
        diagnostics.append(dict(start_s=float(time_s[start]), end_s=float(time_s[end - 1]),
                                period_s=period, autocorrelation=confidence,
                                distance_s=distance_s, prominence_deg=prominence,
                                min_width_s=settings.min_width_s, candidates=len(peaks), accepted=accepted))
    return dict(reps=len(completions), rep_times=[round(t, 3) for t in completions],
                peak_times=[round(t, 3) for t in peak_times], runs=diagnostics)


def count_peaks_for_exercise(df, exercise, start_s, end_s, *, settings=PeakSettings(),
                             end_pad_s=0., use_3d=False):
    """Drop-in experimental count API using the existing signal-selection policy."""
    cfg = counter.EXERCISES[exercise]
    segment = df[(df.time_s >= start_s) & (df.time_s < end_s + end_pad_s)]
    result = dict(exercise=exercise, method="peaks", settings=asdict(settings),
                  reps=0, rep_times=[], peak_times=[], runs=[], flag="", signal="")
    if len(segment) < 8:
        return result | {"flag": "too_short"}
    candidates = counter.prepare_candidates(segment, cfg, use_3d)
    chosen = next((c for c in candidates if c["usable"]), None)
    if chosen is None:
        return result | {"flag": "mostly_missing" if candidates[0]["valid_fraction"] < counter.MIN_VALID_FRACTION else "low_motion"}
    return result | detect_peaks(chosen["values"], segment.time_s.to_numpy(), cfg["pattern"],
                                  cfg["min_gap_s"], settings) | {"signal": chosen["signal"]}
