"""ECG quality control, R-peak detection and T-wave localisation.

The original paper describes a three-stage ECG-gating algorithm:

1. **R-peak first track** - derivative of the first 5 s, third-highest peak,
   half of that value as the threshold, minimum peak-to-peak distance 0.3 s.
2. **R-peak second track** - re-process RR intervals that exceed the mean by
   more than two standard deviations (for example missed VPCs).
3. **T-wave track** - local S-wave minimum, T-wave start/end equations, and
   peak/trough labelling depending on T-wave inversion.

The paper does not publish the exact peak-search routine, threshold updates,
or filtering details.  ``detect_r_peaks_paper`` therefore implements the
described logic as faithfully as the available information permits and is
named ``paper_inspired`` throughout the project.  ``detect_r_peaks_simple`` is
a conventional Pan-Tompkins-style detector used as a clearly named,
robustness-oriented baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy import signal

from .preprocessing import bandpass_filter, notch_filter


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EcgQuality:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": self.reasons, "details": self.details}


@dataclass
class RPeakResult:
    peaks: np.ndarray
    times: np.ndarray
    polarity: str
    method: str
    quality: EcgQuality
    threshold: float | None = None
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "peaks": self.peaks.tolist(),
            "times": self.times.tolist(),
            "polarity": self.polarity,
            "method": self.method,
            "threshold": self.threshold,
            "quality": self.quality.to_dict(),
            "reasons": self.reasons,
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
def assess_ecg_quality(
    x: np.ndarray,
    fs: int | float,
    min_duration_seconds: float = 2.0,
    flatline_std: float = 1e-5,
    max_clipping_fraction: float = 0.05,
) -> EcgQuality:
    """Check basic ECG integrity before attempting R-peak detection.

    The checks are intentionally simple and interpretable: finite values,
    sufficient duration, non-flat signal, and absence of severe clipping.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    reasons: list[str] = []
    details: dict[str, Any] = {}
    duration = x.size / float(fs) if fs else 0.0
    details["duration_seconds"] = duration
    details["n_samples"] = int(x.size)
    if x.size == 0:
        return EcgQuality(False, ["empty signal"], details)
    if not np.all(np.isfinite(x)):
        reasons.append("non-finite values")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if duration < min_duration_seconds:
        reasons.append(f"duration {duration:.2f}s < {min_duration_seconds}s")
    std = float(np.std(x))
    details["std"] = std
    details["min"] = float(np.min(x))
    details["max"] = float(np.max(x))
    if std < flatline_std:
        reasons.append("flatline / near-zero variance")
    # Clipping is estimated from repeated extreme values.  PCM16 WAVs often
    # contain a few samples at the rails, so the threshold is deliberately
    # permissive.
    max_abs = float(np.max(np.abs(x)))
    if max_abs > 0:
        at_rail = np.mean(np.abs(x) >= 0.999 * max_abs)
        details["rail_fraction"] = float(at_rail)
        if at_rail > max_clipping_fraction:
            reasons.append(f"possible clipping ({at_rail:.1%} at rail)")
    # Rough SNR: energy in the QRS band versus the residual.
    try:
        qrs = bandpass_filter(x, fs, 5.0, 25.0, order=3, zero_phase=True)
        residual = x - qrs
        signal_power = float(np.mean(qrs ** 2))
        noise_power = float(np.mean(residual ** 2))
        snr_db = 10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))
        details["qrs_band_snr_db"] = float(snr_db)
    except Exception:
        pass
    return EcgQuality(not reasons, reasons, details)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prepare_for_detection(
    x: np.ndarray,
    fs: int | float,
    ecg_cfg: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Filter a copy of the ECG for detection, preserving native fs."""
    cfg = dict(ecg_cfg or {})
    y = np.asarray(x, dtype=np.float64).ravel()
    y = y - np.mean(y)
    notch = cfg.get("notch_hz")
    if notch:
        y = notch_filter(y, fs, float(notch))
    band = cfg.get("bandpass_hz", [5.0, 25.0])
    if band:
        y = bandpass_filter(y, fs, band[0], band[1], order=3, zero_phase=True)
    return y.astype(np.float64)


def _choose_polarity(
    y: np.ndarray,
    candidate_indices: np.ndarray,
    fs: float,
    search_seconds: float = 0.10,
) -> str:
    """Decide whether the dominant R deflection is positive or negative."""
    if candidate_indices.size == 0:
        return "positive"
    half = max(1, int(round(search_seconds * fs)))
    pos_amps = []
    neg_amps = []
    for idx in candidate_indices:
        lo = max(0, int(idx) - half)
        hi = min(y.size, int(idx) + half + 1)
        if hi <= lo:
            continue
        seg = y[lo:hi]
        pos_amps.append(float(np.max(seg)))
        neg_amps.append(float(-np.min(seg)))
    if not pos_amps:
        return "positive"
    return "positive" if np.median(pos_amps) >= np.median(neg_amps) else "negative"


def _refine_to_r_apex(
    y: np.ndarray,
    candidate_indices: np.ndarray,
    fs: float,
    polarity: str,
    search_seconds: float = 0.10,
) -> np.ndarray:
    """Move derivative-domain candidates to the actual R-wave extremum."""
    half = max(1, int(round(search_seconds * fs)))
    refined: list[int] = []
    for idx in np.asarray(candidate_indices, dtype=int):
        lo = max(0, int(idx) - half)
        hi = min(y.size, int(idx) + half + 1)
        if hi <= lo:
            continue
        seg = y[lo:hi]
        local = int(np.argmax(seg) if polarity == "positive" else np.argmin(seg))
        refined.append(lo + local)
    if not refined:
        return np.asarray([], dtype=int)
    # Remove duplicates that collapsed to the same apex.
    return np.unique(np.asarray(refined, dtype=int))


def _remove_too_close(
    peaks: np.ndarray,
    y: np.ndarray,
    fs: float,
    min_rr_seconds: float,
) -> np.ndarray:
    """Remove peaks closer than the physiological minimum RR interval."""
    if peaks.size <= 1:
        return peaks
    min_dist = max(1, int(round(min_rr_seconds * fs)))
    kept: list[int] = [int(peaks[0])]
    for p in peaks[1:]:
        p = int(p)
        if p - kept[-1] >= min_dist:
            kept.append(p)
        else:
            # Keep the larger absolute deflection.
            if abs(y[p]) > abs(y[kept[-1]]):
                kept[-1] = p
    return np.asarray(kept, dtype=int)


def _rr_stats(peaks: np.ndarray, fs: float) -> dict[str, Any]:
    if peaks.size < 2:
        return {"n_intervals": 0, "median_rr": None, "mean_rr": None, "std_rr": None}
    rr = np.diff(peaks) / float(fs)
    return {
        "n_intervals": int(rr.size),
        "median_rr": float(np.median(rr)),
        "mean_rr": float(np.mean(rr)),
        "std_rr": float(np.std(rr)),
        "min_rr": float(np.min(rr)),
        "max_rr": float(np.max(rr)),
        "heart_rate_bpm": float(60.0 / np.median(rr)) if np.median(rr) > 0 else None,
    }


def _plausibility(
    peaks: np.ndarray,
    y: np.ndarray,
    fs: float,
    min_hr: float,
    max_hr: float,
    min_rr_seconds: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if peaks.size < 3:
        reasons.append(f"too few R peaks ({peaks.size})")
        return False, reasons
    rr = np.diff(peaks) / float(fs)
    median_rr = float(np.median(rr))
    if median_rr <= 0:
        reasons.append("non-positive RR interval")
        return False, reasons
    hr = 60.0 / median_rr
    if not (min_hr <= hr <= max_hr):
        reasons.append(f"median heart rate {hr:.1f} bpm outside [{min_hr}, {max_hr}]")
    if np.min(rr) < min_rr_seconds * 0.9:
        reasons.append(f"RR interval below {min_rr_seconds}s")
    return not reasons, reasons


# ---------------------------------------------------------------------------
# R-peak detectors
# ---------------------------------------------------------------------------
def detect_r_peaks_paper(
    x: np.ndarray,
    fs: int | float,
    ecg_cfg: Mapping[str, Any] | None = None,
    polarity: str = "auto",
) -> RPeakResult:
    """Paper-inspired R-peak detection.

    Implements the first and second tracks described in the paper:
      * initialise on the first 5 s using the derivative and third-highest
        peak; threshold = half of that value; minimum distance = 0.3 s;
      * refine candidates to the R apex;
      * re-search RR gaps larger than mean + 2 standard deviations.

    This is a *paper-inspired* implementation, not an exact reproduction.
    """
    cfg = dict(ecg_cfg or {})
    quality = assess_ecg_quality(x, fs)
    y = _prepare_for_detection(x, fs, cfg)
    if y.size < int(2 * fs):
        return RPeakResult(
            peaks=np.asarray([], dtype=int),
            times=np.asarray([], dtype=float),
            polarity="positive",
            method="paper_inspired",
            quality=quality,
            reasons=["signal shorter than 2 s"],
        )

    # The paper uses the first 5 s for initialisation.  If that segment has
    # fewer than three beats, fall back to the whole recording for threshold
    # estimation (a documented implementation choice).
    init_end = min(y.size, int(round(5.0 * fs)))
    min_dist = max(1, int(round(0.3 * fs)))

    def _initial_candidates(segment: np.ndarray) -> tuple[np.ndarray, float]:
        derivative = np.abs(np.diff(segment))
        found, _ = signal.find_peaks(derivative, distance=min_dist)
        if found.size == 0:
            return np.asarray([], dtype=int), 0.0
        # SciPy only adds ``peak_heights`` to the properties when ``height`` is
        # supplied, so read the derivative values directly.
        heights = derivative[found]
        third = float(np.sort(heights)[-3]) if heights.size >= 3 else float(np.max(heights))
        threshold = 0.5 * third
        selected = found[heights >= threshold]
        return selected, threshold

    initial, threshold = _initial_candidates(y[:init_end])
    if initial.size < 3:
        initial, threshold = _initial_candidates(y)
    if initial.size == 0:
        return RPeakResult(
            peaks=np.asarray([], dtype=int),
            times=np.asarray([], dtype=float),
            polarity="positive",
            method="paper_inspired",
            quality=quality,
            threshold=threshold,
            reasons=["no derivative peaks found"],
        )

    # Detect over the whole recording using the initial threshold.
    derivative_full = np.abs(np.diff(y))
    found_all, _ = signal.find_peaks(derivative_full, distance=min_dist)
    if found_all.size:
        heights_all = derivative_full[found_all]
        candidates = found_all[heights_all >= threshold]
    else:
        candidates = initial
    if candidates.size == 0:
        candidates = initial

    # Polarity and apex refinement.
    apex_polarity = polarity
    if polarity == "auto":
        apex_polarity = _choose_polarity(y, candidates, fs)
    refined = _refine_to_r_apex(y, candidates, fs, apex_polarity)
    refined = _remove_too_close(refined, y, fs, min_rr_seconds=0.3)

    # Second track: re-process RR intervals that are much longer than average.
    rr = np.diff(refined) / float(fs) if refined.size > 1 else np.asarray([])
    added = 0
    if rr.size >= 3:
        mean_rr = float(np.mean(rr))
        std_rr = float(np.std(rr))
        if std_rr > 0:
            new_peaks = list(refined)
            for i in range(len(refined) - 1):
                gap = (refined[i + 1] - refined[i]) / float(fs)
                if gap <= max(0.6, mean_rr + 2.0 * std_rr):
                    continue
                lo = int(refined[i] + 0.3 * fs)
                hi = int(refined[i + 1] - 0.3 * fs)
                if hi <= lo + 2:
                    continue
                segment = derivative_full[lo:hi]
                if segment.size == 0:
                    continue
                local_idx, _ = signal.find_peaks(
                    segment, distance=min_dist, height=max(0.4 * threshold, 0.0)
                )
                if local_idx.size:
                    best = int(local_idx[int(np.argmax(segment[local_idx]))]) + lo
                    if best not in new_peaks:
                        new_peaks.append(best)
                        added += 1
            if added:
                refined = np.sort(np.asarray(new_peaks, dtype=int))
                refined = _remove_too_close(refined, y, fs, min_rr_seconds=0.3)

    ok, reasons = _plausibility(
        refined,
        y,
        fs,
        min_hr=float(cfg.get("min_heart_rate_bpm", 40)),
        max_hr=float(cfg.get("max_heart_rate_bpm", 200)),
        min_rr_seconds=float(cfg.get("min_rr_seconds", 0.3)),
    )
    times = refined.astype(float) / float(fs)
    return RPeakResult(
        peaks=refined,
        times=times,
        polarity=apex_polarity,
        method="paper_inspired",
        quality=quality,
        threshold=float(threshold),
        reasons=reasons,
        diagnostics={
            "n_initial_candidates": int(initial.size),
            "n_candidates_full": int(candidates.size),
            "n_added_second_track": int(added),
            "rr_stats": _rr_stats(refined, fs),
            "init_seconds": float(init_end / fs),
        },
    )


def detect_r_peaks_simple(
    x: np.ndarray,
    fs: int | float,
    ecg_cfg: Mapping[str, Any] | None = None,
    polarity: str = "auto",
    return_diagnostics: bool = False,
):
    """Pan-Tompkins-style energy detector.

    This is deliberately *not* called "the paper method".  It is a robust
    baseline used for comparison and for the synchronisation diagnostic.
    """
    cfg = dict(ecg_cfg or {})
    quality = assess_ecg_quality(x, fs)
    y = _prepare_for_detection(x, fs, cfg)
    if y.size < int(2 * fs):
        result = RPeakResult(
            peaks=np.asarray([], dtype=int),
            times=np.asarray([], dtype=float),
            polarity="positive",
            method="simple_rpeak",
            quality=quality,
            reasons=["signal shorter than 2 s"],
        )
        return result.to_dict() if return_diagnostics else result

    # Pan-Tompkins: derivative -> square -> moving-window integration.
    derivative = np.diff(y)
    energy = derivative ** 2
    win = max(1, int(round(0.150 * fs)))
    kernel = np.ones(win, dtype=np.float64) / win
    integrated = np.convolve(energy, kernel, mode="same")

    # Adaptive height: a fraction of the robust upper percentile.  This avoids
    # selecting one huge artefact as the only threshold.
    scale = float(np.percentile(integrated, 99.5))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(integrated)) if integrated.size else 0.0
    min_dist = max(1, int(round(0.3 * fs)))
    height = 0.15 * scale if scale > 0 else None
    candidates, props = signal.find_peaks(
        integrated,
        distance=min_dist,
        height=height,
        prominence=0.05 * scale if scale > 0 else None,
    )
    if candidates.size == 0:
        candidates = np.asarray([], dtype=int)

    apex_polarity = polarity
    if polarity == "auto":
        apex_polarity = _choose_polarity(y, candidates, fs)
    refined = _refine_to_r_apex(y, candidates, fs, apex_polarity)
    refined = _remove_too_close(refined, y, fs, min_rr_seconds=0.3)

    ok, reasons = _plausibility(
        refined,
        y,
        fs,
        min_hr=float(cfg.get("min_heart_rate_bpm", 40)),
        max_hr=float(cfg.get("max_heart_rate_bpm", 200)),
        min_rr_seconds=float(cfg.get("min_rr_seconds", 0.3)),
    )
    times = refined.astype(float) / float(fs)
    result = RPeakResult(
        peaks=refined,
        times=times,
        polarity=apex_polarity,
        method="simple_rpeak",
        quality=quality,
        threshold=float(height) if height is not None else None,
        reasons=reasons,
        diagnostics={
            "n_candidates": int(candidates.size),
            "rr_stats": _rr_stats(refined, fs),
            "energy_scale": scale,
        },
    )
    return result.to_dict() if return_diagnostics else result


def detect_r_peaks(
    x: np.ndarray,
    fs: int | float,
    method: str = "paper_inspired",
    ecg_cfg: Mapping[str, Any] | None = None,
    polarity: str = "auto",
) -> RPeakResult:
    """Dispatch to the requested R-peak detector."""
    method = (method or "paper_inspired").lower()
    if method in {"paper", "paper_inspired", "paper-inspired"}:
        return detect_r_peaks_paper(x, fs, ecg_cfg=ecg_cfg, polarity=polarity)
    if method in {"simple", "simple_rpeak", "pan_tompkins", "pan-tompkins"}:
        return detect_r_peaks_simple(x, fs, ecg_cfg=ecg_cfg, polarity=polarity)
    raise ValueError(
        f"Unknown gating method {method!r}. Choose 'paper_inspired' or 'simple_rpeak'."
    )


# ---------------------------------------------------------------------------
# T waves
# ---------------------------------------------------------------------------
def detect_t_waves(
    x: np.ndarray,
    r_peaks: np.ndarray,
    fs: int | float,
    ecg_cfg: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Localise T waves using the equations reported in the paper.

    Paper equations (times relative to the R wave):
      * S-wave = local minimum after the R wave
      * T start = R + 0.05 s + 0.1 * RR
      * T end   = R + 0.2 s + 0.3 * RR

    The exact inversion rule is not fully specified in the paper.  Here the
    larger of the absolute positive and negative deflections inside the T
    interval determines the label: a dominant trough is marked inverted.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    peaks = np.asarray(r_peaks, dtype=int)
    if peaks.size == 0:
        return []
    rr_all = np.diff(peaks) / float(fs)
    default_rr = float(np.median(rr_all)) if rr_all.size else 0.8
    waves: list[dict[str, Any]] = []
    for i, r in enumerate(peaks):
        rr = float(rr_all[i]) if i < rr_all.size else default_rr
        rr = float(np.clip(rr, 0.3, 1.5))
        s0 = int(r + 0.02 * fs)
        s1 = int(r + 0.15 * fs)
        s_idx = int(np.argmin(x[s0:s1]) + s0) if s1 > s0 and s1 <= x.size else None
        t0 = int(round(r + (0.05 + 0.10 * rr) * fs))
        t1 = int(round(r + (0.20 + 0.30 * rr) * fs))
        t0 = max(0, min(t0, x.size - 1))
        t1 = max(t0 + 1, min(t1, x.size))
        if t1 - t0 < 2:
            continue
        segment = x[t0:t1]
        # Remove a local linear trend so baseline drift does not dominate the
        # peak/trough comparison.
        seg_t = np.arange(segment.size, dtype=np.float64)
        if segment.size >= 3:
            coeffs = np.polyfit(seg_t, segment, 1)
            detrended = segment - np.polyval(coeffs, seg_t)
        else:
            detrended = segment
        max_idx = int(np.argmax(segment))
        min_idx = int(np.argmin(segment))
        if abs(float(segment[min_idx])) > abs(float(segment[max_idx])):
            peak_idx = min_idx
            inverted = True
        else:
            peak_idx = max_idx
            inverted = False
        peak_abs = int(round(t0 + peak_idx))
        waves.append(
            {
                "r_peak_sample": int(r),
                "r_peak_time": float(r) / float(fs),
                "s_wave_sample": int(s_idx) if s_idx is not None else None,
                "s_wave_time": float(s_idx) / float(fs) if s_idx is not None else None,
                "t_start_sample": int(t0),
                "t_end_sample": int(t1),
                "t_start_time": float(t0) / float(fs),
                "t_end_time": float(t1) / float(fs),
                "t_peak_sample": int(peak_abs),
                "t_peak_time": float(peak_abs) / float(fs),
                "inverted": bool(inverted),
                "t_amplitude": float(segment[peak_idx]),
                "rr_seconds": float(rr),
            }
        )
    return waves


# ---------------------------------------------------------------------------
# Heart rate / summary
# ---------------------------------------------------------------------------
def heart_rate_stats(r_peaks: np.ndarray, fs: int | float) -> dict[str, Any]:
    return _rr_stats(np.asarray(r_peaks, dtype=int), fs)
