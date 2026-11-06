"""PCG-specific processing: envelope, S1/S2 search and window quality.

The ECG tells us *when* a beat happens.  The PCG envelope is then used to
locate the first and second heart sounds inside a small search interval.  This
is the "ECG-informed PCG segmentation" part of the original paper's idea.

These S1/S2 locations are used for quality control and for checking that a
gating window contains the expected sounds.  The main E2 window itself is
anchored to the R peak (a configurable choice; see RESEARCH_DESIGN.md).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal

from .preprocessing import bandpass_filter


def pcg_envelope(x: np.ndarray, fs: int | float, smooth_hz: float = 20.0) -> np.ndarray:
    """Return a smooth amplitude envelope of the PCG signal.

    The Hilbert transform gives a robust analytic amplitude.  A low-pass filter
    removes rapid fluctuations so that S1/S2 appear as broad energy bumps.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x.astype(np.float32)
    analytic = signal.hilbert(x)
    env = np.abs(analytic)
    if smooth_hz and smooth_hz < 0.5 * float(fs):
        env = bandpass_filter(env, fs, None, smooth_hz, order=3, zero_phase=True)
    return env.astype(np.float32)


def _safe_peak(
    env: np.ndarray,
    fs: float,
    start_seconds: float,
    end_seconds: float,
) -> tuple[int | None, float | None]:
    """Return the sample index and value of the maximum in a time interval."""
    n = env.size
    lo = int(round(start_seconds * fs))
    hi = int(round(end_seconds * fs))
    lo = max(0, min(lo, n - 1))
    hi = max(lo + 1, min(hi, n))
    if hi - lo < 2:
        return None, None
    idx = int(np.argmax(env[lo:hi])) + lo
    return idx, float(env[idx])


def find_s1_s2(
    envelope: np.ndarray,
    fs: int | float,
    r_peaks: np.ndarray,
    t_waves: Sequence[Mapping[str, Any]] | None = None,
    s1_after_r_seconds: tuple[float, float] = (0.0, 0.25),
    s2_after_t_seconds: tuple[float, float] = (-0.10, 0.20),
    s2_fallback_after_r_seconds: tuple[float, float] = (0.25, 0.50),
) -> list[dict[str, Any]]:
    """Locate S1 and S2 in the PCG envelope for each detected R peak.

    The paper states that S1 and S2 are searched in intervals beginning
    0.2 seconds after the QRS and T-wave markers, respectively, but does not
    give the exact interval length.  The windows here are configurable and
    documented as an implementation choice.
    """
    env = np.asarray(envelope, dtype=np.float64).ravel()
    fs = float(fs)
    peaks = np.asarray(r_peaks, dtype=int)
    t_by_r = {int(w["r_peak_sample"]): w for w in (t_waves or [])}
    out: list[dict[str, Any]] = []
    for r in peaks:
        r_time = float(r) / fs
        s1_idx, s1_amp = _safe_peak(
            env, fs, r_time + s1_after_r_seconds[0], r_time + s1_after_r_seconds[1]
        )
        t_wave = t_by_r.get(int(r))
        if t_wave is not None and t_wave.get("t_end_sample") is not None:
            t_end_time = float(t_wave["t_end_time"])
            s2_idx, s2_amp = _safe_peak(
                env,
                fs,
                t_end_time + s2_after_t_seconds[0],
                t_end_time + s2_after_t_seconds[1],
            )
        else:
            s2_idx, s2_amp = None, None
        if s2_idx is None:
            s2_idx, s2_amp = _safe_peak(
                env,
                fs,
                r_time + s2_fallback_after_r_seconds[0],
                r_time + s2_fallback_after_r_seconds[1],
            )
        valid = s1_idx is not None and s2_idx is not None
        if valid:
            interval = (s2_idx - s1_idx) / fs
            valid = 0.12 <= interval <= 0.70 and s2_idx > s1_idx
        out.append(
            {
                "r_peak_sample": int(r),
                "r_peak_time": r_time,
                "s1_sample": int(s1_idx) if s1_idx is not None else None,
                "s1_time": float(s1_idx) / fs if s1_idx is not None else None,
                "s1_amplitude": s1_amp,
                "s2_sample": int(s2_idx) if s2_idx is not None else None,
                "s2_time": float(s2_idx) / fs if s2_idx is not None else None,
                "s2_amplitude": s2_amp,
                "s1_s2_interval_seconds": (
                    float(s2_idx - s1_idx) / fs if valid else None
                ),
                "valid": bool(valid),
            }
        )
    return out


def window_quality(
    x: np.ndarray,
    fs: int | float,
    min_rms: float = 0.0,
    max_clipping_fraction: float = 0.02,
) -> dict[str, Any]:
    """Compute simple quality statistics for one PCG window."""
    x = np.asarray(x, dtype=np.float64).ravel()
    reasons: list[str] = []
    if x.size == 0:
        return {"ok": False, "reasons": ["empty window"], "rms": 0.0, "peak": 0.0}
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    if min_rms > 0 and rms < min_rms:
        reasons.append(f"rms {rms:.6f} < {min_rms}")
    if peak > 0:
        rail = float(np.mean(np.abs(x) >= 0.999 * peak))
        if rail > max_clipping_fraction:
            reasons.append(f"possible clipping ({rail:.1%})")
    else:
        reasons.append("flat window")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "rms": rms,
        "peak": peak,
        "n_samples": int(x.size),
    }


def extract_window(
    signal: np.ndarray,
    fs: int | float,
    start_seconds: float,
    duration_seconds: float,
) -> tuple[np.ndarray, int, int]:
    """Extract a window by time, returning ``(window, start_sample, end_sample)``.

    Calls that fall outside the signal raise ``ValueError``; callers must record
    the rejection rather than silently padding or falling back.
    """
    x = np.asarray(signal)
    start = int(round(float(start_seconds) * float(fs)))
    end = start + int(round(float(duration_seconds) * float(fs)))
    if start < 0 or end > x.size or end <= start:
        raise ValueError(
            f"Window [{start_seconds:.3f}, {start_seconds + duration_seconds:.3f}] s "
            f"is outside the signal (length {x.size / float(fs):.3f} s)."
        )
    return x[start:end], start, end
