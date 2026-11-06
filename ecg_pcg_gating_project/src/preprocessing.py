"""Signal preprocessing shared by the ECG and PCG pipelines.

Two PCG profiles are supported explicitly:

``paper_haar``
    Inspired by the original paper: resample to 1440 Hz, band-pass 20-400 Hz,
    two-level Haar DWT keeping the approximation coefficients (cA2), and
    z-score normalisation.  A 1.5 s window at 1440 Hz becomes 540 samples.

``hfref_2000``
    The dataset paper's published preprocessing: resample to 2000 Hz,
    25-400 Hz Butterworth band-pass, no DWT, 1.28 s windows (2560 samples).

The profiles are deliberately separate.  The code does not silently mix the
two configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import signal

try:  # PyWavelets is a required dependency, but a clear error is friendlier.
    import pywt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyWavelets is required.  Install it with: pip install PyWavelets"
    ) from exc


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------
@dataclass
class NormalizationStats:
    """Mean/std estimated on the *training* partition only."""

    mean: float
    std: float
    count: int = 0
    source: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "count": int(self.count),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizationStats":
        return cls(
            mean=float(data["mean"]),
            std=float(data["std"]),
            count=int(data.get("count", 0)),
            source=str(data.get("source", "train")),
        )


def compute_normalization_stats(
    arrays: Iterable[np.ndarray],
    source: str = "train",
    eps: float = 1e-8,
) -> NormalizationStats:
    """Compute global mean/std from an iterable of 1-D arrays.

    Using a streaming sum/sum-of-squares avoids keeping every training window
    in memory.  The caller is responsible for passing *training* arrays only.
    """
    total = 0
    total_sq = 0.0
    count = 0
    for arr in arrays:
        x = np.asarray(arr, dtype=np.float64).ravel()
        if x.size == 0:
            continue
        total += float(np.sum(x))
        total_sq += float(np.dot(x, x))
        count += int(x.size)
    if count == 0:
        raise ValueError("Cannot compute normalization statistics from zero samples.")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if std < eps:
        std = 1.0
    return NormalizationStats(mean=float(mean), std=std, count=count, source=source)


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------
def resample_signal(x: np.ndarray, fs_in: int | float, fs_out: int | float) -> np.ndarray:
    """Polyphase resampling with a rational ratio.

    ``scipy.signal.resample_poly`` is used instead of FFT resampling because it
    is local, fast for long signals and preserves the time axis better.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    fs_in = int(round(float(fs_in)))
    fs_out = int(round(float(fs_out)))
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(f"Sampling rates must be positive, got {fs_in} and {fs_out}")
    if fs_in == fs_out or x.size == 0:
        return x.astype(np.float32)
    g = gcd(fs_in, fs_out)
    up = fs_out // g
    down = fs_in // g
    # A short signal can be shorter than the filter; fall back to linear
    # interpolation in that rare case so tests and edge cases do not crash.
    if x.size < max(up, down) * 4:
        n_out = int(round(x.size * fs_out / fs_in))
        old_t = np.arange(x.size) / float(fs_in)
        new_t = np.arange(n_out) / float(fs_out)
        return np.interp(new_t, old_t, x).astype(np.float32)
    return signal.resample_poly(x, up, down).astype(np.float32)


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low_hz: float | None,
    high_hz: float | None,
    order: int = 5,
    zero_phase: bool = True,
) -> np.ndarray:
    """Butterworth band-pass (or low/high-pass) filter.

    ``zero_phase=True`` uses ``sosfiltfilt`` (forward-backward filtering), which
    avoids the phase delay of a causal filter.  Offline analysis is not
    real-time, so zero-phase filtering is the sensible default.  The official
    dataset code used a causal ``lfilter``; that difference is documented in
    RESEARCH_DESIGN.md.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x.astype(np.float32)
    nyq = 0.5 * float(fs)
    low = None if low_hz is None else max(float(low_hz), 1e-6)
    high = None if high_hz is None else min(float(high_hz), nyq * 0.999)
    if low is not None and high is not None and low >= high:
        raise ValueError(f"Invalid band-pass: low={low}, high={high}, fs={fs}")
    if low is None and high is None:
        return x.astype(np.float32)
    if low is None:
        sos = signal.butter(order, high / nyq, btype="lowpass", output="sos")
    elif high is None:
        sos = signal.butter(order, low / nyq, btype="highpass", output="sos")
    else:
        sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    # sosfiltfilt needs a signal longer than the padding length.
    padlen = 3 * (2 * len(sos) + 1)
    if zero_phase and x.size > padlen:
        y = signal.sosfiltfilt(sos, x)
    else:
        y = signal.sosfilt(sos, x)
    return y.astype(np.float32)


def notch_filter(x: np.ndarray, fs: float, freq_hz: float, q: float = 30.0) -> np.ndarray:
    """Remove mains interference (50 or 60 Hz)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or freq_hz is None:
        return x.astype(np.float32)
    b, a = signal.iirnotch(float(freq_hz), q, float(fs))
    if x.size > 3 * max(len(a), len(b)):
        return signal.filtfilt(b, a, x).astype(np.float32)
    return signal.lfilter(b, a, x).astype(np.float32)


def zscore(x: np.ndarray, mean: float | None = None, std: float | None = None, eps: float = 1e-8) -> np.ndarray:
    """Z-score a signal using supplied statistics or its own statistics."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == 0:
        return x
    mu = float(np.mean(x)) if mean is None else float(mean)
    sigma = float(np.std(x)) if std is None else float(std)
    if not np.isfinite(sigma) or sigma < eps:
        sigma = 1.0
    return ((x - mu) / sigma).astype(np.float32)


def fix_length(x: np.ndarray, length: int, pad_value: float = 0.0) -> np.ndarray:
    """Center-crop or zero-pad a 1-D signal to exactly *length* samples."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == length:
        return x
    if x.size > length:
        start = (x.size - length) // 2
        return x[start : start + length]
    out = np.full(length, pad_value, dtype=np.float32)
    start = (length - x.size) // 2
    out[start : start + x.size] = x
    return out


# ---------------------------------------------------------------------------
# DWT
# ---------------------------------------------------------------------------
def dwt_output_length(n_samples: int, level: int = 2, keep: str = "approximation") -> int:
    """Length produced by the Haar DWT used in this project."""
    n = int(n_samples)
    if keep == "approximation":
        for _ in range(int(level)):
            n = int(np.ceil(n / 2))
        return n
    if keep == "all":
        # cA_L plus all detail bands.  With periodisation the total is the
        # original length (up to rounding).
        total = int(np.ceil(n / (2 ** level)))
        for l in range(1, int(level) + 1):
            total += int(np.ceil(n / (2 ** l)))
        return total
    raise ValueError(f"Unknown DWT keep mode: {keep}")


def haar_dwt(
    x: np.ndarray,
    level: int = 2,
    keep: str = "approximation",
    wavelet: str = "haar",
    mode: str = "periodization",
) -> np.ndarray:
    """Apply a discrete wavelet transform.

    ``keep='approximation'`` returns only the level-*L* approximation
    coefficients (cA_L).  This is the interpretation that matches the original
    paper's statement that the input size is reduced to one quarter after two
    levels.  ``keep='all'`` concatenates cA_L, cD_L, ..., cD_1 and is provided
    only for an explicit sensitivity analysis.

    ``mode='periodization'`` avoids boundary extension and gives an exact
    length reduction, which is important for reproducible tensor shapes.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x.astype(np.float32)
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=int(level), mode=mode)
    if keep == "approximation":
        out = coeffs[0]
    elif keep == "all":
        out = np.concatenate([np.asarray(c, dtype=np.float64) for c in coeffs])
    else:
        raise ValueError(f"Unknown DWT keep mode: {keep}")
    return out.astype(np.float32)


def transform_window(
    x: np.ndarray,
    profile: Mapping[str, Any],
    expected_length: int | None = None,
) -> np.ndarray:
    """Apply the profile's DWT (if enabled) to an already-filtered window.

    The input is assumed to already be at ``profile['pcg_target_fs']`` and to
    have been band-pass filtered.  No z-score is applied here; normalisation
    is handled separately so training statistics can be used.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    if bool(profile.get("dwt_enabled", False)):
        x = haar_dwt(
            x,
            level=int(profile.get("dwt_level", 2)),
            keep=str(profile.get("dwt_keep", "approximation")),
            wavelet=str(profile.get("dwt_wavelet", "haar")),
            mode=str(profile.get("dwt_mode", "periodization")),
        )
    if expected_length is not None:
        x = fix_length(x, int(expected_length))
    return x.astype(np.float32)


def expected_window_length(profile: Mapping[str, Any], window_seconds: float | None = None) -> int:
    """Expected number of samples fed to the CNN for one window."""
    fs = int(profile["pcg_target_fs"])
    seconds = float(window_seconds if window_seconds is not None else profile["window_seconds"])
    n = int(round(seconds * fs))
    if bool(profile.get("dwt_enabled", False)):
        n = dwt_output_length(
            n,
            level=int(profile.get("dwt_level", 2)),
            keep=str(profile.get("dwt_keep", "approximation")),
        )
    return int(n)


# ---------------------------------------------------------------------------
# Full-recording and window preprocessing
# ---------------------------------------------------------------------------
def preprocess_pcg_full(
    x: np.ndarray,
    fs_in: int | float,
    profile: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    """Resample and band-pass a complete PCG recording.

    DWT and z-score are intentionally *not* applied here: they are applied per
    window so that the same window transformation is used for E1 and E2, and so
    normalisation statistics can be estimated from training windows.
    """
    target_fs = int(profile["pcg_target_fs"])
    y = resample_signal(x, fs_in, target_fs)
    band = profile.get("pcg_bandpass_hz")
    if band:
        y = bandpass_filter(
            y,
            target_fs,
            low_hz=band[0],
            high_hz=band[1],
            order=int(profile.get("pcg_filter_order", 5)),
            zero_phase=bool(profile.get("pcg_filter_zero_phase", True)),
        )
    return y.astype(np.float32), target_fs


def preprocess_ecg_full(
    x: np.ndarray,
    fs_in: int | float,
    ecg_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    """Prepare a complete ECG recording.

    The native sampling rate is kept.  The signal is stored without the
    detection band-pass; ``ecg_processing`` filters a copy for R-peak
    detection.  An optional notch filter can be applied here.
    """
    y = np.asarray(x, dtype=np.float32).ravel()
    notch = ecg_cfg.get("notch_hz")
    if notch:
        y = notch_filter(y, float(fs_in), float(notch))
    return y.astype(np.float32), int(round(float(fs_in)))


def preprocess_ecg_window(
    x: np.ndarray,
    fs_in: int | float,
    ecg_cfg: Mapping[str, Any],
    target_fs: int | None = None,
    expected_length: int | None = None,
) -> np.ndarray:
    """Prepare an ECG window for the optional E3 fusion branch."""
    y = np.asarray(x, dtype=np.float32).ravel()
    target_fs = int(target_fs or ecg_cfg.get("target_fs", fs_in))
    if int(round(float(fs_in))) != target_fs:
        y = resample_signal(y, fs_in, target_fs)
    band = ecg_cfg.get("fusion_bandpass_hz") or ecg_cfg.get("bandpass_hz")
    if band:
        y = bandpass_filter(y, target_fs, band[0], band[1], order=3, zero_phase=True)
    if expected_length is not None:
        y = fix_length(y, int(expected_length))
    return y.astype(np.float32)


def normalise_window(
    x: np.ndarray,
    profile: Mapping[str, Any],
    stats: NormalizationStats | None,
) -> np.ndarray:
    """Apply the configured normalisation to a single window."""
    mode = str(profile.get("normalization", "train_global"))
    if mode == "per_window":
        return zscore(x)
    if mode == "train_global":
        if stats is None:
            raise ValueError(
                "This profile requires training-set normalisation statistics, "
                "but none were supplied.  Load the checkpoint's stats or pass "
                "--normalization per_window explicitly."
            )
        return zscore(x, mean=stats.mean, std=stats.std)
    if mode == "none":
        return np.asarray(x, dtype=np.float32)
    raise ValueError(f"Unknown normalization mode: {mode}")
