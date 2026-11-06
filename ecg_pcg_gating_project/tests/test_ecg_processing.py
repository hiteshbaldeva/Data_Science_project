"""ECG quality control, R-peak detection and T-wave localisation."""

from __future__ import annotations

import numpy as np
import pytest

from src.ecg_processing import (
    assess_ecg_quality,
    detect_r_peaks,
    detect_r_peaks_paper,
    detect_r_peaks_simple,
    detect_t_waves,
    heart_rate_stats,
)
from src.synthetic import generate_synthetic_ecg


def _match(detected: np.ndarray, truth: np.ndarray, tolerance: float = 0.06) -> tuple[int, int, int]:
    """Greedy one-to-one matching; returns (matched, false positives, missed)."""
    detected = np.sort(np.asarray(detected, dtype=float))
    truth = np.sort(np.asarray(truth, dtype=float))
    used = np.zeros(detected.size, dtype=bool)
    matched = 0
    for value in truth:
        distances = np.abs(detected - value)
        distances[used] = np.inf
        idx = int(np.argmin(distances))
        if distances[idx] <= tolerance:
            used[idx] = True
            matched += 1
    return matched, int(detected.size - matched), int(truth.size - matched)


@pytest.mark.parametrize("method", ["paper_inspired", "simple_rpeak"])
def test_r_peak_detection_on_synthetic_ecg(method):
    ecg, truth = generate_synthetic_ecg(duration=12.0, heart_rate_bpm=70.0, noise=0.01, seed=11)
    result = detect_r_peaks(ecg, 500, method=method)
    assert result.quality.ok, result.quality.reasons
    matched, false_positives, missed = _match(result.times, truth)
    assert matched >= int(0.9 * truth.size)
    assert false_positives <= 2
    assert missed <= 2
    assert result.method == method


def test_inverted_ecg_polarity_is_detected():
    ecg, truth = generate_synthetic_ecg(duration=10.0, heart_rate_bpm=65.0, invert=True, seed=4)
    result = detect_r_peaks_paper(ecg, 500, ecg_cfg={"bandpass_hz": [5, 25]})
    assert result.polarity == "negative"
    matched, _, _ = _match(result.times, truth)
    assert matched >= int(0.85 * truth.size)


def test_t_wave_localisation_uses_rr_equations():
    ecg, _ = generate_synthetic_ecg(duration=10.0, heart_rate_bpm=60.0, seed=8)
    result = detect_r_peaks_paper(ecg, 500)
    waves = detect_t_waves(ecg, result.peaks, 500)
    assert len(waves) == len(result.peaks)
    for wave in waves[:5]:
        rr = wave["rr_seconds"]
        assert wave["t_start_time"] > wave["r_peak_time"] + 0.05
        assert wave["t_end_time"] < wave["r_peak_time"] + 0.25 + 0.35 * rr
        assert wave["t_start_sample"] < wave["t_end_sample"]


def test_quality_rejects_flatline():
    quality = assess_ecg_quality(np.zeros(5000), 500)
    assert not quality.ok
    assert any("flatline" in reason for reason in quality.reasons)


def test_quality_accepts_normal_synthetic_ecg():
    ecg, _ = generate_synthetic_ecg(duration=8.0, seed=2)
    assert assess_ecg_quality(ecg, 500).ok


def test_heart_rate_stats():
    peaks = np.array([0, 500, 1000, 1500, 2000])
    stats = heart_rate_stats(peaks, 500)
    assert stats["median_rr"] == pytest.approx(1.0)
    assert stats["heart_rate_bpm"] == pytest.approx(60.0)
    assert stats["n_intervals"] == 4


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="Unknown gating method"):
        detect_r_peaks(np.zeros(5000), 500, method="mystery")
