"""Preprocessing: resampling, filtering, DWT, lengths and normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import get_profile
from src.preprocessing import (
    NormalizationStats,
    bandpass_filter,
    compute_normalization_stats,
    dwt_output_length,
    expected_window_length,
    fix_length,
    haar_dwt,
    normalise_window,
    preprocess_pcg_full,
    resample_signal,
    transform_window,
    zscore,
)


def test_resample_length_and_rate():
    fs_in, fs_out, seconds = 4000, 1440, 2.0
    x = np.sin(2 * np.pi * 5 * np.arange(int(fs_in * seconds)) / fs_in)
    y = resample_signal(x, fs_in, fs_out)
    assert abs(y.size - int(fs_out * seconds)) <= 1
    assert np.isfinite(y).all()


def test_bandpass_filter_rejects_low_and_high():
    fs = 1000
    t = np.arange(fs) / fs
    low = np.sin(2 * np.pi * 2 * t)
    mid = np.sin(2 * np.pi * 50 * t)
    high = np.sin(2 * np.pi * 300 * t)
    filtered_low = bandpass_filter(low, fs, 20, 200)
    filtered_mid = bandpass_filter(mid, fs, 20, 200)
    filtered_high = bandpass_filter(high, fs, 20, 200)
    assert np.std(filtered_low) < 0.1
    assert np.std(filtered_mid) > 0.5
    assert np.std(filtered_high) < 0.1


def test_dwt_length_and_shape():
    x = np.random.default_rng(0).normal(size=2160)
    approx = haar_dwt(x, level=2, keep="approximation")
    assert approx.size == 540
    all_coeffs = haar_dwt(x, level=2, keep="all")
    assert all_coeffs.size == dwt_output_length(2160, level=2, keep="all")
    assert np.isfinite(approx).all()
    assert np.isfinite(all_coeffs).all()


def test_expected_window_lengths(cfg):
    paper = get_profile(cfg, "paper_haar")
    hfref = get_profile(cfg, "hfref_2000")
    assert expected_window_length(paper) == 540
    assert expected_window_length(hfref) == 2560


def test_transform_window_matches_expected_length(cfg):
    profile = get_profile(cfg, "paper_haar")
    x = np.random.default_rng(1).normal(size=2160).astype(np.float32)
    transformed = transform_window(x, profile, expected_length=540)
    assert transformed.shape == (540,)
    # 1.5 s at 1440 Hz -> 2160 samples; 2-level Haar -> 540.
    assert profile["window_seconds"] * profile["pcg_target_fs"] == 2160


def test_normalization_stats_and_application():
    stats = compute_normalization_stats([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])])
    assert stats.count == 6
    assert stats.mean == pytest.approx(3.5)
    assert stats.std == pytest.approx(np.std([1, 2, 3, 4, 5, 6]))
    out = zscore(np.array([1.0, 2.0, 3.0]), mean=2.0, std=1.0)
    assert out == pytest.approx([-1.0, 0.0, 1.0])
    profile = {"normalization": "train_global"}
    normalised = normalise_window(np.array([1.0, 2.0, 3.0]), profile, stats)
    assert np.isfinite(normalised).all()


def test_train_global_requires_stats():
    profile = {"normalization": "train_global"}
    with pytest.raises(ValueError, match="training-set normalisation"):
        normalise_window(np.ones(10), profile, None)


def test_fix_length_crop_and_pad():
    assert fix_length(np.arange(10), 6).size == 6
    padded = fix_length(np.arange(4), 8)
    assert padded.size == 8
    assert padded[2:6] == pytest.approx([0, 1, 2, 3])


def test_preprocess_pcg_full_resamples_and_filters(cfg):
    profile = get_profile(cfg, "paper_haar")
    fs = 4000
    t = np.arange(2 * fs) / fs
    x = np.sin(2 * np.pi * 50 * t)
    y, fs_out = preprocess_pcg_full(x, fs, profile)
    assert fs_out == 1440
    assert abs(y.size - 2 * 1440) <= 2
    assert np.std(y) > 0


def test_stats_serialisation_roundtrip():
    stats = NormalizationStats(mean=0.1, std=2.0, count=10, source="train")
    restored = NormalizationStats.from_dict(stats.to_dict())
    assert restored == stats
