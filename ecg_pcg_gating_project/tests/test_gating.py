"""ECG-guided gating, timestamp conversion and failure handling."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import get_profile
from src.gating import estimate_pcg_delay, gate_recording
from src.synthetic import generate_synthetic_pair


def test_gating_uses_ecg_rate_and_pcg_rate_correctly(cfg):
    pair = generate_synthetic_pair(duration=12.0, heart_rate_bpm=70.0, seed=21)
    profile = get_profile(cfg, "paper_haar")
    result = gate_recording(
        pair["ecg"],
        pair["pcg"],
        pair["fs_ecg"],
        pair["fs_pcg"],
        cfg,
        subject="S1",
        site="APEX",
        fold=0,
        window_seconds=float(profile["window_seconds"]),
    )
    assert result.recording_ok, result.reasons
    assert result.n_windows >= 5
    assert result.ecg_fs == 500
    assert result.pcg_fs == 4000

    for window in result.windows:
        # Window is anchored 0.25 s before the R peak.
        assert window.start_time == pytest.approx(window.r_peak_time - 0.25, abs=1e-6)
        assert window.end_time - window.start_time == pytest.approx(
            float(profile["window_seconds"]), abs=1e-6
        )
        # Time -> sample conversion must use the PCG sampling rate (4000 Hz).
        assert window.start_sample == int(round(window.start_time * 4000))
        assert window.end_sample == int(round(window.end_time * 4000))
        assert window.end_sample - window.start_sample == int(
            round(float(profile["window_seconds"]) * 4000)
        )


def test_gating_windows_contain_s1_and_s2(cfg):
    pair = generate_synthetic_pair(duration=12.0, heart_rate_bpm=65.0, seed=22)
    result = gate_recording(
        pair["ecg"],
        pair["pcg"],
        pair["fs_ecg"],
        pair["fs_pcg"],
        cfg,
        subject="S1",
        site="LLSB",
        window_seconds=1.5,
    )
    assert result.n_windows > 0
    contained_s1 = 0
    contained_s2 = 0
    for window in result.windows:
        if window.s1_time is not None and window.start_time <= window.s1_time <= window.end_time:
            contained_s1 += 1
        if window.s2_time is not None and window.start_time <= window.s2_time <= window.end_time:
            contained_s2 += 1
    assert contained_s1 >= 0.8 * len(result.windows)
    assert contained_s2 >= 0.8 * len(result.windows)


def test_duration_mismatch_is_recorded_not_silently_fixed(cfg):
    pair = generate_synthetic_pair(duration=12.0, seed=23)
    pcg_short = pair["pcg"][: 5 * 4000]
    result = gate_recording(
        pair["ecg"],
        pcg_short,
        pair["fs_ecg"],
        pair["fs_pcg"],
        cfg,
        subject="S1",
        site="RUSB",
        window_seconds=1.5,
    )
    assert not result.recording_ok
    assert any("duration mismatch" in reason for reason in result.reasons)
    # Windows beyond the truncated PCG must be rejected, not padded.
    for window in result.rejected:
        if "reasons" in window:
            assert window["reasons"]


def test_short_pcg_produces_no_fabricated_windows(cfg):
    pair = generate_synthetic_pair(duration=12.0, seed=24)
    pcg_short = pair["pcg"][: int(0.5 * 4000)]
    result = gate_recording(
        pair["ecg"],
        pcg_short,
        pair["fs_ecg"],
        pair["fs_pcg"],
        cfg,
        subject="S1",
        site="APEX",
        window_seconds=1.5,
    )
    assert result.n_windows == 0
    assert not result.recording_ok
    assert result.rejected


def test_estimate_pcg_delay_is_near_zero(cfg):
    pair = generate_synthetic_pair(duration=12.0, heart_rate_bpm=70.0, seed=25)
    cfg = dict(cfg)
    cfg["gating"] = {**cfg["gating"], "delay": {"estimate": True, "apply": False}}
    result = gate_recording(
        pair["ecg"],
        pair["pcg"],
        pair["fs_ecg"],
        pair["fs_pcg"],
        cfg,
        subject="S1",
        site="APEX",
        window_seconds=1.5,
    )
    assert result.delay is not None
    assert result.delay["ok"]
    assert abs(result.delay["best_lag_seconds"]) <= 0.06
    assert result.delay["contrast"] > 1.0
    assert result.delay["applied"] is False


def test_delay_is_not_applied_by_default(cfg):
    assert cfg["gating"]["delay"]["apply"] is False


def test_gating_records_rejection_reasons(cfg):
    # The first window starts before t=0 for an R peak at 0.6 s only when the
    # pre-anchor is larger than the R time; force that via the window config.
    pair = generate_synthetic_pair(duration=8.0, seed=26)
    cfg = dict(cfg)
    cfg["gating"] = {
        **cfg["gating"],
        "window": {**cfg["gating"]["window"], "pre_r_seconds": 0.9},
    }
    result = gate_recording(
        pair["ecg"],
        pair["pcg"],
        pair["fs_ecg"],
        pair["fs_pcg"],
        cfg,
        subject="S1",
        site="APEX",
        window_seconds=1.5,
    )
    reasons = [reason for item in result.rejected for reason in item.get("reasons", [])]
    assert "window_start_before_recording" in reasons
