"""ECG-guided gating of PCG recordings.

For each detected R peak we create a PCG window of a configurable duration.
The window is anchored relative to the R peak (or optionally S1).  We also
detect T waves and search for S1/S2 in the PCG envelope, but those locations
are used for quality control rather than silently changing the window.

Crucially:

* ECG and PCG must come from the same subject/site recording.
* R peaks are detected at the ECG's native sampling rate (500 Hz).
* Window boundaries are converted to PCG samples using the PCG sampling rate.
* Out-of-bounds windows are rejected and recorded; there is no silent fallback
  to ungated PCG.
* The four auscultation sites are processed independently, never combined as
  if they were simultaneous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .ecg_processing import (
    EcgQuality,
    assess_ecg_quality,
    detect_r_peaks,
    detect_t_waves,
    heart_rate_stats,
)
from .pcg_processing import find_s1_s2, pcg_envelope, window_quality
from .utils import save_json

LOGGER = logging.getLogger(__name__)


@dataclass
class GatingWindow:
    window_id: int
    r_peak_time: float
    start_time: float
    end_time: float
    start_sample: int
    end_sample: int
    s1_time: float | None = None
    s2_time: float | None = None
    pcg_rms: float | None = None
    valid: bool = True
    reasons: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return float(self.end_time - self.start_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": int(self.window_id),
            "r_peak_time": float(self.r_peak_time),
            "start_time": float(self.start_time),
            "end_time": float(self.end_time),
            "start_sample": int(self.start_sample),
            "end_sample": int(self.end_sample),
            "s1_time": self.s1_time,
            "s2_time": self.s2_time,
            "pcg_rms": self.pcg_rms,
            "valid": bool(self.valid),
            "reasons": list(self.reasons),
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class GatingResult:
    record_id: str
    subject: str
    site: str
    fold: int | None
    ecg_fs: int
    pcg_fs: int
    duration_seconds: float
    method: str
    params: dict[str, Any]
    quality: EcgQuality
    r_peaks: list[int]
    r_peak_times: list[float]
    t_waves: list[dict[str, Any]]
    s1_s2: list[dict[str, Any]]
    windows: list[GatingWindow]
    rejected: list[dict[str, Any]]
    recording_ok: bool
    reasons: list[str]
    delay: dict[str, Any] | None = None
    rr_stats: dict[str, Any] | None = None

    @property
    def n_rpeaks(self) -> int:
        return len(self.r_peaks)

    @property
    def n_windows(self) -> int:
        return sum(1 for w in self.windows if w.valid)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    @property
    def success_rate(self) -> float:
        return float(self.n_windows) / float(self.n_rpeaks) if self.n_rpeaks else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "subject": self.subject,
            "site": self.site,
            "fold": self.fold,
            "ecg_fs": int(self.ecg_fs),
            "pcg_fs": int(self.pcg_fs),
            "duration_seconds": float(self.duration_seconds),
            "method": self.method,
            "params": self.params,
            "quality": self.quality.to_dict(),
            "n_rpeaks": self.n_rpeaks,
            "n_windows": self.n_windows,
            "n_rejected": self.n_rejected,
            "success_rate": self.success_rate,
            "recording_ok": bool(self.recording_ok),
            "reasons": list(self.reasons),
            "r_peaks": [int(x) for x in self.r_peaks],
            "r_peak_times": [float(x) for x in self.r_peak_times],
            "t_waves": self.t_waves,
            "s1_s2": self.s1_s2,
            "windows": [w.to_dict() for w in self.windows],
            "rejected": self.rejected,
            "delay": self.delay,
            "rr_stats": self.rr_stats,
        }

    def summary_row(self) -> dict[str, Any]:
        reason_counts: dict[str, int] = {}
        for item in self.rejected:
            for reason in item.get("reasons", []):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return {
            "record_id": self.record_id,
            "subject": self.subject,
            "site": self.site,
            "fold": self.fold,
            "method": self.method,
            "n_rpeaks": self.n_rpeaks,
            "n_windows": self.n_windows,
            "n_rejected": self.n_rejected,
            "success_rate": self.success_rate,
            "recording_ok": self.recording_ok,
            "reasons": "; ".join(self.reasons),
            "rejection_reasons": "; ".join(
                f"{k}:{v}" for k, v in sorted(reason_counts.items())
            ),
            **{k: v for k, v in (self.rr_stats or {}).items() if k in {"median_rr", "heart_rate_bpm"}},
            "best_lag_seconds": (self.delay or {}).get("best_lag_seconds"),
            "lag_contrast": (self.delay or {}).get("contrast"),
        }


# ---------------------------------------------------------------------------
# Delay diagnostic
# ---------------------------------------------------------------------------
def estimate_pcg_delay(
    pcg: np.ndarray,
    fs_pcg: int | float,
    r_times: np.ndarray,
    max_abs_seconds: float = 0.2,
    step_seconds: float = 0.005,
) -> dict[str, Any]:
    """Estimate the lag that maximises PCG energy after R peaks.

    This is a *diagnostic*.  The dataset declares within-recording ECG/PCG
    synchrony, so the default is not to apply any shift.  A consistent,
    large lag may indicate a header/alignment problem.
    """
    if len(r_times) < 3:
        return {
            "ok": False,
            "reason": "fewer than 3 R peaks",
            "best_lag_seconds": None,
            "contrast": None,
        }
    env = pcg_envelope(pcg, fs_pcg)
    lags = np.arange(-max_abs_seconds, max_abs_seconds + 1e-12, step_seconds)
    scores = []
    for lag in lags:
        on_values = []
        off_values = []
        for r in r_times:
            on0 = int(round((r + lag + 0.02) * fs_pcg))
            on1 = int(round((r + lag + 0.18) * fs_pcg))
            off0 = int(round((r + lag - 0.25) * fs_pcg))
            off1 = int(round((r + lag - 0.08) * fs_pcg))
            if on0 < 0 or on1 >= env.size or off0 < 0 or off1 >= env.size:
                continue
            on_values.append(float(np.mean(env[on0:on1])))
            off_values.append(float(np.mean(env[off0:off1])))
        if on_values and off_values:
            scores.append(float(np.mean(on_values) / (np.mean(off_values) + 1e-12)))
        else:
            scores.append(np.nan)
    scores_arr = np.asarray(scores, dtype=float)
    if np.all(np.isnan(scores_arr)):
        return {
            "ok": False,
            "reason": "not enough complete beats for lag scan",
            "best_lag_seconds": None,
            "contrast": None,
        }
    best_idx = int(np.nanargmax(scores_arr))
    return {
        "ok": True,
        "best_lag_seconds": float(lags[best_idx]),
        "contrast": float(scores_arr[best_idx]),
        "zero_lag_contrast": float(scores_arr[np.argmin(np.abs(lags))]),
        "max_abs_seconds": float(max_abs_seconds),
        "step_seconds": float(step_seconds),
        "lag_curve": {"lags": lags.tolist(), "contrast": scores_arr.tolist()},
        "applied": False,
        "note": "Diagnostic only unless gating.delay.apply=true.",
    }


# ---------------------------------------------------------------------------
# Main gating function
# ---------------------------------------------------------------------------
def gate_recording(
    ecg: np.ndarray,
    pcg: np.ndarray,
    fs_ecg: int | float,
    fs_pcg: int | float,
    cfg: Mapping[str, Any],
    subject: str = "",
    site: str = "",
    fold: int | None = None,
    record_id: str | None = None,
    window_seconds: float | None = None,
) -> GatingResult:
    """Run ECG-guided gating on one synchronized ECG/PCG recording.

    Parameters
    ----------
    ecg, pcg:
        Raw samples from the *same* subject/site recording.
    fs_ecg, fs_pcg:
        Native sampling rates.  They may differ (500 Hz and 4000 Hz here).
    cfg:
        Full project configuration.  The ``gating`` and
        ``preprocessing.ecg`` sections are used.
    window_seconds:
        Window duration.  The pipeline passes the active pre-processing
        profile's window length so E1 and E2 use exactly the same duration.
    """
    gating_cfg = dict(cfg.get("gating", {}))
    ecg_cfg = dict(cfg.get("preprocessing", {}).get("ecg", {}))
    data_cfg = dict(cfg.get("data", {}))
    rec_id = record_id or f"{subject}_{site}"
    method = str(gating_cfg.get("method", "paper_inspired"))
    polarity = str(gating_cfg.get("polarity", "auto"))
    window_cfg = dict(gating_cfg.get("window", {}))
    duration = float(window_seconds or window_cfg.get("duration_seconds", 1.5))
    pre_anchor = float(window_cfg.get("pre_r_seconds", 0.25))
    anchor = str(window_cfg.get("anchor", "r_peak")).lower()
    max_oor = float(window_cfg.get("max_out_of_bounds_seconds", 0.0))

    ecg = np.asarray(ecg, dtype=np.float64).ravel()
    pcg = np.asarray(pcg, dtype=np.float64).ravel()
    fs_ecg = int(round(float(fs_ecg)))
    fs_pcg = int(round(float(fs_pcg)))
    dur_ecg = ecg.size / fs_ecg if fs_ecg else 0.0
    dur_pcg = pcg.size / fs_pcg if fs_pcg else 0.0
    duration_seconds = min(dur_ecg, dur_pcg)
    reasons: list[str] = []

    tolerance = float(data_cfg.get("duration_tolerance_seconds", 0.25))
    if abs(dur_ecg - dur_pcg) > tolerance:
        reasons.append(
            f"ECG/PCG duration mismatch: {dur_ecg:.3f}s vs {dur_pcg:.3f}s "
            f"(tolerance {tolerance}s)"
        )

    quality = assess_ecg_quality(ecg, fs_ecg)
    if not quality.ok:
        reasons.append("ECG quality check failed: " + "; ".join(quality.reasons))

    # Detect R peaks at the ECG's native sampling rate.  Do not upsample the
    # ECG first: upsampling does not create physiological information.
    peak_result = detect_r_peaks(
        ecg,
        fs_ecg,
        method=method,
        ecg_cfg={**ecg_cfg, **{k: gating_cfg.get(k) for k in ("min_rr_seconds", "max_heart_rate_bpm", "min_heart_rate_bpm") if gating_cfg.get(k) is not None}},
        polarity=polarity,
    )
    r_peaks = np.asarray(peak_result.peaks, dtype=int)
    r_times = np.asarray(peak_result.times, dtype=float)
    if peak_result.reasons:
        reasons.extend(peak_result.reasons)

    # T waves and S1/S2 are computed on the native signals.
    t_waves = detect_t_waves(ecg, r_peaks, fs_ecg, ecg_cfg=ecg_cfg) if r_peaks.size else []
    envelope = pcg_envelope(pcg, fs_pcg) if pcg.size else np.asarray([], dtype=np.float32)
    search = dict(gating_cfg.get("search", {}))
    s1_window = tuple(search.get("s1_after_r_seconds", [0.0, 0.25]))
    s2_window = tuple(search.get("s2_after_t_seconds", [-0.10, 0.20]))
    # R peaks were detected at the ECG sampling rate.  The PCG envelope and
    # the S1/S2 search run at the PCG sampling rate, so convert the timestamps
    # instead of reusing ECG sample indices (this is the central timebase
    # conversion of the gating step).
    r_peaks_pcg = np.round(r_times * fs_pcg).astype(int)
    t_waves_pcg = []
    for wave in t_waves:
        converted = dict(wave)
        converted["r_peak_sample"] = int(round(float(wave["r_peak_time"]) * fs_pcg))
        t_waves_pcg.append(converted)
    s1_s2 = (
        find_s1_s2(
            envelope,
            fs_pcg,
            r_peaks_pcg,
            t_waves=t_waves_pcg,
            s1_after_r_seconds=s1_window,
            s2_after_t_seconds=s2_window,
        )
        if r_peaks.size
        else []
    )
    s1_s2_by_r = {int(item["r_peak_sample"]): item for item in s1_s2}

    # Optional delay diagnostic / application.
    delay_cfg = dict(gating_cfg.get("delay", {}))
    delay_result = None
    applied_delay = 0.0
    if bool(delay_cfg.get("estimate", True)) and r_times.size >= 3:
        delay_result = estimate_pcg_delay(
            pcg,
            fs_pcg,
            r_times,
            max_abs_seconds=float(delay_cfg.get("max_abs_seconds", 0.2)),
            step_seconds=float(delay_cfg.get("step_seconds", 0.005)),
        )
        if bool(delay_cfg.get("apply", False)) and delay_result.get("ok"):
            best = float(delay_result["best_lag_seconds"])
            contrast = float(delay_result.get("contrast") or 0.0)
            if abs(best) <= float(delay_cfg.get("max_abs_seconds", 0.2)) and contrast >= 1.5:
                applied_delay = best
                delay_result["applied"] = True
            else:
                delay_result["applied"] = False
                delay_result["apply_skipped_reason"] = (
                    "lag outside limit or contrast below 1.5"
                )

    min_rpeaks = int(gating_cfg.get("quality", {}).get("min_rpeaks", 3))
    if r_peaks.size < min_rpeaks:
        reasons.append(f"only {r_peaks.size} R peaks (minimum {min_rpeaks})")

    windows: list[GatingWindow] = []
    rejected: list[dict[str, Any]] = []
    min_rms = float(gating_cfg.get("quality", {}).get("min_window_rms", 0.0))
    require_s1_s2 = bool(gating_cfg.get("require_s1_s2", False))

    for beat_index, (r_sample, r_time) in enumerate(zip(r_peaks, r_times)):
        marker_time = float(r_time + applied_delay)
        r_pcg_sample = int(round(float(r_time) * fs_pcg))
        s1s2 = s1_s2_by_r.get(r_pcg_sample, {})
        if anchor == "s1":
            if s1s2.get("s1_time") is None:
                rejected.append(
                    {
                        "window_id": None,
                        "r_peak_time": float(r_time),
                        "reasons": ["s1_not_detected"],
                    }
                )
                continue
            base_time = float(s1s2["s1_time"])
            pre = float(window_cfg.get("pre_s1_seconds", 0.10))
        else:
            base_time = marker_time
            pre = pre_anchor
        start_time = base_time - pre
        end_time = start_time + duration

        # Strict bounds by default.  If a tolerance is configured, clamp small
        # overhangs and record that fact rather than silently doing it.
        clamped = False
        if start_time < 0.0:
            if -start_time <= max_oor:
                start_time = 0.0
                clamped = True
            else:
                rejected.append(
                    {
                        "window_id": len(windows),
                        "r_peak_time": float(r_time),
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "reasons": ["window_start_before_recording"],
                    }
                )
                continue
        if end_time > duration_seconds:
            if end_time - duration_seconds <= max_oor:
                end_time = duration_seconds
                clamped = True
            else:
                rejected.append(
                    {
                        "window_id": len(windows),
                        "r_peak_time": float(r_time),
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "reasons": ["window_end_after_recording"],
                    }
                )
                continue

        start_sample = int(round(start_time * fs_pcg))
        end_sample = int(round(end_time * fs_pcg))
        if start_sample < 0 or end_sample > pcg.size or end_sample - start_sample < 2:
            rejected.append(
                {
                    "window_id": len(windows),
                    "r_peak_time": float(r_time),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "reasons": ["sample_conversion_out_of_bounds"],
                }
            )
            continue

        segment = pcg[start_sample:end_sample]
        wq = window_quality(segment, fs_pcg, min_rms=min_rms)
        seg_reasons: list[str] = []
        if not wq["ok"]:
            seg_reasons.extend(wq["reasons"])

        if require_s1_s2:
            if s1s2.get("s1_time") is None or s1s2.get("s2_time") is None:
                seg_reasons.append("s1_or_s2_not_detected")
            else:
                if not (start_time <= float(s1s2["s1_time"]) <= end_time):
                    seg_reasons.append("s1_outside_window")
                if not (start_time <= float(s1s2["s2_time"]) <= end_time):
                    seg_reasons.append("s2_outside_window")

        window = GatingWindow(
            window_id=len(windows),
            r_peak_time=float(r_time),
            start_time=float(start_time),
            end_time=float(end_time),
            start_sample=int(start_sample),
            end_sample=int(end_sample),
            s1_time=s1s2.get("s1_time"),
            s2_time=s1s2.get("s2_time"),
            pcg_rms=float(wq.get("rms", 0.0)),
            valid=not seg_reasons,
            reasons=seg_reasons,
        )
        if window.valid:
            windows.append(window)
        else:
            rejected.append(
                {
                    "window_id": len(windows),
                    "r_peak_time": float(r_time),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "start_sample": int(start_sample),
                    "end_sample": int(end_sample),
                    "reasons": seg_reasons,
                    "clamped": clamped,
                }
            )

    n_total = max(len(r_peaks), 1)
    rejection_fraction = len(rejected) / float(n_total)
    quality_cfg = dict(gating_cfg.get("quality", {}))
    min_windows = int(quality_cfg.get("min_successful_windows", 2))
    max_rej = float(quality_cfg.get("max_rejection_fraction", 0.5))
    recording_ok = True
    if not quality.ok:
        recording_ok = False
    if r_peaks.size < min_rpeaks:
        recording_ok = False
    if len(windows) < min_windows:
        recording_ok = False
        reasons.append(f"only {len(windows)} successful windows (minimum {min_windows})")
    if rejection_fraction > max_rej:
        recording_ok = False
        reasons.append(
            f"rejection fraction {rejection_fraction:.1%} exceeds {max_rej:.1%}"
        )
    if not windows:
        recording_ok = False
        reasons.append("no valid gated windows; no fallback to ungated PCG")

    params = {
        "method": method,
        "polarity": peak_result.polarity,
        "anchor": anchor,
        "pre_anchor_seconds": pre_anchor,
        "window_seconds": duration,
        "max_out_of_bounds_seconds": max_oor,
        "min_rr_seconds": gating_cfg.get("min_rr_seconds"),
        "s1_search_seconds": list(s1_window),
        "s2_search_seconds": list(s2_window),
        "require_s1_s2": require_s1_s2,
        "applied_delay_seconds": applied_delay,
    }

    return GatingResult(
        record_id=rec_id,
        subject=subject,
        site=site,
        fold=fold,
        ecg_fs=fs_ecg,
        pcg_fs=fs_pcg,
        duration_seconds=duration_seconds,
        method=method,
        params=params,
        quality=quality,
        r_peaks=[int(x) for x in r_peaks],
        r_peak_times=[float(x) for x in r_times],
        t_waves=t_waves,
        s1_s2=s1_s2,
        windows=windows,
        rejected=rejected,
        recording_ok=recording_ok,
        reasons=reasons,
        delay=delay_result,
        rr_stats=heart_rate_stats(r_peaks, fs_ecg),
    )


def save_gating_result(result: GatingResult, output_dir: str | Path) -> Path:
    """Save one gating result as JSON (one file per recording)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.record_id}.json"
    save_json(result.to_dict(), path)
    return path
