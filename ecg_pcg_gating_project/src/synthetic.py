"""Synthetic ECG/PCG generators used by the unit tests and smoke tests.

The synthetic signals are *not* a substitute for the real dataset.  They are
deliberately simple, fully known signals that let us verify:
  * that R-peak detection finds the peaks we inserted,
  * that time-to-sample conversion uses the correct sampling rate,
  * that gated windows have the expected duration,
  * that the model forward pass and a tiny training loop work.

The generators are deterministic for a given seed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SITES = ("APEX", "LLSB", "LUSB", "RUSB")


def _beat_times(duration: float, heart_rate_bpm: float, start: float = 0.6) -> np.ndarray:
    period = 60.0 / float(heart_rate_bpm)
    n = int(np.floor((duration - start) / period)) + 1
    return start + np.arange(max(n, 1)) * period


def _gaussian(t: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def generate_synthetic_ecg(
    duration: float = 12.0,
    fs: int = 500,
    heart_rate_bpm: float = 70.0,
    noise: float = 0.015,
    invert: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ecg, r_times_seconds)`` for a clean synthetic ECG.

    The morphology is a simplified P-QRS-T complex.  It is intentionally
    narrow in the QRS so that derivative-based detectors have a clear target.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    t = np.arange(n, dtype=np.float64) / fs
    ecg = np.zeros(n, dtype=np.float64)

    r_times = _beat_times(duration, heart_rate_bpm)
    for r in r_times:
        ecg += _gaussian(t, r - 0.20, 0.035, 0.10)   # P wave
        ecg += _gaussian(t, r - 0.030, 0.009, -0.15)  # Q wave
        ecg += _gaussian(t, r, 0.011, 1.00)           # R wave
        ecg += _gaussian(t, r + 0.030, 0.010, -0.25)  # S wave
        ecg += _gaussian(t, r + 0.28, 0.075, 0.35)    # T wave

    # Slow baseline wander plus measurement noise.
    ecg += 0.05 * np.sin(2 * np.pi * 0.33 * t)
    ecg += rng.normal(0.0, noise, size=n)
    if invert:
        ecg = -ecg
    return ecg.astype(np.float32), r_times


def _burst(
    t: np.ndarray,
    center: float,
    duration: float,
    frequency: float,
    amplitude: float,
    decay: float = 28.0,
) -> np.ndarray:
    """A short damped sinusoid used as an S1/S2 heart-sound burst."""
    out = np.zeros_like(t)
    mask = (t >= center) & (t <= center + duration)
    local = t[mask] - center
    envelope = np.exp(-decay * local)
    out[mask] = amplitude * envelope * np.sin(2 * np.pi * frequency * local)
    return out


def generate_synthetic_pcg(
    duration: float = 12.0,
    fs: int = 4000,
    heart_rate_bpm: float = 70.0,
    noise: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(pcg, r_times, s1_times, s2_times)``.

    S1 is placed shortly after the R wave and S2 roughly one-third of a second
    later.  The exact delays are returned so tests can check gating.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    t = np.arange(n, dtype=np.float64) / fs
    pcg = np.zeros(n, dtype=np.float64)

    r_times = _beat_times(duration, heart_rate_bpm)
    period = 60.0 / heart_rate_bpm
    s2_delay = min(0.34, 0.55 * period)
    s1_times = r_times + 0.05
    s2_times = r_times + s2_delay

    for s1 in s1_times:
        pcg += _burst(t, s1, duration=0.12, frequency=50.0, amplitude=1.0)
    for s2 in s2_times:
        pcg += _burst(t, s2, duration=0.10, frequency=75.0, amplitude=0.8)

    # Low-frequency chest-wall motion plus sensor noise.
    pcg += 0.015 * np.sin(2 * np.pi * 0.5 * t)
    pcg += rng.normal(0.0, noise, size=n)
    return pcg.astype(np.float32), r_times, s1_times, s2_times


def generate_synthetic_pair(
    duration: float = 12.0,
    fs_ecg: int = 500,
    fs_pcg: int = 4000,
    heart_rate_bpm: float = 70.0,
    seed: int = 42,
) -> dict:
    """Return a dictionary with synthetic ECG, PCG and ground-truth times."""
    ecg, r_times = generate_synthetic_ecg(
        duration=duration, fs=fs_ecg, heart_rate_bpm=heart_rate_bpm, seed=seed
    )
    pcg, _, s1_times, s2_times = generate_synthetic_pcg(
        duration=duration, fs=fs_pcg, heart_rate_bpm=heart_rate_bpm, seed=seed + 1
    )
    return {
        "ecg": ecg,
        "pcg": pcg,
        "fs_ecg": fs_ecg,
        "fs_pcg": fs_pcg,
        "r_times": r_times,
        "s1_times": s1_times,
        "s2_times": s2_times,
        "duration": duration,
    }


def _write_wav(path: Path, signal: np.ndarray, fs: int) -> None:
    """Write a mono int16 WAV file, clipping safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(signal))) if signal.size else 1.0
    scale = 0.9 * 32767.0 / peak if peak > 0 else 1.0
    wavfile.write(path, fs, np.clip(signal * scale, -32768, 32767).astype(np.int16))


def write_synthetic_dataset(
    root: str | Path,
    n_subjects: int = 10,
    duration: float = 12.0,
    external: bool = False,
    seed: int = 42,
) -> Path:
    """Create a miniature dataset with the same layout as the HFrEF dataset.

    Parameters
    ----------
    root:
        Destination folder.  For ``external=False`` the layout is
        ``root/fold_0/...`` plus ``root/fold_0_label.csv``.  For
        ``external=True`` it is ``root/test_set/...`` plus
        ``root/test_set_label.csv``.
    n_subjects:
        Number of synthetic subjects.  Use at least 10 for a five-fold test.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)

    for i in range(n_subjects):
        subject = f"S{i + 1:03d}"
        fold = i % 5
        if external:
            folder = root / "test_set"
        else:
            folder = root / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)

        # Spread the two classes across consecutive groups of five subjects so
        # that every fold contains both classes even in a tiny smoke dataset.
        # The real dataset is much more imbalanced; this generator is only a
        # plumbing and signal-processing test fixture.
        lvef_percent = 30.0 if (i // 5) % 2 == 0 else 60.0
        lvef_value = lvef_percent / 100.0  # official CSV uses fractions

        for site in SITES:
            hr = float(rng.uniform(58, 92))
            pair = generate_synthetic_pair(
                duration=duration,
                heart_rate_bpm=hr,
                seed=seed + 1000 * i + SITES.index(site),
            )
            _write_wav(folder / f"{subject}_{site}_ECG.wav", pair["ecg"], pair["fs_ecg"])
            _write_wav(folder / f"{subject}_{site}_PCG.wav", pair["pcg"], pair["fs_pcg"])

        # The official CSVs contain recording-level rows; our parser also
        # accepts subject-level rows.  Writing one row per ECG recording
        # exercises the de-duplication path.
        label_path = (root / "test_set_label.csv") if external else (root / f"fold_{fold}_label.csv")
        write_header = not label_path.exists()
        with label_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(["recording", "LVEF"])
            for site in SITES:
                writer.writerow([f"{subject}_{site}_ECG", f"{lvef_value:.2f}"])

    return root


if __name__ == "__main__":  # pragma: no cover - manual convenience
    import argparse

    parser = argparse.ArgumentParser(description="Generate a synthetic HFrEF-like dataset")
    parser.add_argument("--output", default="data/synthetic_smoke")
    parser.add_argument("--subjects", type=int, default=10)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    path = write_synthetic_dataset(
        args.output, n_subjects=args.subjects, duration=args.duration, external=args.external
    )
    print(f"Synthetic dataset written to {path}")
