"""Shared pytest fixtures.

The tests are designed to run on a CPU-only machine and never touch the real
dataset.  Synthetic fixtures are written to pytest's temporary directories.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

from src.config import load_config  # noqa: E402
from src.synthetic import generate_synthetic_pair, write_synthetic_dataset  # noqa: E402


@pytest.fixture()
def cfg() -> dict:
    """A small, CPU-only configuration for tests."""
    config = copy.deepcopy(load_config(PROJECT_ROOT / "config" / "default.yaml"))
    config["project"]["device"] = "cpu"
    config["project"]["deterministic"] = True
    config["project"]["seed"] = 7
    config["training"].update(
        {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "amp": False,
            "stats_max_windows": 20,
            "early_stopping_patience": 1,
            "scheduler_patience": 1,
        }
    )
    config["evaluation"]["bootstrap_iterations"] = 50
    config["gating"]["delay"]["estimate"] = False
    return config


@pytest.fixture()
def synthetic_pair() -> dict:
    return generate_synthetic_pair(duration=12.0, heart_rate_bpm=72.0, seed=123)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> Path:
    """A 10-subject synthetic dataset with the official folder layout."""
    root = tmp_path_factory.mktemp("synthetic_dataset")
    write_synthetic_dataset(root, n_subjects=10, duration=8.0, seed=99)
    return root


@pytest.fixture()
def tiny_cache(tmp_path):
    """A hand-built signal cache with six subjects and one record each.

    Returns ``(cache, windows, cfg)``.  The PCG is a synthetic 8 s signal at
    1440 Hz and the ECG at 500 Hz; two 1.5 s windows are indexed per subject.
    """
    from src.data_loader import load_signal_cache

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    n_subjects = 6
    pcg_fs, ecg_fs = 1440, 500
    pcg_len = int(8 * pcg_fs)
    ecg_len = int(8 * ecg_fs)
    rng = np.random.default_rng(5)
    ecg_mem = np.zeros((n_subjects, ecg_len), dtype=np.float32)
    pcg_mem = np.zeros((n_subjects, pcg_len), dtype=np.float32)
    rows = []
    windows = []
    for i in range(n_subjects):
        subject = f"S{i:02d}"
        fold = i % 3
        label = 1 if i % 2 == 0 else 0
        pair = generate_synthetic_pair(duration=8.0, heart_rate_bpm=70 + i, seed=200 + i)
        ecg_mem[i] = pair["ecg"]
        # Resample the PCG from 4000 to 1440 by simple decimation for this
        # fixture; window shapes matter more than the exact waveform here.
        pcg = pair["pcg"]
        idx = np.linspace(0, pcg.size - 1, pcg_len)
        pcg_mem[i] = np.interp(idx, np.arange(pcg.size), pcg).astype(np.float32)
        rows.append(
            {
                "record_id": f"{subject}_APEX",
                "subject": subject,
                "site": "APEX",
                "fold": fold,
                "label": label,
                "lvef_percent": 30.0 if label else 60.0,
                "ecg_path": "",
                "pcg_path": "",
                "ecg_fs": ecg_fs,
                "pcg_fs": pcg_fs,
                "target_pcg_fs": pcg_fs,
                "ecg_offset": i * ecg_len,
                "ecg_length": ecg_len,
                "pcg_offset": i * pcg_len,
                "pcg_length": pcg_len,
                "duration_ecg": 8.0,
                "duration_pcg": 8.0,
                "status": "ok",
                "reason": "",
            }
        )
        for j, start in enumerate((1.0, 3.5)):
            windows.append(
                {
                    "window_id": len(windows),
                    "record_id": f"{subject}_APEX",
                    "subject": subject,
                    "site": "APEX",
                    "fold": fold,
                    "label": label,
                    "lvef_percent": 30.0 if label else 60.0,
                    "experiment": "e1",
                    "profile": "paper_haar",
                    "start_sec": start,
                    "end_sec": start + 1.5,
                    "start_sample": int(start * pcg_fs),
                    "length_samples": int(1.5 * pcg_fs),
                    "ecg_start_sample": int(start * ecg_fs),
                    "ecg_length_samples": int(1.5 * ecg_fs),
                    "gating_window_id": -1,
                    "r_peak_time": np.nan,
                    "s1_time": np.nan,
                    "s2_time": np.nan,
                    "pcg_rms": np.nan,
                    "quality_ok": True,
                }
            )
    # The real cache is a flat 1-D memmap; offsets in metadata refer to that
    # flat array.  Flattening here is required for the fixture to behave like
    # the production cache.
    np.save(cache_dir / "ecg.npy", ecg_mem.reshape(-1))
    np.save(cache_dir / "pcg.npy", pcg_mem.reshape(-1))
    import pandas as pd

    pd.DataFrame(rows).to_csv(cache_dir / "metadata.csv", index=False)
    (cache_dir / "cache_config.json").write_text(
        '{"profile": {"name": "paper_haar"}, "target_pcg_fs": 1440}',
        encoding="utf-8",
    )
    cache = load_signal_cache(cache_dir)
    return cache, pd.DataFrame(windows)
