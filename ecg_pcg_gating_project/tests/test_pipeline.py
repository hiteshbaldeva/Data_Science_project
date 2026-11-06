"""Pipeline helpers and the ECG/PCG synchronisation diagnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import config_fingerprint, get_profile, load_config, save_config
from src.data_inspection import check_synchronization
from src.data_loader import read_wav
from src.pipeline import results_path, smoke_subset
from src.synthetic import generate_synthetic_pair, write_synthetic_dataset


def test_synchronisation_diagnostic_finds_near_zero_lag():
    pair = generate_synthetic_pair(duration=12.0, heart_rate_bpm=70.0, seed=31)
    result = check_synchronization(
        pair["ecg"], pair["pcg"], pair["fs_ecg"], pair["fs_pcg"],
        max_lag_seconds=0.15, step_seconds=0.005,
    )
    assert result["ok"]
    assert abs(result["best_lag_seconds"]) <= 0.05
    assert result["contrast"] > 1.0


def test_read_wav_roundtrip(synthetic_dataset: Path):
    path = next(synthetic_dataset.rglob("*_ECG.wav"))
    signal, fs = read_wav(path)
    assert fs == 500
    assert signal.dtype == np.float32
    assert signal.ndim == 1
    assert signal.size > 0
    assert np.max(np.abs(signal)) <= 1.0


def test_smoke_subset_is_balanced_and_bounded(synthetic_dataset: Path):
    from src.data_inspection import discover_dataset

    index = discover_dataset(synthetic_dataset)
    rows = []
    for pair in index.pairs:
        rows.append(
            {
                "subject": pair.subject,
                "record_id": f"{pair.subject}_{pair.site}",
                "site": pair.site,
                "fold": pair.fold,
                "label": pair.label,
                "prob": 0.5,
            }
        )
    frame = pd.DataFrame(rows)
    subset = smoke_subset(frame, max_subjects=6, seed=1)
    assert subset["subject"].nunique() <= 6
    assert subset["label"].nunique() == 2


def test_profile_selection_and_fingerprint(cfg):
    paper = get_profile(cfg, "paper_haar")
    hfref = get_profile(cfg, "hfref_2000")
    assert paper["dwt_enabled"] is True
    assert paper["window_seconds"] == 1.5
    assert hfref["dwt_enabled"] is False
    assert hfref["window_seconds"] == 1.28
    cfg_a = {**cfg, "preprocessing": {**cfg["preprocessing"], "profile": "paper_haar"}}
    cfg_b = {**cfg, "preprocessing": {**cfg["preprocessing"], "profile": "hfref_2000"}}
    assert config_fingerprint(cfg_a) != config_fingerprint(cfg_b)


def test_save_and_load_config_roundtrip(cfg, tmp_path: Path):
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded["preprocessing"]["profile"] == cfg["preprocessing"]["profile"]
    assert loaded["model"]["num_classes"] == 2


def test_results_path_is_profile_specific(cfg):
    cfg = dict(cfg)
    cfg["preprocessing"] = {**cfg["preprocessing"], "profile": "paper_haar"}
    a = results_path(cfg, "e1", run_name="run")
    cfg["preprocessing"] = {**cfg["preprocessing"], "profile": "hfref_2000"}
    b = results_path(cfg, "e1", run_name="run")
    assert "paper_haar" in str(a)
    assert "hfref_2000" in str(b)
    assert a != b
