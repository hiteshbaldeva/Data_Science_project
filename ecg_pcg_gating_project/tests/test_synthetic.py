"""Synthetic generators and dataset discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data_inspection import discover_dataset, parse_label_csv, parse_recording_filename, validate_dataset
from src.synthetic import generate_synthetic_ecg, generate_synthetic_pair, generate_synthetic_pcg


def test_synthetic_shapes_and_ground_truth():
    pair = generate_synthetic_pair(duration=10.0, heart_rate_bpm=60.0, seed=1)
    assert pair["ecg"].dtype == np.float32
    assert pair["pcg"].dtype == np.float32
    assert pair["ecg"].size == 10 * 500
    assert pair["pcg"].size == 10 * 4000
    assert pair["r_times"].size >= 8
    assert np.all(np.diff(pair["r_times"]) > 0)
    assert np.all(pair["s2_times"] > pair["s1_times"])


def test_synthetic_ecg_and_pcg_are_deterministic():
    a = generate_synthetic_ecg(duration=5.0, seed=3)
    b = generate_synthetic_ecg(duration=5.0, seed=3)
    assert np.array_equal(a[0], b[0])
    c = generate_synthetic_pcg(duration=5.0, seed=3)
    d = generate_synthetic_pcg(duration=5.0, seed=3)
    assert np.array_equal(c[0], d[0])


def test_filename_parser():
    parsed = parse_recording_filename("1005263204_APEX_ECG.wav")
    assert parsed is not None
    assert parsed.subject == "1005263204"
    assert parsed.site == "APEX"
    assert parsed.modality == "ECG"
    assert parse_recording_filename("notes.txt") is None
    assert parse_recording_filename("1005263204_APEX_PCG.wav").modality == "PCG"


def test_synthetic_dataset_discovery(synthetic_dataset: Path):
    index = discover_dataset(synthetic_dataset)
    assert index.n_subjects == 10
    assert index.n_pairs == 40
    assert index.label_counts()["positive"] == 5
    assert index.label_counts()["negative"] == 5
    folds = index.fold_counts()
    assert set(folds) == {"0", "1", "2", "3", "4"}
    assert all(counts["subjects"] == 2 for counts in folds.values())
    # Every subject must be in one fold and one fold only.
    for subject in index.subjects.values():
        assert subject.fold in {0, 1, 2, 3, 4}
    frame, summary = validate_dataset(
        index,
        expected_ecg_fs=500,
        expected_pcg_fs=4000,
        min_duration_seconds=5.0,
    )
    assert summary["n_errors"] == 0
    assert summary["n_pairs"] == 40
    assert len(frame) == 40


def test_label_csv_parser_accepts_fraction_and_percent(tmp_path: Path):
    path = tmp_path / "fold_0_label.csv"
    path.write_text("record,LVEF\nS1_APEX,0.35\nS1_LLSB,35\n", encoding="utf-8")
    rows = parse_label_csv(path)
    assert len(rows) == 2
    assert rows[0]["subject"] == "S1"
    assert pytest.approx(rows[0]["lvef_percent"]) == 35.0
    assert rows[1]["fold"] == 0
