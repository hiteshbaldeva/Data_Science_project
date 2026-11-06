"""Window indexing, cached datasets and subject-level helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import (
    WindowDataset,
    build_window_index,
    check_no_subject_leakage,
    compute_pcg_stats,
    subject_table,
    subset_by_subjects,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "S1_APEX",
                "subject": "S1",
                "site": "APEX",
                "fold": 0,
                "label": 1,
                "lvef_percent": 30.0,
                "pcg_length": 30 * 1440,
                "ecg_length": 30 * 500,
                "ecg_fs": 500,
                "pcg_fs": 1440,
                "status": "ok",
            },
            {
                "record_id": "S2_APEX",
                "subject": "S2",
                "site": "APEX",
                "fold": 0,
                "label": 0,
                "lvef_percent": 60.0,
                "pcg_length": 30 * 1440,
                "ecg_length": 30 * 500,
                "ecg_fs": 500,
                "pcg_fs": 1440,
                "status": "ok",
            },
        ]
    )


def _gating_payload(record_id: str, n_windows: int = 2, ok: bool = True) -> dict:
    windows = []
    for i in range(n_windows):
        start = 2.0 + 3.0 * i
        windows.append(
            {
                "window_id": i,
                "r_peak_time": start + 0.25,
                "start_time": start,
                "end_time": start + 1.5,
                "start_sample": int(round(start * 1440)),
                "end_sample": int(round((start + 1.5) * 1440)),
                "s1_time": start + 0.3,
                "s2_time": start + 0.6,
                "pcg_rms": 0.1,
                "valid": True,
                "reasons": [],
                "duration_seconds": 1.5,
            }
        )
    return {
        "record_id": record_id,
        "subject": record_id.split("_")[0],
        "site": "APEX",
        "fold": 0,
        "recording_ok": ok,
        "reasons": [] if ok else ["only 1 successful windows (minimum 2)"],
        "windows": windows,
        "rejected": [],
    }


def test_e1_tiles_windows(cfg):
    frame, report = build_window_index(_metadata(), cfg, experiment="e1")
    # 30 s, 1.5 s windows, 0.75 s hop -> 39 full windows per recording.
    assert report["window_samples"] == 2160
    assert report["n_recordings_used"] == 2
    assert len(frame) == 78
    assert frame["start_sample"].min() == 0
    assert frame["label"].isin([0, 1]).all()


def test_e2_uses_only_gated_windows_and_skips_failed_recordings(cfg, tmp_path: Path):
    gating_dir = tmp_path / "gating"
    gating_dir.mkdir()
    (gating_dir / "S1_APEX.json").write_text(
        json.dumps(_gating_payload("S1_APEX", n_windows=2, ok=True)), encoding="utf-8"
    )
    (gating_dir / "S2_APEX.json").write_text(
        json.dumps(_gating_payload("S2_APEX", n_windows=0, ok=False)), encoding="utf-8"
    )
    frame, report = build_window_index(
        _metadata(), cfg, experiment="e2", gating_dir=gating_dir
    )
    assert len(frame) == 2
    assert set(frame["record_id"]) == {"S1_APEX"}
    assert report["n_recordings_skipped"] == 1
    assert report["gating_rejections"] == 1
    # Explicit no-fallback check: the failed recording contributes zero windows.
    assert not (frame["record_id"] == "S2_APEX").any()
    assert any("gating rejected recording" in item["reason"] for item in report["skipped"])


def test_e2_missing_gating_json_is_skipped(cfg, tmp_path: Path):
    gating_dir = tmp_path / "gating"
    gating_dir.mkdir()
    (gating_dir / "S1_APEX.json").write_text(
        json.dumps(_gating_payload("S1_APEX")), encoding="utf-8"
    )
    frame, report = build_window_index(
        _metadata(), cfg, experiment="e2", gating_dir=gating_dir
    )
    assert len(frame) == 2
    assert report["n_recordings_skipped"] == 1
    assert any("missing gating" in item["reason"] for item in report["skipped"])


def test_window_dataset_shapes_and_normalisation(cfg, tiny_cache):
    cache, windows = tiny_cache
    dataset = WindowDataset(windows, cache, cfg, normalize=False)
    assert len(dataset) == 12
    sample = dataset[0]
    assert sample["x"].shape == (1, 540)
    assert sample["label"].item() in (0.0, 1.0)
    assert isinstance(sample["subject"], str)

    stats = compute_pcg_stats(dataset, seed=1)
    dataset.stats = stats
    dataset.normalize = True
    values = np.stack([dataset[i]["x"].numpy().ravel() for i in range(len(dataset))])
    assert abs(float(values.mean())) < 1e-4
    assert abs(float(values.std()) - 1.0) < 0.05


def test_subject_helpers_and_leakage_detection(tiny_cache):
    _, windows = tiny_cache
    table = subject_table(windows)
    assert len(table) == 6
    subset = subset_by_subjects(windows, ["S00", "S01"])
    assert set(subset["subject"]) == {"S00", "S01"}
    check_no_subject_leakage(subset[subset["subject"] == "S00"], subset[subset["subject"] == "S01"])
    with pytest.raises(ValueError, match="leakage"):
        check_no_subject_leakage(subset, subset[subset["subject"] == "S00"])
