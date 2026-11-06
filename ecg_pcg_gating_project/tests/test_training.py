"""Subject splits, class weights and a tiny end-to-end training run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import subject_table
from src.training import (
    build_model_from_checkpoint,
    compute_class_weights,
    ensure_folds,
    load_checkpoint,
    make_subject_splits,
    train_one_fold,
)


def test_subject_level_split_has_no_leakage(tiny_cache):
    _, windows = tiny_cache
    train, val, test, info = make_subject_splits(
        windows, test_fold=0, val_fraction=0.25, seed=3, min_val_positives=1
    )
    assert set(train["subject"]).isdisjoint(set(val["subject"]))
    assert set(train["subject"]).isdisjoint(set(test["subject"]))
    assert set(val["subject"]).isdisjoint(set(test["subject"]))
    assert set(test["subject"]) == {"S00", "S03"}
    assert info["n_test_subjects"] == 2
    assert info["n_train_subjects"] + info["n_val_subjects"] + info["n_test_subjects"] == 6
    # Every recording of a subject stays in the same split.
    for frame in (train, val, test):
        grouped = frame.groupby("subject")["fold"].nunique()
        assert (grouped <= 1).all()


def test_official_folds_are_kept_when_complete(tiny_cache):
    _, windows = tiny_cache
    result, source = ensure_folds(windows, n_splits=3)
    assert source == "official_folds"
    assert result["fold"].notna().all()


def test_fallback_folds_are_explicit(tiny_cache):
    _, windows = tiny_cache
    without = windows.copy()
    without["fold"] = np.nan
    result, source = ensure_folds(without, n_splits=3, seed=5)
    assert source != "official_folds"
    assert result["fold"].notna().all()
    # Each subject is assigned to exactly one fold.
    assert result.groupby("subject")["fold"].nunique().max() == 1
    assert set(result["fold"].astype(int).unique()) == {0, 1, 2}


def test_class_weights_are_balanced():
    weights = compute_class_weights([0] * 8 + [1] * 2)
    assert weights.shape == (2,)
    assert weights[1] > weights[0]
    assert float(weights[0] * 8) == pytest.approx(float(weights[1] * 2), rel=1e-6)


def test_tiny_training_run(tmp_path: Path, cfg, tiny_cache):
    cache, windows = tiny_cache
    result = train_one_fold(
        windows,
        cache,
        cfg,
        fold=0,
        experiment="e1",
        output_dir=tmp_path / "fold_0",
        device=torch.device("cpu"),
        save_window_predictions=False,
    )
    assert result["n_parameters"] == 1150
    assert 0.0 <= result["threshold"] <= 1.0
    assert Path(result["checkpoint"]).exists()
    assert (tmp_path / "fold_0" / "fold_metrics.json").exists()
    assert (tmp_path / "fold_0" / "history.csv").exists()
    assert len(result["test_subjects"]) == 2
    assert len(result["val_subjects"]) == 2
    assert result["split"]["n_test_subjects"] == 2
    assert result["test_windows"]["subject"].nunique() == 2

    checkpoint = load_checkpoint(result["checkpoint"], device="cpu")
    model = build_model_from_checkpoint(checkpoint, cfg, device="cpu")
    assert sum(p.numel() for p in model.parameters()) == 1150
    assert checkpoint["profile_name"] == "paper_haar"
    assert "mean" in checkpoint["normalization"]
    assert checkpoint["normalization"]["source"] == "train"
