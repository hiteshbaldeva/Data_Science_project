"""Metrics, aggregation, threshold selection and paired comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    aggregate_windows_to_recordings,
    aggregate_windows_to_subjects,
    bootstrap_metric_ci,
    build_comparison_table,
    compute_binary_metrics,
    confusion_counts,
    paired_bootstrap_difference,
    restrict_to_common_subjects,
    select_threshold,
    summarise_predictions,
)


def test_confusion_counts_and_metrics():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.8, 0.6]
    counts = confusion_counts(y_true, y_score, 0.5)
    assert counts == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    metrics = compute_binary_metrics(y_true, y_score, 0.5)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall_sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["npv"] == pytest.approx(1.0)


def test_postprocessing_never_degenerates_to_accuracy_only():
    metrics = compute_binary_metrics([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], 0.5)
    for key in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "balanced_accuracy"):
        assert key in metrics


def test_single_class_auc_is_nan():
    metrics = compute_binary_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.5)
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])


def test_aggregate_windows_to_subjects():
    windows = pd.DataFrame(
        {
            "subject": ["A", "A", "B"],
            "record_id": ["A_1", "A_2", "B_1"],
            "site": ["APEX", "LLSB", "APEX"],
            "label": [1, 1, 0],
            "fold": [0, 0, 1],
            "prob": [0.8, 0.6, 0.2],
        }
    )
    subjects = aggregate_windows_to_subjects(windows)
    assert len(subjects) == 2
    row = subjects[subjects["subject"] == "A"].iloc[0]
    assert row["prob"] == pytest.approx(0.7)
    assert row["n_windows"] == 2
    assert row["n_recordings"] == 2

    recordings = aggregate_windows_to_recordings(windows)
    assert len(recordings) == 3
    assert set(recordings["record_id"]) == {"A_1", "A_2", "B_1"}


def test_min_windows_per_subject_filter():
    windows = pd.DataFrame(
        {
            "subject": ["A", "B", "B"],
            "record_id": ["A_1", "B_1", "B_2"],
            "label": [1, 0, 0],
            "fold": [0, 0, 0],
            "prob": [0.7, 0.2, 0.4],
        }
    )
    subjects = aggregate_windows_to_subjects(windows, min_windows=2)
    assert list(subjects["subject"]) == ["B"]


def test_threshold_selection_on_separable_data(cfg):
    y = np.array([0] * 10 + [1] * 10)
    p = np.array([0.05 + 0.03 * i for i in range(10)] + [0.7 + 0.025 * i for i in range(10)])
    info = select_threshold(y, p, metric="f1", cfg=cfg)
    assert 0.3 < info["threshold"] < 0.75
    assert info["score"] == pytest.approx(1.0)


def test_threshold_selection_single_class_falls_back(cfg):
    info = select_threshold([0, 0, 0], [0.1, 0.2, 0.3], cfg=cfg)
    assert info["threshold"] == 0.5
    assert "only one class" in info["reason"]


def test_bootstrap_ci_and_paired_difference(cfg):
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    good = np.concatenate([rng.uniform(0.0, 0.5, 40), rng.uniform(0.5, 1.0, 40)])
    worse = np.concatenate([rng.uniform(0.0, 0.8, 40), rng.uniform(0.2, 1.0, 40)])
    ci = bootstrap_metric_ci(y, good, "roc_auc", n_boot=100)
    assert ci["n"] == 80
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    diff = paired_bootstrap_difference(
        y, good, worse, metric="roc_auc", n_boot=100
    )
    assert diff["difference"] > 0
    assert diff["n_subjects"] == 80


def test_restrict_to_common_subjects():
    a = pd.DataFrame({"subject": ["A", "B", "C"], "prob": [0.1, 0.2, 0.3]})
    b = pd.DataFrame({"subject": ["B", "C", "D"], "prob": [0.2, 0.3, 0.4]})
    ra, rb = restrict_to_common_subjects(a, b)
    assert list(ra["subject"]) == ["B", "C"]
    assert list(rb["subject"]) == ["B", "C"]


def test_comparison_table_paired_cohort(cfg):
    subjects = [f"S{i}" for i in range(20)]
    labels = [0] * 10 + [1] * 10
    e1 = pd.DataFrame(
        {"subject": subjects, "label": labels, "fold": [0] * 20,
         "prob": [0.1] * 10 + [0.6] * 10}
    )
    e2 = pd.DataFrame(
        {"subject": subjects, "label": labels, "fold": [0] * 20,
         "prob": [0.05] * 10 + [0.9] * 10}
    )
    table, paired = build_comparison_table(
        {
            "E1": {"subjects": e1, "recordings": None, "threshold": 0.5},
            "E2": {"subjects": e2, "recordings": None, "threshold": 0.5},
        },
        cfg,
        paired=True,
    )
    assert set(table["cohort"]) == {"all_available", "paired"}
    assert table["level"].eq("subject").all()
    assert len(paired) == 5  # roc_auc, pr_auc, f1, sensitivity, specificity
    assert (paired["difference"] <= 1e-9).all()  # E1 reference minus E2
