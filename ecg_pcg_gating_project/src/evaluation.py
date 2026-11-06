"""Metrics, aggregation and statistical comparison.

All reporting is done at two levels:

* **recording level** - one auscultation site (one ECG/PCG pair);
* **subject level** - the mean predicted probability across that subject's
  available recordings/windows.

The subject level is the clinically meaningful unit and the primary result.
Confidence intervals use a subject-level bootstrap, so recordings from the same
subject are never treated as independent observations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import safe_div

LOGGER = logging.getLogger(__name__)

METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "npv",
    "roc_auc",
    "pr_auc",
    "brier",
]


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------
def _as_arrays(y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.size != y_score.size:
        raise ValueError(f"y_true and y_score differ in length: {y_true.size} vs {y_score.size}")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on zero samples.")
    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("y_true must contain only 0/1 labels.")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score contains non-finite values.")
    return y_true.astype(int), y_score.astype(float)


def confusion_counts(y_true, y_score, threshold: float = 0.5) -> dict[str, int]:
    """Return ``tn``, ``fp``, ``fn``, ``tp`` for a probability threshold."""
    y_true, y_score = _as_arrays(y_true, y_score)
    pred = (y_score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_binary_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute a full set of binary classification metrics.

    Accuracy is reported for completeness but is never the only metric.
    ROC-AUC and PR-AUC are threshold-free.  If a split contains only one class
    the corresponding AUC is ``nan`` rather than an invented number.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    threshold = float(threshold)
    pred = (y_score >= threshold).astype(int)
    counts = confusion_counts(y_true, y_score, threshold)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    n = int(y_true.size)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    roc_auc = float(roc_auc_score(y_true, y_score)) if n_pos and n_neg else float("nan")
    pr_auc = float(average_precision_score(y_true, y_score)) if n_pos else float("nan")
    brier = float(brier_score_loss(y_true, y_score)) if n_pos and n_neg else float("nan")

    return {
        **counts,
        "threshold": threshold,
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": safe_div(tn, tn + fp, 0.0),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "npv": safe_div(tn, tn + fn, 0.0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_windows_to_subjects(
    windows: pd.DataFrame,
    prob_col: str = "prob",
    min_windows: int = 1,
    require_columns: Iterable[str] = ("subject", "label", "fold"),
    site_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate window predictions to one row per subject.

    The primary aggregation is the arithmetic mean of window probabilities,
    matching the dataset paper's subject-level decision rule. If site_weights
    are provided (e.g. APEX=2.0), a weighted average across auscultation
    sites is used.
    """
    for col in require_columns:
        if col not in windows.columns:
            raise ValueError(f"Missing required column {col!r} in window predictions.")
    if prob_col not in windows.columns:
        raise ValueError(f"Missing probability column {prob_col!r}.")
    if windows.empty:
        return pd.DataFrame(
            columns=[
                "subject",
                "label",
                "fold",
                "prob",
                "n_windows",
                "n_recordings",
                "max_prob",
            ]
        )
    if site_weights and "site" in windows.columns:
        w_df = windows.copy()
        w_df["_sw"] = w_df["site"].astype(str).map(lambda s: float(site_weights.get(s, 1.0)))
        w_df["_weighted_p"] = w_df[prob_col] * w_df["_sw"]
        subj_weighted_sum = w_df.groupby("subject")["_weighted_p"].sum()
        subj_weight_sum = w_df.groupby("subject")["_sw"].sum().clip(lower=1e-6)
        subj_prob = (subj_weighted_sum / subj_weight_sum).reset_index(name="prob")
        grouped = (
            windows.groupby("subject", as_index=False)
            .agg(
                label=("label", "first"),
                fold=("fold", "first"),
                max_prob=(prob_col, "max"),
                n_windows=(prob_col, "size"),
                n_recordings=("record_id", "nunique")
                if "record_id" in windows.columns
                else (prob_col, "size"),
            )
            .merge(subj_prob, on="subject")
        )
    else:
        grouped = (
            windows.groupby("subject", as_index=False)
            .agg(
                label=("label", "first"),
                fold=("fold", "first"),
                prob=(prob_col, "mean"),
                max_prob=(prob_col, "max"),
                n_windows=(prob_col, "size"),
                n_recordings=("record_id", "nunique")
                if "record_id" in windows.columns
                else (prob_col, "size"),
            )
            .reset_index(drop=True)
        )
    if min_windows > 1:
        before = len(grouped)
        grouped = grouped[grouped["n_windows"] >= int(min_windows)].reset_index(drop=True)
        if len(grouped) < before:
            LOGGER.info(
                "Excluded %d subjects with fewer than %d windows.",
                before - len(grouped),
                min_windows,
            )
    return grouped


def aggregate_windows_to_recordings(
    windows: pd.DataFrame,
    prob_col: str = "prob",
) -> pd.DataFrame:
    """Aggregate window predictions to one row per subject/site recording."""
    if "record_id" not in windows.columns:
        raise ValueError("record_id column is required for recording-level analysis.")
    if windows.empty:
        return pd.DataFrame(
            columns=["record_id", "subject", "site", "label", "fold", "prob", "n_windows"]
        )
    return (
        windows.groupby("record_id", as_index=False)
        .agg(
            subject=("subject", "first"),
            site=("site", "first") if "site" in windows.columns else ("subject", "first"),
            label=("label", "first"),
            fold=("fold", "first"),
            prob=(prob_col, "mean"),
            n_windows=(prob_col, "size"),
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Threshold selection (validation only)
# ---------------------------------------------------------------------------
def threshold_grid(cfg: Mapping[str, Any] | None = None) -> np.ndarray:
    cfg = dict(cfg or {})
    ev = dict(cfg.get("evaluation", {}))
    start = float(ev.get("threshold_grid_start", 0.01))
    stop = float(ev.get("threshold_grid_stop", 0.99))
    step = float(ev.get("threshold_grid_step", 0.01))
    return np.round(np.arange(start, stop + step / 2.0, step), 6)


def _threshold_score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, metric: str) -> float:
    pred = (y_prob >= threshold).astype(int)
    if metric == "f1":
        return float(f1_score(y_true, pred, zero_division=0))
    if metric == "youden":
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return float(tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1.0)
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, pred))
    if metric == "sensitivity_plus_specificity":
        return float(recall_score(y_true, pred, zero_division=0) + safe_div(tn, tn + fp, 0.0))
    raise ValueError(f"Unknown threshold metric: {metric}")


def select_threshold(
    y_true,
    y_prob,
    metric: str = "f1",
    cfg: Mapping[str, Any] | None = None,
    fallback: float = 0.5,
) -> dict[str, Any]:
    """Select a decision threshold on **validation** data only.

    Ties are broken towards the threshold closest to 0.5, which is a neutral
    choice and avoids selecting an extreme value by accident.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.size == 0:
        return {"threshold": float(fallback), "metric": metric, "score": float("nan")}
    if y_true.min() == y_true.max():
        return {
            "threshold": float(fallback),
            "metric": metric,
            "score": float("nan"),
            "reason": "validation set contains only one class",
        }
    grid = threshold_grid(cfg)
    scores = np.asarray(
        [_threshold_score(y_true, y_prob, t, metric) for t in grid], dtype=float
    )
    best_score = float(np.nanmax(scores))
    candidates = grid[np.isclose(scores, best_score, atol=1e-12)]
    best = float(candidates[np.argmin(np.abs(candidates - 0.5))])
    return {
        "threshold": best,
        "metric": metric,
        "score": best_score,
        "grid_size": int(grid.size),
        "criterion": metric,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def _metric_value(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str,
    threshold: float,
) -> float:
    """Scalar metric used by the bootstrap routines."""
    metric = metric.lower()
    if metric in {"roc_auc", "auroc", "auc"}:
        if y_true.min() == y_true.max():
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    if metric in {"pr_auc", "auprc", "average_precision"}:
        if y_true.sum() == 0:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    pred = (y_score >= float(threshold)).astype(int)
    if metric in {"f1", "f1_score"}:
        return float(f1_score(y_true, pred, zero_division=0))
    if metric in {"recall", "sensitivity", "recall_sensitivity"}:
        return float(recall_score(y_true, pred, zero_division=0))
    if metric in {"specificity"}:
        tn, fp, _, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return safe_div(tn, tn + fp, 0.0)
    if metric in {"accuracy"}:
        return float(accuracy_score(y_true, pred))
    if metric in {"precision"}:
        return float(precision_score(y_true, pred, zero_division=0))
    if metric in {"balanced_accuracy"}:
        return float(balanced_accuracy_score(y_true, pred))
    raise ValueError(f"Unknown bootstrap metric: {metric}")


def bootstrap_metric_ci(
    y_true,
    y_score,
    metric: str = "roc_auc",
    threshold: float = 0.5,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Percentile bootstrap confidence interval at the subject level.

    Resampling is over rows.  Callers must pass **subject-level** predictions
    (one row per subject), not windows, so that repeated recordings from one
    subject do not inflate the effective sample size.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    point = _metric_value(y_true, y_score, metric, threshold)
    n = int(y_true.size)
    if n < 3:
        return {
            "metric": metric,
            "point": point,
            "lower": float("nan"),
            "upper": float("nan"),
            "n": n,
            "n_valid_boot": 0,
        }
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        value = _metric_value(y_true[idx], y_score[idx], metric, threshold)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {
            "metric": metric,
            "point": point,
            "lower": float("nan"),
            "upper": float("nan"),
            "n": n,
            "n_valid_boot": 0,
        }
    alpha = (1.0 - float(confidence)) / 2.0
    lower = float(np.percentile(values, 100.0 * alpha))
    upper = float(np.percentile(values, 100.0 * (1.0 - alpha)))
    return {
        "metric": metric,
        "point": point,
        "lower": lower,
        "upper": upper,
        "n": n,
        "n_valid_boot": len(values),
        "confidence": float(confidence),
    }


def paired_bootstrap_difference(
    y_true,
    score_a,
    score_b,
    metric: str = "roc_auc",
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired subject-level bootstrap of ``metric(A) - metric(B)``.

    A confidence interval that excludes zero is evidence of a difference in
    this sample.  It is *not* a substitute for a pre-registered hypothesis test;
    the project reports the interval and the paired cohort size explicitly.
    """
    y_true, score_a = _as_arrays(y_true, score_a)
    _, score_b = _as_arrays(y_true, score_b)
    if score_a.size != score_b.size:
        raise ValueError("Paired scores must have the same length.")
    point_a = _metric_value(y_true, score_a, metric, threshold_a)
    point_b = _metric_value(y_true, score_b, metric, threshold_b)
    n = int(y_true.size)
    rng = np.random.default_rng(int(seed))
    diffs: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        a = _metric_value(y_true[idx], score_a[idx], metric, threshold_a)
        b = _metric_value(y_true[idx], score_b[idx], metric, threshold_b)
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    if not diffs:
        lower = upper = float("nan")
    else:
        alpha = (1.0 - float(confidence)) / 2.0
        lower = float(np.percentile(diffs, 100.0 * alpha))
        upper = float(np.percentile(diffs, 100.0 * (1.0 - alpha)))
    return {
        "metric": metric,
        "value_a": point_a,
        "value_b": point_b,
        "difference": point_a - point_b,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_subjects": n,
        "n_valid_boot": len(diffs),
        "confidence": float(confidence),
        "threshold_a": float(threshold_a),
        "threshold_b": float(threshold_b),
    }


# ---------------------------------------------------------------------------
# Reporting tables
# ---------------------------------------------------------------------------
def _summary_row(
    experiment: str,
    level: str,
    cohort: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metrics = compute_binary_metrics(y_true, y_score, threshold)
    ev = dict((cfg or {}).get("evaluation", {}))
    n_boot = int(ev.get("bootstrap_iterations", 1000))
    conf = float(ev.get("bootstrap_confidence", 0.95))
    seed = int(ev.get("bootstrap_seed", 42))
    row: dict[str, Any] = {
        "experiment": experiment,
        "level": level,
        "cohort": cohort,
        **metrics,
    }
    for metric in ("roc_auc", "pr_auc", "f1", "recall_sensitivity", "specificity"):
        ci = bootstrap_metric_ci(
            y_true,
            y_score,
            metric=metric,
            threshold=threshold,
            n_boot=n_boot,
            confidence=conf,
            seed=seed,
        )
        key = metric if metric in {"roc_auc", "pr_auc"} else metric
        row[f"{key}_ci_lower"] = ci["lower"]
        row[f"{key}_ci_upper"] = ci["upper"]
        row[f"{key}_n_valid_boot"] = ci["n_valid_boot"]
    return row


def summarise_predictions(
    experiment: str,
    subject_predictions: pd.DataFrame,
    cfg: Mapping[str, Any] | None = None,
    threshold: float = 0.5,
    cohort: str = "all_available",
    recording_predictions: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Return subject-level and recording-level metric rows."""
    rows: list[dict[str, Any]] = []
    if subject_predictions is not None and len(subject_predictions):
        rows.append(
            _summary_row(
                experiment,
                "subject",
                cohort,
                subject_predictions["label"].to_numpy(),
                subject_predictions["prob"].to_numpy(),
                threshold,
                cfg,
            )
        )
    if recording_predictions is not None and len(recording_predictions):
        rows.append(
            _summary_row(
                experiment,
                "recording",
                cohort,
                recording_predictions["label"].to_numpy(),
                recording_predictions["prob"].to_numpy(),
                threshold,
                cfg,
            )
        )
    return rows


def restrict_to_common_subjects(
    *frames: pd.DataFrame,
    subject_col: str = "subject",
) -> tuple[pd.DataFrame, ...]:
    """Restrict prediction frames to subjects present in every frame."""
    if not frames:
        return tuple()
    common = set(frames[0][subject_col].astype(str))
    for frame in frames[1:]:
        common &= set(frame[subject_col].astype(str))
    out = []
    for frame in frames:
        sub = frame[frame[subject_col].astype(str).isin(common)].copy()
        sub = sub.sort_values(subject_col).reset_index(drop=True)
        out.append(sub)
    return tuple(out)


def build_comparison_table(
    experiments: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
    paired: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``comparison_results.csv`` and ``paired_comparison.csv``.

    Parameters
    ----------
    experiments:
        Mapping ``name -> {"subjects": DataFrame, "recordings": DataFrame,
        "threshold": float}``.  DataFrames must contain ``subject``,
        ``label`` and ``prob``.
    paired:
        If true, additionally evaluate every experiment on the intersection of
        subjects that have predictions in all experiments, and compute paired
        bootstrap differences against the first experiment (the reference,
        normally E1).
    """
    rows: list[dict[str, Any]] = []
    for name, payload in experiments.items():
        subjects = payload["subjects"]
        recordings = payload.get("recordings")
        threshold = float(payload.get("threshold", 0.5))
        rows.extend(
            summarise_predictions(
                name,
                subjects,
                cfg=cfg,
                threshold=threshold,
                cohort="all_available",
                recording_predictions=recordings,
            )
        )

    paired_rows: list[dict[str, Any]] = []
    if paired and len(experiments) >= 2:
        names = list(experiments)
        subject_frames = [experiments[n]["subjects"] for n in names]
        common = restrict_to_common_subjects(*subject_frames)
        for name, subjects in zip(names, common):
            payload = experiments[name]
            recordings = payload.get("recordings")
            if recordings is not None and len(recordings):
                recordings = recordings[
                    recordings["subject"].astype(str).isin(set(subjects["subject"].astype(str)))
                ]
            rows.extend(
                summarise_predictions(
                    name,
                    subjects,
                    cfg=cfg,
                    threshold=float(payload.get("threshold", 0.5)),
                    cohort="paired",
                    recording_predictions=recordings,
                )
            )

        reference = names[0]
        ref = experiments[reference]["subjects"]
        ref = ref.set_index(ref["subject"].astype(str))
        for name in names[1:]:
            other = experiments[name]["subjects"]
            other = other.set_index(other["subject"].astype(str))
            common_index = ref.index.intersection(other.index)
            ref_aligned = ref.loc[common_index]
            other_aligned = other.loc[common_index]
            y_true = ref_aligned["label"].to_numpy()
            ev = dict(cfg.get("evaluation", {}))
            for metric in ("roc_auc", "pr_auc", "f1", "recall_sensitivity", "specificity"):
                diff = paired_bootstrap_difference(
                    y_true,
                    ref_aligned["prob"].to_numpy(),
                    other_aligned["prob"].to_numpy(),
                    metric=metric,
                    threshold_a=float(experiments[reference].get("threshold", 0.5)),
                    threshold_b=float(experiments[name].get("threshold", 0.5)),
                    n_boot=int(ev.get("bootstrap_iterations", 1000)),
                    confidence=float(ev.get("bootstrap_confidence", 0.95)),
                    seed=int(ev.get("bootstrap_seed", 42)),
                )
                paired_rows.append(
                    {
                        "reference_experiment": reference,
                        "comparison_experiment": name,
                        "level": "subject",
                        "metric": metric,
                        **diff,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def format_metric(value: float, digits: int = 3) -> str:
    """Format a metric, displaying ``nan`` as ``NA``."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def metrics_to_text(metrics: Mapping[str, Any]) -> str:
    """Human-readable metric summary used in console output."""
    return (
        f"AUROC={format_metric(metrics.get('roc_auc'))} "
        f"AUPRC={format_metric(metrics.get('pr_auc'))} "
        f"F1={format_metric(metrics.get('f1'))} "
        f"Sens={format_metric(metrics.get('recall_sensitivity'))} "
        f"Spec={format_metric(metrics.get('specificity'))} "
        f"Acc={format_metric(metrics.get('accuracy'))}"
    )
