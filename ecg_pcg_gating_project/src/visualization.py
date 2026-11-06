"""Figures for quality control, gating diagnostics and model evaluation.

All plotting uses the non-interactive Agg backend so the scripts run on a
headless Windows session or in CI.  Figures are saved as PNG; no display is
required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .utils import ensure_dir  # noqa: E402

LOGGER = logging.getLogger(__name__)

COLORS = {
    "blue": "#2b6cb0",
    "green": "#2f855a",
    "orange": "#dd6b20",
    "red": "#c53030",
    "grey": "#718096",
    "purple": "#6b46c1",
}


def _save(fig, path: str | Path, dpi: int = 140) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _downsample(x: np.ndarray, fs: float, max_points: int = 12000) -> tuple[np.ndarray, float]:
    """Return a decimated signal and its effective sampling rate."""
    x = np.asarray(x)
    factor = max(1, int(np.ceil(x.size / float(max_points))))
    if factor == 1:
        return x, float(fs)
    return x[::factor], float(fs) / factor


# ---------------------------------------------------------------------------
# Gating QC
# ---------------------------------------------------------------------------
def plot_signal_with_gating(
    ecg: np.ndarray,
    pcg: np.ndarray,
    fs_ecg: int | float,
    fs_pcg: int | float,
    gating: Any,
    path: str | Path,
    max_seconds: float = 12.0,
    title: str | None = None,
) -> Path:
    """Plot ECG R peaks, PCG windows and S1/S2 markers for one recording.

    *gating* may be a :class:`src.gating.GatingResult` or its ``to_dict`` form.
    """
    if hasattr(gating, "to_dict"):
        data = gating.to_dict()
    else:
        data = dict(gating)
    ecg = np.asarray(ecg, dtype=float).ravel()
    pcg = np.asarray(pcg, dtype=float).ravel()
    fs_ecg = float(fs_ecg)
    fs_pcg = float(fs_pcg)
    n_ecg = min(ecg.size, int(round(max_seconds * fs_ecg)))
    n_pcg = min(pcg.size, int(round(max_seconds * fs_pcg)))
    t_ecg = np.arange(n_ecg) / fs_ecg
    t_pcg = np.arange(n_pcg) / fs_pcg
    ecg_plot, ecg_fs_plot = _downsample(ecg[:n_ecg], fs_ecg, max_points=8000)
    pcg_plot, pcg_fs_plot = _downsample(pcg[:n_pcg], fs_pcg, max_points=16000)
    t_ecg_plot = np.arange(ecg_plot.size) / ecg_fs_plot
    t_pcg_plot = np.arange(pcg_plot.size) / pcg_fs_plot

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=False)
    ax = axes[0]
    ax.plot(t_ecg_plot, ecg_plot, color=COLORS["blue"], lw=0.8, label="ECG")
    r_times = np.asarray(data.get("r_peak_times", []), dtype=float)
    r_times = r_times[r_times <= max_seconds]
    if r_times.size:
        ax.vlines(r_times, ymin=np.min(ecg_plot), ymax=np.max(ecg_plot),
                  color=COLORS["red"], lw=0.8, alpha=0.6, label="R peaks")
    ax.set_xlim(0, min(max_seconds, ecg.size / fs_ecg))
    ax.set_ylabel("ECG")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or f"Gating QC: {data.get('record_id', '')}")

    ax = axes[1]
    ax.plot(t_pcg_plot, pcg_plot, color=COLORS["grey"], lw=0.5, label="PCG")
    accepted = 0
    rejected = 0
    for window in data.get("windows", []):
        start = float(window["start_time"])
        end = float(window["end_time"])
        if start > max_seconds:
            continue
        if window.get("valid", True):
            ax.axvspan(start, min(end, max_seconds), color=COLORS["green"], alpha=0.10, lw=0)
            accepted += 1
        else:
            ax.axvspan(start, min(end, max_seconds), color=COLORS["red"], alpha=0.10, lw=0)
            rejected += 1
    for item in data.get("rejected", []):
        start = item.get("start_time")
        end = item.get("end_time")
        if start is None or end is None or start > max_seconds:
            continue
        ax.axvspan(float(start), min(float(end), max_seconds),
                   color=COLORS["red"], alpha=0.10, lw=0)
    ax.set_xlim(0, min(max_seconds, pcg.size / fs_pcg))
    ax.set_ylabel("PCG")
    ax.set_xlabel("Time (s)")
    ax.text(
        0.99,
        0.95,
        f"accepted windows: {accepted} | rejected: {rejected}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    ax = axes[2]
    if pcg.size:
        try:
            from .pcg_processing import pcg_envelope

            env = pcg_envelope(pcg[:n_pcg], fs_pcg)
            ax.plot(t_pcg_plot, env[:: max(1, int(np.ceil(env.size / 16000)))],
                    color=COLORS["purple"], lw=0.7, label="PCG envelope")
        except Exception:
            pass
    s1 = [float(x["s1_time"]) for x in data.get("s1_s2", []) if x.get("s1_time") is not None]
    s2 = [float(x["s2_time"]) for x in data.get("s1_s2", []) if x.get("s2_time") is not None]
    if s1:
        ax.scatter(s1, [0] * len(s1), marker="v", s=9, color=COLORS["green"], label="S1", zorder=3)
    if s2:
        ax.scatter(s2, [0] * len(s2), marker="^", s=9, color=COLORS["orange"], label="S2", zorder=3)
    ax.set_xlim(0, min(max_seconds, pcg.size / fs_pcg))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope")
    ax.legend(loc="upper right", fontsize=8)

    return _save(fig, path)


def plot_gated_window_examples(
    pcg: np.ndarray,
    fs_pcg: int | float,
    gating: Any,
    path: str | Path,
    n_windows: int = 6,
) -> Path:
    """Plot several accepted gated PCG windows side by side."""
    data = gating.to_dict() if hasattr(gating, "to_dict") else dict(gating)
    windows = [w for w in data.get("windows", []) if w.get("valid", True)]
    windows = windows[: int(n_windows)]
    if not windows:
        windows = []
    fig, axes = plt.subplots(
        max(1, len(windows)),
        1,
        figsize=(8, 1.5 * max(1, len(windows))),
        sharex=False,
    )
    if len(windows) == 1:
        axes = [axes]
    pcg = np.asarray(pcg, dtype=float).ravel()
    fs_pcg = float(fs_pcg)
    for ax, window in zip(axes, windows):
        start = int(window["start_sample"])
        end = int(window["end_sample"])
        segment = pcg[start:end]
        t = np.arange(segment.size) / fs_pcg
        ax.plot(t, segment, color=COLORS["blue"], lw=0.6)
        ax.set_ylabel("PCG", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(
            f"window {window.get('window_id')} | R={window.get('r_peak_time'):.3f}s | "
            f"start={window.get('start_time'):.3f}s",
            fontsize=8,
        )
    if not windows:
        axes[0].text(0.5, 0.5, "No accepted windows", ha="center", va="center")
        axes[0].axis("off")
    axes[-1].set_xlabel("Time within window (s)")
    return _save(fig, path)


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric column, or an empty Series when it is absent.

    Error-only gating summaries do not contain every optional diagnostic
    column (lag, heart rate), so plotting must tolerate missing keys instead of
    assuming ``DataFrame.get`` returns a Series.
    """
    if name not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").dropna()


def plot_gating_summary(summary: pd.DataFrame, path: str | Path) -> Path:
    """Histograms of gating success, R peaks, delay and heart rate."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    if summary.empty:
        axes = axes.ravel()
        for ax in axes:
            ax.text(0.5, 0.5, "No gating results", ha="center", va="center")
            ax.axis("off")
        return _save(fig, path)

    ax = axes[0, 0]
    values = _numeric_column(summary, "success_rate")
    ax.hist(values, bins=np.linspace(0, 1, 21), color=COLORS["green"], alpha=0.8)
    ax.set_title("Window success rate")
    ax.set_xlabel("accepted / detected R peaks")
    ax.set_ylabel("recordings")

    ax = axes[0, 1]
    values = _numeric_column(summary, "n_windows")
    ax.hist(values, bins=20, color=COLORS["blue"], alpha=0.8)
    ax.set_title("Accepted windows per recording")
    ax.set_xlabel("windows")
    ax.set_ylabel("recordings")

    ax = axes[1, 0]
    values = _numeric_column(summary, "best_lag_seconds")
    if len(values):
        ax.hist(values, bins=30, color=COLORS["purple"], alpha=0.8)
    ax.set_title("ECG-PCG lag diagnostic (not applied)")
    ax.set_xlabel("best lag (s)")
    ax.set_ylabel("recordings")

    ax = axes[1, 1]
    values = _numeric_column(summary, "heart_rate_bpm")
    if len(values):
        ax.hist(values, bins=30, color=COLORS["orange"], alpha=0.8)
    ax.set_title("Median heart rate")
    ax.set_xlabel("bpm")
    ax.set_ylabel("recordings")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray | Sequence[Sequence[int]],
    path: str | Path,
    title: str = "Confusion matrix",
    labels: Sequence[str] = ("LVEF > 40%", "LVEF <= 40%"),
) -> Path:
    cm = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                fontsize=13,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def _roc_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    from sklearn.metrics import roc_curve

    return roc_curve(y_true, y_score)


def _pr_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    from sklearn.metrics import precision_recall_curve

    return precision_recall_curve(y_true, y_score)


def plot_roc_pr(
    y_true,
    y_score,
    path: str | Path,
    title: str = "",
    threshold: float | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> Path:
    """ROC and precision-recall curves with bootstrap AUC confidence bands."""
    from .evaluation import bootstrap_metric_ci

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    if y_true.size == 0 or y_true.min() == y_true.max():
        for ax in axes:
            ax.text(0.5, 0.5, "Need both classes", ha="center", va="center")
            ax.axis("off")
        return _save(fig, path)

    fpr, tpr, _ = _roc_curve_points(y_true, y_score)
    precision, recall, _ = _pr_curve_points(y_true, y_score)
    ev = dict((cfg or {}).get("evaluation", {}))
    n_boot = int(ev.get("bootstrap_iterations", 1000))
    auc_ci = bootstrap_metric_ci(
        y_true,
        y_score,
        "roc_auc",
        n_boot=n_boot,
        confidence=float(ev.get("bootstrap_confidence", 0.95)),
        seed=int(ev.get("bootstrap_seed", 42)),
    )
    pr_ci = bootstrap_metric_ci(
        y_true,
        y_score,
        "pr_auc",
        n_boot=n_boot,
        confidence=float(ev.get("bootstrap_confidence", 0.95)),
        seed=int(ev.get("bootstrap_seed", 42)),
    )

    ax = axes[0]
    ax.plot(fpr, tpr, color=COLORS["blue"], lw=1.8)
    ax.plot([0, 1], [0, 1], "--", color=COLORS["grey"], lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(
        f"ROC (AUROC {auc_ci['point']:.3f}; "
        f"95% CI {auc_ci['lower']:.3f}-{auc_ci['upper']:.3f})"
    )
    if threshold is not None:
        tn, fp, fn, tp = 0, 0, 0, 0
        pred = (y_score >= float(threshold)).astype(int)
        from sklearn.metrics import confusion_matrix

        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ax.scatter([1 - spec], [sens], color=COLORS["red"], zorder=4,
                   label=f"threshold {threshold:.2f}")
        ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    baseline = float(y_true.mean())
    ax.plot(recall, precision, color=COLORS["green"], lw=1.8)
    ax.axhline(baseline, ls="--", color=COLORS["grey"], lw=0.8,
               label=f"prevalence {baseline:.2f}")
    ax.set_xlabel("Recall / sensitivity")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"PR (AUPRC {pr_ci['point']:.3f}; "
        f"95% CI {pr_ci['lower']:.3f}-{pr_ci['upper']:.3f})"
    )
    ax.legend(loc="upper right", fontsize=8)
    if title:
        fig.suptitle(title)
    return _save(fig, path)


def plot_probability_distribution(
    subject_predictions: pd.DataFrame,
    path: str | Path,
    threshold: float = 0.5,
    title: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if subject_predictions.empty:
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    for label, name, color in ((0, "LVEF > 40%", COLORS["blue"]), (1, "LVEF <= 40%", COLORS["red"])):
        values = subject_predictions.loc[subject_predictions["label"] == label, "prob"]
        if len(values):
            ax.hist(values, bins=np.linspace(0, 1, 21), alpha=0.55, color=color, label=name)
    ax.axvline(float(threshold), color="black", ls="--", lw=1.0,
               label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Subjects")
    ax.set_title(title or "Subject-level probability distribution")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_training_curves(history: pd.DataFrame, path: str | Path, title: str = "") -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    if history.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No history", ha="center", va="center")
            ax.axis("off")
        return _save(fig, path)
    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], label="train", color=COLORS["blue"])
    ax.plot(history["epoch"], history["val_loss"], label="validation", color=COLORS["orange"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss")
    ax.legend(fontsize=8)

    ax = axes[1]
    for col, label, color in (
        ("val_subject_auroc", "subject AUROC", COLORS["green"]),
        ("val_subject_auprc", "subject AUPRC", COLORS["purple"]),
        ("val_window_auprc", "window AUPRC", COLORS["grey"]),
    ):
        if col in history.columns:
            ax.plot(history["epoch"], history[col], label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation discrimination")
    ax.legend(fontsize=8)

    ax = axes[2]
    if "val_subject_f1_at_0.5" in history.columns:
        ax.plot(history["epoch"], history["val_subject_f1_at_0.5"],
                color=COLORS["red"], label="subject F1 @ 0.5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation F1")
    ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    return _save(fig, path)


def plot_fold_metrics(fold_metrics: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if fold_metrics.empty:
        ax.text(0.5, 0.5, "No fold metrics", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    test = fold_metrics[fold_metrics["split"] == "test"].sort_values("fold")
    metrics = [
        ("subject_roc_auc", "AUROC"),
        ("subject_pr_auc", "AUPRC"),
        ("subject_f1", "F1"),
        ("subject_recall_sensitivity", "Sensitivity"),
        ("subject_specificity", "Specificity"),
    ]
    x = np.arange(len(test))
    width = 0.15
    for i, (column, label) in enumerate(metrics):
        if column in test.columns:
            ax.bar(x + (i - 2) * width, test[column], width=width, label=label)
    ax.set_xticks(x, [f"fold {int(v)}" for v in test["fold"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Subject-level test metrics by fold")
    ax.legend(fontsize=7, ncol=5, loc="upper center")
    return _save(fig, path)


def plot_model_comparison(
    comparison: pd.DataFrame,
    path: str | Path,
    level: str = "subject",
    cohort: str = "paired",
) -> Path:
    """Bar chart of E1/E2 metrics with bootstrap confidence intervals."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    subset = comparison[(comparison["level"] == level) & (comparison["cohort"] == cohort)]
    if subset.empty:
        ax.text(0.5, 0.5, "No comparison rows", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    metrics = [
        ("roc_auc", "AUROC"),
        ("pr_auc", "AUPRC"),
        ("f1", "F1"),
        ("recall_sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]
    experiments = list(subset["experiment"].unique())
    x = np.arange(len(metrics))
    width = 0.8 / max(len(experiments), 1)
    for i, experiment in enumerate(experiments):
        row = subset[subset["experiment"] == experiment].iloc[0]
        values = [row.get(m, np.nan) for m, _ in metrics]
        lower = [row.get(f"{m}_ci_lower", np.nan) for m, _ in metrics]
        upper = [row.get(f"{m}_ci_upper", np.nan) for m, _ in metrics]
        values_arr = np.asarray(values, dtype=float)
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)
        err = np.vstack([
            np.nan_to_num(values_arr - lower_arr, nan=0.0),
            np.nan_to_num(upper_arr - values_arr, nan=0.0),
        ])
        ax.bar(x + (i - (len(experiments) - 1) / 2) * width, values_arr, width=width,
               yerr=err, capsize=3, label=experiment)
    ax.set_xticks(x, [name for _, name in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{level.capitalize()}-level comparison ({cohort} cohort)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_gating_rejections(summary: pd.DataFrame, path: str | Path) -> Path:
    """Bar chart of rejection reasons collected in the gating summary."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if summary.empty or "rejection_reasons" not in summary.columns:
        ax.text(0.5, 0.5, "No rejection reasons", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    counts: dict[str, int] = {}
    for value in summary["rejection_reasons"].dropna():
        for part in str(value).split(";"):
            part = part.strip()
            if not part or ":" not in part:
                continue
            reason, number = part.rsplit(":", 1)
            try:
                counts[reason] = counts.get(reason, 0) + int(float(number))
            except ValueError:
                continue
    if not counts:
        ax.text(0.5, 0.5, "No window rejections", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    ax.barh(labels, values, color=COLORS["red"], alpha=0.85)
    ax.set_xlabel("Rejected windows")
    ax.set_title("Gating rejection reasons")
    return _save(fig, path)
