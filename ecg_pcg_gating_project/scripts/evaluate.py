"""Evaluate a completed cross-validation run from its saved predictions.

This script does not retrain anything.  It applies the fold-specific
thresholds (mean threshold for the pooled view), computes subject-level and
recording-level metrics with bootstrap confidence intervals, and creates
confusion-matrix, ROC/PR and probability-distribution figures.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from _common import add_common_arguments, configuration_from_args, initialise

from src.evaluation import (
    compute_binary_metrics,
    format_metric,
    metrics_to_text,
    summarise_predictions,
)
from src.pipeline import results_path
from src.utils import ensure_dir, load_json, save_json

LOGGER = logging.getLogger("scripts.evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--experiment", required=True, choices=["e1", "e2", "e3"])
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (default: mean of validation thresholds)")
    args = parser.parse_args()

    cfg = configuration_from_args(args)
    initialise(args, cfg)
    run_dir = Path(args.run_dir) if args.run_dir else results_path(
        cfg, args.experiment, run_name=args.run_name
    )
    if not run_dir.exists():
        LOGGER.error("Run directory not found: %s", run_dir)
        return 2

    subject_path = run_dir / "oof_subject_predictions.csv"
    if not subject_path.exists():
        LOGGER.error("Missing %s. Run scripts/train.py first.", subject_path)
        return 2
    subjects = pd.read_csv(subject_path)
    if subjects.empty:
        LOGGER.error("No subject predictions in %s", subject_path)
        return 2

    recording_path = run_dir / "recording_level_predictions.csv"
    recordings = pd.read_csv(recording_path) if recording_path.exists() else None

    threshold_path = run_dir / "thresholds.csv"
    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "command line"
    elif threshold_path.exists():
        thresholds = pd.read_csv(threshold_path)
        threshold = float(pd.to_numeric(thresholds["threshold"], errors="coerce").mean())
        threshold_source = "mean of validation-selected fold thresholds"
    else:
        threshold = 0.5
        threshold_source = "default 0.5 (no thresholds.csv)"

    rows = summarise_predictions(
        args.experiment.upper(),
        subjects,
        cfg=cfg,
        threshold=threshold,
        cohort="pooled_oof",
        recording_predictions=recordings,
    )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)

    subject_metrics = rows[0]
    recording_metrics = rows[1] if len(rows) > 1 else None
    metrics_at_05 = compute_binary_metrics(
        subjects["label"].to_numpy(), subjects["prob"].to_numpy(), 0.5
    )
    payload = {
        "experiment": args.experiment,
        "run_dir": str(run_dir),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "n_subjects": int(len(subjects)),
        "n_positive_subjects": int((subjects["label"] == 1).sum()),
        "n_negative_subjects": int((subjects["label"] == 0).sum()),
        "subject_metrics": subject_metrics,
        "recording_metrics": recording_metrics,
        "subject_metrics_at_0.5": metrics_at_05,
    }
    cv_summary_path = run_dir / "cv_summary.json"
    if cv_summary_path.exists():
        payload["fold_source"] = load_json(cv_summary_path).get("fold_source")
    save_json(payload, run_dir / "evaluation_summary.json")

    try:
        from src.visualization import (
            plot_confusion_matrix,
            plot_probability_distribution,
            plot_roc_pr,
        )

        cm = np.array(
            [
                [subject_metrics["tn"], subject_metrics["fp"]],
                [subject_metrics["fn"], subject_metrics["tp"]],
            ]
        )
        plot_confusion_matrix(
            cm,
            run_dir / "confusion_matrix_subject.png",
            title=f"{args.experiment.upper()} subject level (threshold {threshold:.2f})",
        )
        plot_roc_pr(
            subjects["label"].to_numpy(),
            subjects["prob"].to_numpy(),
            run_dir / "roc_pr_subject.png",
            title=f"{args.experiment.upper()} pooled out-of-fold subjects",
            threshold=threshold,
            cfg=cfg,
        )
        plot_probability_distribution(
            subjects,
            run_dir / "probability_distribution.png",
            threshold=threshold,
            title=f"{args.experiment.upper()} subject-level probabilities",
        )
    except Exception as exc:
        LOGGER.warning("Could not create evaluation figures: %s", exc)

    print()
    print("=" * 78)
    print(f"EVALUATION: {args.experiment.upper()} ({run_dir.name})")
    print("=" * 78)
    print(f"subjects              : {len(subjects)} "
          f"({int((subjects['label'] == 1).sum())} positive)")
    print(f"threshold             : {threshold:.3f} ({threshold_source})")
    print(f"subject level         : {metrics_to_text(subject_metrics)}")
    print(f"  AUROC 95% CI        : {format_metric(subject_metrics['roc_auc_ci_lower'])} - "
          f"{format_metric(subject_metrics['roc_auc_ci_upper'])}")
    print(f"  AUPRC 95% CI        : {format_metric(subject_metrics['pr_auc_ci_lower'])} - "
          f"{format_metric(subject_metrics['pr_auc_ci_upper'])}")
    print(f"  confusion (tn,fp,fn,tp): "
          f"{subject_metrics['tn']}, {subject_metrics['fp']}, "
          f"{subject_metrics['fn']}, {subject_metrics['tp']}")
    if recording_metrics:
        print(f"recording level       : {metrics_to_text(recording_metrics)}")
    print(f"at threshold 0.5      : {metrics_to_text(metrics_at_05)}")
    print()
    print(f"metrics.csv           : {run_dir / 'metrics.csv'}")
    print(f"figures               : {run_dir / 'confusion_matrix_subject.png'}")
    print(f"                        {run_dir / 'roc_pr_subject.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
