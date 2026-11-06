"""Evaluate a trained cross-validation ensemble on the external community set.

The external set is kept completely separate.  It is never used for training,
early stopping, model selection or threshold tuning.  Its threshold is the mean
of the internal validation thresholds, and predictions are averaged over the
fold checkpoints.

The external set contains only five positive subjects.  All sensitivity-like
metrics therefore have very wide confidence intervals and are reported only as
a limited generalisation check.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _common import (
    add_common_arguments,
    configuration_from_args,
    device_from_config,
    initialise,
)

from src.data_loader import load_signal_cache
from src.evaluation import format_metric, metrics_to_text
from src.pipeline import (
    cache_path,
    evaluate_external,
    gating_path,
    prepare_dataset,
    prepare_windows,
    results_path,
)
from src.utils import save_json

LOGGER = logging.getLogger("scripts.evaluate_external")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--experiment", required=True, choices=["e1", "e2", "e3"])
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-gating", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = configuration_from_args(args)
    initialise(args, cfg)
    device = device_from_config(cfg)

    run_dir = results_path(cfg, args.experiment, run_name=args.run_name)
    checkpoints = sorted(run_dir.glob("fold_*/checkpoint.pt"))
    if not checkpoints:
        LOGGER.error(
            "No checkpoints found under %s. Train the experiment first:\n"
            "  python scripts/train.py --experiment %s --run-name %s",
            run_dir,
            args.experiment,
            args.run_name,
        )
        return 2

    # 1. External cache and gating.
    index, cache = prepare_dataset(
        cfg,
        root=None,
        dataset_name="external",
        external=True,
        overwrite=bool(args.overwrite_cache),
        max_recordings=None,
        validate=True,
    )
    gating_dir = gating_path(cfg, dataset_name="external")
    from src.pipeline import run_gating

    run_gating(
        cfg,
        cache,
        output_dir=gating_dir,
        overwrite=bool(args.overwrite_gating),
        max_recordings=None,
        make_plots=True,
        plot_count=5,
        example_count=2,
        plot_seconds=10.0,
    )

    # 2. Window table using the experiment definition.
    ext_window_dir = Path(cfg["_project_root"]) / "data" / "windows" / str(cfg["preprocessing"]["profile"]) / "external" / args.experiment
    windows, report = prepare_windows(
        cfg,
        cache,
        experiment=args.experiment,
        gating_dir=gating_dir if args.experiment in {"e2", "e3"} else None,
        output_dir=ext_window_dir,
        max_windows_per_recording=None,
    )
    if windows.empty:
        LOGGER.error("External window table is empty. Check the gating summary.")
        return 1

    output = Path(args.out) if args.out else run_dir / "external"
    summary = evaluate_external(
        cfg,
        cache,
        windows,
        checkpoints,
        output_dir=output,
        experiment=args.experiment,
        device=device,
        batch_size=int(args.batch_size),
    )
    save_json(
        {
            "checkpoints": [str(p) for p in checkpoints],
            "window_report": report,
            "external_index": index.summary(),
            "note": (
                "External evaluation is a limited generalisation check. "
                "No external data were used for training or threshold selection."
            ),
        },
        output / "external_run_context.json",
    )

    metrics = summary["subject_metrics"]
    bootstrap = summary.get("bootstrap", {})
    auroc_ci = bootstrap.get("roc_auc", {})
    auprc_ci = bootstrap.get("pr_auc", {})
    print()
    print("=" * 78)
    print(f"EXTERNAL EVALUATION: {args.experiment.upper()}")
    print("=" * 78)
    print(f"subjects              : {summary['n_subjects']} "
          f"({summary['n_positive_subjects']} positive)")
    print(f"recordings            : {summary['n_recordings']}")
    print(f"threshold             : {summary['threshold']:.3f} "
          f"({summary['threshold_source']})")
    print(f"subject level         : {metrics_to_text(metrics)}")
    print(f"  AUROC 95% CI        : {format_metric(auroc_ci.get('lower'))} - "
          f"{format_metric(auroc_ci.get('upper'))}")
    print(f"  AUPRC 95% CI        : {format_metric(auprc_ci.get('lower'))} - "
          f"{format_metric(auprc_ci.get('upper'))}")
    print()
    print("LIMITATION: only 5 positive external subjects. Do not present these")
    print("estimates as precise performance; the confidence intervals are wide.")
    print()
    print(f"outputs               : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
