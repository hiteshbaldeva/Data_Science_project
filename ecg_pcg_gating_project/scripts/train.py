"""Train and evaluate one experiment with subject-independent cross-validation.

E1 and E2 are run with exactly the same model, loss, optimiser and evaluation
code.  A validation subset is carved out of the training folds for early
stopping and threshold selection; the test fold is untouched.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from _common import (
    add_common_arguments,
    configuration_from_args,
    device_from_config,
    initialise,
    save_effective_config,
)

from src.evaluation import format_metric, metrics_to_text
from src.pipeline import (
    load_windows,
    results_path,
    smoke_subset,
    windows_path,
)
from src.data_loader import load_signal_cache
from src.pipeline import cache_path
from src.training import run_cross_validation
from src.utils import save_json

LOGGER = logging.getLogger("scripts.train")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--experiment", required=True, choices=["e1", "e2", "e3"])
    parser.add_argument("--dataset-name", default="main")
    parser.add_argument("--windows", default=None,
                        help="Explicit windows.csv path (default: data/windows/...)")
    parser.add_argument("--run-name", default=None,
                        help="Output sub-directory (default: run, or smoke)")
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="Fold indices (default: 0 1 2 3 4 from the config)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run for pipeline verification; NOT a result")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Subset subjects (smoke testing only)")
    parser.add_argument("--no-window-predictions", action="store_true",
                        help="Do not write per-window CSVs (saves disk space)")
    args = parser.parse_args()

    overrides = {}
    if args.smoke:
        overrides = {
            "training": {
                "epochs": int(args.epochs if args.epochs is not None else 2),
                "batch_size": 2,
                "stats_max_windows": 200,
                "max_train_windows": 400,
                "early_stopping_patience": 2,
                "scheduler_patience": 1,
                "num_workers": 0,
            },
            "evaluation": {"bootstrap_iterations": 100},
        }
    elif args.epochs is not None:
        overrides = {"training": {"epochs": int(args.epochs)}}

    cfg = configuration_from_args(args, overrides=overrides)
    initialise(args, cfg)
    device = device_from_config(cfg)
    profile = str(cfg["preprocessing"]["profile"])

    window_dir = Path(args.windows) if args.windows else windows_path(
        cfg, args.experiment, dataset_name=args.dataset_name
    )
    windows = load_windows(window_dir)
    windows = windows[windows["label"].notna()].copy()
    if args.smoke or args.max_subjects:
        max_subjects = int(
            args.max_subjects if args.max_subjects is not None
            else cfg.get("smoke", {}).get("max_subjects", 8)
        )
        windows = smoke_subset(windows, max_subjects=max_subjects, seed=cfg["project"].get("seed", 42))
        LOGGER.warning(
            "SMOKE MODE: using %d subjects. These outputs are pipeline checks, not scientific results.",
            windows["subject"].nunique(),
        )

    cache_dir = cache_path(cfg, dataset_name=args.dataset_name)
    if not (cache_dir / "metadata.csv").exists():
        LOGGER.error("Signal cache not found at %s. Run scripts/prepare_cache.py first.", cache_dir)
        return 2
    cache = load_signal_cache(cache_dir)

    run_name = args.run_name or ("smoke" if args.smoke else "run")
    output = results_path(cfg, args.experiment, run_name=run_name)
    output.mkdir(parents=True, exist_ok=True)
    save_effective_config(cfg, output)
    save_json(vars(args), output / "train_arguments.json")

    requested_folds = list(args.folds) if args.folds else list(cfg["data"]["folds"])
    available_folds = sorted(int(v) for v in windows["fold"].dropna().unique())
    folds = [f for f in requested_folds if f in available_folds]
    if not folds:
        LOGGER.error(
            "None of the requested folds %s have windows. Available folds: %s",
            requested_folds,
            available_folds,
        )
        return 2
    if len(folds) != len(requested_folds):
        LOGGER.warning("Running folds %s; folds %s have no windows in this subset.",
                       folds, sorted(set(requested_folds) - set(folds)))

    try:
        result = run_cross_validation(
            windows,
            cache,
            cfg,
            experiment=args.experiment,
            output_dir=output,
            folds=folds,
            seed=int(cfg["project"].get("seed", 42)),
            save_window_predictions=not args.no_window_predictions,
        )
    except Exception:
        LOGGER.exception("Training failed")
        return 1

    summary = result["summary"]
    print()
    print("=" * 78)
    print(f"CROSS-VALIDATION COMPLETE: {args.experiment.upper()} ({profile})")
    print("=" * 78)
    print(f"folds                 : {summary['folds']}")
    print(f"fold source           : {summary['fold_source']}")
    print(f"model parameters      : {summary['n_parameters']:,}")
    print(f"thresholds            : {[round(t, 3) for t in summary['thresholds']]}")
    print(f"mean threshold        : {summary['mean_threshold']:.3f}")
    print()
    if summary.get("pooled_oof_metrics"):
        pooled = summary["pooled_oof_metrics"]
        print(f"pooled OOF subject    : {metrics_to_text(pooled)}")
        print(f"  AUROC 95% CI        : {format_metric(pooled.get('roc_auc_ci_lower'))} - "
              f"{format_metric(pooled.get('roc_auc_ci_upper'))}")
        print(f"  F1 95% CI           : {format_metric(pooled.get('f1_ci_lower'))} - "
              f"{format_metric(pooled.get('f1_ci_upper'))}")
    print()
    for metric in ("subject_roc_auc", "subject_pr_auc", "subject_f1"):
        if f"mean_{metric}" in summary:
            print(f"per-fold {metric:<18s}: "
                  f"{format_metric(summary[f'mean_{metric}'])} "
                  f"+/- {format_metric(summary[f'std_{metric}'])}")
    print()
    print(f"outputs               : {output}")
    print(f"  fold_metrics.csv    : {output / 'fold_metrics.csv'}")
    print(f"  oof predictions     : {output / 'oof_subject_predictions.csv'}")
    print(f"  checkpoints         : {output / 'fold_*' / 'checkpoint.pt'}")
    print()
    if args.smoke:
        print("SMOKE RUN ONLY: no scientific conclusion should be drawn.")
    print("Next: scripts/evaluate.py and scripts/compare.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
