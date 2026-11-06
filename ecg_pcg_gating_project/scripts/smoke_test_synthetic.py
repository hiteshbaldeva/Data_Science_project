"""End-to-end synthetic smoke test.

This script verifies that every stage runs on synthetic data:

    synthetic dataset -> cache -> gating -> E1/E2 windows -> 5-fold training
    -> comparison -> (optionally) external evaluation

It is a plumbing test, not an experiment.  The resulting metrics are
meaningless and are written under ``results_synthetic`` so they cannot be
confused with real results.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from _common import (
    PROJECT_ROOT,
    add_common_arguments,
    configuration_from_args,
    initialise,
)

from src.evaluation import build_comparison_table
from src.pipeline import (
    cache_path,
    evaluate_external,
    gating_path,
    load_windows,
    prepare_dataset,
    prepare_windows,
    results_path,
    run_gating,
    windows_path,
)
from src.synthetic import write_synthetic_dataset
from src.training import run_cross_validation
from src.utils import ensure_dir, save_json

LOGGER = logging.getLogger("scripts.smoke")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--root", default=None,
                        help="Synthetic dataset directory (default: data/synthetic_smoke)")
    parser.add_argument("--subjects", type=int, default=10)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--external", action="store_true",
                        help="Also generate and evaluate a synthetic external set")
    args = parser.parse_args()

    root = Path(args.root) if args.root else PROJECT_ROOT / "data" / "synthetic_smoke"
    overrides = {
        "data": {
            "root": str(root),
            "cache_dir": str(PROJECT_ROOT / "data" / "cache_synthetic"),
            "folds": [0, 1, 2, 3, 4],
        },
        "project": {"output_dir": "results_synthetic"},
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "stats_max_windows": 200,
            "max_train_windows": 400,
            "early_stopping_patience": 2,
            "scheduler_patience": 1,
            "num_workers": 0,
        },
        "evaluation": {"bootstrap_iterations": 100, "min_windows_per_subject": 1},
        "smoke": {"max_windows_per_recording": 4},
    }
    cfg = configuration_from_args(args, overrides=overrides)
    initialise(args, cfg)
    profile = str(cfg["preprocessing"]["profile"])

    if not (root / "fold_0").exists():
        LOGGER.info("Synthetic dataset not found at %s; generating it.", root)
        write_synthetic_dataset(
            root,
            n_subjects=int(args.subjects),
            duration=float(args.duration),
            seed=int(cfg["project"].get("seed", 42)),
        )

    # 1. Cache.
    index, cache = prepare_dataset(
        cfg,
        root=root,
        dataset_name="synthetic",
        external=False,
        overwrite=True,
        validate=True,
    )
    if index.n_pairs == 0:
        LOGGER.error("Synthetic dataset discovery found no pairs.")
        return 1

    # 2. Gating.
    gate_dir = gating_path(cfg, dataset_name="synthetic")
    gating_summary, gating_stats = run_gating(
        cfg,
        cache,
        output_dir=gate_dir,
        overwrite=True,
        make_plots=True,
        plot_count=2,
        example_count=1,
        plot_seconds=8.0,
    )
    if gating_stats["n_windows_total"] == 0:
        LOGGER.error("Gating produced no windows on synthetic data.")
        return 1

    # 3. Window tables and training for E1/E2.
    experiments = {}
    runs = {}
    max_windows = int(cfg.get("smoke", {}).get("max_windows_per_recording", 4))
    for experiment in ("e1", "e2"):
        windows, report = prepare_windows(
            cfg,
            cache,
            experiment=experiment,
            gating_dir=gate_dir if experiment == "e2" else None,
            output_dir=windows_path(cfg, experiment, dataset_name="synthetic"),
            max_windows_per_recording=max_windows,
        )
        if windows.empty:
            LOGGER.error("No windows for %s; smoke test cannot continue.", experiment)
            return 1
        result = run_cross_validation(
            windows,
            cache,
            cfg,
            experiment=experiment,
            output_dir=results_path(cfg, experiment, run_name="smoke"),
            folds=[0, 1, 2, 3, 4],
            save_window_predictions=True,
        )
        runs[experiment] = result
        experiments[experiment.upper()] = {
            "subjects": result["oof_subject_predictions"],
            "recordings": result["recording_predictions"],
            "threshold": float(result["summary"]["mean_threshold"]),
        }

    # 4. Paired comparison (synthetic only).
    comparison, paired = build_comparison_table(experiments, cfg, paired=True)
    comparison_dir = ensure_dir(
        Path(cfg["_project_root"]) / cfg["project"]["output_dir"] / profile / "comparison" / "smoke"
    )
    comparison.to_csv(comparison_dir / "comparison_results.csv", index=False)
    paired.to_csv(comparison_dir / "paired_comparison.csv", index=False)

    # 5. Sanity checks that matter even in a smoke test.
    params = {k: v["summary"]["n_parameters"] for k, v in runs.items()}
    if params.get("e1") != params.get("e2"):
        LOGGER.error("E1 and E2 used different parameter counts: %s", params)
        return 1
    for experiment, result in runs.items():
        out = Path(result["output_dir"])
        required = [
            out / "fold_metrics.csv",
            out / "oof_subject_predictions.csv",
            out / "cv_summary.json",
            out / "fold_0" / "checkpoint.pt",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            LOGGER.error("Missing smoke-test artefacts for %s: %s", experiment, missing)
            return 1

    external_summary = None
    if args.external:
        external_root = root / "external"
        if not (external_root / "test_set_label.csv").exists():
            write_synthetic_dataset(
                external_root,
                n_subjects=max(5, int(args.subjects) // 2),
                duration=float(args.duration),
                external=True,
                seed=int(cfg["project"].get("seed", 42)) + 1,
            )
        _, external_cache = prepare_dataset(
            cfg,
            root=external_root,
            dataset_name="synthetic_external",
            external=True,
            overwrite=True,
            validate=True,
        )
        ext_gate = gating_path(cfg, dataset_name="synthetic_external")
        run_gating(
            cfg,
            external_cache,
            output_dir=ext_gate,
            overwrite=True,
            make_plots=False,
        )
        ext_windows, _ = prepare_windows(
            cfg,
            external_cache,
            experiment="e2",
            gating_dir=ext_gate,
            output_dir=windows_path(cfg, "e2", dataset_name="synthetic_external"),
            max_windows_per_recording=max_windows,
        )
        checkpoints = sorted(
            (results_path(cfg, "e2", run_name="smoke")).glob("fold_*/checkpoint.pt")
        )
        external_summary = evaluate_external(
            cfg,
            external_cache,
            ext_windows,
            checkpoints,
            output_dir=results_path(cfg, "e2", run_name="smoke") / "external_synthetic",
            experiment="e2",
        )

    save_json(
        {
            "status": "smoke_test_passed",
            "profile": profile,
            "gating": gating_stats,
            "parameters": params,
            "note": "Synthetic plumbing test only; metrics are not scientific results.",
            "external_tested": bool(args.external),
        },
        comparison_dir / "smoke_summary.json",
    )

    print()
    print("=" * 78)
    print("SYNTHETIC SMOKE TEST PASSED")
    print("=" * 78)
    print(f"profile                 : {profile}")
    print(f"synthetic subjects      : {index.n_subjects}")
    print(f"gated windows           : {gating_stats['n_windows_total']}")
    print(f"parameters E1 / E2      : {params['e1']} / {params['e2']}")
    for experiment, result in runs.items():
        pooled = result["summary"].get("pooled_oof_metrics", {})
        print(
            f"{experiment.upper()} pooled OOF subjects : n={pooled.get('n', 'NA')} "
            f"AUROC={pooled.get('roc_auc', float('nan')):.3f} "
            f"F1={pooled.get('f1', float('nan')):.3f}"
        )
    print()
    print(f"outputs                 : {comparison_dir}")
    print("REMINDER: synthetic metrics are meaningless. Do not report them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
