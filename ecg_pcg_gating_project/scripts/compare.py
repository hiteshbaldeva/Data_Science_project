"""Compare E1 (ungated PCG) and E2 (ECG-gated PCG) on identical subjects.

The primary comparison is paired: only subjects with predictions in every
experiment are used, and paired bootstrap confidence intervals are reported for
the difference.  Full-availability metrics are also written.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from _common import add_common_arguments, configuration_from_args, initialise

from src.evaluation import (
    build_comparison_table,
    compute_binary_metrics,
    format_metric,
    metrics_to_text,
)
from src.pipeline import results_path
from src.utils import ensure_dir, load_json, save_json

LOGGER = logging.getLogger("scripts.compare")

EXPERIMENT_LABELS = {
    "e1": ("E1", "ungated fixed PCG windows"),
    "e2": ("E2", "ECG-gated PCG windows"),
    "e3": ("E3", "ECG-gated PCG + small ECG branch (exploratory)"),
}

REFERENCE_MODELS = [
    {
        "kind": "reference",
        "model": "Original paper, Configuration 5 (5 classes)",
        "parameters": 1537,
        "source": "Original paper Table IV; reconstructed layer-by-layer",
    },
    {
        "kind": "reference",
        "model": "Original paper, Configuration 5 (binary projection)",
        "parameters": 1150,
        "source": "Same encoder, 128->2 fully connected head",
    },
    {
        "kind": "reference",
        "model": "LGC2-Net full (dataset paper)",
        "parameters": 41_650_000,
        "source": "HFrEF dataset paper, approximately 41.65 M parameters",
    },
    {
        "kind": "reference",
        "model": "LGC2-Net PCG-only branch (dataset paper)",
        "parameters": 25_450_000,
        "source": "HFrEF dataset paper, approximately 25.45 M parameters",
    },
]


def _load_run(run_dir: Path, experiment: str) -> dict:
    subject_path = run_dir / "oof_subject_predictions.csv"
    if not subject_path.exists():
        raise FileNotFoundError(
            f"Missing {subject_path}. Run scripts/train.py --experiment {experiment} first."
        )
    subjects = pd.read_csv(subject_path)
    recording_path = run_dir / "recording_level_predictions.csv"
    recordings = pd.read_csv(recording_path) if recording_path.exists() else None
    threshold_path = run_dir / "thresholds.csv"
    if threshold_path.exists():
        thresholds = pd.read_csv(threshold_path)
        threshold = float(pd.to_numeric(thresholds["threshold"], errors="coerce").mean())
    else:
        threshold = 0.5
    summary_path = run_dir / "cv_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    return {
        "subjects": subjects,
        "recordings": recordings,
        "threshold": threshold,
        "summary": summary,
        "run_dir": run_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--run-name-e1", default="run")
    parser.add_argument("--run-name-e2", default="run")
    parser.add_argument("--run-name-e3", default=None,
                        help="Optional exploratory fusion run")
    parser.add_argument("--comparison-name", default=None)
    parser.add_argument("--no-paired", action="store_true",
                        help="Do not compute the paired-cohort comparison")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = configuration_from_args(args)
    initialise(args, cfg)
    profile = str(cfg["preprocessing"]["profile"])
    name = args.comparison_name or ("e1_e2_e3" if args.run_name_e3 else "e1_vs_e2")
    output = Path(args.out) if args.out else (
        Path(cfg["_project_root"]) / "results" / profile / "comparison" / name
    )
    ensure_dir(output)

    experiments = {}
    summaries = {}
    descriptions = {}
    for experiment, run_name in (
        ("e1", args.run_name_e1),
        ("e2", args.run_name_e2),
        ("e3", args.run_name_e3),
    ):
        if experiment == "e3" and not run_name:
            continue
        run_dir = results_path(cfg, experiment, run_name=run_name)
        try:
            payload = _load_run(run_dir, experiment)
        except FileNotFoundError as exc:
            LOGGER.error("%s", exc)
            return 2
        key, description = EXPERIMENT_LABELS[experiment]
        descriptions[key] = description
        experiments[key] = {
            "subjects": payload["subjects"],
            "recordings": payload["recordings"],
            "threshold": payload["threshold"],
        }
        summaries[key] = {**payload["summary"], "run_dir": str(run_dir), "description": description}

    comparison, paired = build_comparison_table(
        experiments,
        cfg,
        paired=not args.no_paired,
    )
    comparison["description"] = comparison["experiment"].map(descriptions)
    comparison.to_csv(output / "comparison_results.csv", index=False)
    paired.to_csv(output / "paired_comparison.csv", index=False)

    # Model-size comparison: implemented runs + published reference points.
    rows = []
    for key, summary in summaries.items():
        rows.append(
            {
                "kind": "implemented",
                "experiment": key,
                "model": summary.get("model_type", "pcg"),
                "profile": summary.get("profile", profile),
                "parameters": summary.get("n_parameters"),
                "input_length": summary.get("input_length"),
                "source": summary.get("run_dir", ""),
            }
        )
    rows.extend(REFERENCE_MODELS)
    pd.DataFrame(rows).to_csv(output / "model_parameter_comparison.csv", index=False)

    # Figures per experiment.
    try:
        from src.visualization import (
            plot_confusion_matrix,
            plot_model_comparison,
            plot_probability_distribution,
            plot_roc_pr,
        )

        for key, payload in experiments.items():
            subjects = payload["subjects"]
            threshold = float(payload["threshold"])
            metrics = compute_binary_metrics(
                subjects["label"].to_numpy(), subjects["prob"].to_numpy(), threshold
            )
            plot_confusion_matrix(
                np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]),
                output / f"confusion_{key}.png",
                title=f"{key} subject level (threshold {threshold:.2f})",
            )
            plot_roc_pr(
                subjects["label"].to_numpy(),
                subjects["prob"].to_numpy(),
                output / f"roc_pr_{key}.png",
                title=f"{key} pooled out-of-fold subjects",
                threshold=threshold,
                cfg=cfg,
            )
            plot_probability_distribution(
                subjects,
                output / f"probability_distribution_{key}.png",
                threshold=threshold,
                title=f"{key} subject-level probabilities",
            )
        plot_model_comparison(comparison, output / "comparison_subject.png",
                              level="subject", cohort="paired")
        plot_model_comparison(comparison, output / "comparison_recording.png",
                              level="recording", cohort="paired")
    except Exception as exc:
        LOGGER.warning("Could not create comparison figures: %s", exc)

    save_json(
        {
            "comparison_name": name,
            "profile": profile,
            "experiments": {
                key: {
                    "n_subjects": int(len(payload["subjects"])),
                    "n_positive_subjects": int((payload["subjects"]["label"] == 1).sum()),
                    "threshold": float(payload["threshold"]),
                    "description": descriptions.get(key, ""),
                }
                for key, payload in experiments.items()
            },
            "paired_cohort": (
                int(paired["n_subjects"].iloc[0]) if len(paired) else None
            ),
            "note": (
                "Paired comparison uses subjects evaluable in all experiments. "
                "External data are never included here."
            ),
        },
        output / "comparison_summary.json",
    )

    print()
    print("=" * 78)
    print(f"COMPARISON: {name} ({profile})")
    print("=" * 78)
    for cohort in ("all_available", "paired"):
        subset = comparison[(comparison["level"] == "subject") & (comparison["cohort"] == cohort)]
        if subset.empty:
            continue
        print(f"{cohort} cohort, subject level:")
        for _, row in subset.iterrows():
            print(
                f"  {row['experiment']:<5s} n={int(row['n']):>3d} "
                f"pos={int(row['n_positive']):>2d} thr={row['threshold']:.2f} | "
                f"AUROC={format_metric(row['roc_auc'])} "
                f"AUPRC={format_metric(row['pr_auc'])} "
                f"F1={format_metric(row['f1'])} "
                f"Sens={format_metric(row['recall_sensitivity'])} "
                f"Spec={format_metric(row['specificity'])}"
            )
    if len(paired):
        print()
        print("Paired bootstrap differences (E1 reference; positive favours E1):")
        for _, row in paired.iterrows():
            print(
                f"  {row['metric']:<20s} E1={format_metric(row['value_a'])} "
                f"vs {row['comparison_experiment']}={format_metric(row['value_b'])} "
                f"diff={row['difference']:+.3f} "
                f"[{format_metric(row['ci_lower'])}, {format_metric(row['ci_upper'])}]"
            )
    print()
    print(f"comparison_results.csv    : {output / 'comparison_results.csv'}")
    print(f"paired_comparison.csv     : {output / 'paired_comparison.csv'}")
    print(f"model parameters          : {output / 'model_parameter_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
