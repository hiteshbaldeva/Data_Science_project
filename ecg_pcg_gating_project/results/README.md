# Results directory

All machine-readable metrics and figures are written here by the scripts. The
layout is deterministic so that E1 and E2 can be compared without guesswork.

```
results/
├── figures/                 # optional top-level copies for reports
├── metrics/                 # optional top-level copies for reports
├── checkpoints/             # optional top-level copies for reports
├── inspection/
│   ├── main/                # discovery + header validation for the primary set
│   └── external/            # same for the external community set
└── <profile>/
    ├── e1/<run_name>/       # ungated PCG
    ├── e2/<run_name>/       # ECG-gated PCG
    ├── e3/<run_name>/       # optional gated PCG + small ECG branch
    └── comparison/<name>/   # E1 vs E2 (and optional E3) outputs
```

`<profile>` is `paper_haar` for the main comparison or `hfref_2000` for the
dataset-paper sensitivity profile.

## Per-run artefacts

Each `e1/e2/e3` run directory contains:

| File | Meaning |
|---|---|
| `effective_config.yaml` | exact configuration used (reproducibility) |
| `fold_0/` … `fold_4/` | one directory per fold |
| `fold_*/checkpoint.pt` | model weights, normalisation stats, threshold, split summary |
| `fold_*/fold_metrics.json` | subject and recording metrics for that fold |
| `fold_*/history.csv` | per-epoch loss and validation metrics |
| `fold_*/training_curves.png` | loss / AUROC / AUPRC / F1 curves |
| `fold_*/test_subject_predictions.csv` | held-out subject predictions |
| `fold_*/test_recording_predictions.csv` | held-out recording predictions |
| `fold_*/validation_subject_predictions.csv` | validation predictions used for early stopping and threshold choice |
| `fold_metrics.csv` | one row per fold and split |
| `thresholds.csv` | validation-selected threshold per fold |
| `subject_level_predictions.csv` | validation + test subject predictions |
| `oof_subject_predictions.csv` | out-of-fold test predictions (one row per subject) |
| `recording_level_predictions.csv` | out-of-fold recording predictions |
| `window_predictions_test.csv` | optional per-window predictions |
| `oof_metrics.csv` | pooled out-of-fold metrics with bootstrap CIs |
| `cv_summary.json` | fold source, parameter count, thresholds, mean/std metrics |
| `evaluation_summary.json` | created by `scripts/evaluate.py` |
| `metrics.csv` | subject/recording metrics from `scripts/evaluate.py` |
| `confusion_matrix_subject.png`, `roc_pr_subject.png`, `probability_distribution.png` | evaluation figures |

## Comparison artefacts

`results/<profile>/comparison/<name>/` contains:

| File | Meaning |
|---|---|
| `comparison_results.csv` | all experiments, subject and recording level, all-available and paired cohorts |
| `paired_comparison.csv` | paired subject-level bootstrap differences (E1 reference) |
| `model_parameter_comparison.csv` | implemented parameter counts plus published reference points |
| `comparison_summary.json` | cohort sizes and thresholds used |
| `confusion_E1.png`, `confusion_E2.png`, … | confusion matrices |
| `roc_pr_E1.png`, `roc_pr_E2.png`, … | ROC and PR curves with bootstrap bands |
| `probability_distribution_E1.png`, … | subject probability histograms |
| `comparison_subject.png`, `comparison_recording.png` | metric bar charts with CIs |

## Current status (important)

This repository has been **verified end-to-end only at smoke-test scale**:

- synthetic 10-subject dataset: cache → gating → E1/E2 windows → 5-fold
  training → paired comparison → external-path smoke: **passed**;
- real dataset subset of 200 recordings (50 subjects): cache → gating → E1/E2
  windows → smoke training → evaluation/comparison: **passed**;
- real external set (120 subjects, 5 positives): cache → gating → external
  evaluation path: **passed** using smoke checkpoints.

**The full five-fold experiments have NOT been run.** No scientific E1-vs-E2
result exists in this repository yet. Any metrics currently under `results/`
come from 1-2 epoch smoke runs and must not be reported as findings. See the
root `README.md` for the exact verification status.

When the full experiments are run, keep smoke outputs under a separate
`--run-name smoke` and use `--run-name run` for the real analysis.
