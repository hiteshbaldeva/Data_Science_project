# ECG-Gated PCG for HFrEF Classification

A reproducible research codebase that compares **ungated PCG windows (E1)**
with **ECG-gated PCG windows (E2)** for binary classification of heart failure
with reduced ejection fraction (HFrEF) on the public HFrEF phonocardiogram
dataset. An optional **E3** experiment adds a small ECG branch to the gated PCG
branch.

The PCG model is a faithful reconstruction of **Configuration 5** from the
original ECG-gating PCG paper: a lightweight 1-D CNN with **1,150 parameters**
for the binary task. E1 and E2 use exactly the same architecture, loss,
optimiser, subject splits and evaluation code; only the window definition
differs, so any difference can be attributed to gating rather than capacity.

> **Status:** the pipeline is implemented and verified end-to-end at
> smoke-test scale, including a 200-recording real-data subset. **The full
> five-fold experiments have NOT been run.** No scientific E1-vs-E2 result is
> claimed. See [Verification status](#verification-status).

## Contents

- [Research question](#research-question)
- [Experiments](#experiments)
- [Dataset](#dataset)
- [Signal processing](#signal-processing)
- [ECG gating](#ecg-gating)
- [Model](#model)
- [Evaluation protocol](#evaluation-protocol)
- [Project structure](#project-structure)
- [Installation and quick start](#installation-and-quick-start)
- [Outputs](#outputs)
- [Verification status](#verification-status)
- [Limitations and what this project does not do](#limitations-and-what-this-project-does-not-do)
- [Documentation map](#documentation-map)

## Research question

Does ECG-guided segmentation of phonocardiogram recordings improve
subject-level HFrEF classification compared with fixed, ungated PCG windows,
when the model and evaluation are held identical?

The comparison is paired at the subject level. Because gating can fail on some
recordings, E2 may cover fewer subjects than E1; the primary comparison is
therefore restricted to subjects evaluable in both experiments, and the
all-available cohort is reported as a secondary view. The external community
test set is never used for training, model selection or threshold tuning.

## Experiments

| ID | PCG windows | Extra input | Purpose |
|---|---|---|---|
| **E1** | fixed 1.5 s windows, 0.75 s hop, tiled from the start of each recording | none | ungated baseline |
| **E2** | windows anchored 0.25 s before each ECG R peak, same duration as E1 | none | primary ECG-gated experiment |
| **E3** | identical to E2 | time-aligned ECG window through a small CNN branch | optional exploratory fusion |

E3 is deliberately small and is **not** a reproduction of LGC2-Net. It exists
only to test whether a little ECG context adds information beyond gating.

## Dataset

Primary set (verified during development):

- 500 subjects, 60 with LVEF <= 40% (class 1) and 440 with LVEF > 40% (class 0);
- four sequential auscultation sites per subject (APEX, LLSB, LUSB, RUSB);
- 2,000 synchronised ECG/PCG pairs, 4,000 WAV files, 30 s each;
- ECG 500 Hz, PCG 4,000 Hz, mono PCM_16;
- official five subject-level folds, 100 subjects and 12 positives per fold.

External community set: 120 subjects, 480 pairs, only **5 positive subjects**.
It is used once, after all model selection is complete, with the threshold
frozen from internal validation. Its sensitivity and AUPRC estimates have very
wide uncertainty and are reported only as a limited generalisation check.

The dataset is not bundled. See [`data/README.md`](data/README.md) for the
expected layout, label definition and environment variables.

## Signal processing

Two PCG profiles are implemented and kept strictly separate. The active profile
is selected in `config/default.yaml` or with `--profile`.

| Setting | `paper_haar` (primary) | `hfref_2000` (sensitivity) |
|---|---|---|
| PCG resampling | 1,440 Hz | 2,000 Hz |
| Band-pass | 20-400 Hz Butterworth, zero-phase | 25-400 Hz Butterworth, zero-phase |
| DWT | 2-level Haar, keep cA2 | none |
| Window / hop | 1.5 s / 0.75 s | 1.28 s / 0.64 s |
| Samples into CNN | 540 | 2,560 |
| Purpose | original-paper-inspired main comparison | dataset-paper sensitivity check |

For `paper_haar`, a 1.5 s window at 1,440 Hz contains 2,160 samples. A
two-level Haar DWT with periodisation and approximation-only output gives
`ceil(ceil(2160/2)/2) = 540` samples, i.e. one quarter of the input. This is
the shape that reproduces the paper's 128-dimensional feature head and its
1,537-parameter five-class network.

DWT and z-score normalisation are applied **per window, after** full-recording
resampling and band-pass filtering. Z-score statistics are estimated from the
training subjects only and stored in every checkpoint.

ECG preprocessing keeps the native 500 Hz timebase. R peaks are detected on a
5-25 Hz band-passed copy at 500 Hz; the ECG is never upsampled to the PCG rate
for detection. Window boundaries are converted to PCG samples with the PCG
sampling rate, not the ECG rate.

## ECG gating

For each subject/site recording:

1. Validate that ECG and PCG come from the same pair and have compatible
   durations (default tolerance 0.25 s).
2. Assess ECG quality (flatline, clipping, implausible amplitude).
3. Detect R peaks at 500 Hz with one of two explicitly named methods:
   - `paper_inspired`: derivative, third-highest-peak threshold initialisation,
     half-threshold selection, minimum-RR constraint, RR-outlier re-search, and
     T-wave timing equations from the original paper;
   - `simple_rpeak`: a Pan-Tompkins-style energy detector used as a simplified
     baseline.
4. Estimate T waves and search the PCG envelope for S1/S2 in configurable
   intervals. S1/S2 are used for quality control and optional validation; they
   do not silently move the window.
5. Create a window of the **profile's** duration, anchored 0.25 s before the R
   peak (or optionally before S1). Reject windows that start before the
   recording or end after it; rejections and reasons are recorded.
6. An optional lag diagnostic estimates the ECG-PCG delay but **does not apply
   it** by default, because the dataset declares within-recording synchrony. A
   fixed blind delay is never inserted.

Recordings are never combined across sites. If gating fails, the recording is
excluded from E2; there is **no fallback to ungated PCG**. One JSON file per
recording stores R-peak samples and times, T waves, S1/S2, accepted windows,
rejected windows with reasons, delay diagnostics and quality flags. Summary
CSVs and QC figures are written alongside.

## Model

Configuration 5 from the original paper, reconstructed layer by layer:

| Layer | Output shape | Parameters |
|---|---|---|
| Conv1d(1 -> 4, k=4, s=2) + BN + ReLU | 4 x 269 | 20 + 8 = 28 |
| Conv1d(4 -> 8, k=4, s=1) + BN + ReLU | 8 x 266 | 136 + 16 = 152 |
| Conv1d(8 -> 16, k=4, s=1) + BN + ReLU | 16 x 263 | 528 + 32 = 560 |
| AvgPool1d(16) | 16 x 16 | 0 |
| Conv1d(16 -> 8, k=1) + BN + ReLU | 8 x 16 | 136 + 16 = 152 |
| Flatten | 128 | 0 |
| Linear(128 -> 2) | 2 | 258 |
| **Total (binary)** | | **1,150** |

The five-class version used by the original paper has `Linear(128 -> 5)` and
**1,537** parameters, matching the reported Table IV value. The binary head is
the only change. For non-540 inputs (for example the 2,560-sample sensitivity
profile), adaptive average pooling keeps the same 16-frame, 128-dimensional
feature head; the pooling mode is written into the model card and checkpoint.

The optional E3 model adds a small ECG branch (three convolutions, adaptive
pooling, 32-dimensional embedding) concatenated with the 128 PCG features and a
32-unit fusion head. It is intentionally much smaller than LGC2-Net and uses no
attention modules.

## Evaluation protocol

- **Subject-independent folds.** The official five folds are used when present.
  If fold information is genuinely missing, a subject-stratified fallback is
  generated and recorded in `cv_summary.json`; the fallback is never silent.
- **No leakage.** All recordings and windows from a subject stay in one split.
  A validation subset (15% of training subjects, stratified) is carved out of
  the training folds for early stopping and threshold selection. The test fold
  is untouched until final evaluation.
- **Normalisation from training only.** Mean/std are estimated on the training
  subset and reused for validation, test and external data.
- **Window-to-subject aggregation.** The primary decision is the arithmetic
  mean of window probabilities per subject, matching a subject-level reading.
  Maximum probability and window counts are kept for diagnostics.
- **Class imbalance.** Cross-entropy uses balanced class weights; optional
  weighted sampling is available. Accuracy is reported but never alone.
- **Threshold selection.** A threshold grid is searched on validation
  subject-level predictions (F1 by default; Youden or balanced accuracy are
  alternatives). The chosen threshold is applied unchanged to the test fold.
- **Metrics.** Accuracy, precision, sensitivity/recall, specificity, F1,
  balanced accuracy, NPV, ROC-AUC, PR-AUC and Brier score, at both subject and
  recording level. Confidence intervals use a subject-level percentile
  bootstrap.
- **Paired comparison.** E1 and E2 are compared on the intersection of
  evaluable subjects with a paired bootstrap of metric differences. The
  interval is reported as evidence in this sample, not as a formal
  pre-registered hypothesis test.
- **External set.** No threshold tuning, no model selection, no early stopping.
  The threshold is the mean of the internal validation thresholds and
  predictions are averaged over the fold checkpoints. The five-positive
  limitation is reported explicitly.

## Project structure

```
ecg_pcg_gating_project/
├── config/
│   └── default.yaml              # all profiles, gating, model, training, evaluation settings
├── data/
│   └── README.md                 # dataset layout and generated-data locations
├── results/
│   └── README.md                 # output layout and current status
├── scripts/
│   ├── _common.py                # shared CLI/config/logging helpers
│   ├── inspect_dataset.py        # discovery, labels, WAV headers, sync diagnostic
│   ├── prepare_cache.py          # preprocessed memmap cache
│   ├── run_gating.py             # ECG gating + QC figures
│   ├── prepare_windows.py        # E1/E2/E3 window tables
│   ├── train.py                  # subject-independent cross-validation
│   ├── evaluate.py               # metrics and figures from saved predictions
│   ├── compare.py                # paired E1/E2 comparison
│   ├── evaluate_external.py      # external community set, frozen threshold
│   ├── make_synthetic_dataset.py # synthetic data generator
│   └── smoke_test_synthetic.py   # full synthetic plumbing test
├── src/
│   ├── config.py                 # YAML loading, profiles, path resolution
│   ├── utils.py                  # paths, seeds, JSON, logging
│   ├── synthetic.py              # synthetic ECG/PCG generators
│   ├── data_inspection.py        # dataset discovery, labels, validation, sync
│   ├── data_loader.py            # WAV I/O and memmap cache
│   ├── preprocessing.py          # resampling, filters, Haar DWT, normalisation
│   ├── ecg_processing.py         # quality, R peaks, T waves
│   ├── pcg_processing.py         # envelope, S1/S2 search, window quality
│   ├── gating.py                 # timestamp conversion and window construction
│   ├── dataset.py                # window tables and PyTorch datasets
│   ├── model.py                  # Config-5 PCG CNN, ECG branch, fusion model
│   ├── training.py               # splits, training loop, checkpoints, CV
│   ├── evaluation.py             # metrics, aggregation, bootstrap, comparison
│   ├── visualization.py          # QC, ROC/PR, confusion, training figures
│   └── pipeline.py               # stage orchestration and external evaluation
├── tests/                        # pytest suite (65 tests)
├── README.md
├── RESEARCH_DESIGN.md
├── EXECUTION.md
├── VERIFICATION.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Installation and quick start

See **[`EXECUTION.md`](EXECUTION.md)** for the complete beginner-friendly
PowerShell walkthrough with 25 numbered steps and a separate 10-step
"ANTIGRAVITY EXECUTION INSTRUCTIONS" section.

Short version:

```powershell
cd "C:\Users\<you>\Desktop\new_ds_project\ecg_pcg_gating_project"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python scripts/smoke_test_synthetic.py
```

Then set the dataset paths and follow the stages:

```powershell
$env:HFREF_DATA_ROOT = "C:\path\to\HFrEF_5_folds\HFrEF_5_folds"
$env:HFREF_EXTERNAL_ROOT = "C:\path\to\HFrEF_community_scenario_test_set\HFrEF_community_scenario_test_set"

python scripts/inspect_dataset.py
python scripts/prepare_cache.py --overwrite
python scripts/run_gating.py --overwrite
python scripts/prepare_windows.py --experiment e1
python scripts/prepare_windows.py --experiment e2
python scripts/train.py --experiment e1
python scripts/train.py --experiment e2
python scripts/evaluate.py --experiment e1
python scripts/evaluate.py --experiment e2
python scripts/compare.py
```

Do not start the full five-fold runs until the inspection, tests and synthetic
smoke test pass. The full runs are long; see EXECUTION.md for expected
run times and disk use.

## Outputs

Every run writes `effective_config.yaml`, per-fold checkpoints, per-fold
metrics, subject/recording/window predictions, training curves and a
`cv_summary.json`. The comparison stage writes:

- `comparison_results.csv`;
- `paired_comparison.csv`;
- `model_parameter_comparison.csv`;
- confusion matrices, ROC/PR curves, probability distributions and comparison
  bar charts.

See [`results/README.md`](results/README.md) for the complete layout.

## Verification status

What was actually executed during development (2026-11-02):

| Check | Result |
|---|---|
| `python -m compileall src scripts tests` | passed |
| `pytest -q` | **65 passed** |
| Synthetic 10-subject end-to-end (cache → gating → E1/E2 → 5 folds → comparison) | passed |
| Synthetic external path (`smoke_test_synthetic.py --external`) | passed |
| Optional E3 fusion, 1 synthetic fold | passed (plumbing only) |
| Real primary dataset inspection | 500 subjects, 60 positives, 2,000 pairs, 0 header errors |
| Real external dataset inspection | 120 subjects, 5 positives, 480 pairs, 0 header errors |
| Real 200-recording subset: cache → gating → E1/E2 windows | passed; 100% recordings gated, 6,494 windows |
| Real subset smoke training + evaluation + comparison | passed |
| Real external evaluation path (smoke checkpoints) | passed |
| **Full five-fold E1/E2 training** | **NOT RUN** |
| **Full external evaluation with final models** | **NOT RUN** |
| **`hfref_2000` sensitivity profile end-to-end** | **NOT RUN** |

The smoke test environment used Anaconda Python 3.13 on Windows; the documented
target is Python 3.10/3.11. The code avoids 3.12+-only syntax. Full training
was not run, so no scientific metrics are available and none are invented here.

The exact commands, versions and observed outputs for every row above are in
[`VERIFICATION.md`](VERIFICATION.md). That file also lists the experiments that
were **not** run.

## Limitations and what this project does not do

- It does **not** reproduce LGC2-Net or any large multimodal architecture.
- It does **not** use the four auscultation sites as simultaneous channels.
- It does **not** tune anything on the external set.
- It does **not** claim a significant gating benefit before the paired
  five-fold experiment is run.
- The exact original paper's DWT coefficient choice, peak-search function and
  S1/S2 interval definitions are not fully specified; the implementation choices
  are documented as a **paper-inspired approximation**, not an exact replica.
- The external set has only 5 positive subjects; all sensitivity-like metrics
  have very wide confidence intervals.
- The dataset labels are LVEF-based; the task is HFrEF vs preserved ejection
  fraction, not a multi-class heart-failure diagnosis.

## Documentation map

- [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md) - scientific design, hypotheses,
  deviations from the papers, statistical plan, threats to validity.
- [`EXECUTION.md`](EXECUTION.md) - step-by-step PowerShell instructions,
  expected outputs, verification checks and the Antigravity run order.
- [`VERIFICATION.md`](VERIFICATION.md) - exact commands, environment versions
  and observed outputs for the checks that were executed, plus the list of
  experiments that were not run.
- [`data/README.md`](data/README.md) - dataset layout, labels, environment
  variables, disk use.
- [`results/README.md`](results/README.md) - artefact layout and status.
