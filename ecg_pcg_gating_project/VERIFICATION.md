# Verification Record

**Date:** 2026-11-02
**Purpose:** record what was actually executed in this repository, so that no
smoke-test output is mistaken for a scientific result.
**Bottom line:** the code and pipeline are verified at smoke/preflight scale.
**Full five-fold E1/E2 training and the full external evaluation have NOT been
run.**

---

## 1. Environment used for verification

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26300) |
| Python | 3.13.9 (Anaconda; the documented target is 3.10/3.11) |
| PyTorch | 2.14.1+cpu |
| CUDA available in the verification environment | **No** (CPU only) |
| NumPy | 2.3.5 |
| Pandas | 2.3.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.7.2 |
| SoundFile | 0.14.0 |
| PyWavelets | 1.8.0 |
| Matplotlib | 3.10.6 |
| Seaborn | 0.13.2 |
| PyYAML | 6.0.3 |

The RTX 2050 was **not** used for these checks (the verification interpreter had
a CPU-only PyTorch build). GPU execution, mixed precision and the 4 GB VRAM
budget are implemented but were not exercised end-to-end at full scale.

Anaconda's MKL and PyTorch can each load an Intel OpenMP runtime, which produces
`OMP: Error #15`. The project sets `KMP_DUPLICATE_LIB_OK=TRUE` inside
`src/__init__.py` and `scripts/_common.py`. Direct `python -c` checks outside the
package need the same environment variable; this is documented in
`EXECUTION.md`.

---

## 2. Checks that were executed

### 2.1 Syntax, imports and tests

| Check | Command | Observed result |
|---|---|---|
| Byte-compile | `python -m compileall -q src scripts tests` | exit 0 |
| Imports | import every module in `src` | `ALL SRC IMPORTS OK` |
| CLI help | `python scripts/<script>.py --help` for all 10 scripts | 10/10 exit 0 |
| Unit tests | `python -m pytest -q` | **65 passed** in 21.34 s |

The pytest suite covers:

- dataset discovery, label parsing and pairing;
- PCG resampling, filtering, Haar DWT and normalisation;
- ECG quality checks, R-peak detection and T-wave detection;
- gating timebase conversion, window placement and rejection;
- window-table construction and `WindowDataset` shapes;
- CNN forward pass, output shapes and parameter counts;
- subject-level split separation and leakage guards;
- metrics, threshold selection, aggregation and bootstrap;
- a tiny training loop;
- synthetic generators and pipeline integration.

### 2.2 Synthetic end-to-end smoke test

Command:

```powershell
python scripts\smoke_test_synthetic.py
```

Observed result: **exit 0**, final banner:

```
SYNTHETIC SMOKE TEST PASSED
profile                 : paper_haar
synthetic subjects      : 10
gated windows           : 536
parameters E1 / E2      : 1150 / 1150
```

The run covered: synthetic dataset → cache → gating → E1/E2/E3 window tables →
five tiny folds → evaluation → paired comparison. Synthetic metrics are
meaningless and are not reproduced here as findings.

The synthetic external path was also exercised:

```powershell
python scripts\smoke_test_synthetic.py --external
```

Observed result: **exit 0**, external cache, gating and window table created.

The optional E3 fusion path was exercised on synthetic data for one fold:
**exit 0**, 12,080 parameters. This is plumbing evidence only.

### 2.3 Real dataset inspection

Primary set:

```powershell
python scripts\inspect_dataset.py --data-root "C:\Users\Hitesh m\Downloads\HFrEF_5_folds\HFrEF_5_folds"
```

Observed:

| Quantity | Value |
|---|---|
| Subjects | 500 |
| Positive (LVEF ≤ 40 %) | 60 |
| Negative (LVEF > 40 %) | 440 |
| ECG/PCG pairs | 2,000 |
| Unpaired | 0 |
| Unparsed WAV files | 0 |
| Header errors | 0 |
| Fold counts | 100 subjects / 12 positives per fold |

External set:

```powershell
python scripts\inspect_dataset.py --external --external-root "C:\Users\Hitesh m\Downloads\HFrEF_community_scenario_test_set\HFrEF_community_scenario_test_set"
```

Observed:

| Quantity | Value |
|---|---|
| Subjects | 120 |
| Positive (LVEF ≤ 40 %) | 5 |
| Negative (LVEF > 40 %) | 115 |
| ECG/PCG pairs | 480 |
| Unpaired | 0 |
| Header errors | 0 |

Outputs are under `results/inspection/main/` and `results/inspection/external/`.

### 2.4 Real-data preflight on a 200-recording subset

A separate cache name (`smoke_main`) was used so that the full `main` cache
remains clean.

| Stage | Command | Observed result |
|---|---|---|
| Cache | `prepare_cache.py --dataset-name smoke_main --max-recordings 200 --overwrite` | 200/200 pairs cached, exit 0 |
| Gating | `run_gating.py --dataset-name smoke_main --overwrite` | 200/200 recordings accepted; 6,494 windows; 352 rejected; 94.9 % window success |
| E1 windows | `prepare_windows.py --dataset-name smoke_main --experiment e1 --max-windows-per-recording 6` | 1,200 windows, 50 subjects |
| E2 windows | `prepare_windows.py --dataset-name smoke_main --experiment e2 --max-windows-per-recording 6` | 1,200 windows, 50 subjects |
| E1 smoke training | `train.py --dataset-name smoke_main --experiment e1 --smoke` | 5 folds, 2 epochs, 1,150 parameters, exit 0 |
| E2 smoke training | `train.py --dataset-name smoke_main --experiment e2 --smoke` | 5 folds, 2 epochs, 1,150 parameters, exit 0 |
| Evaluation | `evaluate.py --experiment e1/e2 --run-name smoke` | exit 0, metrics and figures written |
| Comparison | `compare.py --run-name-e1 smoke --run-name-e2 smoke --comparison-name smoke` | exit 0, comparison tables written |

Gating rejection reasons were only edge effects:

```
window_end_after_recording   296
window_start_before_recording 56
```

The preflight class balance was 72 positive windows out of 1,200. The smoke
models saw 1–2 epochs on 50 subjects, so **their metrics are not results**. They
are retained under `results/paper_haar/e1/smoke`, `results/paper_haar/e2/smoke`
and `results/paper_haar/comparison/smoke` only to demonstrate that the output
files are produced correctly.

### 2.5 External evaluation path with smoke checkpoints

Command:

```powershell
python scripts\evaluate_external.py --experiment e2 --run-name smoke `
  --external-root "C:\Users\Hitesh m\Downloads\HFrEF_community_scenario_test_set\HFrEF_community_scenario_test_set" `
  --out "results\paper_haar\e2\smoke\external_smoke"
```

Observed: exit 0, 120 subjects, 480 recordings, 5 positive subjects, threshold
`0.236` frozen from internal validation, outputs under
`results/paper_haar/e2/smoke/external_smoke/`.

The external metrics from these 2-epoch smoke checkpoints are **meaningless**
and must not be reported. The run verifies only that the external path loads the
cache, gates the recordings, loads all five checkpoints, applies the frozen
threshold and writes the expected files.

The observed smoke values (quarantined here only for completeness) were subject
AUROC ≈ 0.55, sensitivity 1.00 and specificity 0.00. A specificity of zero with
five positives and a low threshold is exactly the kind of unstable behaviour
expected from an untrained smoke model; it is not evidence about gating.

---

## 3. What was NOT run

| Item | Status |
|---|---|
| Full primary cache (2,000 pairs) | **NOT RUN** |
| Full primary gating (2,000 recordings) | **NOT RUN** |
| Full E1 five-fold training (25 epochs) | **NOT RUN** |
| Full E2 five-fold training (25 epochs) | **NOT RUN** |
| Full external evaluation with final models | **NOT RUN** |
| `hfref_2000` sensitivity profile end-to-end | **NOT RUN** |
| E3 on the real dataset | **NOT RUN** |
| GPU / mixed-precision timing and memory measurement | **NOT RUN** |

Consequently:

- There is no scientific E1-vs-E2 result in this repository.
- No claim about the benefit or harm of ECG gating is supported yet.
- No external generalisation claim is supported yet.
- The smoke metrics under `results/` are plumbing artefacts, not findings.

---

## 4. Reproducing the verification

From the project root, with the environment active:

```powershell
python -m compileall -q src scripts tests
python -m pytest -q
python scripts\smoke_test_synthetic.py
python scripts\inspect_dataset.py
python scripts\inspect_dataset.py --external
```

For the real-data preflight, see Steps 14–17 of `EXECUTION.md`. For the full
experiments, see Steps 18–25, but do not start them until the preflight and
quality-control review are complete.

---

## 5. Honest reporting rule

Any future report must state which of the following produced the numbers:

1. synthetic smoke test (`results_synthetic/`);
2. real-data 200-recording preflight (`--dataset-name smoke_main`, `--run-name smoke`);
3. full five-fold run (`--run-name run`);
4. external evaluation (`external/` vs `external_smoke/`).

Categories 1 and 2 are engineering checks. Only category 3, produced by the full
official folds with the final configuration, may be presented as the main
internal result. Category 4 is a limited generalisation check with five positive
subjects and must always carry that caveat.
