# Execution Guide (Windows PowerShell)

This is the practical, beginner-friendly guide to running the project. It assumes
Windows 10/11, PowerShell, and the project folder:

```
C:\Users\Hitesh m\OneDrive\Desktop\new_ds_project\ecg_pcg_gating_project
```

> **Important status note.** This guide describes how to run the project. It does
> **not** claim that the full experiments have already been run. In this
> repository, only smoke-scale verification has been executed. **Full five-fold
> E1/E2 training, the full external evaluation, and the `hfref_2000` sensitivity
> profile are NOT RUN.** Do not describe smoke metrics as findings.

---

## Before you start

- Every command below is run **from the project root** in **PowerShell**, with the
  virtual environment **active**. The prompt should begin with `(.venv)`.
- If a path contains spaces, keep it inside double quotes.
- The commands use relative paths (`scripts\...`, `config\...`). Do not run them
  from `scripts\` or from your home directory.
- The dataset is not bundled. Set the two environment variables in Step 9.
- The project targets **Python 3.10 or 3.11**. Newer interpreters often work, but
  they are not the documented target.
- The full pipeline can use several GB of disk. The cache is the largest
  generated artefact.

### Opening PowerShell in the right folder

Use one of these:

```powershell
cd "C:\Users\Hitesh m\OneDrive\Desktop\new_ds_project\ecg_pcg_gating_project"
```

or, in File Explorer, hold **Shift**, right-click the project folder, and choose
**Open PowerShell window here**.

Verify:

```powershell
Get-Location
Test-Path config\default.yaml
```

Expected: the location is the project folder and `True` is printed.

---

## 25-Step Execution Guide

### Step 1 — Open PowerShell in the project root

```powershell
cd "C:\Users\Hitesh m\OneDrive\Desktop\new_ds_project\ecg_pcg_gating_project"
Get-Location
Get-ChildItem
```

**Expected:** the path ends in `ecg_pcg_gating_project`, and you see `README.md`,
`config`, `scripts`, `src`, `tests`, `requirements.txt`, `pyproject.toml`.

**Success check:** `Test-Path config\default.yaml` prints `True`.

---

### Step 2 — Check that Python is installed

```powershell
py -3.11 --version
```

If that fails, try:

```powershell
py --list
python --version
```

**Expected:** `Python 3.11.x` (or another 3.10/3.11 interpreter). If `python`
opens the Microsoft Store, use the `py` launcher instead.

**Success check:** a Python version is printed without an error.

---

### Step 3 — Create the virtual environment

```powershell
py -3.11 -m venv .venv
```

**Expected:** a `.venv` folder appears. Nothing else is printed.

**Success check:**

```powershell
Test-Path .venv\Scripts\python.exe
```

prints `True`.

---

### Step 4 — Activate the virtual environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Expected:** the prompt changes to include `(.venv)`.

**Success check:**

```powershell
python -c "import sys; print(sys.executable)"
```

prints a path inside your project's `.venv\Scripts\python.exe`, not the system
Python.

> If you use `cmd.exe` instead of PowerShell, activate with
> `.venv\Scripts\activate.bat`.
>
> If you open a new terminal later, activate the environment again. Environment
> variables and activation do not persist across terminals.

---

### Step 5 — Upgrade the packaging tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

**Expected:** pip reports that it upgraded (or is already up to date).

**Success check:** the command exits with code `0`. In PowerShell:

```powershell
$LASTEXITCODE
```

prints `0`.

---

### Step 6 — Install the project requirements

```powershell
python -m pip install -r requirements.txt
```

**Expected:** NumPy, Pandas, SciPy, scikit-learn, PyTorch, SoundFile, PyWavelets,
Matplotlib, Seaborn, tqdm, PyYAML and pytest are installed. This is usually the
longest installation step.

**Success check:** the command exits with code `0` and the following prints `ok`:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python -c "import torch, numpy, pandas, scipy, sklearn, soundfile, pywt; print('ok')"
```

The `KMP_DUPLICATE_LIB_OK` line prevents the Anaconda/MKL + PyTorch OpenMP
conflict described in [Troubleshooting](#omp-error-15-on-windows). It is set
automatically inside the project scripts, but a direct `python -c` check is
outside that protection.

---

### Step 7 — Install the CUDA build of PyTorch (recommended for RTX 2050)

`requirements.txt` may install a CPU-only PyTorch wheel. For the RTX 2050
(~4 GB VRAM), install the CUDA build explicitly. Open
<https://pytorch.org/get-started/locally/> and use the selector
**Windows / Pip / Python / CUDA 12.x**, then run the generated command. An
example (the exact CUDA version may change):

```powershell
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

If you do not have a supported GPU or the CUDA install fails, skip this step and
use the CPU build from Step 6. The code runs on CPU; it is slower.

**Expected:** pip installs a `torch` build whose filename contains `cu12` (or
similar).

**Success check:**

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

prints a version and `True` when CUDA is working, or `False` for CPU.

---

### Step 8 — Verify the complete environment

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python -c "import sys, torch, numpy, pandas, scipy, sklearn, soundfile, pywt, matplotlib, seaborn, yaml; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Expected:** all imports succeed; Python is 3.10/3.11 (3.13 was used for the
development smoke tests but is not the documented target); the device is your
RTX 2050 or CPU.

**Success check:** no `ModuleNotFoundError`. Record the printed versions; they
belong in your report.

---

### Step 9 — Point the project at the datasets

```powershell
$env:HFREF_DATA_ROOT = "C:\Users\Hitesh m\Downloads\HFrEF_5_folds\HFrEF_5_folds"
$env:HFREF_EXTERNAL_ROOT = "C:\Users\Hitesh m\Downloads\HFrEF_community_scenario_test_set\HFrEF_community_scenario_test_set"

Test-Path $env:HFREF_DATA_ROOT
Test-Path $env:HFREF_EXTERNAL_ROOT
```

**Expected:** `True` for both paths.

**Success check:** both `Test-Path` calls print `True`. If not, fix the path; do
not continue until they do.

> These variables last only for the current PowerShell session. Re-run this step
> after opening a new terminal. You can also pass the paths with `--data-root`
> and `--external-root` on individual commands.

---

### Step 10 — Inspect the primary dataset

```powershell
python scripts\inspect_dataset.py
```

**Expected output (verified on the development copy):**

```
subjects             : 500
ECG/PCG pairs        : 2000
positive (LVEF<=40%) : 60
negative (LVEF>40%)  : 440
unpaired             : 0
unparsed WAV files   : 0
header errors        : 0
```

It also writes `results\inspection\main\`.

**Success check:**

```powershell
Test-Path results\inspection\main\validation_summary.json
```

prints `True`, and the printed subject count is 500 with 60 positives.

---

### Step 11 — Inspect the external community set

```powershell
python scripts\inspect_dataset.py --external
```

**Expected output:** 120 subjects, 480 pairs, 5 positive and 115 negative
subjects, 0 header errors. The external set has no official folds, so the fold
is reported as `unknown`.

**Success check:**

```powershell
Test-Path results\inspection\external\validation_summary.json
```

prints `True`, and the printed positive count is **5**. Remember this limitation
for every external result.

---

### Step 12 — Run the unit test suite

```powershell
python -m pytest -q
```

**Expected:** all tests pass. At the time of writing this is **65 passed** with
one benign scikit-learn single-label warning from a deliberately degenerate test.

**Success check:** the last line contains `passed` and the exit code is `0`:

```powershell
if ($LASTEXITCODE -eq 0) { "TESTS PASSED" } else { "TESTS FAILED - do not continue" }
```

If any test fails, stop and fix the environment before touching the data.

---

### Step 13 — Run the synthetic end-to-end smoke test

```powershell
python scripts\smoke_test_synthetic.py
```

**Expected:** the script generates a small synthetic dataset, builds a cache,
gates it, creates E1/E2 windows, trains five tiny folds, and writes a comparison.
The final lines include:

```
SYNTHETIC SMOKE TEST PASSED
parameters E1 / E2      : 1150 / 1150
```

Outputs go to `results_synthetic\paper_haar\comparison\smoke\`.

**Success check:** the command exits with code `0` and the `SYNTHETIC SMOKE TEST
PASSED` banner is printed.

> The synthetic metrics are **meaningless**. They verify plumbing only. Do not
> put them in a paper or report.

Optional external-path check:

```powershell
python scripts\smoke_test_synthetic.py --external
```

---

### Step 14 — Build a small real-data cache for preflight

```powershell
python scripts\prepare_cache.py --dataset-name smoke_main --max-recordings 200 --overwrite
```

**Expected:** 200 ECG/PCG pairs are read, validated and written to
`data\cache\paper_haar\smoke_main\`. The log ends with `Signal cache written to
...` and a summary with `n_pairs_ok` around 200.

**Success check:**

```powershell
Test-Path data\cache\paper_haar\smoke_main\metadata.csv
```

prints `True`. This is a **separate** cache from the full `main` cache, so the
full cache stays clean.

---

### Step 15 — Gate the real-data preflight subset

```powershell
python scripts\run_gating.py --dataset-name smoke_main --overwrite --plot-count 5 --example-count 2
```

**Expected (on the development subset):** 200/200 recordings accepted, about
6,000+ windows, about 350 windows rejected at the recording edges, mean window
success rate around 95 %. Exact numbers depend on the 200 recordings selected.

**Success check:**

```powershell
Test-Path data\gating\paper_haar\smoke_main\gating_summary.csv
```

prints `True`. Open the summary and confirm that `recording_ok` is mostly `True`
and that rejection reasons are sensible (edge windows are expected).

---

### Step 16 — Build the preflight E1 and E2 window tables

```powershell
python scripts\prepare_windows.py --dataset-name smoke_main --experiment e1 --max-windows-per-recording 6
python scripts\prepare_windows.py --dataset-name smoke_main --experiment e2 --max-windows-per-recording 6
```

**Expected:** about 1,200 windows per experiment from 200 recordings and 50
subjects. The script prints the window/hop, sample count, subject count and
class balance.

**Success check:**

```powershell
Test-Path data\windows\paper_haar\smoke_main\e1\windows.csv
Test-Path data\windows\paper_haar\smoke_main\e2\windows.csv
```

Both print `True`.

---

### Step 17 — Run preflight smoke training, evaluation and comparison

```powershell
python scripts\train.py --dataset-name smoke_main --experiment e1 --smoke
python scripts\train.py --dataset-name smoke_main --experiment e2 --smoke
python scripts\evaluate.py --experiment e1 --run-name smoke
python scripts\evaluate.py --experiment e2 --run-name smoke
python scripts\compare.py --run-name-e1 smoke --run-name-e2 smoke --comparison-name smoke
```

**Expected:** each command exits `0`; the model is `PCGCNN` with **1,150
parameters**; five tiny folds run (2 epochs by default for `--smoke`); comparison
files are written to `results\paper_haar\comparison\smoke\`.

**Success check:**

```powershell
Test-Path results\paper_haar\comparison\smoke\comparison_results.csv
```

prints `True`.

> Preflight metrics come from 1–2 epochs on a 50-subject subset. They are
> **plumbing evidence only**, not results.

---

### Step 18 — Build the full primary cache

```powershell
python scripts\prepare_cache.py --overwrite
```

**Expected:** all 2,000 pairs (4,000 WAV files) are read, validated, resampled
and band-pass filtered into `data\cache\paper_haar\main\`. The cache is roughly
0.4–0.6 GB (estimate; measure on your machine). Expect this to take a while; the
exact duration was **not measured at full scale** in this repository.

**Success check:**

```powershell
python -c "import json; d=json.load(open('data/cache/paper_haar/main/cache_config.json')); print('pairs ok', d['n_pairs_ok'], 'errors', d['n_pairs_error'])"
```

prints `pairs ok 2000 errors 0` on the verified dataset. If a few files error,
inspect `data\cache\paper_haar\main_index\recording_validation.csv` before
continuing.

---

### Step 19 — Run full gating and review the quality control

```powershell
python scripts\run_gating.py --overwrite --plot-count 10 --example-count 5
```

**Expected:** one JSON per recording and a summary for all 2,000 recordings in
`data\gating\paper_haar\main\`. The printed summary shows accepted recordings,
detected R peaks, accepted/rejected windows, success rate and rejection reasons.
QC figures show R peaks, T waves, S1/S2 candidates and window boundaries.

**Success check:**

```powershell
python -c "import pandas as pd; d=pd.read_csv('data/gating/paper_haar/main/gating_summary.csv'); print('recordings', len(d)); print('accepted', int(d['recording_ok'].sum())); print('windows', int(d['n_windows'].sum()))"
```

prints a recording count of 2,000. **Review the QC figures before training.**
Recordings that fail gating are excluded from E2; do not force them to pass.
There is no fallback to ungated PCG.

---

### Step 20 — Build the full E1 and E2 window tables

```powershell
python scripts\prepare_windows.py --experiment e1
python scripts\prepare_windows.py --experiment e2
```

**Expected:** roughly 75,000–80,000 E1 windows (39 windows per 30 s recording at
1.5 s / 0.75 s) and a similar order of magnitude for E2, depending on heart rate
and gating success. Both tables are written under
`data\windows\paper_haar\main\`.

**Success check:**

```powershell
Test-Path data\windows\paper_haar\main\e1\windows.csv
Test-Path data\windows\paper_haar\main\e2\windows.csv
```

Both print `True`. Note the E2 subject and window counts; they may be lower than
E1 if gating failed on some recordings. That difference is part of the result,
not something to hide.

---

### Step 21 — Train E1 (ungated) with the full five folds

```powershell
python scripts\train.py --experiment e1 --run-name run
```

**Expected:** five folds, up to 25 epochs each, early stopping on validation
subject AUPRC, batch size 128, AdamW, learning rate 0.004, mixed precision when
CUDA is available. Per-fold checkpoints, histories, predictions and
`cv_summary.json` are written to `results\paper_haar\e1\run\`.

**Success check:**

```powershell
Test-Path results\paper_haar\e1\run\cv_summary.json
Test-Path results\paper_haar\e1\run\fold_metrics.csv
```

Both print `True`. The log should show `Model PCGCNN | 1150 parameters`.

> This is a long run. The full five-fold training has **not** been executed in
> this repository, so no honest runtime estimate or metric can be given here.
> Measure it on your machine and record the wall-clock time.

---

### Step 22 — Train E2 (ECG-gated) with the full five folds

```powershell
python scripts\train.py --experiment e2 --run-name run
```

**Expected:** exactly the same model, loss, optimiser, folds and evaluation as
E1; only the window table differs. Outputs go to
`results\paper_haar\e2\run\`.

**Success check:**

```powershell
Test-Path results\paper_haar\e2\run\cv_summary.json
Test-Path results\paper_haar\e2\run\fold_metrics.csv
```

Both print `True`. Confirm that the parameter count is again 1,150.

---

### Step 23 — Evaluate E1 and E2

```powershell
python scripts\evaluate.py --experiment e1 --run-name run
python scripts\evaluate.py --experiment e2 --run-name run
```

**Expected:** subject-level and recording-level metrics with bootstrap intervals,
confusion matrices, ROC/PR curves and probability distributions. Files include
`metrics.csv`, `evaluation_summary.json` and PNG figures in each run directory.

**Success check:**

```powershell
Test-Path results\paper_haar\e1\run\metrics.csv
Test-Path results\paper_haar\e2\run\metrics.csv
```

Both print `True`. Read `metrics.csv`; do not report accuracy without
sensitivity, specificity, F1 and AUROC.

---

### Step 24 — Compare E1 and E2

```powershell
python scripts\compare.py --run-name-e1 run --run-name-e2 run --comparison-name main
```

**Expected:** `comparison_results.csv`, `paired_comparison.csv`,
`model_parameter_comparison.csv`, `comparison_summary.json`, confusion matrices,
ROC/PR curves, probability distributions and comparison bar charts in
`results\paper_haar\comparison\main\`.

**Success check:**

```powershell
Test-Path results\paper_haar\comparison\main\comparison_results.csv
Test-Path results\paper_haar\comparison\main\paired_comparison.csv
```

Both print `True`. The paired table is the confirmatory comparison; the
all-available rows are secondary.

---

### Step 25 — Evaluate the external set, audit, and archive

```powershell
python scripts\evaluate_external.py --experiment e2 --run-name run
```

Optionally repeat for E1:

```powershell
python scripts\evaluate_external.py --experiment e1 --run-name run
```

**Expected:** the script builds an external cache and gating output, evaluates
the five fold checkpoints, and writes `external_summary.json`,
`external_metrics.csv`, predictions and figures to
`results\paper_haar\e2\run\external\`. The threshold is the mean of the internal
validation thresholds and is **not** tuned on the external set.

**Success check:**

```powershell
Test-Path results\paper_haar\e2\run\external\external_summary.json
Test-Path results\paper_haar\e2\run\external\external_metrics.csv
```

Both print `True`. Confirm the printed subject count is 120 with **5 positive
subjects**, and quote the limitation with every external number.

Final audit checklist:

```powershell
Get-ChildItem results\paper_haar\e1\run -File | Select-Object Name
Get-ChildItem results\paper_haar\e2\run -File | Select-Object Name
Get-ChildItem results\paper_haar\comparison\main -File | Select-Object Name
```

Record:

- Python, PyTorch, CUDA and GPU driver versions;
- wall-clock time and peak GPU memory for E1 and E2;
- gating acceptance/rejection counts and reasons;
- the exact commands used;
- the fact that the external set has only five positives.

To archive the lightweight summary artefacts without raw data or checkpoints:

```powershell
Compress-Archive -Path results\paper_haar\e1\run\*.csv,results\paper_haar\e1\run\*.json,results\paper_haar\e2\run\*.csv,results\paper_haar\e2\run\*.json,results\paper_haar\comparison\main -DestinationPath results_summary.zip -Force
```

Do **not** archive `data\`, `data\cache\*`, or the checkpoint files into the
source repository.

---

## Troubleshooting

### `OMP: Error #15` on Windows

Anaconda's MKL and PyTorch can each supply an Intel OpenMP runtime. The project
sets `KMP_DUPLICATE_LIB_OK=TRUE` at import time in `src\__init__.py` and
`scripts\_common.py`, which is the documented workaround. If you still see the
error, set it for the current terminal before running anything:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

This is a compatibility measure, not a numerical setting.

### `Activate.ps1 cannot be loaded because running scripts is disabled`

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again. This changes the policy only for the current PowerShell
process.

### `python` opens the Microsoft Store

Use the Windows launcher:

```powershell
py -3.11 -m venv .venv
py -3.11 -m pip install -r requirements.txt
```

or install Python 3.11 from python.org and disable the Store alias.

### CUDA out of memory on the RTX 2050 (~4 GB)

Lower the batch size in `config\default.yaml`:

```yaml
training:
  batch_size: 4
```

You can also run on CPU with `--device cpu`, or reduce the number of windows for
a debug run with `--max-recordings` / `--max-windows-per-recording`.

### `FileNotFoundError: Signal cache not found`

Build the cache first:

```powershell
python scripts\prepare_cache.py --overwrite
```

For the preflight subset, use `--dataset-name smoke_main` consistently in every
command.

### `ValueError: ... is not a valid .npy file (missing header)`

An old raw-memmap cache is present. Rebuild it:

```powershell
python scripts\prepare_cache.py --overwrite
```

### `unrecognized arguments`

You are probably running from the wrong folder or have a quoting problem. Run
`Get-Location`, ensure it is the project root, and copy the command exactly. For
paths with spaces, keep the double quotes.

### `ModuleNotFoundError`

The virtual environment is not active, or requirements were installed into a
different interpreter. Activate `.\.venv\Scripts\Activate.ps1`, then re-run
`python -m pip install -r requirements.txt`.

### Full training is slow

That is expected. The model is tiny; the cost is dominated by reading WAV files,
resampling, filtering and gating. Use the preflight subset to verify the route,
and only then start the full run. Do not reduce the full run to a subset and call
it the final experiment.

---

## What success looks like at the end

- [ ] `pytest -q` passes (65 tests at the time of writing).
- [ ] The synthetic smoke test prints `SYNTHETIC SMOKE TEST PASSED`.
- [ ] Primary inspection reports 500 subjects, 60 positives, 2,000 pairs and 0 header errors.
- [ ] External inspection reports 120 subjects and 5 positives.
- [ ] Full gating summary exists and its rejection reasons have been reviewed.
- [ ] E1 and E2 window tables exist and E2 subject/window losses are documented.
- [ ] E1 and E2 both trained with 1,150 parameters and five official folds.
- [ ] `metrics.csv` and `evaluation_summary.json` exist for both experiments.
- [ ] `comparison_results.csv` and `paired_comparison.csv` exist.
- [ ] External evaluation used the frozen threshold and reports the five-positive limitation.
- [ ] The report distinguishes smoke/preflight outputs from full results.

---

## ANTIGRAVITY EXECUTION INSTRUCTIONS

This is the exact 10-step order for Antigravity. Run these in order and **stop
after Step 10**. Do **not** start the full five-fold training in this phase. The
full runs are described in Steps 18–25 above and require explicit human approval.

1. **Confirm the working directory and files.** From the project root, run:

   ```powershell
   Get-Location
   Get-ChildItem
   Test-Path config\default.yaml
   ```

   Expected: the project root, the expected folders/files, and `True`. If the
   location is wrong, stop and correct it.

2. **Check Python and create the virtual environment if it does not exist.**

   ```powershell
   py -3.11 --version
   if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.11 -m venv .venv }
   ```

   Expected: a Python 3.11 version and an existing `.venv\Scripts\python.exe`.

3. **Activate the environment and install the requirements.**

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   $env:KMP_DUPLICATE_LIB_OK = "TRUE"
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt
   ```

   Expected: the prompt shows `(.venv)` and the installation exits with code 0.
   The `KMP_DUPLICATE_LIB_OK` line is a Windows/Anaconda OpenMP compatibility
   workaround; the project scripts set it themselves, but direct Python checks
   in this terminal need it too. If CUDA is available and the user wants GPU
   execution, install the CUDA PyTorch wheel from the official selector before
   continuing; otherwise the CPU build is acceptable for this verification
   phase.

4. **Set the dataset environment variables.**

   ```powershell
   $env:HFREF_DATA_ROOT = "C:\Users\Hitesh m\Downloads\HFrEF_5_folds\HFrEF_5_folds"
   $env:HFREF_EXTERNAL_ROOT = "C:\Users\Hitesh m\Downloads\HFrEF_community_scenario_test_set\HFrEF_community_scenario_test_set"
   Test-Path $env:HFREF_DATA_ROOT
   Test-Path $env:HFREF_EXTERNAL_ROOT
   ```

   Expected: `True` for both. If either is `False`, stop and report the missing
   path; do not guess.

5. **Compile-check the source.**

   ```powershell
   python -m compileall src scripts tests
   ```

   Expected: no syntax errors and exit code 0.

6. **Run the unit tests.**

   ```powershell
   python -m pytest -q
   ```

   Expected: all tests pass (65 at the time of writing). If any test fails, stop
   and report the failure; do not proceed to the data pipeline.

7. **Inspect the primary dataset.**

   ```powershell
   python scripts\inspect_dataset.py
   ```

   Expected: 500 subjects, 60 positives, 440 negatives, 2,000 pairs, 0 header
   errors, and output in `results\inspection\main\`. If the counts differ, stop
   and report them instead of continuing.

8. **Run the synthetic end-to-end smoke test.**

   ```powershell
   python scripts\smoke_test_synthetic.py
   ```

   Expected: `SYNTHETIC SMOKE TEST PASSED`, 1,150 parameters for E1 and E2, and
   outputs under `results_synthetic\paper_haar\comparison\smoke\`. The metrics
   are meaningless; this step verifies plumbing only.

9. **Run the real-data preflight on a 200-recording subset.** Run these commands
   in order and stop if any exits non-zero:

   ```powershell
   python scripts\prepare_cache.py --dataset-name smoke_main --max-recordings 200 --overwrite
   python scripts\run_gating.py --dataset-name smoke_main --overwrite --plot-count 5 --example-count 2
   python scripts\prepare_windows.py --dataset-name smoke_main --experiment e1 --max-windows-per-recording 6
   python scripts\prepare_windows.py --dataset-name smoke_main --experiment e2 --max-windows-per-recording 6
   python scripts\train.py --dataset-name smoke_main --experiment e1 --smoke
   python scripts\train.py --dataset-name smoke_main --experiment e2 --smoke
   python scripts\evaluate.py --experiment e1 --run-name smoke
   python scripts\evaluate.py --experiment e2 --run-name smoke
   python scripts\compare.py --run-name-e1 smoke --run-name-e2 smoke --comparison-name smoke
   ```

   Expected: the cache and gating summaries are created; both window tables have
   about 1,200 windows and 50 subjects; both models have 1,150 parameters; the
   comparison CSV exists at
   `results\paper_haar\comparison\smoke\comparison_results.csv`. These are
   preflight outputs only.

10. **Stop and report.** Do **not** run any of the following in this phase:

    ```powershell
    python scripts\prepare_cache.py --overwrite
    python scripts\run_gating.py --overwrite
    python scripts\train.py --experiment e1 --run-name run
    python scripts\train.py --experiment e2 --run-name run
    python scripts\evaluate_external.py --experiment e2 --run-name run
    ```

    Summarise the exit codes, test count, inspection counts, gating acceptance
    rate, window counts and parameter counts. State explicitly that the full
    five-fold training and the full external evaluation are **NOT RUN**, and ask
    for approval before starting them.

---

## Where to go next

- Scientific rationale, hypotheses and deviations: [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md).
- Dataset layout, labels and environment variables: [`data/README.md`](data/README.md).
- Output file layout: [`results/README.md`](results/README.md).
- Project overview: [`README.md`](README.md).
