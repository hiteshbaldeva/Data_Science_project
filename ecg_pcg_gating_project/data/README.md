# Data directory

This folder is a placeholder for local data. **No raw recordings, caches,
windows, gating outputs or checkpoints should be committed to Git.**

The project reads the official HFrEF dataset from a path you provide at run
time. Nothing is copied into the repository by default.

## 1. Dataset used

The primary dataset is the public **HFrEF five-fold PCG dataset** distributed
with the HFrEF phonocardiogram paper (Zenodo). It contains synchronised
four-site ECG/PCG recordings and subject-level left-ventricular ejection
fraction (LVEF) labels.

The expected extracted layout is flexible; the discovery code searches nested
folders. A typical layout is:

```
HFrEF_5_folds/
├── fold_0/
│   ├── 1001025030_APEX_ECG.wav
│   ├── 1001025030_APEX_PCG.wav
│   ├── 1001025030_LLSB_ECG.wav
│   ├── 1001025030_LLSB_PCG.wav
│   ├── ...
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
├── fold_0_label.csv
├── fold_1_label.csv
├── fold_2_label.csv
├── fold_3_label.csv
└── fold_4_label.csv
```

Verified properties of the copy used during development:

| Property | Value |
|---|---|
| Subjects | 500 |
| Positive subjects (LVEF <= 40%) | 60 |
| Negative subjects (LVEF > 40%) | 440 |
| Auscultation sites per subject | 4 (APEX, LLSB, LUSB, RUSB) |
| ECG/PCG pairs | 2,000 |
| WAV files | 4,000 |
| Recording duration | 30 s |
| ECG sampling rate | 500 Hz |
| PCG sampling rate | 4,000 Hz |
| Format | mono PCM_16 WAV |
| Official folds | 5 subject-level folds, 100 subjects each, 12 positives each |

The separate **community-scenario test set** contains 120 subjects, 480 pairs,
and only **5 positive subjects**. It is kept completely separate from
cross-validation and is never used for training, early stopping, model
selection or threshold tuning.

## 2. Label definition

The label column in the official CSVs is LVEF. The code accepts either a
fraction (`0.35`) or a percentage (`35`) and normalises everything to percent.

```
class 0: LVEF > 40%   (preserved ejection fraction / non-HFrEF in this task)
class 1: LVEF <= 40%  (HFrEF)
```

The threshold is set by `data.lvef_threshold_percent` in
`config/default.yaml` (default `40.0`). Do not change it between E1 and E2:
the comparison would no longer be valid.

## 3. How to point the project at the data

Use environment variables (recommended) or CLI arguments.

PowerShell:

```powershell
$env:HFREF_DATA_ROOT = "C:\path\to\HFrEF_5_folds\HFrEF_5_folds"
$env:HFREF_EXTERNAL_ROOT = "C:\path\to\HFrEF_community_scenario_test_set\HFrEF_community_scenario_test_set"
```

The same values can be passed per command:

```powershell
python scripts/inspect_dataset.py --data-root "C:\path\to\HFrEF_5_folds\HFrEF_5_folds"
python scripts/inspect_dataset.py --external --external-root "C:\path\to\external"
```

Environment variables only last for the current PowerShell session. Re-set
them after opening a new terminal.

## 4. Generated data written here

| Path | Contents | Approximate size |
|---|---|---|
| `data/cache/<profile>/<dataset>/` | memory-mapped, preprocessed full recordings (`ecg.npy`, `pcg.npy`, metadata) | ~0.4-0.6 GB for the full primary set per profile |
| `data/gating/<profile>/<dataset>/` | one JSON per recording (R peaks, T waves, S1/S2, windows, rejection reasons), summary CSV, QC figures | tens of MB |
| `data/windows/<profile>/<dataset>/<experiment>/` | `windows.csv` used by training | ~5-20 MB per experiment |
| `data/synthetic_smoke/` | synthetic dataset generated for plumbing tests | a few MB |

These folders are in `.gitignore`. They can always be regenerated with the
scripts, so deleting them is safe (but costs time).

The cache is keyed by preprocessing profile and dataset name. Rebuilding with
different preprocessing settings requires `--overwrite` or a different
`--profile`; the code will not silently reuse a cache built with another
profile.

## 5. What not to do

- Do not train on the external community set.
- Do not choose a threshold on the external set.
- Do not merge E1 (ungated) windows into E2 (gated) windows.
- Do not combine the four auscultation sites into a single simultaneous
  multi-channel recording: they are sequential recordings of the same subject,
  not four synchronised channels.
- Do not report synthetic smoke-test metrics as scientific results.
