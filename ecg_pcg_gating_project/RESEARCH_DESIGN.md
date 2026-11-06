# Research Design

**Project:** Cardiovascular disease classification with ECG-gated PCG windows and a
lightweight programmable-AI-friendly 1-D CNN
**Task:** binary detection of heart failure with reduced ejection fraction (HFrEF)
**Status:** design and implementation complete; full experiments **NOT RUN**
(last updated 2026-11-02)

This document explains the scientific reasoning behind the code. It is written so
that a reader can judge whether the comparison is fair, what was reproduced from
the source papers, what was approximated, and what may and may not be concluded
from the outputs. Numerical results are deliberately absent: the full five-fold
runs have not been executed.

---

## 1. Research question

> In a subject-independent evaluation on the public HFrEF phonocardiogram
> dataset, does **ECG-guided segmentation of PCG recordings** improve
> subject-level HFrEF classification compared with **fixed, ungated PCG windows**,
> when the model, preprocessing profile, loss, optimiser, splits and evaluation
> are held identical?

The practical motivation is that a PCG recording contains several cardiac cycles,
and the diagnostic content (S1, S2, murmurs) is not uniformly distributed in
time. Ungated windows may land in systole, diastole, noise, or on a signal
boundary. An ECG channel, recorded synchronously with the PCG, provides R-peak
timestamps that let every window start at the same physiological phase. If
gating helps, the gain should appear without increasing model capacity, which
matters for a small accelerator.

A second, engineering motivation is that the classifier must remain small enough
for a programmable AI accelerator. The primary CNN therefore has **1,150
parameters** for the binary task, and E1 and E2 share it exactly.

### 1.1 Primary and secondary hypotheses

| ID | Hypothesis | Primary endpoint |
|---|---|---|
| **H1** | E2 (gated) has a higher subject-level ROC-AUC than E1 (ungated) | pooled out-of-fold subject ROC-AUC |
| **H2** | The paired subject-level difference in ROC-AUC (E2 − E1) is positive | paired bootstrap mean difference and 95% interval |
| **H3** (secondary) | E2 has a higher subject-level F1 at the validation-selected threshold | pooled out-of-fold subject F1 |
| **H0** | There is no difference between E1 and E2 | — |

H1 and H2 are the confirmatory analysis. F1, sensitivity, specificity and PR-AUC
are secondary because they depend on a threshold, and the threshold is chosen on
a small internal validation split. The optional E3 fusion experiment is
**exploratory only** and is not part of the confirmatory family.

No multiplicity correction is applied to the secondary metrics; they are
reported descriptively. With 60 positive subjects in the primary set, the study
is powered to detect only fairly large differences. The paired design reduces
between-subject variance but does not create information that is not in the data.

---

## 2. Data

### 2.1 Primary dataset

The primary dataset is the public **HFrEF five-fold PCG dataset** distributed
with the HFrEF phonocardiogram paper (Zenodo). Verified properties of the copy
used during development:

| Property | Value |
|---|---|
| Subjects | 500 |
| Positive subjects (LVEF ≤ 40 %) | 60 |
| Negative subjects (LVEF > 40 %) | 440 |
| Auscultation sites per subject | 4: APEX, LLSB, LUSB, RUSB |
| ECG/PCG pairs | 2,000 |
| WAV files | 4,000 |
| Duration per recording | 30 s |
| ECG sampling rate | 500 Hz |
| PCG sampling rate | 4,000 Hz |
| Format | mono PCM_16 WAV |
| Official folds | 5 subject-level folds, 100 subjects each, 12 positives each |

The four sites are **sequential recordings of the same subject**, not four
simultaneous channels. The code therefore treats each `subject × site` pair as
an independent recording, keeps all four in the same fold, and never stacks them
as a multi-channel input.

### 2.2 Label definition

Labels are read from the official `fold_i_label.csv` files, whose rows are keyed
by `{ID}_{SITE}` and contain `lvef_fraction` (for example `0.35`). The code
accepts a fraction or a percentage and normalises to percent.

```
class 0: LVEF >  40 %   (preserved ejection fraction)
class 1: LVEF <= 40 %   (HFrEF)
```

The threshold is `data.lvef_threshold_percent` (default `40.0`). It is not
changed between E1 and E2. This is a **binary LVEF-defined** task, not a
multi-class heart-failure diagnosis. A subject with LVEF exactly 40 % is
positive (`<=`), matching the usual HFrEF definition.

The dataset provides one LVEF value per subject, repeated across sites. Subject
labels are therefore constant across that subject's recordings. Any label
disagreement within a subject is treated as a data error and surfaced during
inspection rather than silently resolved.

### 2.3 External community set

The separate community-scenario test set contains 120 subjects, 480 pairs and
only **5 positive subjects**. It is kept completely separate:

- never used for training, early stopping, model selection or threshold tuning;
- evaluated once, after the internal cross-validation is complete;
- threshold frozen to the mean of the internal validation thresholds;
- predictions averaged over the five fold checkpoints;
- reported with an explicit five-positive limitation.

No result from this set can support a precise sensitivity estimate. It is a
plumbing and limited-generalisation check, not a validation study.

### 2.4 Synchronisation

The dataset declares ECG and PCG to be synchronised within a recording. The
inspection script estimates the lag by maximising the cross-correlation between
the PCG envelope and the ECG R-peak train. Observed lags on the primary set are
approximately −0.02 to −0.045 s, consistent with zero plus filter/onset effects.

The lag is a **diagnostic**, not a correction that is blindly applied. The
default `gating.delay.apply` is `false`. If it is enabled, the code only applies
a lag that is within `max_abs_seconds` and has a sufficient correlation
contrast, and it records the applied value per recording. No fixed delay (for
example a hard-coded 1.5 s) is ever inserted.

---

## 3. Relationship to the two source papers

Two papers inform the design and must not be conflated.

### 3.1 Original ECG-gating PCG paper

This paper contributes:

- PCG band-pass filtering and resampling;
- a two-level Haar discrete wavelet transform (DWT) with approximation
  coefficients;
- z-score normalisation;
- a lightweight 1-D CNN, **Configuration 5** in its Table IV;
- the idea of using ECG R peaks to place PCG windows;
- R-peak detection based on a derivative, an initial threshold from the
  third-highest peak, a half-threshold selection rule, a minimum-RR constraint,
  RR-outlier re-search and T-wave timing equations.

### 3.2 HFrEF dataset paper and official implementation

This paper contributes:

- the dataset and the five subject-level folds;
- the HFrEF label definition;
- the published PCG preprocessing: 2,000 Hz, 25–400 Hz band-pass, 1.28 s
  segments with 50 % overlap, z-score normalisation;
- a large multimodal reference model (LGC2-Net, ~41.65 M parameters; a PCG-only
  variant ~25.45 M), which this project deliberately does **not** reproduce.

### 3.3 Mapping from paper statements to implementation

| Paper element | Implementation | Exactness |
|---|---|---|
| PCG band-pass + resample | `paper_haar`: 1,440 Hz, 20–400 Hz Butterworth order 5, zero-phase | paper-inspired; exact order/edge handling unspecified |
| Two-level Haar DWT | `pywt.wavedec(..., wavelet="haar", level=2, mode="periodization")`, keep `cA2` | approximation choice documented; the paper does not state which coefficients are kept |
| Z-score normalisation | per-window z-score using **training-subject** mean/std stored in the checkpoint | exact intent; the paper does not state the statistics source |
| Configuration 5 CNN | three Conv1D+BN+ReLU stages, AvgPool(16), 1×1 Conv+BN+ReLU, flatten, FC | layer sequence and parameter counts match Table IV |
| ECG R-peak gating | R peaks detected at native ECG rate, window anchored before R | paper-inspired; peak-search and window anchor are not fully specified |
| T-wave equations | `detect_t_waves` with configurable offsets | approximate; the paper gives equations but not all constants |
| S1/S2 search | envelope peaks in configured intervals around R and T | approximate; intervals are explicit implementation choices |
| HFrEF preprocessing | `hfref_2000` profile: 2,000 Hz, 25–400 Hz, 1.28 s / 0.64 s, no DWT | matches the dataset paper's stated settings |
| HFrEF model | **not reproduced** | out of scope by design |

The **main comparison uses `paper_haar`**, because it is the profile that
reproduces the original paper's tensor shape and parameter count. `hfref_2000`
is a pre-specified sensitivity analysis, not the primary analysis.

---

## 4. Experiments

| ID | Window source | Extra input | Role |
|---|---|---|---|
| **E1** | fixed windows tiled from the start of each PCG recording, 50 % overlap | none | ungated baseline |
| **E2** | one window per detected R peak, anchored at R − 0.25 s, same duration as E1 | none | primary gated experiment |
| **E3** | identical to E2 | time-aligned ECG window → small CNN branch → concatenation | optional exploratory fusion |

### 4.1 Why E1 and E2 are comparable

E1 and E2 must differ **only** in how the window start is chosen. The code
enforces this by:

- using the same preprocessing profile and therefore the same PCG resampling,
  band-pass filter and DWT;
- using the same window duration, taken from the active profile (1.5 s for
  `paper_haar`);
- using the same `WindowDataset`, model class, loss, optimiser, epoch budget,
  early-stopping rule, class weights and evaluation functions;
- using the same subject folds and the same subject-level aggregation;
- applying the same minimum-window filter.

E1 uses a hop of 0.75 s (50 % overlap). E2 uses one window per R peak, so its
windows are not evenly spaced; this is the intended difference. E2 does not
subsample windows by default, so a 30 s recording with ~35 beats can contribute
more windows than E1. Window counts are reported, and the subject-level
aggregation is a mean, so a subject with more windows does not automatically
receive more weight than another subject.

### 4.2 Optional E3

E3 adds a small ECG branch (three convolutions, adaptive pooling, 32-dimensional
embedding) concatenated with the 128 PCG features and a 32-unit fusion head
(about 12 k parameters on synthetic data). It is **not** LGC2-Net and contains
no attention modules. E3 is reported only if it is run; it cannot replace E2 as
the primary experiment.

---

## 5. PCG preprocessing profiles

The two profiles are selected with `preprocessing.profile` or `--profile`. They
are never mixed silently, and the cache is keyed by profile.

| Setting | `paper_haar` (main) | `hfref_2000` (sensitivity) |
|---|---|---|
| Resampling target | 1,440 Hz | 2,000 Hz |
| Band-pass | 20–400 Hz, Butterworth order 5, zero-phase | 25–400 Hz, order 5, zero-phase |
| DWT | 2-level Haar, keep `cA2` | disabled |
| Window | 1.5 s | 1.28 s |
| Hop (E1) | 0.75 s | 0.64 s |
| Samples into CNN | 540 | 2,560 |
| Pooling | `AvgPool(16)` → exactly 16 frames | adaptive average pooling → 16 frames |
| Purpose | original-paper-inspired main comparison | dataset-paper sensitivity check |

### 5.1 DWT shape

For a 1.5 s window at 1,440 Hz, the pre-DWT length is 2,160 samples. A two-level
Haar DWT with periodisation and approximation-only output gives

```
ceil(ceil(2160 / 2) / 2) = ceil(1080 / 2) = 540 samples
```

i.e. one quarter of the input. This is the input length that produces a
`8 × 16 = 128`-dimensional flattened feature vector and reproduces the paper's
parameter count.

The `dwt_keep: all` option (concatenating `cA2`, `cD2`, `cD1`) exists only for a
sensitivity analysis. It changes the input length and is not the primary
setting.

### 5.2 Filtering order and phase

Filtering is zero-phase (`sosfiltfilt`) so that gating timestamps and window
content are not shifted by the filter. Edge padding is handled by SciPy; the
first and last few milliseconds are not used as a basis for any claim.

### 5.3 Normalisation

Z-score statistics are estimated from **training subjects only**, using up to
`training.stats_max_windows` windows, and stored in every checkpoint. The same
statistics are applied to validation, test and external windows. No subject's
own mean or standard deviation is used to normalise that subject's windows.
This prevents a subtle form of leakage that is common when per-recording
normalisation is used and then evaluated on the same recording.

---

## 6. ECG preprocessing and gating

### 6.1 ECG preprocessing

- The ECG is kept at its native **500 Hz** timebase for R-peak detection.
  Upsampling to the PCG rate would add no physiological information and would
  change the peak detector's behaviour.
- A 5–25 Hz band-pass copy is used for detection; the original ECG is retained
  for the optional E3 branch (1–40 Hz).
- Quality checks flag flatline, clipping, implausible amplitude and too few
  beats. A failed quality check is recorded as a gating failure.
- R peaks are refined to the local extremum on the filtered signal, and
  too-close peaks are removed using a minimum-RR constraint.

### 6.2 Two explicitly named detectors

| Method | Description | Status |
|---|---|---|
| `paper_inspired` | derivative, third-highest-peak threshold initialisation, half-threshold selection, min-RR constraint, RR-outlier re-search, T-wave equations | primary; approximates the paper |
| `simple_rpeak` | Pan-Tompkins-style energy detector | simplified baseline; explicitly named |

The method is written into every gating JSON and summary. The project never
claims that `paper_inspired` is an exact replica of an under-specified algorithm.

### 6.3 Timebase conversion

This is the central correctness requirement:

1. R peaks are detected on the ECG at `fs_ecg` (500 Hz), producing sample
   indices `r_ecg`.
2. Times are computed as `r_time = r_ecg / fs_ecg`.
3. PCG sample positions are computed as `r_pcg = round(r_time * fs_pcg)`
   (4,000 Hz before resampling; the cached PCG is at the profile rate).
4. Window boundaries are computed in **seconds** relative to the R time and then
   converted with `fs_pcg`:
   `start_pcg = round((r_time - pre_r_seconds) * fs_pcg)`,
   `end_pcg = start_pcg + round(window_seconds * fs_pcg)`.

No ECG sample index is ever reused directly as a PCG index. The conversion is
covered by unit tests with different sampling rates.

### 6.4 Window placement and rejection

- Default anchor: R peak, `pre_r_seconds = 0.25`, duration = active profile
  window (1.5 s for `paper_haar`).
- Optional anchor `s1` uses `pre_s1_seconds = 0.10`; it is not the primary
  setting.
- A window that starts before the recording or ends after it is **rejected** and
  the reason is recorded (`window_start_before_recording`,
  `window_end_after_recording`).
- A recording is accepted only if it has at least `quality.min_rpeaks` R peaks
  and at least `quality.min_successful_windows` accepted windows, and if the
  rejection fraction is below `quality.max_rejection_fraction`.
- If gating fails, the recording is **excluded from E2**. There is **no
  fallback to ungated PCG**. This is essential: silently substituting ungated
  windows would contaminate the comparison.

### 6.5 S1/S2 and delay

S1/S2 are searched in the PCG envelope in configurable intervals around the R
peak and the T wave. They are used for quality control and optional validation,
and they are stored per beat. They do **not** silently move the window; moving
the anchor is an explicit configuration change.

The ECG–PCG lag is estimated and stored. It is not applied by default. This
respects the dataset's declared synchrony and avoids inserting a blind delay.

### 6.6 What is saved for audit

For every recording, a JSON file stores:

- subject, site, fold, record ID, method;
- R-peak sample indices and times;
- T-wave times;
- S1/S2 candidates and the intervals used;
- accepted windows (start/end samples and times, anchor);
- rejected windows with reasons;
- delay estimate and whether it was applied;
- quality flags and the recording-level decision.

The summary CSV stores counts, success rate, heart-rate statistics, lag and
reasons. QC figures show R peaks, T waves, S1/S2 candidates, window boundaries
and example gated PCG windows. These artefacts are the visual quality control
for the gating step.

---

## 7. Model

### 7.1 Configuration 5 reconstruction

| Layer | Output shape (540 input) | Parameters |
|---|---|---|
| Conv1d(1 → 4, k=4, s=2) + BN + ReLU | 4 × 269 | 20 + 8 = 28 |
| Conv1d(4 → 8, k=4, s=1) + BN + ReLU | 8 × 266 | 136 + 16 = 152 |
| Conv1d(8 → 16, k=4, s=1) + BN + ReLU | 16 × 263 | 528 + 32 = 560 |
| AvgPool1d(16) | 16 × 16 | 0 |
| Conv1d(16 → 8, k=1) + BN + ReLU | 8 × 16 | 136 + 16 = 152 |
| Flatten | 128 | 0 |
| Linear(128 → 2) | 2 | 258 |
| **Total (binary)** | | **1,150** |

The five-class version used in the original paper replaces the final layer with
`Linear(128 → 5)` and has **1,537** parameters, matching the reported Table IV
value. The binary head is the only architectural change.

### 7.2 Pooling for non-540 inputs

The paper's `AvgPool(16)` assumes a post-convolution length of 256. The
`hfref_2000` profile produces a much longer post-convolution sequence. Rather
than silently changing the classifier head, the model uses adaptive average
pooling to produce the same 16 frames when `AvgPool(16)` would not. The chosen
mode (`avg_kernel` or `adaptive`) is recorded in the model card and checkpoint.

### 7.3 E3 fusion model

The optional ECG branch has three convolutional stages, adaptive pooling to 8
frames, and a 32-dimensional embedding. It is concatenated with the 128 PCG
features and passed through a 32-unit fusion head. No attention, no recurrence,
no pretrained weights. It is deliberately small.

### 7.4 Model summary

Every training run logs and saves the parameter count, trainable parameter
count, input length, flatten dimension, pooling mode and profile. The comparison
stage writes `model_parameter_comparison.csv`, including published reference
points (LGC2-Net and its PCG-only variant) clearly marked as **not implemented
here**.

---

## 8. Training protocol

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Loss | cross-entropy with balanced class weights |
| Optional sampling | weighted sampler (off by default) |
| Optimiser | AdamW |
| Learning rate | 4e-3 (sqrt-scaled from 1e-3 at batch 8) |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Epochs | 25 (early stopping may end sooner) |
| Scheduler | ReduceLROnPlateau, patience 3, factor 0.5 |
| Early stopping | patience 7, monitor validation subject AUPRC |
| Mixed precision | enabled when CUDA is available |
| Gradient clipping | 1.0 |
| Validation fraction | 15 % of training subjects, stratified |
| Seeds | 42 (configurable) |
| DataLoader workers | 0 (safest on Windows) |

Class imbalance is handled with balanced class weights because the positive
prevalence is 12 %. Accuracy is never used as the deciding metric. Weighted
sampling is available but off by default, so that E1 and E2 remain identical in
this respect.

### 8.1 Splits

For each official fold `f`:

- **test** = all subjects in fold `f`;
- **train + validation** = all subjects in the other four folds;
- **validation** = a 15 % subject-level, label-stratified holdout from the
  training folds;
- **train** = the remainder.

All recordings and windows of a subject stay in one split. `check_no_subject_leakage`
raises if a subject appears in more than one split. The test fold is not used
for early stopping or threshold selection.

If the official fold column is missing or incomplete for some subjects, the code
generates subject-level stratified folds and records
`fallback_stratified_subject_folds` (or `fallback_modulo_folds` for degenerate
data) in `cv_summary.json`. The fallback is never silent.

### 8.2 Threshold selection

A threshold grid from 0.01 to 0.99 in steps of 0.01 is searched on **validation
subject-level** predictions. The default objective is F1; Youden's J and balanced
accuracy are alternatives. The selected threshold is applied unchanged to the
test fold and saved in the checkpoint and `thresholds.csv`.

For the pooled out-of-fold view and the external evaluation, the threshold is
the mean of the fold-specific validation thresholds. No test label is used to
choose a threshold.

---

## 9. Evaluation protocol

### 9.1 Aggregation

- Window probabilities are averaged per **subject** (primary) and per
  **recording** (secondary). Maximum probability and window counts are kept for
  diagnostics.
- Subjects with fewer than `evaluation.min_windows_per_subject` windows are
  excluded from subject-level aggregation and counted in the output. The default
  is 1, so every subject with at least one window is retained.
- E1 and E2 are compared on the **intersection** of evaluable subjects (paired
  cohort). The all-available cohort is reported as a secondary view so that the
  cost of gating failures is visible.

### 9.2 Metrics

At both subject and recording level:

- accuracy;
- precision;
- sensitivity / recall;
- specificity;
- F1;
- balanced accuracy;
- negative predictive value;
- ROC-AUC;
- PR-AUC (average precision);
- Brier score;
- confusion counts (TN, FP, FN, TP).

Confusion matrices, ROC curves, PR curves and probability distributions are
plotted. Accuracy is always reported next to the class-balanced metrics, never
alone.

### 9.3 Uncertainty

Confidence intervals use a **subject-level percentile bootstrap** with 1,000
resamples and a 95 % interval by default. Resampling subjects (not windows)
respects the clustering of windows within subjects.

The paired comparison uses a paired bootstrap: the same resampled subject set is
used for E1 and E2, and the distribution of the difference is reported. This is
an interval for the difference in this sample, not a formal pre-registered
hypothesis test. No p-value is invented from a bootstrap that was not designed
for one.

### 9.4 What is not done

- No threshold tuning on test or external data.
- No feature selection on the full dataset.
- No early stopping on the test fold.
- No per-subject normalisation.
- No reporting of the best epoch chosen on test.
- No synthetic or smoke metric is presented as a scientific result.

---

## 10. External evaluation policy

The external set is processed only after the internal cross-validation is
complete:

1. Build the cache and gate the external recordings with the same profile and
   gating configuration.
2. Load the five fold checkpoints.
3. For each fold checkpoint, predict external windows using that checkpoint's
   training-only normalisation statistics.
4. Average the five checkpoint probabilities per window, then aggregate windows
   to subjects.
5. Apply the mean internal validation threshold, unchanged.
6. Report metrics with bootstrap intervals and the five-positive limitation.

No external label is used for training, early stopping, checkpoint selection,
threshold selection or feature selection. If the external gate fails on a
recording, that recording is excluded; there is no fallback.

---

## 11. Deviations, assumptions and threats to validity

### 11.1 Deviations from the original paper

| Item | Deviation | Consequence |
|---|---|---|
| DWT coefficients | `cA2` only is kept | The paper does not specify; this choice gives the 540-sample input and 1,537-parameter five-class model |
| Peak-search function | Implemented from the described rules, not copied from unavailable code | `paper_inspired` is an approximation, not an exact replica |
| Third-highest-peak threshold | Implemented as described, but exact indexing/edge rules are choices | R-peak counts may differ slightly from the paper |
| Window anchor | R − 0.25 s | The paper does not fully specify the anchor; this is explicit and applied identically to all subjects |
| S1/S2 intervals | Explicit configuration values | Used for QC, not for silently moving windows |
| Output classes | Binary instead of five | Required by the HFrEF task; the five-class parameter count is still reproduced for validation |
| Dataset | HFrEF set, not the original paper's dataset | The comparison is about gating on this dataset, not a replication of the paper's accuracy |

### 11.2 Deviations from the HFrEF dataset paper

| Item | Deviation | Consequence |
|---|---|---|
| Model | Small Config-5 CNN instead of LGC2-Net | Intentional: the project studies gating with a fixed small model, not multimodal accuracy |
| Preprocessing | `paper_haar` for the main comparison | Required to reproduce the original paper's tensor shape; `hfref_2000` is the sensitivity profile |
| Segmentation | ECG-gated, not fixed 1.28 s | This is the intervention being studied |
| Sample weighting | Balanced class weights | Appropriate for 12 % prevalence; applied identically to E1 and E2 |

### 11.3 Statistical limitations

- Only **60 positive subjects** in the primary set. Subject-level AUC and F1
  intervals will be wide.
- Only **5 positive subjects** in the external set. Sensitivity and PR-AUC there
  are essentially anecdotal; the interval will span most of [0, 1].
- Five-fold cross-validation on the same 500 subjects is not an independent
  external validation.
- The paired comparison reduces variance but cannot correct for a small number
  of positive subjects.
- The threshold is selected on a 15 % internal validation split, which contains
  roughly 9 positive subjects per fold. Threshold instability is expected and is
  why F1 is secondary to AUC.
- No formal power calculation was performed; the study should be read as
  descriptive and exploratory with a pre-specified primary endpoint.

### 11.4 Gating-specific threats

- If gating fails disproportionately on difficult recordings, E2 may appear
  better or worse for reasons unrelated to gating. The paired cohort and the
  gating summary are reported to make this visible.
- R-peak detection errors can shift or reject windows. The QC figures and
  rejection reasons are the audit trail.
- The four sites are sequential; a site with poor signal may dominate a
  subject's window average if it contributes more windows. Recording-level
  results and window counts are reported alongside subject-level results.
- The lag diagnostic is not a substitute for a proper synchronisation study.

### 11.5 What would falsify the hypothesis

H1/H2 would not be supported if the paired subject-level AUC difference is
centred near zero or negative, or if its confidence interval includes zero. A
negative result is still informative: it would suggest that, for this dataset
and this small model, fixed overlapping windows already capture enough
phase-independent information, or that gating failures and window placement
offset any benefit. The project must report that outcome honestly rather than
tune until a positive result appears.

---

## 12. Reproducibility

- One YAML configuration controls profiles, gating, model, training and
  evaluation.
- Seeds are set for Python, NumPy and PyTorch; deterministic algorithms are
  requested where available.
- Every run writes `effective_config.yaml` next to its outputs.
- Checkpoints contain model weights, normalisation statistics, threshold, split
  summary and model card.
- Caches are keyed by preprocessing profile and dataset name and carry a
  configuration fingerprint; a cache built with different settings is not
  silently reused.
- The synthetic generator and smoke test verify the entire pipeline without the
  real data.
- The full experiments are long-running and are **not** claimed to have been run
  in this repository.

---

## 13. Reporting plan

When the full runs are complete, the report will contain:

1. Dataset inspection and fold counts.
2. Gating audit: recordings accepted/rejected, windows accepted/rejected,
   reasons, lag, heart rate, QC figures.
3. Model summary and parameter comparison.
4. E1 and E2 fold-level and pooled out-of-fold subject/recording metrics with
   bootstrap intervals.
5. Paired E1-vs-E2 differences with paired bootstrap intervals.
6. Confusion matrices, ROC/PR curves, probability distributions and training
   curves.
7. External evaluation with the frozen threshold and the five-positive caveat.
8. A limitations section that states what was approximated and what was not run.

Until those runs exist, every metrics file under `results/` is a smoke-test
artefact and must be labelled as such.
