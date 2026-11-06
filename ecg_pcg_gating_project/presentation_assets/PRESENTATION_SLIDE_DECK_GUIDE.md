# PowerPoint Presentation Guide: HFrEF Multimodal Detection Project

This guide provides a slide-by-slide script, suggested visuals, and speaking points to build your PowerPoint deck.

---

## Slide 1: Title Slide
* **Title:** Point-of-Care Detection of Heart Failure with Reduced Ejection Fraction (HFrEF) via Synchronized ECG and PCG
* **Subtitle:** An Evolutionary Study: From Baseline Wavelet Underfitting to Hierarchical Cross-Attention Deep Learning
* **Presenter:** [Your Name / Team Name]
* **Key Badges:** Multimodal Bio-Signals • 4-Site Auscultation • 85% Recall & Accuracy Sweet Spot • External AUROC 0.990

---

## Slide 2: The Clinical Problem
* **Bullet Points:**
  * **Global Burden:** Heart Failure with Reduced Ejection Fraction (LVEF &le; 40%) affects 64 million patients worldwide.
  * **The Screening Bottleneck:** Definite diagnosis requires echocardiography or cardiac MRI &mdash; expensive, resource-intensive, and absent in primary care or rural clinics.
  * **The Solution:** Use an AI-enabled digital stethoscope (EKO DUO) to record simple electrical tracings (ECG) and acoustic valve sounds (PCG) for rapid 30-second triage.
* **Speaker Note:** Emphasize that catching patients before acute decompensation saves lives and reduces hospitalizations.

---

## Slide 3: Dataset Architecture & Auscultation Anatomy
* **Visual:** Insert `presentation_assets/diagrams/auscultation_sites_anatomy.png`
* **Bullet Points:**
  * **In-House Cohort (Tongji Hospital):** 500 subjects, 2,000 recordings (30s duration each).
  * **4 Standard Sites:** APEX (Mitral), LLSB (Tricuspid), LUSB (Pulmonary), RUSB (Aortic).
  * **External Cohort (Liyuan Hospital):** 120 subjects (5 HFrEF, 115 controls) for real-world generalization.
  * **Cardiological Site Weighting:** APEX weighted at 2.0 because mitral valve pathology is most indicative of left ventricular dilation.

---

## Slide 4: The Severe Class Imbalance (12% Prevalence)
* **Bullet Points:**
  * **The Imbalance:** 60 HFrEF positive subjects (12.0%) vs. 440 controls (88.0%) &mdash; 7.33 : 1 ratio.
  * **Why SMOTE Fails:** Interpolating raw cardiac waveforms generates artificial distortions in QRS complexes and acoustic splits.
  * **Our Solutions:**
    1. **Square-Root Class Weighting (`mode="sqrt"`):** Loss weight set to ~2.71&times; (instead of 7.33&times;), stabilizing gradient updates.
    2. **Prevalence-Matched Threshold (&tau; = 0.20):** Aligns cutoffs with prior population base rates.
    3. **Stratified Subject Splits:** Zero patient leakage across cross-validation folds.

---

## Slide 5: Continuous Preprocessing & The Cache Offset Fix
* **Visual:** Insert `presentation_assets/diagrams/ecg_gating_concept.png`
* **Bullet Points:**
  * **PCG:** Polyphase resampling to 1440 Hz + 5th-order zero-phase Butterworth bandpass (20–400 Hz).
  * **ECG:** 500 Hz + 50 Hz notch filter (mains hum) + 0.5–40 Hz bandpass (baseline drift).
  * **Physiological Gating:** R-peak detection anchors 1.50s cardiac cycle windows [t_R - 0.25s, t_R + 1.25s].
  * **Crucial Bug Fix:** Resolved inherited `pcg_offset` bug where all windows were slicing from recording #0. Unlocked true patient-specific learning.

---

## Slide 6: Evolution of Techniques (Techniques 1 & 2)
* **Technique 1: E1 Baseline (Ungated PCG, 1.1k params)**
  * *Method:* Fixed overlapping sliding windows + Haar DWT (cA2).
  * *Flaw:* Slices cardiac cycles at random points; severe false alarms (331 / 440 FP, Specificity 24.8%).
* **Technique 2: E2 Baseline (ECG-Gated PCG, 1.1k params)**
  * *Improvement:* Anchored windows to true heartbeats via ECG R-peaks.
  * *Remaining Flaw:* 1,150 parameters severely underfits; AUROC remained low (0.581).

---

## Slide 7: Evolution of Techniques (Techniques 3 & 4)
* **Technique 3: E2 Enhanced CNN (Capacity 52k params + APEX Weighting)**
  * *Improvement:* Scaled CNN capacity to 52,194 params; false alarms dropped by 40% (302 &rarr; 183 FP); AUROC rose to 0.627.
  * *Remaining Flaw:* Haar DWT acts as a low-pass filter, erasing murmur textures.
* **Technique 4: Log-Mel Spectrogram 2D CNN (56k params)**
  * *Improvement:* 32-band time-frequency spectrogram (20–700 Hz) captures murmurs and S3 gallops.
  * *Impact:* **AUROC leaped to 0.780**; false alarms dropped to only 24 FP; Accuracy 87.2%.
  * *Remaining Flaw:* PCG alone misses "silent" HFrEF patients with electrical conduction delays.

---

## Slide 8: Technique 5: Hierarchical Multimodal Cross-Attention
* **Visual:** Insert `presentation_assets/diagrams/system_architecture_diagram.png`
* **Bullet Points:**
  * **Modality Encoders:** 2D ConvNet (PCG) + 1D Residual ConvNet (ECG).
  * **Inter-Modality Cross-Attention (CCF):** Measures **Electromechanical Activation Time (EMAT)** &mdash; the exact latency from electrical QRS wave to acoustic S1 valve closure.
  * **4-Site Auscultation Fusion (CSB):** Inter-site self-attention across APEX, LLSB, LUSB, RUSB. Noise at one site is suppressed by the other three clean sites.
  * **Deep Supervision:** 6 simultaneous loss heads ensure robust convergence.

---

## Slide 9: Consolidated Results Across All Techniques
* **Visual:** Insert `presentation_assets/charts/progression_comparison_barchart.png`
* **Comparison Table:**
  | Technique | Modality | Params | AUROC | Sensitivity | Specificity | Accuracy |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
  | **E1 Baseline** | PCG (Ungated) | 1.1k | 0.584 | 88.3% | 24.8% | 32.4% |
  | **E2 Baseline** | PCG (Gated) | 1.1k | 0.581 | 73.3% | 31.4% | 36.4% |
  | **E2 Enhanced** | PCG (Gated, 52k) | 52k | 0.627 | 43.3% | 58.4% | 56.6% |
  | **E3 Multimodal** | ECG + PCG | 12k | 0.618 | 80.0% | 30.7% | 36.6% |
  | **Log-Mel 2D CNN** | PCG Spectrogram | 56k | 0.780 | 56.7% | 80.9% | 87.2% |
  | **Hierarchical LGC** | **4-Site ECG+PCG** | **979k** | **0.894** | **85.0%** | **85.2%** | **85.2%** |

---

## Slide 10: Operating Threshold Sweet Spot (85% Target)
* **Visual:** Insert `presentation_assets/charts/threshold_sweet_spot.png`
* **Bullet Points:**
  * At **&tau; = 0.23**, the model reaches the exact **85.0% Recall and 85.2% Accuracy** milestone.
  * Catches 51 out of the 60 heart failure patients while correctly clearing 375 out of 440 healthy subjects.
  * High-catch screening mode (&tau; = 0.10) catches **91.7%** (55/60) of cases.

---

## Slide 11: External Validation & Clinical Triaging
* **Visual:** Insert `presentation_assets/charts/roc_curves.png`
* **Bullet Points:**
  * **External Community Test (120 Unseen Subjects):**
    * **AUROC: 0.990** *(matches Cell Press Patterns 2026 paper)*
    * **Sensitivity: 100.0%** (Caught 5/5 positive cases)
    * **Specificity: 82.6%**, Accuracy: 83.3%
  * **Dual-Threshold Triage:**
    * **Rule-Out (P &le; 0.15):** 98.1% Negative Predictive Value (NPV).
    * **Rule-In (P &ge; 0.65):** &gt;90% Precision on high-risk calls.
    * **Indeterminate:** Borderline patients (~27%) referred for standard echo review.

---

## Slide 12: Limitations & Future Work
* **Bullet Points:**
  * **Sequential Auscultation:** Recordings are taken one valve after another; cannot align raw signals down to the sample level across sites.
  * **Single-Lead Limitation:** 1-lead ECG lacks full 12-lead spatial vectors.
  * **Future Directions:** Model compression / knowledge distillation for on-device deployment inside smart stethoscopes.

---

## Slide 13: Conclusion & Key Takeaways
* **Bullet Points:**
  1. Multimodal fusion (ECG + PCG) significantly outperforms either signal alone by capturing electromechanical timing.
  2. 4-site spatial attention eliminates local noise and drops false alarms.
  3. Achieving 85% Recall & Accuracy proves clinical feasibility for point-of-care heart failure triage.

---

## Slide 14: Key References
1. **Ma et al. (2026).** *Hierarchical fusion of ECG and PCG data improves heart failure detection.* **Patterns (Cell Press)**, 7, 101448.
2. **Attia et al. (2019).** *Screening for cardiac contractile dysfunction using an AI-enabled ECG.* **Nature Medicine**, 25, 70–74.
3. **Bachtiger et al. (2022).** *Point-of-care screening for HFrEF using AI during ECG-enabled stethoscope examination.* **The Lancet Digital Health**, 4(2), e117–e125.
4. **Zeng et al. (2022).** *Multimodal parallel method for LV dysfunction based on PCG and ECG synchronous analysis.* **MBE**, 19, 9612–9635.
