# Presentation Package: Multimodal ECG-PCG Heart Failure Screening (HFrEF)

This archive contains all the materials needed to create and deliver an academic or professional presentation/slide deck on the Multimodal ECG-PCG Heart Failure Detection project.

---

## Directory Structure

```
presentation_assets/
├── COMPLETE_PRESENTATION_REPORT.pdf    <-- Comprehensive 7-page PDF report with embedded figures and tables
├── PRESENTATION_SLIDE_DECK_GUIDE.md    <-- Slide-by-slide presenter notes and slide layout guide
├── README.md                           <-- Package overview (this file)
│
├── diagrams/                           <-- High-resolution architecture & clinical concept diagrams (PNG)
│   ├── system_architecture_diagram.png   (Hierarchical Cross-Attention Net with 4-Site Spatial Fusion)
│   ├── ecg_gating_concept.png            (Continuous Pan-Tompkins ECG gating and PCG cycle anchoring)
│   └── auscultation_sites_anatomy.png    (4 standard cardiac auscultation positions on chest)
│
├── charts/                             <-- High-resolution publication plots & visual results (PNG)
│   ├── progression_comparison_barchart.png (AUROC, Recall, Specificity, Accuracy across all 6 models)
│   ├── threshold_sweet_spot.png          (Operating threshold sweep highlighting 85% Recall / 85% Accuracy)
│   └── roc_curves.png                    (Evolution of ROC curves: Baseline -> Log-Mel -> Cross-Attention)
│
└── data/                               <-- Raw prediction CSVs and performance summaries
    ├── all_models_progression_metrics.csv (Consolidated progression metrics table across all techniques)
    ├── threshold_sweep.csv                (Threshold sweep data from tau=0.01 to tau=0.99)
    ├── final_summary.json                 (In-house 5-fold cross-validation aggregate metrics)
    ├── oof_subject_predictions.csv        (Out-of-fold subject-level prediction scores for all 500 subjects)
    ├── external_summary.json              (External Liyuan hospital cohort validation metrics)
    └── external_subject_predictions.csv   (Subject-level predictions on the 120 external subjects)
```

---

## Key Numbers for Your Presentation

- **In-House Cross-Validation (Tongji 500 Subjects):**
  - **AUROC:** 0.894 (surge of +31.0 points over baseline 0.584)
  - **Sweet Spot Operating Point ($\tau = 0.23$):**
    - **Sensitivity / Recall:** **85.0%** (51 / 60 HFrEF subjects caught)
    - **Accuracy:** **85.2%** (426 / 500 correct classifications)
    - **Specificity:** **85.2%** (375 / 440 healthy subjects correctly cleared)
- **External Independent Validation (Liyuan 120 Subjects):**
  - **AUROC:** **0.990**
  - **Sensitivity / Recall:** **100.0%** (5 / 5 HFrEF patients caught)
  - **Accuracy:** **83.3%**
  - **Specificity:** **82.6%**

---

## How to Build Your Presentation Slides

Refer to `PRESENTATION_SLIDE_DECK_GUIDE.md` for the suggested 10-slide PowerPoint sequence:
1. **Title & Problem Statement** (Heart failure epidemic, echocardiogram bottleneck, non-invasive acoustic-electrical screening).
2. **Dataset & Severe Imbalance** (Tongji & Liyuan cohorts, 12% prevalence, 7.33:1 imbalance, why SMOTE corrupts medical waves).
3. **Core Preprocessing Pipeline** (Zero-phase filtering, Pan-Tompkins R-peak anchoring, 32-band Log-Mel spectrogram generation).
4. **The 5-Technique Evolution** (Step-by-step diagnostic flaw and engineering solution).
5. **Deep Dive: Hierarchical Multimodal Architecture** (Use `diagrams/system_architecture_diagram.png`).
6. **Experimental Progression Results** (Use `charts/progression_comparison_barchart.png` and `data/all_models_progression_metrics.csv`).
7. **Clinical Sweet Spot Analysis** (Use `charts/threshold_sweet_spot.png` showing $\tau = 0.23$).
8. **Comparative ROC Analysis** (Use `charts/roc_curves.png`).
9. **External Hospital Cohort Validation** (AUROC 0.990, 100% sensitivity on independent cohort).
10. **Limitations & Future Directions** (Hospital noise, valvular confounders, future real-time edge stethoscope deployment).
