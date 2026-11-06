"""Generate detailed Preprocessing and Fusion Pipeline PDF using ReportLab."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, text)
        self.drawString(54, 36, "HFrEF ECG+PCG Preprocessing & Multimodal Fusion Pipeline")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 48, letter[0] - 54, 48)
        self.restoreState()


def build_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=60,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#0d3b66")
    secondary_color = colors.HexColor("#00509d")
    accent_color = colors.HexColor("#2b6cb0")
    dark_text = colors.HexColor("#0f172a")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=15,
    )

    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=secondary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=accent_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=dark_text,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=3,
    )

    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("Complete Preprocessing & Multimodal Fusion Pipeline", title_style))
    story.append(Paragraph(
        "Detailed Technical Flow: Signal Conditioning, ECG-Guided Physiological Gating, "
        "Time-Frequency Transformations, and Hierarchical Multimodal Cross-Attention Fusion",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=14))

    # Section 1: Overview
    story.append(Paragraph("1. High-Level Architectural Pipeline", h1_style))
    story.append(Paragraph(
        "The automated screening pipeline ingests synchronous 30-second phonocardiogram (PCG, acoustic) "
        "and electrocardiogram (ECG, electrical) recordings acquired via an <b>EKO DUO ECG+ digital stethoscope</b> "
        "across four standard clinical auscultation positions: <b>APEX (Mitral)</b>, <b>LLSB (Tricuspid)</b>, "
        "<b>LUSB (Pulmonary)</b>, and <b>RUSB (Aortic)</b>. It detects <b>Heart Failure with Reduced Ejection Fraction (HFrEF, LVEF &le; 40%)</b>.",
        body_style
    ))

    # ASCII Architecture Table
    flow_text = """[Raw Stethoscope Signals (30s Synchronous ECG + PCG across 4 Auscultation Sites)]
  │
  ├── 1. Continuous Signal Preprocessing (Resampling + Zero-Phase Butterworth + Mains Notch Filter)
  ├── 2. Physiological Gating (Pan-Tompkins R-Peak Detection -> Anchored [t_R - 0.25s, t_R + 1.25s] Slices)
  ├── 3. Feature Representations (PCG -> 32-Band Log-Mel 2D Spectrogram; ECG -> Normalized 1D Vector)
  ├── 4. Dual-Branch Encoders (PCG 2D ConvNet + ECG 1D Conv-Residual Encoders -> Latent Sequences)
  ├── 5. Level 1 Fusion: Inter-Modality Cross-Attention (Q_ECG x K/V_PCG -> Detects EMAT Timing Delay)
  ├── 6. Level 2 Fusion: Inter-Site Spatial Fusion (Multi-Head Self-Attention across APEX, LLSB, LUSB, RUSB)
  └── 7. Feature-Decision Double-Level Multi-Heads (6 Simultaneous Supervisions -> Ensembled Prediction)"""

    flow_table = Table([[Paragraph(flow_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style)]], colWidths=[letter[0] - 108])
    flow_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(flow_table)
    story.append(Spacer(1, 10))

    # Section 2: Continuous Preprocessing
    story.append(Paragraph("2. Stage-by-Stage Detailed Preprocessing", h1_style))
    story.append(Paragraph("Stage 1: Continuous Full-Recording Filtering & Caching", h2_style))
    story.append(Paragraph(
        "Raw 30-second recordings are conditioned as complete continuous waveforms prior to any window extraction:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Acoustic PCG Stream:</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Polyphase Rational Resampling:</b> Resampled to <b>1440 Hz</b> using polyphase filtering (<code>scipy.signal.resample_poly</code>) to avoid phase artifacts.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Zero-Phase Butterworth Bandpass:</b> 5th-order filter passing <b>20 Hz to 400 Hz</b> using forward-backward filtering (<code>sosfiltfilt</code>). This removes low-frequency respiratory motion (&lt;20 Hz) and ambient room hiss (&gt;400 Hz) with strictly zero phase distortion.", bullet_style))
    story.append(Paragraph("&bull; <b>Electrical ECG Stream:</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Sampling Rate:</b> Maintained at <b>500 Hz</b> (standard clinical monitoring resolution).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Mains Notch Filter:</b> IIR notch filter at <b>50 Hz</b> (Q = 30) eliminates electrical powerline hum.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Baseline Wander Removal:</b> 3rd-order Butterworth bandpass filter (0.5 Hz to 40 Hz) removes respiration baseline drift for clean R-peak detection.", bullet_style))
    story.append(Paragraph("&bull; <b>Flat Memory-Mapped Cache:</b> Signals are stored sequentially in <code>pcg.npy</code> and <code>ecg.npy</code>. Metadata records per-recording coordinates (<code>pcg_offset</code>, <code>ecg_offset</code>), enabling sub-millisecond retrieval.", bullet_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Stage 2: ECG-Guided Physiological Gating", h2_style))
    story.append(Paragraph(
        "Instead of arbitrary sliding windows that slice heart sounds mid-cycle, the synchronized ECG serves as a physiological clock:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Pan-Tompkins R-Peak Detection:</b> Evaluates 500 Hz lead-I ECG via derivative and moving-average integration to identify ventricular depolarization peaks (t<sub>R</sub>).", bullet_style))
    story.append(Paragraph("&bull; <b>Anchored Cardiac Cycle Windowing:</b> Extracts windows from <b>[t<sub>R</sub> - 0.25 s, t<sub>R</sub> + 1.25 s]</b> (fixed duration: <b>1.50 seconds</b>).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>PCG Slice:</b> Exactly <b>2,160 samples</b> (1.50 s &times; 1440 Hz).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>ECG Slice:</b> Exactly <b>750 samples</b> (1.50 s &times; 500 Hz).", bullet_style))
    story.append(Paragraph("&bull; <b>Acoustic Quality Verification:</b> Evaluates the Hilbert envelope to confirm S1/S2 heart sound peaks; rejects low-RMS (&lt;0.005) or clipped microphone friction artifacts.", bullet_style))

    story.append(PageBreak())

    # Section 3: Feature Representation
    story.append(Paragraph("Stage 3: Window Feature Representation per Site", h2_style))
    story.append(Paragraph(
        "Each valid cardiac cycle window is converted into structured feature tensors for deep learning:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>PCG Stream &rarr; 32-Band Log-Mel 2D Spectrogram:</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Short-Time Fourier Transform (STFT):</b> 128-point Hann window with 64-sample hop (50% overlap).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Mel Filterbank:</b> 32 triangular filterbanks spanning <b>20 Hz to 700 Hz</b>.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Logarithmic Compression:</b> <code>Log-Mel = log(max(Mel Power, 1e-6))</code> followed by standardization.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; <b>Output Shape:</b> <b>(1, 32, 35)</b> &mdash; 1 channel, 32 frequency bins, 35 time frames.", bullet_style))
    story.append(Paragraph("&bull; <b>ECG Stream &rarr; 1D Normalized Temporal Signal:</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&bull; Standardized: <code>(x - mean) / std</code> &rarr; <b>Output Shape: (1, 750)</b>.", bullet_style))

    story.append(Spacer(1, 10))

    # Section 4: Fusion Architecture
    story.append(Paragraph("3. Deep Multimodal & Multi-Site Fusion Mechanics", h1_style))

    story.append(Paragraph("Level 1: Inter-Modality Cross-Attention (CCF / EMAT Detection)", h2_style))
    story.append(Paragraph(
        "In HFrEF, ventricular dyssynchrony prolongs the <b>Electromechanical Activation Time (EMAT)</b> &mdash; "
        "the latency between electrical excitation (Q-wave) and mechanical mitral valve closure (S1). "
        "The network discovers this latency through multi-head cross-attention:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>PCG 2D Conv Encoder:</b> 3 Conv2d stages (32 &rarr; 64 &rarr; 128 channels) &rarr; sequence <code>(B, 8, 128)</code>.", bullet_style))
    story.append(Paragraph("&bull; <b>ECG 1D Conv-Residual Encoder:</b> 3 Conv1d stages (kernels 15, 7, 5; channels 32 &rarr; 64 &rarr; 128) &rarr; sequence <code>(B, 8, 128)</code>.", bullet_style))
    story.append(Paragraph("&bull; <b>Bidirectional Cross-Attention:</b> <code>Q_ECG &times; K/V_PCG</code> and <code>Q_PCG &times; K/V_ECG</code> adaptively query the timing between electrical spikes and acoustic heart sounds.", bullet_style))
    story.append(Paragraph("&bull; <b>Fused Site Embedding:</b> Projected and pooled to a 128-D vector: <code>v_site &isin; &reals;<sup>128</sup></code> (for APEX, LLSB, LUSB, RUSB).", bullet_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Level 2: 4-Site Auscultation Spatial Fusion (CSB)", h2_style))
    story.append(Paragraph(
        "Because recordings from the 4 auscultation sites are collected sequentially, raw waveforms are asynchronous and cannot be concatenated in time. "
        "The network fuses them at the semantic embedding level:",
        body_style
    ))
    story.append(Paragraph("&bull; Stack the 4 site embeddings: <code>[v_APEX, v_LLSB, v_LUSB, v_RUSB] &rarr; (B, 4, 128)</code>.", bullet_style))
    story.append(Paragraph("&bull; <b>Inter-Site Multi-Head Self-Attention:</b> Models cross-valve correlations across the chest surface.", bullet_style))
    story.append(Paragraph("&bull; <b>Noise Suppression:</b> If one site (e.g. APEX) contains local chest wall rubbing artifacts, the other 3 clean sites cross-reference and <b>suppress false alarms</b>.", bullet_style))
    story.append(Paragraph("&bull; Produces the Channel-Shared Embedding: <code>v_CS &isin; &reals;<sup>128</sup></code>.", bullet_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Level 3: Feature-Decision Double-Level Multi-Branch Heads", h2_style))
    story.append(Paragraph(
        "The architecture applies deep multi-task auxiliary supervision across 6 distinct heads:",
        body_style
    ))
    story.append(Paragraph("&bull; 4 Site Heads (<code>P_APEX, P_LLSB, P_LUSB, P_RUSB</code>) + 1 Shared Head (<code>P_CS</code>) + 1 Joint Multi-Branch Head (<code>P_joint</code>).", bullet_style))
    story.append(Paragraph("&bull; <b>Total Training Loss:</b> &Lscr;<sub>total</sub> = &sum; &Lscr;<sub>site</sub> + &Lscr;<sub>CS</sub> + &Lscr;<sub>joint</sub> (with square-root class weights).", bullet_style))
    story.append(Paragraph("&bull; <b>Final Decision Ensemble:</b> <code>P_final = (P_APEX + P_LLSB + P_LUSB + P_RUSB + P_CS + P_joint) / 6.0</code>.", bullet_style))

    story.append(Spacer(1, 10))

    # Section 5: Comparative Matrix Table
    story.append(Paragraph("4. Method Comparison Matrix", h1_style))
    table_data = [
        ["Model Architecture", "Input Signals", "Acoustic Feature", "Cross-Attn", "Multi-Site", "AUROC", "Recall", "Accuracy"],
        ["E1 Baseline", "PCG Only", "Haar DWT (cA2)", "No", "Post-hoc", "0.584", "88.3%", "32.4%"],
        ["E2 Baseline", "PCG (Gated)", "Haar DWT (cA2)", "No", "Post-hoc", "0.581", "73.3%", "36.4%"],
        ["E2 Enhanced", "PCG (Gated)", "Haar DWT (cA2)", "No", "Post-hoc", "0.627", "43.3%", "56.6%"],
        ["E3 Multimodal Fusion", "ECG + PCG", "Haar DWT + 1D ECG", "No (Concat)", "Post-hoc", "0.618", "80.0%", "36.6%"],
        ["Log-Mel 2D CNN", "PCG (Gated)", "32-Band Log-Mel 2D", "No", "Post-hoc", "0.780", "56.7%", "87.2%"],
        ["Hierarchical Cross-Attn", "4-Site ECG+PCG", "Log-Mel + 1D ECG", "Yes (EMAT)", "Self-Attn", "0.894", "85.0%", "85.2%"],
    ]

    summary_table = Table(table_data, colWidths=[105, 75, 85, 55, 55, 42, 42, 45])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d3b66")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (2, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f8fafc")]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#ecfdf5")),
        ("TEXTCOLOR", (0, -1), (-1, -1), colors.HexColor("#065f46")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Key Takeaway:</b> At &tau; = 0.23, the Hierarchical Multimodal Cross-Attention network achieved "
        "<b>85.0% Recall and 85.2% Accuracy</b> simultaneously on the in-house cohort (AUROC 0.894), "
        "and <b>100.0% Sensitivity with 0.990 AUROC</b> on the independent 120-subject community scenario test set.",
        body_style
    ))

    # Build PDF with dynamic page numbers
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built: {output_path}")


if __name__ == "__main__":
    out_pdf = Path("PREPROCESSING_AND_FUSION_PIPELINE.pdf")
    build_pdf(out_pdf)

