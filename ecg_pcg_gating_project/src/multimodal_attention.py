"""Hierarchical Multimodal Cross-Attention Network for 4-Site ECG+PCG HFrEF Detection.

Implements the key architectural innovations from Ma et al., Patterns (Cell Press, 2026):
1. Dual-Branch Modality Encoders:
   - PCG Branch: 2D Log-Mel Spectrogram ConvNet
   - ECG Branch: 1D ResNet/Conv Temporal Encoder
2. Cross-Modality Cross-Attention (CCF):
   - Bidirectional multi-head cross-attention between electrical (ECG) and acoustic (PCG) features
     to capture electromechanical activation timing (EMAT).
3. 4-Site Auscultation Fusion (CSB):
   - Inter-site self-attention across APEX (Mitral), LLSB (Tricuspid), LUSB (Pulmonary), and RUSB (Aortic).
4. Feature-Decision Double-Level Multi-Branch Heads:
   - 4 site-specific heads + 1 channel-shared head + 1 joint fusion head with deep supervision.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .logmel import LogMelExtractor

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modality Feature Encoders
# ---------------------------------------------------------------------------
class PCGLogMelEncoder(nn.Module):
    """Encodes 2D Log-Mel spectrograms (B, 1, 32, 35) into feature sequence (B, 8, embed_dim)."""

    def __init__(self, embed_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 32, 16, 17)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 64, 8, 8)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),  # (B, embed_dim, 1, 8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # Reshape to (B, 8, embed_dim)
        return out.squeeze(2).transpose(1, 2)


class ECG1DEncoder(nn.Module):
    """Encodes 1D filtered ECG signals (B, 1, 750) into feature sequence (B, 8, embed_dim)."""

    def __init__(self, embed_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # (B, 32, 187)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # (B, 64, 46)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, embed_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            nn.AdaptiveAvgPool1d(8),  # (B, embed_dim, 8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # Permute to (B, 8, embed_dim)
        return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Cross-Modality Cross-Attention
# ---------------------------------------------------------------------------
class CrossModalityAttentionBlock(nn.Module):
    """Bidirectional Multi-Head Cross-Attention between ECG and PCG."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.ecg_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.pcg_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ecg = nn.LayerNorm(embed_dim)
        self.norm_pcg = nn.LayerNorm(embed_dim)
        self.fusion_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, f_ecg: torch.Tensor, f_pcg: torch.Tensor) -> torch.Tensor:
        # ECG attends to PCG
        ecg_corr, _ = self.ecg_cross_attn(query=f_ecg, key=f_pcg, value=f_pcg)
        f_ecg_ref = self.norm_ecg(f_ecg + ecg_corr)

        # PCG attends to ECG
        pcg_corr, _ = self.pcg_cross_attn(query=f_pcg, key=f_ecg, value=f_ecg)
        f_pcg_ref = self.norm_pcg(f_pcg + pcg_corr)

        # Pool over sequence length (dim 1)
        ecg_vec = f_ecg_ref.mean(dim=1)
        pcg_vec = f_pcg_ref.mean(dim=1)

        # Concatenate and fuse
        fused = torch.cat([ecg_vec, pcg_vec], dim=-1)
        return self.fusion_fc(fused)  # (B, embed_dim)


# ---------------------------------------------------------------------------
# Inter-Site Channel-Shared Fusion
# ---------------------------------------------------------------------------
class ChannelSharedEncoder(nn.Module):
    """Learns spatial inter-correlations across the 4 chest auscultation sites."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.site_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, site_stack: torch.Tensor) -> torch.Tensor:
        # site_stack: (B, 4, embed_dim)
        attn_out, _ = self.site_attn(query=site_stack, key=site_stack, value=site_stack)
        res = self.norm(site_stack + attn_out)
        out = self.mlp(res)
        return out.mean(dim=1)  # (B, embed_dim)


# ---------------------------------------------------------------------------
# Complete Hierarchical LGC2-Net
# ---------------------------------------------------------------------------
class HierarchicalLGCNet(nn.Module):
    """End-to-End Hierarchical Cross-Modality Cross-Channel Fusion Network.

    Processes 4 auscultation channels [APEX, LLSB, LUSB, RUSB] each containing
    synchronized ECG (1D) and PCG (2D Log-Mel).
    """

    def __init__(self, embed_dim: int = 128, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        # Shared modality encoders (shared across the 4 sites to prevent overfitting, as in the paper)
        self.pcg_encoder = PCGLogMelEncoder(embed_dim=embed_dim, dropout=dropout)
        self.ecg_encoder = ECG1DEncoder(embed_dim=embed_dim, dropout=dropout)

        # 4 Channel-Specific Cross-Attention Modules
        self.ccf_apex = CrossModalityAttentionBlock(embed_dim=embed_dim, dropout=dropout)
        self.ccf_llsb = CrossModalityAttentionBlock(embed_dim=embed_dim, dropout=dropout)
        self.ccf_lusb = CrossModalityAttentionBlock(embed_dim=embed_dim, dropout=dropout)
        self.ccf_rusb = CrossModalityAttentionBlock(embed_dim=embed_dim, dropout=dropout)

        # Channel-Shared Encoder across 4 sites
        self.channel_shared_encoder = ChannelSharedEncoder(embed_dim=embed_dim, dropout=dropout)

        # Feature-Decision Multi-Branch Heads
        self.head_apex = nn.Linear(embed_dim, num_classes)
        self.head_llsb = nn.Linear(embed_dim, num_classes)
        self.head_lusb = nn.Linear(embed_dim, num_classes)
        self.head_rusb = nn.Linear(embed_dim, num_classes)
        self.head_cs = nn.Linear(embed_dim, num_classes)

        # Joint fusion head
        self.head_joint = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self,
        ecg_apex: torch.Tensor,
        pcg_apex: torch.Tensor,
        ecg_llsb: torch.Tensor,
        pcg_llsb: torch.Tensor,
        ecg_lusb: torch.Tensor,
        pcg_lusb: torch.Tensor,
        ecg_rusb: torch.Tensor,
        pcg_rusb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # 1. Feature extraction per site
        # APEX (Mitral)
        f_p_apex = self.pcg_encoder(pcg_apex)
        f_e_apex = self.ecg_encoder(ecg_apex)
        v_apex = self.ccf_apex(f_e_apex, f_p_apex)

        # LLSB (Tricuspid)
        f_p_llsb = self.pcg_encoder(pcg_llsb)
        f_e_llsb = self.ecg_encoder(ecg_llsb)
        v_llsb = self.ccf_llsb(f_e_llsb, f_p_llsb)

        # LUSB (Pulmonary)
        f_p_lusb = self.pcg_encoder(pcg_lusb)
        f_e_lusb = self.ecg_encoder(ecg_lusb)
        v_lusb = self.ccf_lusb(f_e_lusb, f_p_lusb)

        # RUSB (Aortic)
        f_p_rusb = self.pcg_encoder(pcg_rusb)
        f_e_rusb = self.ecg_encoder(ecg_rusb)
        v_rusb = self.ccf_rusb(f_e_rusb, f_p_rusb)

        # 2. Inter-site shared encoding
        site_stack = torch.stack([v_apex, v_llsb, v_lusb, v_rusb], dim=1)  # (B, 4, embed_dim)
        v_cs = self.channel_shared_encoder(site_stack)  # (B, embed_dim)

        # 3. Individual branch logits
        logits_apex = self.head_apex(v_apex)
        logits_llsb = self.head_llsb(v_llsb)
        logits_lusb = self.head_lusb(v_lusb)
        logits_rusb = self.head_rusb(v_rusb)
        logits_cs = self.head_cs(v_cs)

        # 4. Joint fusion logits
        joint_feat = torch.cat([v_apex, v_llsb, v_lusb, v_rusb, v_cs], dim=-1)
        logits_joint = self.head_joint(joint_feat)

        # 5. Decision ensemble (average of probability distributions)
        p_apex = F.softmax(logits_apex, dim=-1)
        p_llsb = F.softmax(logits_llsb, dim=-1)
        p_lusb = F.softmax(logits_lusb, dim=-1)
        p_rusb = F.softmax(logits_rusb, dim=-1)
        p_cs = F.softmax(logits_cs, dim=-1)
        p_joint = F.softmax(logits_joint, dim=-1)

        p_final = (p_apex + p_llsb + p_lusb + p_rusb + p_cs + p_joint) / 6.0

        return {
            "logits_apex": logits_apex,
            "logits_llsb": logits_llsb,
            "logits_lusb": logits_lusb,
            "logits_rusb": logits_rusb,
            "logits_cs": logits_cs,
            "logits_joint": logits_joint,
            "p_final": p_final,
        }


# ---------------------------------------------------------------------------
# Patient-Level Multi-Site Dataset
# ---------------------------------------------------------------------------
class PatientMultiSiteDataset(Dataset):
    """Delivers synchronized 4-site [APEX, LLSB, LUSB, RUSB] ECG & PCG inputs per subject."""

    SITES = ["APEX", "LLSB", "LUSB", "RUSB"]

    def __init__(
        self,
        subjects: list[str],
        windows_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        pcg_memmap: np.ndarray,
        ecg_memmap: np.ndarray,
        extractor: LogMelExtractor | None = None,
        is_train: bool = True,
        seed: int = 42,
    ) -> None:
        self.subjects = sorted(list(set(str(s) for s in subjects)))
        self.is_train = is_train
        self.extractor = extractor or LogMelExtractor()
        self.pcg_mem = pcg_memmap
        self.ecg_mem = ecg_memmap
        self.rng = np.random.default_rng(seed)

        # Pre-index windows by (subject, site)
        meta_copy = metadata_df.copy()
        meta_copy["subject"] = meta_copy["subject"].astype(str)
        meta_copy["site"] = meta_copy["site"].astype(str)
        self.sub_meta = meta_copy.set_index(["subject", "site"])

        # Pre-group window indices for high-speed sampling
        self.win_lookup: dict[tuple[str, str], list[dict[str, int]]] = {}
        wins_copy = windows_df.copy()
        wins_copy["subject"] = wins_copy["subject"].astype(str)
        wins_copy["site"] = wins_copy["site"].astype(str)
        grouped = wins_copy.groupby(["subject", "site"])
        for (subj, site), group in grouped:
            self.win_lookup[(str(subj), str(site))] = group[
                ["start_sample", "length_samples", "ecg_start_sample", "ecg_length_samples"]
            ].to_dict("records")

        # Labels per subject
        subj_labels = meta_copy.drop_duplicates("subject").set_index("subject")["label"].to_dict()
        self.labels = {str(k): int(v) for k, v in subj_labels.items()}

    def __len__(self) -> int:
        return len(self.subjects)

    def _get_site_sample(self, subj: str, site: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
        wins = self.win_lookup.get((subj, site), [])
        if not wins:
            # Fallback zero pads if somehow missing
            return np.zeros((1, 32, 35), dtype=np.float32), np.zeros((1, 750), dtype=np.float32)

        # Select window (random if train, deterministic if eval)
        if self.is_train:
            w = self.rng.choice(wins)
        else:
            w = wins[idx % len(wins)]

        meta_row = self.sub_meta.loc[(subj, site)]
        pcg_off = int(meta_row["pcg_offset"])
        ecg_off = int(meta_row["ecg_offset"])

        # PCG slice -> Log-Mel
        p_start = pcg_off + int(w["start_sample"])
        p_len = int(w["length_samples"])
        pcg_raw = np.array(self.pcg_mem[p_start : p_start + p_len], dtype=np.float32)
        pcg_mel = self.extractor.transform(pcg_raw)

        # ECG slice -> 1D normalized
        e_start = ecg_off + int(w["ecg_start_sample"])
        e_len = int(w["ecg_length_samples"])
        ecg_raw = np.array(self.ecg_mem[e_start : e_start + e_len], dtype=np.float32)
        if len(ecg_raw) != 750:
            if len(ecg_raw) > 750:
                ecg_raw = ecg_raw[:750]
            else:
                ecg_raw = np.pad(ecg_raw, (0, 750 - len(ecg_raw)))
        # Standardize ECG window
        m = float(np.mean(ecg_raw))
        s = float(np.std(ecg_raw))
        s = s if s > 1e-6 else 1.0
        ecg_norm = ((ecg_raw - m) / s)[np.newaxis, :].astype(np.float32)

        return pcg_mel, ecg_norm

    def __getitem__(self, index: int) -> dict[str, Any]:
        subj = self.subjects[index]
        label = self.labels[subj]

        pcg_apex, ecg_apex = self._get_site_sample(subj, "APEX", index)
        pcg_llsb, ecg_llsb = self._get_site_sample(subj, "LLSB", index)
        pcg_lusb, ecg_lusb = self._get_site_sample(subj, "LUSB", index)
        pcg_rusb, ecg_rusb = self._get_site_sample(subj, "RUSB", index)

        return {
            "subject": subj,
            "label": torch.tensor(label, dtype=torch.long),
            "ecg_apex": torch.from_numpy(ecg_apex),
            "pcg_apex": torch.from_numpy(pcg_apex),
            "ecg_llsb": torch.from_numpy(ecg_llsb),
            "pcg_llsb": torch.from_numpy(pcg_llsb),
            "ecg_lusb": torch.from_numpy(ecg_lusb),
            "pcg_lusb": torch.from_numpy(pcg_lusb),
            "ecg_rusb": torch.from_numpy(ecg_rusb),
            "pcg_rusb": torch.from_numpy(pcg_rusb),
        }
