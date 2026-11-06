"""Log-Mel Spectrogram Extraction and 2D CNN Model for PCG Classification."""

from __future__ import annotations

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any, Mapping
import pandas as pd

from .data_loader import SignalCache


def mel_filterbank(
    n_mels: int = 32,
    n_fft: int = 128,
    fs: int = 1440,
    f_min: float = 20.0,
    f_max: float = 400.0,
) -> np.ndarray:
    """Construct a triangular Mel filterbank."""
    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)
    n_freq = n_fft // 2 + 1
    fbank = np.zeros((n_mels, n_freq), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        for j in range(left, center):
            if j < n_freq:
                fbank[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            if j < n_freq:
                fbank[i - 1, j] = (right - j) / max(right - center, 1)
    return fbank


class LogMelExtractor:
    """Extract Log-Mel spectrograms from PCG raw windows."""

    def __init__(
        self,
        fs: int = 1440,
        n_mels: int = 32,
        nperseg: int = 128,
        noverlap: int = 64,
        f_min: float = 20.0,
        f_max: float = 400.0,
    ) -> None:
        self.fs = int(fs)
        self.n_mels = int(n_mels)
        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap)
        self.fbank = mel_filterbank(n_mels, nperseg, fs, f_min, f_max)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Convert a 1-D PCG waveform (e.g. 2160 samples) into a Log-Mel spectrogram (1, n_mels, n_frames)."""
        x = np.asarray(x, dtype=np.float32).ravel()
        _, _, Zxx = signal.stft(x, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
        power = np.abs(Zxx) ** 2
        mel = np.dot(self.fbank, power)
        log_mel = np.log(np.maximum(mel, 1e-6))
        # Per-spectrogram standardization
        m = float(np.mean(log_mel))
        s = float(np.std(log_mel))
        s = s if s > 1e-6 else 1.0
        normalized = (log_mel - m) / s
        return normalized[np.newaxis, :, :].astype(np.float32)


class LogMelDataset(Dataset):
    """Dataset delivering 2D Log-Mel spectrograms for ECG-gated PCG windows."""

    def __init__(
        self,
        windows: pd.DataFrame,
        cache: SignalCache,
        extractor: LogMelExtractor | None = None,
    ) -> None:
        self.windows = windows.reset_index(drop=True)
        self.cache = cache
        self.extractor = extractor or LogMelExtractor()
        self._pcg = np.load(cache.pcg_path, mmap_mode="r")
        offsets = cache.metadata.set_index("record_id")["pcg_offset"].to_dict()
        w = self.windows
        self._col_pcg_offset = np.array(
            [int(offsets.get(rid, 0)) for rid in w["record_id"].astype(str)],
            dtype=np.int64,
        )
        self._col_start_sample = w["start_sample"].to_numpy(dtype=np.int64, copy=True)
        self._col_length_samples = w["length_samples"].to_numpy(dtype=np.int64, copy=True)
        self._col_label = w["label"].to_numpy(dtype=np.float64, copy=True)
        self._col_subject = w["subject"].astype(str).to_numpy(dtype=object)
        self._col_record_id = w["record_id"].astype(str).to_numpy(dtype=object)
        self._col_site = w["site"].astype(str).to_numpy(dtype=object)
        self._col_window_id = w["window_id"].to_numpy(dtype=np.int64, copy=True)
        self._col_start_sec = w["start_sec"].to_numpy(dtype=np.float64, copy=True)
        fold_values = w["fold"].to_numpy(dtype=np.float64, copy=True)
        self._col_fold = np.where(np.isnan(fold_values), -1, fold_values).astype(np.int64)

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_pcg"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._pcg = np.load(self.cache.pcg_path, mmap_mode="r")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start = int(self._col_pcg_offset[idx]) + int(self._col_start_sample[idx])
        length = int(self._col_length_samples[idx])
        slice_pcg = np.array(self._pcg[start : start + length], dtype=np.float32)
        spec = self.extractor.transform(slice_pcg)
        return {
            "x": torch.from_numpy(spec).float(),
            "label": torch.tensor(int(self._col_label[idx]), dtype=torch.long),
            "subject": str(self._col_subject[idx]),
            "record_id": str(self._col_record_id[idx]),
            "site": str(self._col_site[idx]),
            "fold": int(self._col_fold[idx]),
            "window_id": int(self._col_window_id[idx]),
            "start_sec": float(self._col_start_sec[idx]),
        }


class LogMelCNN(nn.Module):
    """Lightweight 2D CNN processing time-frequency Log-Mel spectrograms."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.flatten_dim = 64 * 2 * 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(128, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)

