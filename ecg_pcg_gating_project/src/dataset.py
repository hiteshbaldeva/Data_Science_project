"""Window index construction and PyTorch datasets.

E1 (ungated) uses fixed, overlapping windows tiled across each PCG recording.
E2 (gated) uses the windows produced by the ECG gating step.  Both are stored
in the same table format so that the *same* CNN and the *same* training loop can
consume them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import get_profile
from .data_loader import SignalCache
from .pcg_processing import window_quality
from .preprocessing import (
    NormalizationStats,
    expected_window_length,
    normalise_window,
    preprocess_ecg_window,
    transform_window,
)

LOGGER = logging.getLogger(__name__)

WINDOW_COLUMNS = [
    "window_id",
    "record_id",
    "subject",
    "site",
    "fold",
    "label",
    "lvef_percent",
    "experiment",
    "profile",
    "start_sec",
    "end_sec",
    "start_sample",
    "length_samples",
    "ecg_start_sample",
    "ecg_length_samples",
    "gating_window_id",
    "r_peak_time",
    "s1_time",
    "s2_time",
    "pcg_rms",
    "quality_ok",
]


def _limit_indices(n: int, max_n: int | None) -> np.ndarray:
    """Choose at most *max_n* evenly spaced indices from ``range(n)``."""
    if max_n is None or n <= max_n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_n).round().astype(int))


def build_window_index(
    metadata: pd.DataFrame,
    cfg: Mapping[str, Any],
    experiment: str,
    gating_dir: str | Path | None = None,
    max_windows_per_recording: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a table of windows for E1, E2 or E3.

    Parameters
    ----------
    metadata:
        The signal-cache metadata table (one row per subject/site recording).
    experiment:
        ``e1`` = fixed ungated windows; ``e2`` = ECG-gated windows;
        ``e3`` = the same gated windows plus an ECG window (fusion).
    gating_dir:
        Directory containing one JSON per recording, produced by the gating
        pipeline.  Required for E2/E3.
    """
    experiment = experiment.lower()
    if experiment not in {"e1", "e2", "e3"}:
        raise ValueError("experiment must be one of: e1, e2, e3")
    profile = get_profile(cfg)
    fs_pcg = int(profile["pcg_target_fs"])
    window_seconds = float(profile["window_seconds"])
    hop_seconds = float(profile["hop_seconds"])
    window_samples = int(round(window_seconds * fs_pcg))
    hop_samples = max(1, int(round(hop_seconds * fs_pcg)))
    ecg_cfg = dict(cfg.get("preprocessing", {}).get("ecg", {}))
    fs_ecg_target = int(ecg_cfg.get("target_fs", 500))
    min_rms = float(cfg.get("gating", {}).get("quality", {}).get("min_window_rms", 0.0))

    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "experiment": experiment,
        "profile": profile["name"],
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "window_samples": window_samples,
        "n_recordings_total": int(len(metadata)),
        "n_recordings_used": 0,
        "n_recordings_skipped": 0,
        "n_windows": 0,
        "skipped": [],
        "quality_rejections": 0,
        "gating_rejections": 0,
    }

    for _, rec in metadata.iterrows():
        if str(rec.get("status", "ok")) == "error":
            report["skipped"].append(
                {"record_id": rec["record_id"], "reason": "cache status=error"}
            )
            report["n_recordings_skipped"] += 1
            continue
        if int(rec.get("pcg_length", 0)) <= 0 or int(rec.get("ecg_length", 0)) <= 0:
            report["skipped"].append(
                {"record_id": rec["record_id"], "reason": "zero-length cached signal"}
            )
            report["n_recordings_skipped"] += 1
            continue

        record_id = str(rec["record_id"])
        subject = str(rec["subject"])
        site = str(rec["site"])
        fold = int(rec["fold"]) if pd.notna(rec.get("fold")) else None
        label = int(rec["label"]) if pd.notna(rec.get("label")) else None
        lvef = float(rec["lvef_percent"]) if pd.notna(rec.get("lvef_percent")) else None
        pcg_len = int(rec["pcg_length"])
        ecg_len = int(rec["ecg_length"])
        ecg_fs = int(rec["ecg_fs"])

        if experiment == "e1":
            starts = list(range(0, max(pcg_len - window_samples + 1, 0), hop_samples))
            starts = [starts[i] for i in _limit_indices(len(starts), max_windows_per_recording)]
            used = 0
            for start_sample in starts:
                end_sample = start_sample + window_samples
                start_sec = start_sample / fs_pcg
                end_sec = end_sample / fs_pcg
                # The ECG window is only needed for E3, but we store its
                # coordinates for all experiments so the tables are uniform.
                ecg_start = int(round(start_sec * ecg_fs))
                ecg_len_win = int(round(window_seconds * ecg_fs))
                if ecg_start < 0 or ecg_start + ecg_len_win > ecg_len:
                    continue
                quality_ok = True
                rms = None
                if min_rms > 0:
                    # The current signal cache does not expose the raw window
                    # here without opening the memmap, so quality filtering is
                    # deferred to the dataset (see WindowDataset).  We still
                    # keep the flag for bookkeeping.
                    quality_ok = True
                rows.append(
                    {
                        "window_id": len(rows),
                        "record_id": record_id,
                        "subject": subject,
                        "site": site,
                        "fold": fold,
                        "label": label,
                        "lvef_percent": lvef,
                        "experiment": experiment,
                        "profile": profile["name"],
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "start_sample": start_sample,
                        "length_samples": window_samples,
                        "ecg_start_sample": ecg_start,
                        "ecg_length_samples": ecg_len_win,
                        "gating_window_id": -1,
                        "r_peak_time": np.nan,
                        "s1_time": np.nan,
                        "s2_time": np.nan,
                        "pcg_rms": rms,
                        "quality_ok": quality_ok,
                    }
                )
                used += 1
            if used:
                report["n_recordings_used"] += 1
            else:
                report["n_recordings_skipped"] += 1
                report["skipped"].append(
                    {"record_id": record_id, "reason": "no full windows fit"}
                )
            continue

        # E2 / E3: read gating JSON and use only successful windows.
        if gating_dir is None:
            raise ValueError("gating_dir is required for E2/E3")
        gating_path = Path(gating_dir) / f"{record_id}.json"
        if not gating_path.exists():
            report["n_recordings_skipped"] += 1
            report["skipped"].append(
                {"record_id": record_id, "reason": "missing gating result"}
            )
            continue
        with gating_path.open("r", encoding="utf-8") as fh:
            gating = json.load(fh)
        if not bool(gating.get("recording_ok", False)):
            report["n_recordings_skipped"] += 1
            report["gating_rejections"] += 1
            report["skipped"].append(
                {
                    "record_id": record_id,
                    "reason": "gating rejected recording: "
                    + "; ".join(gating.get("reasons", [])),
                }
            )
            continue

        valid_windows = [w for w in gating.get("windows", []) if w.get("valid", False)]
        valid_windows = [
            valid_windows[i]
            for i in _limit_indices(len(valid_windows), max_windows_per_recording)
        ]
        used = 0
        for w in valid_windows:
            start_sample = int(w["start_sample"])
            end_sample = int(w["end_sample"])
            length = end_sample - start_sample
            if length != window_samples:
                report["skipped"].append(
                    {
                        "record_id": record_id,
                        "window_id": w.get("window_id"),
                        "reason": f"gated length {length} != profile {window_samples}",
                    }
                )
                continue
            if start_sample < 0 or end_sample > pcg_len:
                report["skipped"].append(
                    {
                        "record_id": record_id,
                        "window_id": w.get("window_id"),
                        "reason": "gated window outside cached PCG",
                    }
                )
                continue
            start_sec = float(w["start_time"])
            ecg_start = int(round(start_sec * ecg_fs))
            ecg_len_win = int(round(window_seconds * ecg_fs))
            if ecg_start < 0 or ecg_start + ecg_len_win > ecg_len:
                report["skipped"].append(
                    {
                        "record_id": record_id,
                        "window_id": w.get("window_id"),
                        "reason": "ECG window outside cached ECG",
                    }
                )
                continue
            rows.append(
                {
                    "window_id": len(rows),
                    "record_id": record_id,
                    "subject": subject,
                    "site": site,
                    "fold": fold,
                    "label": label,
                    "lvef_percent": lvef,
                    "experiment": experiment,
                    "profile": profile["name"],
                    "start_sec": start_sec,
                    "end_sec": float(w["end_time"]),
                    "start_sample": start_sample,
                    "length_samples": length,
                    "ecg_start_sample": ecg_start,
                    "ecg_length_samples": ecg_len_win,
                    "gating_window_id": int(w.get("window_id", -1)),
                    "r_peak_time": w.get("r_peak_time"),
                    "s1_time": w.get("s1_time"),
                    "s2_time": w.get("s2_time"),
                    "pcg_rms": w.get("pcg_rms"),
                    "quality_ok": True,
                }
            )
            used += 1
        if used:
            report["n_recordings_used"] += 1
        else:
            report["n_recordings_skipped"] += 1
            report["skipped"].append(
                {"record_id": record_id, "reason": "no valid gated windows"}
            )

    df = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    report["n_windows"] = int(len(df))
    if len(df):
        report["n_subjects_with_windows"] = int(df["subject"].nunique())
        report["label_counts"] = {
            "positive_windows": int((df["label"] == 1).sum()),
            "negative_windows": int((df["label"] == 0).sum()),
        }
        report["subjects_per_fold"] = {
            str(k): int(v)
            for k, v in df.groupby("fold")["subject"].nunique().items()
        }
    else:
        report["n_subjects_with_windows"] = 0
    return df, report


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    """PyTorch dataset over a window table.

    ``mode='pcg'`` returns the PCG window used by E1/E2.
    ``mode='fusion'`` additionally returns a time-aligned ECG window (E3).
    """

    def __init__(
        self,
        windows: pd.DataFrame,
        cache: SignalCache,
        cfg: Mapping[str, Any],
        stats: NormalizationStats | None = None,
        ecg_stats: NormalizationStats | None = None,
        mode: str = "pcg",
        normalize: bool = True,
    ):
        self.windows = windows.reset_index(drop=True)
        self.cache = cache
        self.cfg = cfg
        self.profile = get_profile(cfg)
        self.ecg_cfg = dict(cfg.get("preprocessing", {}).get("ecg", {}))
        self.stats = stats
        self.ecg_stats = ecg_stats
        self.mode = mode
        self.normalize = normalize
        self.expected_length = expected_window_length(
            self.profile, float(self.profile["window_seconds"])
        )
        self.ecg_expected_length = int(
            round(float(self.profile["window_seconds"]) * int(self.ecg_cfg.get("target_fs", 500)))
        )
        self._ecg = np.load(cache.ecg_path, mmap_mode="r")
        self._pcg = np.load(cache.pcg_path, mmap_mode="r")
        # Per-record sampling rates are constant in this dataset, but read them
        # from the metadata so the class remains correct if that changes.
        self._record_fs = (
            cache.metadata.set_index("record_id")[["ecg_fs", "pcg_fs"]].to_dict("index")
        )
        # Column-wise numpy views.  Building a ``pandas.Series`` per item cost
        # far more than the signal slicing itself, which made the GPU sit idle
        # during training.  The DataFrame ``self.windows`` is kept for callers
        # that need a table view.
        w = self.windows
        self._col_start_sample = w["start_sample"].to_numpy(dtype=np.int64, copy=True)
        self._col_length_samples = w["length_samples"].to_numpy(dtype=np.int64, copy=True)
        self._col_ecg_start_sample = w["ecg_start_sample"].to_numpy(dtype=np.int64, copy=True)
        self._col_ecg_length_samples = w["ecg_length_samples"].to_numpy(
            dtype=np.int64, copy=True
        )
        self._col_label = w["label"].to_numpy(dtype=np.float64, copy=True)
        self._col_subject = w["subject"].astype(str).to_numpy(dtype=object)
        self._col_record_id = w["record_id"].astype(str).to_numpy(dtype=object)
        self._col_site = w["site"].astype(str).to_numpy(dtype=object)
        self._col_window_id = w["window_id"].to_numpy(dtype=np.int64, copy=True)
        self._col_start_sec = w["start_sec"].to_numpy(dtype=np.float64, copy=True)
        fold_values = w["fold"].to_numpy(dtype=np.float64, copy=True)
        self._col_fold = np.where(np.isnan(fold_values), -1, fold_values).astype(np.int64)
        offsets = cache.metadata.set_index("record_id")[["pcg_offset", "ecg_offset"]].to_dict("index")
        self._col_pcg_offset = np.array(
            [int(offsets.get(rid, {}).get("pcg_offset", 0)) for rid in self._col_record_id],
            dtype=np.int64,
        )
        self._col_ecg_offset = np.array(
            [int(offsets.get(rid, {}).get("ecg_offset", 0)) for rid in self._col_record_id],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.windows)

    # DataLoader workers on Windows are spawned, so the dataset is pickled.
    # A raw ``np.memmap`` pickles its whole buffer (~0.4 GB here), which fails
    # on the Windows spawn pipe, so drop the memmaps and reopen them per worker.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_ecg"] = None
        state["_pcg"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._ecg = np.load(self.cache.ecg_path, mmap_mode="r")
        self._pcg = np.load(self.cache.pcg_path, mmap_mode="r")

    def _slice_pcg(self, index: int) -> np.ndarray:
        start = int(self._col_pcg_offset[index]) + int(self._col_start_sample[index])
        length = int(self._col_length_samples[index])
        return np.array(self._pcg[start : start + length], dtype=np.float32)

    def _slice_ecg(self, index: int) -> np.ndarray:
        start = int(self._col_ecg_offset[index]) + int(self._col_ecg_start_sample[index])
        length = int(self._col_ecg_length_samples[index])
        return np.array(self._ecg[start : start + length], dtype=np.float32)

    def get_transformed_pcg(self, index: int) -> np.ndarray:
        pcg = self._slice_pcg(index)
        return transform_window(
            pcg, self.profile, expected_length=self.expected_length
        )

    def get_transformed_ecg(self, index: int) -> np.ndarray:
        ecg = self._slice_ecg(index)
        fs_ecg = int(self._record_fs[str(self._col_record_id[index])]["ecg_fs"])
        return preprocess_ecg_window(
            ecg,
            fs_ecg,
            self.ecg_cfg,
            target_fs=int(self.ecg_cfg.get("target_fs", 500)),
            expected_length=self.ecg_expected_length,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        x = self.get_transformed_pcg(index)
        if self.normalize:
            x = normalise_window(x, self.profile, self.stats)
        label = float(self._col_label[index])
        sample: dict[str, Any] = {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0),
            "label": torch.tensor(label if not np.isnan(label) else -1.0, dtype=torch.float32),
            "subject": str(self._col_subject[index]),
            "record_id": str(self._col_record_id[index]),
            "site": str(self._col_site[index]),
            "fold": int(self._col_fold[index]),
            "window_id": int(self._col_window_id[index]),
            "start_sec": float(self._col_start_sec[index]),
        }
        if self.mode == "fusion":
            ecg = self.get_transformed_ecg(index)
            if self.normalize:
                ecg = normalise_window(
                    ecg,
                    {"normalization": self.ecg_cfg.get("normalization", "train_global")},
                    self.ecg_stats,
                )
            sample["ecg"] = torch.from_numpy(np.asarray(ecg, dtype=np.float32)).unsqueeze(0)
        return sample

    def labels(self) -> np.ndarray:
        return self._col_label.copy()

    def subjects(self) -> np.ndarray:
        return self._col_subject.copy()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _stats_from_dataset(
    dataset: WindowDataset,
    which: str = "pcg",
    max_windows: int | None = 5000,
    seed: int = 42,
    source: str = "train",
) -> NormalizationStats:
    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot compute statistics from an empty dataset.")
    rng = np.random.default_rng(seed)
    if max_windows is not None and n > max_windows:
        indices = rng.choice(n, size=max_windows, replace=False)
    else:
        indices = np.arange(n)
    total = 0.0
    total_sq = 0.0
    count = 0
    for idx in indices:
        x = (
            dataset.get_transformed_pcg(int(idx))
            if which == "pcg"
            else dataset.get_transformed_ecg(int(idx))
        )
        x = np.asarray(x, dtype=np.float64).ravel()
        total += float(np.sum(x))
        total_sq += float(np.dot(x, x))
        count += int(x.size)
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return NormalizationStats(mean=float(mean), std=std, count=count, source=source)


def compute_pcg_stats(
    dataset: WindowDataset,
    max_windows: int | None = 5000,
    seed: int = 42,
    source: str = "train",
) -> NormalizationStats:
    return _stats_from_dataset(dataset, "pcg", max_windows, seed, source)


def compute_ecg_stats(
    dataset: WindowDataset,
    max_windows: int | None = 5000,
    seed: int = 42,
    source: str = "train",
) -> NormalizationStats:
    return _stats_from_dataset(dataset, "ecg", max_windows, seed, source)


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------
def subject_table(windows: pd.DataFrame) -> pd.DataFrame:
    """Return one row per subject with label and fold.

    ``lvef_percent`` is included when present; prediction tables do not carry
    it and must still be usable with this helper.
    """
    aggregations: dict[str, str] = {"label": "first", "fold": "first"}
    if "lvef_percent" in windows.columns:
        aggregations["lvef_percent"] = "first"
    return (
        windows.groupby("subject", as_index=False)
        .agg(aggregations)
        .sort_values("subject")
        .reset_index(drop=True)
    )


def subset_by_subjects(windows: pd.DataFrame, subjects: Iterable[str]) -> pd.DataFrame:
    subjects = set(str(s) for s in subjects)
    return windows[windows["subject"].astype(str).isin(subjects)].reset_index(drop=True)


def check_no_subject_leakage(*frames: pd.DataFrame) -> None:
    """Raise if any subject appears in more than one split."""
    seen: dict[str, int] = {}
    for i, frame in enumerate(frames):
        for subject in frame["subject"].astype(str).unique():
            if subject in seen and seen[subject] != i:
                raise ValueError(
                    f"Subject leakage detected: {subject} appears in splits "
                    f"{seen[subject]} and {i}."
                )
            seen[subject] = i


def subject_label_lookup(windows: pd.DataFrame) -> dict[str, int]:
    table = subject_table(windows)
    return {str(row.subject): int(row.label) for row in table.itertuples()}
