"""WAV loading and memory-mapped signal cache.

The HFrEF dataset contains ~2,000 30-second recordings.  Loading every
full-resolution waveform into RAM (or onto the GPU) is unnecessary.  Instead,
we build a one-time cache of *preprocessed full recordings* in two NumPy
memmaps:

* ``ecg.npy`` - native sampling rate (500 Hz), raw-ish ECG;
* ``pcg.npy`` - resampled and band-passed PCG.

Window extraction then becomes a cheap slice.  Memmaps are shared by DataLoader
workers through the operating-system page cache and do not count as Python
heap allocations.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data_inspection import DatasetIndex, RecordingPair
from .preprocessing import (
    preprocess_ecg_full,
    preprocess_pcg_full,
    resample_signal,
)
from .utils import ensure_dir, save_json

LOGGER = logging.getLogger(__name__)

CACHE_METADATA_COLUMNS = [
    "record_id",
    "subject",
    "site",
    "fold",
    "label",
    "lvef_percent",
    "ecg_path",
    "pcg_path",
    "ecg_fs",
    "pcg_fs",
    "target_pcg_fs",
    "ecg_offset",
    "ecg_length",
    "pcg_offset",
    "pcg_length",
    "duration_ecg",
    "duration_pcg",
    "status",
    "reason",
]


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------
def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a mono WAV file as float32 in ``[-1, 1]``.

    ``soundfile`` is preferred because it handles PCM/float WAVs and reports
    useful errors.  A SciPy fallback keeps the loader usable on systems where
    libsndfile is unavailable.
    """
    path = Path(path)
    try:
        import soundfile as sf

        x, fs = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        LOGGER.warning("soundfile could not read %s (%s); trying scipy", path, exc)
        from scipy.io import wavfile

        fs, x = wavfile.read(str(path))
        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            x = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            x = x.astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        # The dataset is mono.  If a stereo file appears, average channels
        # rather than silently selecting one.
        x = np.mean(x, axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.ravel(), int(round(float(fs)))


def read_pair(pair: RecordingPair) -> tuple[np.ndarray, np.ndarray, int, int]:
    ecg, fs_ecg = read_wav(pair.ecg_path)
    pcg, fs_pcg = read_wav(pair.pcg_path)
    return ecg, pcg, fs_ecg, fs_pcg


def _wav_info(path: Path) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(str(path))
    return {
        "fs": int(info.samplerate),
        "frames": int(info.frames),
        "duration": float(info.duration),
        "channels": int(info.channels),
    }


def _resampled_length(n_in: int, fs_in: int, fs_out: int) -> int:
    if fs_in == fs_out:
        return int(n_in)
    g = gcd(int(fs_in), int(fs_out))
    up = int(fs_out) // g
    down = int(fs_in) // g
    return int(math.ceil(int(n_in) * up / down))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@dataclass
class SignalCache:
    cache_dir: Path
    metadata: pd.DataFrame
    profile_name: str
    target_pcg_fs: int

    @property
    def ecg_path(self) -> Path:
        return self.cache_dir / "ecg.npy"

    @property
    def pcg_path(self) -> Path:
        return self.cache_dir / "pcg.npy"

    def open_ecg(self) -> np.ndarray:
        return np.load(self.ecg_path, mmap_mode="r")

    def open_pcg(self) -> np.ndarray:
        return np.load(self.pcg_path, mmap_mode="r")

    def get_row(self, record_id: str) -> pd.Series:
        row = self.metadata[self.metadata["record_id"] == record_id]
        if row.empty:
            raise KeyError(f"record_id not found in cache: {record_id}")
        return row.iloc[0]


def cache_paths(cache_root: str | Path, name: str = "main") -> Path:
    return Path(cache_root) / name


def build_signal_cache(
    index: DatasetIndex,
    cfg: dict[str, Any],
    cache_root: str | Path,
    name: str = "main",
    overwrite: bool = False,
    max_recordings: int | None = None,
) -> SignalCache:
    """Build the memmapped ECG/PCG cache for one dataset split.

    The cache is deliberately keyed by the active preprocessing profile.  A
    different profile must use a different ``cache_root`` (the pipeline does
    this automatically).
    """
    from .config import get_profile

    profile = get_profile(cfg)
    target_fs = int(profile["pcg_target_fs"])
    out_dir = cache_paths(cache_root, name)
    metadata_path = out_dir / "metadata.csv"
    config_path = out_dir / "cache_config.json"
    if metadata_path.exists() and config_path.exists() and not overwrite:
        LOGGER.info("Using existing signal cache at %s", out_dir)
        return load_signal_cache(out_dir)

    ensure_dir(out_dir)
    for old in ("ecg.npy", "pcg.npy"):
        p = out_dir / old
        if p.exists():
            p.unlink()

    pairs = index.pairs[:max_recordings] if max_recordings else index.pairs
    if not pairs:
        raise ValueError("No recording pairs to cache.")

    # ------------------------------------------------------------------
    # Pass 1: read headers, validate pairing and compute upper-bound sizes.
    # ------------------------------------------------------------------
    plan: list[dict[str, Any]] = []
    total_ecg = 0
    total_pcg = 0
    for pair in tqdm(pairs, desc="Scanning WAV headers", unit="pair"):
        row: dict[str, Any] = {
            "record_id": f"{pair.subject}_{pair.site}",
            "subject": pair.subject,
            "site": pair.site,
            "fold": pair.fold,
            "label": pair.label,
            "lvef_percent": pair.lvef_percent,
            "ecg_path": str(pair.ecg_path),
            "pcg_path": str(pair.pcg_path),
            "ecg_fs": np.nan,
            "pcg_fs": np.nan,
            "target_pcg_fs": target_fs,
            "ecg_offset": total_ecg,
            "ecg_length": 0,
            "pcg_offset": total_pcg,
            "pcg_length": 0,
            "duration_ecg": np.nan,
            "duration_pcg": np.nan,
            "status": "ok",
            "reason": "",
        }
        try:
            ecg_info = _wav_info(pair.ecg_path)
            pcg_info = _wav_info(pair.pcg_path)
            row.update(
                {
                    "ecg_fs": ecg_info["fs"],
                    "pcg_fs": pcg_info["fs"],
                    "duration_ecg": ecg_info["duration"],
                    "duration_pcg": pcg_info["duration"],
                }
            )
            if ecg_info["fs"] <= 0 or pcg_info["fs"] <= 0:
                raise ValueError("non-positive sampling rate")
            n_pcg = _resampled_length(pcg_info["frames"], pcg_info["fs"], target_fs)
            # +2 samples guards against a one-sample rounding difference in
            # the polyphase resampler.
            row["_ecg_alloc"] = int(ecg_info["frames"])
            row["_pcg_alloc"] = int(n_pcg + 2)
            total_ecg += int(ecg_info["frames"])
            total_pcg += int(n_pcg + 2)
        except Exception as exc:
            row["status"] = "error"
            row["reason"] = f"header/read error: {exc}"
            row["_ecg_alloc"] = 0
            row["_pcg_alloc"] = 0
        plan.append(row)

    # Pre-allocate the two memmaps.  ``np.lib.format.open_memmap`` writes a
    # proper ``.npy`` header, unlike ``np.memmap``, so the files can be opened
    # with ``np.load(..., mmap_mode="r")`` later.
    ecg_mem = np.lib.format.open_memmap(
        out_dir / "ecg.npy", dtype=np.float32, mode="w+", shape=(max(total_ecg, 1),)
    )
    pcg_mem = np.lib.format.open_memmap(
        out_dir / "pcg.npy", dtype=np.float32, mode="w+", shape=(max(total_pcg, 1),)
    )

    # Offsets must be recomputed sequentially because upper-bound allocations
    # are not the same as the actual resampled lengths.
    ecg_cursor = 0
    pcg_cursor = 0
    rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Pass 2: load, preprocess and write.
    # ------------------------------------------------------------------
    for pair, row in tqdm(list(zip(pairs, plan)), desc="Caching signals", unit="pair"):
        row = dict(row)
        row["ecg_offset"] = ecg_cursor
        row["pcg_offset"] = pcg_cursor
        if row["status"] != "error":
            try:
                ecg, pcg, fs_ecg, fs_pcg = read_pair(pair)
                ecg_pre, fs_ecg_out = preprocess_ecg_full(
                    ecg, fs_ecg, cfg.get("preprocessing", {}).get("ecg", {})
                )
                pcg_pre, fs_pcg_out = preprocess_pcg_full(pcg, fs_pcg, profile)
                ecg_mem[ecg_cursor : ecg_cursor + ecg_pre.size] = ecg_pre
                pcg_mem[pcg_cursor : pcg_cursor + pcg_pre.size] = pcg_pre
                row.update(
                    {
                        "ecg_fs": int(fs_ecg_out),
                        "pcg_fs": int(fs_pcg_out),
                        "ecg_length": int(ecg_pre.size),
                        "pcg_length": int(pcg_pre.size),
                        "duration_ecg": float(ecg_pre.size / fs_ecg_out),
                        "duration_pcg": float(pcg_pre.size / fs_pcg_out),
                    }
                )
                # Warn, but keep the cache.  Gating performs the strict check.
                tolerance = float(cfg.get("data", {}).get("duration_tolerance_seconds", 0.25))
                if abs(row["duration_ecg"] - row["duration_pcg"]) > tolerance:
                    row["status"] = "warning"
                    row["reason"] = (
                        f"duration mismatch {abs(row['duration_ecg'] - row['duration_pcg']):.3f}s"
                    )
                ecg_cursor += int(ecg_pre.size)
                pcg_cursor += int(pcg_pre.size)
            except Exception as exc:
                row["status"] = "error"
                row["reason"] = f"preprocessing error: {exc}"
                row["ecg_length"] = 0
                row["pcg_length"] = 0
        rows.append({k: row.get(k) for k in CACHE_METADATA_COLUMNS})

    # Flush and truncate?  NumPy memmaps cannot be resized portably, so the
    # files remain slightly larger than the used region.  Metadata offsets and
    # lengths define the valid data.
    ecg_mem.flush()
    pcg_mem.flush()
    del ecg_mem, pcg_mem

    metadata = pd.DataFrame(rows, columns=CACHE_METADATA_COLUMNS)
    metadata.to_csv(metadata_path, index=False)
    save_json(
        {
            "name": name,
            "profile": profile,
            "target_pcg_fs": target_fs,
            "n_pairs_requested": len(pairs),
            "n_pairs_ok": int((metadata["status"] == "ok").sum()),
            "n_pairs_warning": int((metadata["status"] == "warning").sum()),
            "n_pairs_error": int((metadata["status"] == "error").sum()),
            "total_ecg_samples_used": int(ecg_cursor),
            "total_pcg_samples_used": int(pcg_cursor),
            "total_ecg_samples_allocated": int(total_ecg),
            "total_pcg_samples_allocated": int(total_pcg),
            "created_unix": time.time(),
        },
        config_path,
    )
    LOGGER.info("Signal cache written to %s", out_dir)
    return load_signal_cache(out_dir)


def load_signal_cache(cache_dir: str | Path) -> SignalCache:
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.csv"
    config_path = cache_dir / "cache_config.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Signal cache metadata not found: {metadata_path}\n"
            "Run scripts/prepare_cache.py first."
        )
    metadata = pd.read_csv(metadata_path)
    import json

    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    # Detect caches written by an older version that used a raw memmap (no
    # ``.npy`` header).  Those files cannot be read with np.load(); rebuilding
    # is cheaper than guessing the array length.
    for name in ("ecg.npy", "pcg.npy"):
        path = cache_dir / name
        if path.exists():
            with path.open("rb") as fh:
                magic = fh.read(6)
            if magic != b"\x93NUMPY":
                raise ValueError(
                    f"{path} is not a valid .npy file (missing header).\n"
                    "Rebuild the cache with scripts/prepare_cache.py --overwrite."
                )
    return SignalCache(
        cache_dir=cache_dir,
        metadata=metadata,
        profile_name=str(config.get("profile", {}).get("name", "unknown")),
        target_pcg_fs=int(config.get("target_pcg_fs", 0)),
    )


def iter_cached_pair(
    cache: SignalCache,
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load one ECG/PCG pair from the cache into memory."""
    if int(row["ecg_length"]) <= 0 or int(row["pcg_length"]) <= 0:
        raise ValueError(f"Cache row has no signal data: {row['record_id']}")
    ecg = np.asarray(
        np.load(cache.ecg_path, mmap_mode="r")[
            int(row["ecg_offset"]) : int(row["ecg_offset"]) + int(row["ecg_length"])
        ]
    ).copy()
    pcg = np.asarray(
        np.load(cache.pcg_path, mmap_mode="r")[
            int(row["pcg_offset"]) : int(row["pcg_offset"]) + int(row["pcg_length"])
        ]
    ).copy()
    return ecg, pcg, int(row["ecg_fs"]), int(row["pcg_fs"])
