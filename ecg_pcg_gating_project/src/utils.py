"""Small shared helpers used across the project.

Keeping these functions here avoids repeating the same boilerplate in the
scripts and makes behaviour (seeds, JSON serialisation, logging) consistent.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def project_root() -> Path:
    """Return the repository root (the folder that contains ``src``)."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | os.PathLike[str], base: Path | None = None) -> Path:
    """Resolve a path, expanding ``~`` and environment variables.

    Relative paths are interpreted relative to *base* (the project root by
    default), never relative to the current working directory.  This is
    important because the scripts may be launched from anywhere.
    """
    base = Path(base) if base is not None else project_root()
    raw = os.path.expandvars(str(path))
    raw = os.path.expanduser(raw)
    p = Path(raw)
    return p if p.is_absolute() else (base / p).resolve()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create *path* as a directory if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Serialisation (NumPy types are common in this project)
# ---------------------------------------------------------------------------
def _to_builtin(value: Any) -> Any:
    """Recursively convert NumPy / Path values into JSON-serialisable ones."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def save_json(obj: Any, path: str | os.PathLike[str], indent: int = 2) -> Path:
    """Write *obj* as UTF-8 JSON, creating parent directories."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(_to_builtin(obj), fh, indent=indent, ensure_ascii=False)
        fh.write("\n")
    return p


def load_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_text(text: str, path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (if installed).

    ``deterministic=True`` asks cuDNN to use deterministic kernels.  This can
    be slightly slower but makes results easier to reproduce.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    except Exception:  # pragma: no cover - torch is a required dependency
        pass


def choose_device(prefer_cuda: bool = True) -> "torch.device":
    """Return CUDA when available and requested, otherwise CPU."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: "torch.device") -> str:
    import torch

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return (
            f"CUDA device {idx}: {props.name}, "
            f"{props.total_memory / (1024 ** 3):.2f} GB VRAM"
        )
    return "CPU"


# ---------------------------------------------------------------------------
# Logging / timing
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Context manager that logs how long a block took."""
    log = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("%s finished in %.1f s", label, elapsed)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def safe_div(numerator: float, denominator: float, default: float = float("nan")) -> float:
    """Divide without raising when the denominator is zero."""
    return float(numerator) / float(denominator) if denominator else default


def chunks(sequence: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield lists of at most *size* items from *sequence*."""
    bucket: list[Any] = []
    for item in sequence:
        bucket.append(item)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def print_config_summary(cfg: Any, logger: logging.Logger | None = None) -> None:
    """Log the handful of settings that most affect a run.

    Printing the whole YAML every time would bury the progress messages, so
    only the values that change results (profile, device, seed, split policy)
    are shown.  The complete effective configuration is saved next to the run
    outputs by the scripts.
    """
    log = logger or logging.getLogger("config")
    cfg = dict(cfg or {})
    project = dict(cfg.get("project", {}))
    preprocessing = dict(cfg.get("preprocessing", {}))
    training = dict(cfg.get("training", {}))
    data = dict(cfg.get("data", {}))
    gating = dict(cfg.get("gating", {}))
    active = str(preprocessing.get("profile", ""))
    profile_cfg = dict(preprocessing.get("profiles", {}).get(active, {}))
    log.info(
        "Config | project=%s | profile=%s | device=%s | seed=%s",
        project.get("name", "?"),
        active or "?",
        project.get("device", "auto"),
        project.get("seed", "?"),
    )
    log.info(
        "Config | folds=%s | window=%.2fs hop=%.2fs | gating=%s",
        data.get("folds", []),
        float(profile_cfg.get("window_seconds", 0.0) or 0.0),
        float(profile_cfg.get("hop_seconds", 0.0) or 0.0),
        gating.get("method", "?"),
    )
    log.info(
        "Config | epochs=%s batch=%s lr=%s optimizer=%s",
        training.get("epochs", "?"),
        training.get("batch_size", "?"),
        training.get("lr", "?"),
        training.get("optimizer", "?"),
    )


def human_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"
