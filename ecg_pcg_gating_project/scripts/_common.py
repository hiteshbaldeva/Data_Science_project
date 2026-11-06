"""Shared helpers for the command-line scripts.

Running ``python scripts/foo.py`` puts ``scripts/`` on ``sys.path`` rather than
the project root, so this module inserts the root before importing ``src``.
Every script starts with ``from _common import ...`` for exactly that reason.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping

# Must be set before NumPy/MKL or PyTorch is imported (see src/__init__.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.config import load_config, save_config  # noqa: E402
from src.utils import (  # noqa: E402
    choose_device,
    describe_device,
    ensure_dir,
    print_config_summary,
    set_global_seed,
    setup_logging,
)

LOGGER = logging.getLogger("scripts")


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
        help="YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument("--profile", default=None, help="Preprocessing profile override")
    parser.add_argument("--data-root", default=None, help="Primary HFrEF dataset root")
    parser.add_argument("--external-root", default=None, help="External test-set root")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override project.output_dir (results are still nested by profile/experiment)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def configuration_from_args(
    args: argparse.Namespace,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the YAML configuration and apply CLI overrides."""
    cfg = load_config(args.config)
    if getattr(args, "profile", None):
        cfg.setdefault("preprocessing", {})["profile"] = args.profile
    if getattr(args, "data_root", None):
        cfg.setdefault("data", {})["root"] = args.data_root
    if getattr(args, "external_root", None):
        cfg.setdefault("data", {})["external_root"] = args.external_root
    if getattr(args, "device", None):
        cfg.setdefault("project", {})["device"] = args.device
    if getattr(args, "seed", None) is not None:
        cfg.setdefault("project", {})["seed"] = int(args.seed)
        cfg.setdefault("training", {})["seed"] = int(args.seed)
    if getattr(args, "output_dir", None):
        cfg.setdefault("project", {})["output_dir"] = args.output_dir
    if overrides:
        from src.config import deep_merge

        cfg = deep_merge(cfg, overrides)
    return cfg


def initialise(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    """Set logging and global seeds."""
    setup_logging(logging.DEBUG if getattr(args, "verbose", False) else logging.INFO)
    project = dict(cfg.get("project", {}))
    set_global_seed(
        int(project.get("seed", 42)),
        deterministic=bool(project.get("deterministic", True)),
    )
    print_config_summary(cfg)


def device_from_config(cfg: Mapping[str, Any]) -> torch.device:
    requested = str(cfg.get("project", {}).get("device", "auto")).lower()
    if requested in {"cpu"}:
        device = torch.device("cpu")
    elif requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = choose_device(prefer_cuda=True)
    LOGGER.info("Device: %s", describe_device(device))
    return device


def save_effective_config(cfg: Mapping[str, Any], output_dir: str | Path) -> Path:
    path = ensure_dir(output_dir) / "effective_config.yaml"
    save_config(cfg, path)
    return path
