"""Configuration loading and path resolution.

The project uses a single YAML file (``config/default.yaml``) plus optional
command-line overrides.  Paths in the YAML are relative to the project root
unless they are absolute.  The dataset location is intentionally provided via
environment variables (``HFREF_DATA_ROOT`` and optionally
``HFREF_EXTERNAL_ROOT``) so that raw data never has to live inside the
repository.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .utils import ensure_dir, project_root, resolve_path, save_json

DEFAULT_CONFIG_PATH = project_root() / "config" / "default.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration is missing a required value."""


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``$VAR`` in string values."""
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* without mutating either."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML configuration, apply environment expansion and overrides.

    Parameters
    ----------
    path:
        YAML file to load.  Defaults to ``config/default.yaml``.
    overrides:
        Optional nested dictionary merged on top of the YAML values.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(
            f"Configuration file not found: {cfg_path}\n"
            "Run commands from the project root or pass --config explicitly."
        )
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"Configuration root must be a mapping, got {type(cfg)}")
    cfg = _expand_env(cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)

    # A few values are convenient to have available as absolute paths.
    cfg["_config_path"] = str(cfg_path.resolve())
    cfg["_project_root"] = str(project_root())
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Save the effective configuration next to experiment outputs."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(clean, fh, sort_keys=False, allow_unicode=True)
    return p


def get_profile(cfg: Mapping[str, Any], name: str | None = None) -> dict[str, Any]:
    """Return a flattened preprocessing profile.

    The YAML has a ``preprocessing.profiles`` mapping.  This helper merges the
    selected profile with the common preprocessing section (for example the
    ECG settings) so downstream code receives one dictionary.
    """
    pre = cfg.get("preprocessing", {})
    profiles = pre.get("profiles", {})
    selected = name or pre.get("profile")
    if selected not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise ConfigError(
            f"Unknown preprocessing profile {selected!r}. Available: {available}"
        )
    profile = copy.deepcopy(dict(profiles[selected]))
    profile["name"] = selected
    profile["ecg"] = copy.deepcopy(pre.get("ecg", {}))
    # Values that are common to every profile can be placed at the parent
    # level; they are copied only when the profile does not define them.
    for key, value in pre.items():
        if key in {"profiles", "profile", "ecg"}:
            continue
        profile.setdefault(key, copy.deepcopy(value))
    return profile


def _resolve_with_fallback(
    raw: str | None,
    fallback: str | os.PathLike[str],
    label: str,
) -> Path:
    """Resolve a dataset path, using *fallback* when the env var is unset."""
    if raw is None or str(raw).strip() == "" or str(raw).startswith("${"):
        path = resolve_path(fallback)
    else:
        path = resolve_path(raw)
    if not path.exists():
        raise ConfigError(
            f"{label} not found at: {path}\n"
            "Set the environment variable shown in EXECUTION.md (for example\n"
            "  $env:HFREF_DATA_ROOT = \"C:\\Users\\you\\hfref_data\"\n"
            ") or pass the corresponding --data-root argument."
        )
    return path


def resolve_data_root(
    cfg: Mapping[str, Any],
    override: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the absolute path to the primary HFrEF dataset."""
    if override:
        return _resolve_with_fallback(str(override), override, "Dataset root")
    data = cfg.get("data", {})
    return _resolve_with_fallback(
        data.get("root"),
        data.get("fallback_root", "data/HFrEF_5_folds"),
        "Dataset root",
    )


def resolve_external_root(
    cfg: Mapping[str, Any],
    override: str | os.PathLike[str] | None = None,
    required: bool = False,
) -> Path | None:
    """Return the external community-test-set path, if present.

    The external archive may be extracted next to the primary dataset, or
    inside the project's ``data`` folder.  Several common layouts are tried.
    """
    if override:
        p = resolve_path(override)
        if p.exists():
            return p
        if required:
            raise ConfigError(f"External dataset not found at: {p}")
        return None

    data = cfg.get("data", {})
    candidates: list[Path] = []
    raw_external = data.get("external_root")
    if raw_external and not str(raw_external).startswith("${"):
        candidates.append(resolve_path(raw_external))

    # Try common siblings / sub-folders.
    try:
        primary = resolve_data_root(cfg)
        candidates.extend(
            [
                primary.parent / "HFrEF_community_scenario_test_set",
                primary.parent / "test_set",
                primary / "HFrEF_community_scenario_test_set",
                primary / "test_set",
            ]
        )
    except ConfigError:
        pass
    candidates.extend(
        [
            project_root() / "data" / "HFrEF_community_scenario_test_set",
            project_root() / "data" / "test_set",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        raise ConfigError(
            "External community test set not found.  Extract "
            "HFrEF_community_scenario_test_set.zip and pass --external-root, or "
            "set $env:HFREF_EXTERNAL_ROOT."
        )
    return None


def get_device_name(cfg: Mapping[str, Any]) -> str:
    """Return the requested device string from the config (``auto`` default)."""
    return str(cfg.get("project", {}).get("device", "auto"))


def config_fingerprint(cfg: Mapping[str, Any], keys: list[str] | None = None) -> str:
    """Return a short stable hash of the configuration.

    Used to avoid silently reusing a cache built with different preprocessing
    settings.  Only the requested top-level sections are included.
    """
    if keys is None:
        keys = ["data", "preprocessing", "gating", "model"]
    subset = {k: cfg.get(k) for k in keys if k in cfg}
    blob = json.dumps(subset, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]
