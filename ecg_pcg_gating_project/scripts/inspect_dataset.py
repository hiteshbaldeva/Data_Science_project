"""Inspect the dataset without training anything.

Checks folder discovery, label parsing, WAV headers, ECG/PCG pairing, fold
counts, and (optionally) the ECG-PCG synchronisation diagnostic.  This is the
first script to run on a new machine.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    add_common_arguments,
    configuration_from_args,
    initialise,
)

from src.config import resolve_data_root, resolve_external_root
from src.data_inspection import (
    check_synchronization,
    discover_dataset,
    save_index,
    validate_dataset,
)
from src.data_loader import read_pair
from src.utils import ensure_dir, save_json

LOGGER = logging.getLogger("scripts.inspect")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--external", action="store_true", help="Inspect the external set")
    parser.add_argument("--max-recordings", type=int, default=None,
                        help="Validate at most this many pairs (default: all)")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip the ECG/PCG lag diagnostic")
    parser.add_argument("--sync-examples", type=int, default=5,
                        help="Number of pairs used for the lag diagnostic")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    cfg = configuration_from_args(args)
    initialise(args, cfg)

    if args.external:
        root = resolve_external_root(cfg, required=True)
    else:
        root = resolve_data_root(cfg)
    output = ensure_dir(
        args.out
        or (
            Path(cfg["_project_root"])
            / "results"
            / "inspection"
            / ("external" if args.external else "main")
        )
    )
    LOGGER.info("Inspecting %s", root)

    index = discover_dataset(
        root,
        lvef_threshold_percent=float(cfg.get("data", {}).get("lvef_threshold_percent", 40.0)),
        external=bool(args.external),
    )
    save_index(index, output)
    summary = index.summary()
    save_json(summary, output / "dataset_index_summary.json")

    data_cfg = dict(cfg.get("data", {}))
    frame, validation = validate_dataset(
        index,
        expected_ecg_fs=int(data_cfg.get("expected_ecg_fs", 500)),
        expected_pcg_fs=int(data_cfg.get("expected_pcg_fs", 4000)),
        min_duration_seconds=float(data_cfg.get("min_duration_seconds", 10.0)),
        max_duration_seconds=float(data_cfg.get("max_duration_seconds", 90.0)),
        duration_tolerance_seconds=float(data_cfg.get("duration_tolerance_seconds", 0.25)),
        max_recordings=args.max_recordings,
    )
    frame.to_csv(output / "recording_validation.csv", index=False)
    save_json(validation, output / "validation_summary.json")

    print()
    print("=" * 78)
    print("DATASET SUMMARY")
    print("=" * 78)
    print(f"root                 : {summary['root']}")
    print(f"external             : {summary['external']}")
    print(f"subjects             : {summary['n_subjects']}")
    print(f"ECG/PCG pairs        : {summary['n_pairs']}")
    print(f"labelled subjects    : {summary['labels']['labelled']}")
    print(f"positive (LVEF<=40%) : {summary['labels']['positive']}")
    print(f"negative (LVEF>40%)  : {summary['labels']['negative']}")
    print("fold counts          :")
    for fold, counts in sorted(summary["folds"].items()):
        print(f"  fold {fold:<2s} subjects={counts['subjects']:>3d} "
              f"pos={counts['positive']:>2d} neg={counts['negative']:>3d}")
    print(f"unpaired             : {summary['n_unpaired']}")
    print(f"unparsed WAV files   : {summary['n_unparsed']}")
    print(f"header errors        : {validation['n_errors']}")
    print(f"header warnings      : {validation['n_warnings']}")
    if summary["warnings"]:
        print("warnings:")
        for warning in summary["warnings"][:20]:
            print(f"  - {warning}")
        if len(summary["warnings"]) > 20:
            print(f"  ... {len(summary['warnings']) - 20} more (see JSON)")

    if not args.skip_sync and index.pairs:
        print()
        print("ECG-PCG synchronisation diagnostic (expected lag ~0 s)")
        diagnostics = []
        for pair in index.pairs[: max(0, args.sync_examples)]:
            try:
                ecg, pcg, fs_ecg, fs_pcg = read_pair(pair)
                result = check_synchronization(ecg, pcg, fs_ecg, fs_pcg)
                result.update(
                    {"subject": pair.subject, "site": pair.site,
                     "ecg_fs": fs_ecg, "pcg_fs": fs_pcg}
                )
                diagnostics.append(result)
                lag = result.get("best_lag_seconds")
                contrast = result.get("contrast")
                print(
                    f"  {pair.subject}_{pair.site}: lag="
                    f"{lag if lag is None else f'{lag:+.3f}s'} "
                    f"contrast={contrast if contrast is None else f'{contrast:.2f}'} "
                    f"peaks={result.get('n_rpeaks')}"
                )
            except Exception as exc:
                LOGGER.warning("Sync diagnostic failed for %s_%s: %s",
                               pair.subject, pair.site, exc)
        save_json(diagnostics, output / "synchronisation_diagnostic.json")

    save_json(vars(args), output / "inspect_arguments.json")
    print()
    print(f"Outputs written to: {output}")
    print("Next: scripts/prepare_cache.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
