"""Build the window table for E1 (ungated), E2 (ECG-gated) or E3 (fusion).

E1 and E2 use exactly the same preprocessing profile and window duration; the
only difference is how the window start is chosen.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _common import add_common_arguments, configuration_from_args, initialise

from src.data_loader import load_signal_cache
from src.pipeline import (
    cache_path,
    gating_path,
    prepare_windows,
    windows_path,
)
from src.utils import save_json

LOGGER = logging.getLogger("scripts.prepare_windows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--experiment", required=True, choices=["e1", "e2", "e3"],
                        help="e1=fixed windows, e2=ECG-gated, e3=gated+ECG")
    parser.add_argument("--dataset-name", default="main")
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--max-windows-per-recording", type=int, default=None,
                        help="Evenly subsample at most N windows per recording")
    parser.add_argument("--gating-dir", default=None,
                        help="Override the gating output directory")
    parser.add_argument("--out", default=None, help="Window-table output directory")
    args = parser.parse_args()

    cfg = configuration_from_args(args)
    initialise(args, cfg)
    dataset_name = args.dataset_name if not args.external else (
        args.dataset_name if args.dataset_name != "main" else "external"
    )
    cache_dir = cache_path(cfg, dataset_name=dataset_name)
    if not (cache_dir / "metadata.csv").exists():
        LOGGER.error("Signal cache not found at %s. Run scripts/prepare_cache.py first.", cache_dir)
        return 2
    cache = load_signal_cache(cache_dir)

    gating_dir = Path(args.gating_dir) if args.gating_dir else gating_path(cfg, dataset_name=dataset_name)
    if args.experiment in {"e2", "e3"} and not gating_dir.exists():
        LOGGER.error(
            "Gating directory not found at %s.\nRun scripts/run_gating.py first.",
            gating_dir,
        )
        return 2

    output = Path(args.out) if args.out else windows_path(
        cfg, args.experiment, dataset_name=dataset_name
    )
    frame, report = prepare_windows(
        cfg,
        cache,
        experiment=args.experiment,
        gating_dir=gating_dir if args.experiment in {"e2", "e3"} else None,
        output_dir=output,
        max_windows_per_recording=args.max_windows_per_recording,
    )

    print()
    print("=" * 78)
    print(f"WINDOW INDEX: {args.experiment.upper()}")
    print("=" * 78)
    print(f"profile                    : {report['profile']}")
    print(f"window / hop (s)           : {report['window_seconds']} / {report['hop_seconds']}")
    print(f"window samples (pre-DWT)   : {report['window_samples']}")
    print(f"recordings used            : {report['n_recordings_used']}")
    print(f"recordings skipped         : {report['n_recordings_skipped']}")
    print(f"windows                    : {report['n_windows']}")
    print(f"subjects with windows      : {report.get('n_subjects_with_windows', 0)}")
    if report.get("label_counts"):
        print(f"positive windows           : {report['label_counts']['positive_windows']}")
        print(f"negative windows           : {report['label_counts']['negative_windows']}")
    if report.get("skipped"):
        print("first skipped recordings   :")
        for item in report["skipped"][:10]:
            print(f"  {item}")
    print()
    print(f"window table               : {output / 'windows.csv'}")
    save_json(vars(args), output / "prepare_windows_arguments.json")
    print()
    print("Next: scripts/train.py --experiment " + args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
