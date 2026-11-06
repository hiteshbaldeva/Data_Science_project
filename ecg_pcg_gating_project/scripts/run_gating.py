"""Run ECG-guided gating on every cached recording.

Writes one JSON per recording (R peaks, T waves, S1/S2, accepted windows,
rejected windows and reasons), a summary CSV, diagnostic plots, and example
gated PCG windows.  Recordings that fail gating are excluded from E2; they are
never replaced by ungated PCG.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from _common import add_common_arguments, configuration_from_args, initialise

from src.data_loader import load_signal_cache
from src.pipeline import cache_path, gating_path, run_gating
from src.utils import ensure_dir, save_json

LOGGER = logging.getLogger("scripts.run_gating")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--dataset-name", default="main")
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=None,
                        help="Gate only the first N recordings (debugging only)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute existing gating JSON files")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-count", type=int, default=8,
                        help="Number of QC figures to create")
    parser.add_argument("--example-count", type=int, default=3,
                        help="Number of gated-window example figures")
    parser.add_argument("--plot-seconds", type=float, default=10.0,
                        help="Time span shown in QC figures")
    parser.add_argument("--out", default=None, help="Gating output directory")
    args = parser.parse_args()

    cfg = configuration_from_args(args)
    initialise(args, cfg)
    dataset_name = args.dataset_name if not args.external else (
        args.dataset_name if args.dataset_name != "main" else "external"
    )
    cache_dir = cache_path(cfg, dataset_name=dataset_name)
    if not (cache_dir / "metadata.csv").exists():
        LOGGER.error(
            "Signal cache not found at %s.\nRun scripts/prepare_cache.py %s first.",
            cache_dir,
            "--external" if args.external else "",
        )
        return 2

    cache = load_signal_cache(cache_dir)
    output = Path(args.out) if args.out else gating_path(cfg, dataset_name=dataset_name)
    ensure_dir(output)
    summary_df, stats = run_gating(
        cfg,
        cache,
        output_dir=output,
        overwrite=bool(args.overwrite),
        max_recordings=args.max_recordings,
        make_plots=not args.no_plots,
        plot_count=int(args.plot_count),
        example_count=int(args.example_count),
        plot_seconds=float(args.plot_seconds),
    )

    print()
    print("=" * 78)
    print("GATING SUMMARY")
    print("=" * 78)
    print(f"method                    : {stats['method']}")
    print(f"window duration           : {stats['window_seconds']} s")
    print(f"recordings                : {stats['n_recordings']}")
    print(f"accepted recordings       : {stats['n_recording_ok']} "
          f"({stats['fraction_recording_ok']:.1%})")
    print(f"detected R peaks          : {stats['n_rpeaks_total']}")
    print(f"accepted windows          : {stats['n_windows_total']}")
    print(f"rejected windows          : {stats['n_rejected_windows_total']}")
    print(f"mean window success rate  : {stats['mean_success_rate']:.1%}")
    if stats["rejection_reason_counts"]:
        print("window rejection reasons  :")
        for reason, count in sorted(stats["rejection_reason_counts"].items(),
                                    key=lambda kv: kv[1], reverse=True):
            print(f"  {reason:<40s} {count:>6d}")
    print()
    print(f"per-recording JSON        : {output}")
    print(f"summary                   : {output / 'gating_summary.csv'}")
    print(f"QC figures                : {output / 'qc'}")
    print(f"gated window examples     : {output / 'examples'}")
    print()
    print("Next: scripts/prepare_windows.py --experiment e2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
