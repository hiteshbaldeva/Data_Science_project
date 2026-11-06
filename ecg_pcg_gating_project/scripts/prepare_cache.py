"""Build the memory-mapped ECG/PCG cache for one preprocessing profile.

The cache stores the full 30 s signals after the profile's resampling and
band-pass filtering.  It is large (a few hundred MB per profile) but makes
window extraction and gating fast and memory-efficient.
"""

from __future__ import annotations

import argparse
import logging

from _common import add_common_arguments, configuration_from_args, initialise

from src.pipeline import cache_path, prepare_dataset

LOGGER = logging.getLogger("scripts.prepare_cache")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--dataset-name", default="main",
                        help="Cache name (default: main)")
    parser.add_argument("--external", action="store_true",
                        help="Cache the external community test set")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild an existing cache")
    parser.add_argument("--max-recordings", type=int, default=None,
                        help="Limit the number of pairs (debugging only)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the per-file header validation pass")
    args = parser.parse_args()

    cfg = configuration_from_args(args)
    initialise(args, cfg)
    dataset_name = args.dataset_name if not args.external else (
        args.dataset_name if args.dataset_name != "main" else "external"
    )

    index, cache = prepare_dataset(
        cfg,
        root=None,
        dataset_name=dataset_name,
        external=bool(args.external),
        overwrite=bool(args.overwrite),
        max_recordings=args.max_recordings,
        validate=not args.skip_validation,
    )
    metadata = cache.metadata
    print()
    print("=" * 78)
    print("SIGNAL CACHE")
    print("=" * 78)
    print(f"cache directory : {cache.cache_dir}")
    print(f"profile         : {cache.profile_name}")
    print(f"PCG target fs   : {cache.target_pcg_fs} Hz")
    print(f"recordings      : {len(metadata)}")
    for status in ("ok", "warning", "error"):
        count = int((metadata["status"] == status).sum())
        print(f"  {status:<8s}: {count}")
    total_sec = float(metadata["duration_pcg"].fillna(0).sum())
    print(f"PCG duration    : {total_sec / 60.0:.1f} min")
    print(f"metadata        : {cache.cache_dir / 'metadata.csv'}")
    print()
    print("Next: scripts/run_gating.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
