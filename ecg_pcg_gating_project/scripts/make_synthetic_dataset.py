"""Create a synthetic HFrEF-like dataset for tests and smoke runs.

The synthetic data are not a substitute for the real recordings.  They let a
new machine verify the complete pipeline (cache, gating, windows, training,
evaluation) before the real dataset is processed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import PROJECT_ROOT  # noqa: F401  (ensures the project root is importable)

from src.synthetic import write_synthetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic_smoke")
    parser.add_argument("--subjects", type=int, default=10)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()

    root = Path(args.output)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    write_synthetic_dataset(
        root,
        n_subjects=int(args.subjects),
        duration=float(args.duration),
        external=bool(args.external),
        seed=int(args.seed),
    )
    print(f"Synthetic dataset written to: {root}")
    print(f"  subjects : {args.subjects}")
    print(f"  duration : {args.duration} s")
    print(f"  layout   : {'external test_set' if args.external else 'fold_0 ... fold_4'}")
    print()
    print("This dataset is for pipeline verification only. Never report its metrics as results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
