"""Create the versioned local FloodNet train/validation/test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.split import create_versioned_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic grouped multi-label split from audit inventory.csv."
        )
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--near-duplicate-review",
        type=Path,
        help=(
            "Reviewed candidate CSV. Defaults to near_duplicate_candidates.csv "
            "beside inventory.csv."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--optimization-steps", type=int, default=10000)
    parser.add_argument(
        "--allow-unreviewed-near-duplicates",
        action="store_true",
        help="Development escape hatch; do not use for the canonical v1 split.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = create_versioned_split(
        args.inventory,
        args.output_dir,
        near_duplicate_review=args.near_duplicate_review,
        seed=args.seed,
        optimization_steps=args.optimization_steps,
        allow_unreviewed=args.allow_unreviewed_near_duplicates,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
