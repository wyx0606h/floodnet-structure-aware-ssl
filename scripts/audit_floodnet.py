"""Audit an extracted FloodNet Track 1 data root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.audit import audit_dataset  # noqa: E402
from floodnet_ssl.constants import EXPECTED_COUNTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate FloodNet layout, labels, counts, hashes, and duplicates."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--near-duplicate-hamming", type=int, default=6)
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Disable official-count enforcement; intended only for development fixtures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit_dataset(
        args.data_root,
        args.output_dir,
        expected_counts=None if args.allow_count_mismatch else EXPECTED_COUNTS,
        near_duplicate_hamming=args.near_duplicate_hamming,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
