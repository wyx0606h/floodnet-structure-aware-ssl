"""Safely plan or execute the merge extraction of FloodNet Track 1 ZIPs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.archive import (  # noqa: E402
    build_merge_plan,
    execute_merge_plan,
    merge_plan_sha256,
    merge_plan_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect exactly seven FloodNet Track 1 ZIP parts. The default is a "
            "read-only dry run; extraction requires --execute and an exact "
            "--confirm-target path."
        )
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument(
        "--destination-dir",
        type=Path,
        help="Defaults to SOURCE_DIR/FloodNet_Track1_Merged.",
    )
    parser.add_argument("--max-expanded-gib", type=float, default=64.0)
    parser.add_argument("--max-compression-ratio", type=float, default=1000.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm-target",
        type=Path,
        help="Must exactly match the resolved destination when --execute is used.",
    )
    parser.add_argument(
        "--writable-files",
        action="store_true",
        help="Do not remove write bits from extracted files (not recommended).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination_dir or args.source_dir / "FloodNet_Track1_Merged"
    plan = build_merge_plan(
        args.source_dir,
        destination,
        max_expanded_gib=args.max_expanded_gib,
        max_compression_ratio=args.max_compression_ratio,
    )
    summary = merge_plan_summary(plan)
    summary["plan_sha256"] = merge_plan_sha256(plan)
    summary["mode"] = "execute" if args.execute else "dry-run"
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.execute:
        print(
            "\nDry run only: no directory was created and no archive was extracted."
        )
        return 0
    if args.confirm_target is None:
        raise SystemExit("--execute requires --confirm-target")
    result = execute_merge_plan(
        plan,
        confirmed_destination=args.confirm_target,
        set_read_only=not args.writable_files,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
