"""Build FloodNet sup398/full1445 protocol split lists and manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.protocols import build_floodnet_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FloodNet protocol split lists.")
    parser.add_argument("--supervised-root", required=True, type=Path, help="FloodNet-Supervised_v1.0 root or parent directory")
    parser.add_argument("--challenge-root", required=True, type=Path, help="EARTHVISION Track 1 challenge root or parent directory")
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated split files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_floodnet_splits(
        supervised_root=args.supervised_root,
        challenge_root=args.challenge_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
