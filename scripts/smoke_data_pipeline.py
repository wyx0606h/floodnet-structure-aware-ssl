"""Run a non-training CPU smoke test on the canonical FloodNet split."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from floodnet_ssl.constants import IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from floodnet_ssl.dataset import FloodNetDataset  # noqa: E402
from floodnet_ssl.transforms import (  # noqa: E402
    CenterCrop,
    build_supervised_train_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load cropped batches from train/validation/test without training."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def _inspect_split(
    data_root: Path,
    manifest: Path,
    split: str,
    *,
    crop_size: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    transform = (
        build_supervised_train_transform(crop_size=crop_size, seed=seed)
        if split == "train"
        else CenterCrop(crop_size)
    )
    dataset = FloodNetDataset(
        data_root,
        manifest,
        split=split,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(loader))
    images = batch["image"]
    masks = batch["mask"]
    valid_mask_values = ((masks >= 0) & (masks < NUM_CLASSES)) | (
        masks == IGNORE_INDEX
    )
    if images.shape[-2:] != (crop_size, crop_size):
        raise AssertionError(f"Unexpected image crop shape: {images.shape}")
    if masks.shape[-2:] != (crop_size, crop_size):
        raise AssertionError(f"Unexpected mask crop shape: {masks.shape}")
    if not torch.isfinite(images).all():
        raise AssertionError("Image batch contains NaN or Inf")
    if not torch.all(valid_mask_values):
        raise AssertionError("Mask batch contains invalid class IDs")
    return {
        "dataset_size": len(dataset),
        "batch_ids": list(batch["id"]),
        "image_shape": list(images.shape),
        "mask_shape": list(masks.shape),
        "image_dtype": str(images.dtype),
        "mask_dtype": str(masks.dtype),
        "image_min": float(images.min()),
        "image_max": float(images.max()),
        "mask_values": sorted(int(value) for value in torch.unique(masks)),
    }


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke report: {output}")
    if args.crop_size <= 0 or args.batch_size <= 0:
        raise ValueError("crop-size and batch-size must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    report = {
        "passed": True,
        "data_root": str(args.data_root.expanduser().resolve()),
        "manifest": str(args.manifest.expanduser().resolve()),
        "crop_size": args.crop_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "splits": {
            split: _inspect_split(
                args.data_root,
                args.manifest,
                split,
                crop_size=args.crop_size,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            for split in ("train", "validation", "test")
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
