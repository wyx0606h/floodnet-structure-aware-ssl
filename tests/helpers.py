from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from floodnet_ssl.constants import CLASS_NAMES, SUPERVISED_ROOT_NAME, TRACK1_ROOT_NAME


def write_class_mapping(path: Path) -> None:
    rows = [
        ["Semi-supervised Image Classification", "", "", "Semi-supervised Semantic Segmentation", ""],
        ["", "", "", "", ""],
        ["Class Index", "Class Name", "", "Class Index", "Class Name"],
        ["0", "Flooded", "", "0", CLASS_NAMES[0]],
        ["1", "Non-flooded", "", "1", CLASS_NAMES[1]],
    ]
    for class_id in range(2, len(CLASS_NAMES)):
        rows.append(["", "", "", str(class_id), CLASS_NAMES[class_id]])
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def create_synthetic_track1(
    parent: Path,
    *,
    flooded: int = 2,
    non_flooded: int = 3,
    unlabeled: int = 1,
    validation: int = 1,
    test: int = 1,
) -> Path:
    root = parent / TRACK1_ROOT_NAME
    root.mkdir(parents=True)
    write_class_mapping(root / "class_mapping.csv")

    next_id = 100
    for scene, count in (("Flooded", flooded), ("Non-Flooded", non_flooded)):
        image_dir = root / "Train" / "Labeled" / scene / "image"
        mask_dir = root / "Train" / "Labeled" / scene / "mask"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for index in range(count):
            sample_id = str(next_id)
            image = np.zeros((6, 8, 3), dtype=np.uint8)
            image[..., 0] = (next_id * 3) % 255
            image[index % 6, index % 8, 1] = 255
            mask = np.full((6, 8), next_id % 10, dtype=np.uint8)
            Image.fromarray(image, mode="RGB").save(image_dir / f"{sample_id}.jpg")
            Image.fromarray(mask, mode="L").save(mask_dir / f"{sample_id}_lab.png")
            next_id += 1

    for split_parts, count in (
        (("Train", "Unlabeled", "image"), unlabeled),
        (("Validation", "image"), validation),
        (("Test", "image"), test),
    ):
        directory = root.joinpath(*split_parts)
        directory.mkdir(parents=True)
        for index in range(count):
            sample_id = str(next_id)
            image = np.zeros((6, 8, 3), dtype=np.uint8)
            image[..., 2] = (next_id * 5) % 255
            image[index % 6, index % 8, 0] = 255
            Image.fromarray(image, mode="RGB").save(directory / f"{sample_id}.jpg")
            next_id += 1
    return root



def create_synthetic_supervised(parent: Path, *, train: int = 2, validation: int = 1, test: int = 1, start_id: int = 100) -> Path:
    root = parent / SUPERVISED_ROOT_NAME
    layout = {
        "train": ("train/train-org-img", "train/train-label-img", train),
        "validation": ("val/val-org-img", "val/val-label-img", validation),
        "test": ("test/test-org-img", "test/test-label-img", test),
    }
    next_id = start_id
    for _split, (image_rel, mask_rel, count) in layout.items():
        image_dir = root / image_rel
        mask_dir = root / mask_rel
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            sample_id = str(next_id)
            image = np.zeros((6, 8, 3), dtype=np.uint8)
            image[..., 0] = (next_id * 7) % 255
            image[index % 6, index % 8, 1] = 255
            mask = np.full((6, 8), next_id % 10, dtype=np.uint8)
            Image.fromarray(image, mode="RGB").save(image_dir / f"{sample_id}.jpg")
            Image.fromarray(mask, mode="L").save(mask_dir / f"{sample_id}_lab.png")
            next_id += 1
    return root
