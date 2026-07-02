from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from floodnet_ssl.dataset import FloodNetDataset
from tests.helpers import create_synthetic_track1


class DatasetTest(unittest.TestCase):
    def test_loads_rgb_and_long_mask_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            track1 = create_synthetic_track1(root, flooded=1, non_flooded=0)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "split",
                        "scene_label",
                        "image_path",
                        "mask_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "100",
                        "split": "train",
                        "scene_label": "Flooded",
                        "image_path": "Train/Labeled/Flooded/image/100.jpg",
                        "mask_path": "Train/Labeled/Flooded/mask/100_lab.png",
                    }
                )
            dataset = FloodNetDataset(root, manifest, split="train")
            sample = dataset[0]
            self.assertEqual((3, 6, 8), tuple(sample["image"].shape))
            self.assertEqual((6, 8), tuple(sample["mask"].shape))
            self.assertEqual(torch.float32, sample["image"].dtype)
            self.assertEqual(torch.int64, sample["mask"].dtype)
            self.assertGreaterEqual(float(sample["image"].min()), 0.0)
            self.assertLessEqual(float(sample["image"].max()), 1.0)
            self.assertEqual(track1.name, Path(sample["image_path"]).parents[4].name)

    def test_paired_transform_keeps_image_and_mask_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_synthetic_track1(root, flooded=1, non_flooded=0)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "image_path", "mask_path"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "100",
                        "image_path": "Train/Labeled/Flooded/image/100.jpg",
                        "mask_path": "Train/Labeled/Flooded/mask/100_lab.png",
                    }
                )

            def flip(image: Image.Image, mask: Image.Image | None):
                assert mask is not None
                return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

            sample = FloodNetDataset(root, manifest, transform=flip)[0]
            self.assertEqual(sample["image"].shape[1:], sample["mask"].shape)

    def test_mismatched_rgb_is_aligned_to_mask_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            track1 = create_synthetic_track1(root, flooded=1, non_flooded=0)
            Image.fromarray(
                np.full((7, 10, 3), 128, dtype=np.uint8), mode="RGB"
            ).save(track1 / "Train" / "Labeled" / "Flooded" / "image" / "100.jpg")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "image_path", "mask_path"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "100",
                        "image_path": "Train/Labeled/Flooded/image/100.jpg",
                        "mask_path": "Train/Labeled/Flooded/mask/100_lab.png",
                    }
                )
            sample = FloodNetDataset(root, manifest)[0]
            self.assertEqual((10, 7), sample["original_image_size"])
            self.assertEqual((8, 6), sample["aligned_image_size"])
            self.assertEqual((3, 6, 8), tuple(sample["image"].shape))
            with self.assertRaises(ValueError):
                FloodNetDataset(root, manifest, align_image_to_mask=False)[0]

    def test_manifest_fixed_crop_is_applied_before_tensor_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_synthetic_track1(root, flooded=1, non_flooded=0)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "image_path",
                        "mask_path",
                        "crop_top",
                        "crop_left",
                        "crop_height",
                        "crop_width",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "100",
                        "image_path": "Train/Labeled/Flooded/image/100.jpg",
                        "mask_path": "Train/Labeled/Flooded/mask/100_lab.png",
                        "crop_top": 1,
                        "crop_left": 2,
                        "crop_height": 3,
                        "crop_width": 4,
                    }
                )
            sample = FloodNetDataset(root, manifest)[0]
            self.assertEqual((3, 3, 4), tuple(sample["image"].shape))
            self.assertEqual((3, 4), tuple(sample["mask"].shape))


    def test_rgb_and_rgba_index_masks_are_collapsed_when_channels_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            track1 = create_synthetic_track1(root, flooded=1, non_flooded=0)
            mask_path = track1 / "Train" / "Labeled" / "Flooded" / "mask" / "100_lab.png"
            mask = np.asarray(Image.open(mask_path))
            rgb = np.stack([mask, mask, mask], axis=-1)
            Image.fromarray(rgb, mode="RGB").save(mask_path)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "image_path", "mask_path"])
                writer.writeheader()
                writer.writerow({"sample_id": "100", "image_path": "Train/Labeled/Flooded/image/100.jpg", "mask_path": "Train/Labeled/Flooded/mask/100_lab.png"})
            sample = FloodNetDataset(root, manifest)[0]
            self.assertEqual(torch.int64, sample["mask"].dtype)
            self.assertEqual((6, 8), tuple(sample["mask"].shape))

            rgba = np.concatenate([rgb, np.full((*mask.shape, 1), 255, dtype=np.uint8)], axis=-1)
            Image.fromarray(rgba, mode="RGBA").save(mask_path)
            sample = FloodNetDataset(root, manifest)[0]
            self.assertEqual((6, 8), tuple(sample["mask"].shape))

    def test_rgb_color_mask_is_rejected_without_palette_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            track1 = create_synthetic_track1(root, flooded=1, non_flooded=0)
            mask_path = track1 / "Train" / "Labeled" / "Flooded" / "mask" / "100_lab.png"
            color = np.zeros((6, 8, 3), dtype=np.uint8)
            color[..., 0] = 1
            color[..., 1] = 2
            Image.fromarray(color, mode="RGB").save(mask_path)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "image_path", "mask_path"])
                writer.writeheader()
                writer.writerow({"sample_id": "100", "image_path": "Train/Labeled/Flooded/image/100.jpg", "mask_path": "Train/Labeled/Flooded/mask/100_lab.png"})
            with self.assertRaises(ValueError):
                FloodNetDataset(root, manifest)[0]


if __name__ == "__main__":
    unittest.main()
