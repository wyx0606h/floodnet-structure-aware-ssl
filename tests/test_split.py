from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from floodnet_ssl.split import create_versioned_split


def write_inventory(
    path: Path, count: int = 30, *, include_unlabeled: bool = False
) -> None:
    fieldnames = [
        "sample_id",
        "official_split",
        "scene_label",
        "image_path",
        "mask_path",
        "exact_duplicate_group",
    ]
    for class_id in range(10):
        fieldnames.extend(
            [f"class_{class_id}_present", f"class_{class_id}_fraction"]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(count):
            row = {
                "sample_id": f"{index:04d}",
                "official_split": "Train/Labeled",
                "scene_label": "Flooded" if index % 6 == 0 else "Non-Flooded",
                "image_path": f"images/{index:04d}.jpg",
                "mask_path": f"masks/{index:04d}.png",
                "exact_duplicate_group": "pair_0" if index in (0, 1) else "",
            }
            for class_id in range(10):
                present = int((index + class_id) % (class_id + 2) != 0)
                row[f"class_{class_id}_present"] = present
                row[f"class_{class_id}_fraction"] = (
                    present * ((index % 5) + 1) / 100
                )
            writer.writerow(row)
        if include_unlabeled:
            row = {
                "sample_id": "u001",
                "official_split": "Train/Unlabeled",
                "scene_label": "",
                "image_path": "unlabeled/u001.jpg",
                "mask_path": "",
                "exact_duplicate_group": "",
            }
            for class_id in range(10):
                row[f"class_{class_id}_present"] = ""
                row[f"class_{class_id}_fraction"] = ""
            writer.writerow(row)


def write_empty_review(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id_a",
                "sample_id_b",
                "hamming_distance_128",
                "decision",
            ],
        )
        writer.writeheader()


class SplitTest(unittest.TestCase):
    def test_split_is_exact_disjoint_grouped_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = root / "inventory.csv"
            review = root / "near_duplicate_candidates.csv"
            write_inventory(inventory)
            write_empty_review(review)
            first = root / "split_a"
            second = root / "split_b"
            create_versioned_split(
                inventory,
                first,
                split_sizes=(20, 5, 5),
                optimization_steps=200,
            )
            create_versioned_split(
                inventory,
                second,
                split_sizes=(20, 5, 5),
                optimization_steps=200,
            )
            self.assertEqual(
                (first / "manifest.csv").read_text(encoding="utf-8"),
                (second / "manifest.csv").read_text(encoding="utf-8"),
            )
            with (first / "manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            counts = {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "test")
            }
            self.assertEqual(
                {"train": 20, "validation": 5, "test": 5}, counts
            )
            assignments = {row["sample_id"]: row["split"] for row in rows}
            self.assertEqual(assignments["0000"], assignments["0001"])

    def test_unreviewed_candidate_blocks_canonical_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = root / "inventory.csv"
            review = root / "near_duplicate_candidates.csv"
            write_inventory(inventory)
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id_a", "sample_id_b", "decision"],
                )
                writer.writeheader()
                writer.writerow(
                    {"sample_id_a": "0002", "sample_id_b": "0003", "decision": ""}
                )
            with self.assertRaises(ValueError):
                create_versioned_split(
                    inventory,
                    root / "split",
                    split_sizes=(20, 5, 5),
                    optimization_steps=0,
                )

    def test_reviewed_labeled_unlabeled_pair_forces_local_train(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = root / "inventory.csv"
            review = root / "near_duplicate_candidates.csv"
            write_inventory(inventory, include_unlabeled=True)
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id_a", "sample_id_b", "decision"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id_a": "0002",
                        "sample_id_b": "u001",
                        "decision": "same_scene",
                    }
                )
            output = root / "split"
            summary = create_versioned_split(
                inventory,
                output,
                split_sizes=(20, 5, 5),
                optimization_steps=100,
            )
            with (output / "manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                assignments = {
                    row["sample_id"]: row["split"] for row in csv.DictReader(handle)
                }
            self.assertEqual("train", assignments["0002"])
            self.assertEqual(["0002"], summary["forced_train_sample_ids"])


if __name__ == "__main__":
    unittest.main()
